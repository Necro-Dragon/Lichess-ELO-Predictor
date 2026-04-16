"""Build a compressed rapid-game corpus keyed by 40-ply board vectors."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import chess.pgn
import numpy as np

from .position import BOARD_SIZE, ChessPosition


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-dir",
        default="data/rapid_2026-03_sample_1500000",
        help="Directory containing the sampled rapid PGN chunks and manifest.",
    )
    parser.add_argument(
        "--outdir",
        default="data/rapid_2026-03_sample_1500000_40ply_corpus",
        help="Directory to write the compressed corpus.",
    )
    parser.add_argument(
        "--target-plies",
        type=int,
        default=40,
        help="Ply depth to encode, using the final position when a game ends earlier.",
    )
    return parser.parse_args(argv)


def load_sample_manifest(sample_dir: Path) -> dict:
    manifest_path = sample_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("chunk_files"):
        raise ValueError(f"sample manifest at {manifest_path} does not list chunk files")
    return manifest


def iter_games_from_zst(chunk_path: Path) -> Iterator[chess.pgn.Game]:
    process = subprocess.Popen(
        ["zstd", "-dc", str(chunk_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    assert process.stderr is not None

    try:
        while True:
            game = chess.pgn.read_game(process.stdout)
            if game is None:
                break
            if game.errors:
                raise ValueError(f"PGN parse errors encountered in {chunk_path}")
            yield game
    finally:
        process.stdout.close()
        stderr_text = process.stderr.read()
        process.stderr.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"zstd failed while reading {chunk_path} with exit code {return_code}: {stderr_text.strip()}"
            )


def parse_rating(value: str, *, field_name: str) -> int:
    value = value.strip()
    if not value or not value.isdigit():
        raise ValueError(f"{field_name} is missing or non-numeric: {value!r}")
    rating = int(value)
    info = np.iinfo(np.int16)
    if rating < info.min or rating > info.max:
        raise ValueError(f"{field_name}={rating} does not fit in int16")
    return rating


def extract_position_vector(game: chess.pgn.Game, target_plies: int) -> tuple[np.ndarray, int]:
    if target_plies <= 0:
        raise ValueError("target_plies must be positive")

    board = game.board()
    recorded_position = ChessPosition.from_board(board)
    plies_recorded = 0

    for move in game.mainline_moves():
        board.push(move)
        plies_recorded += 1
        recorded_position = ChessPosition.from_board(board)
        if plies_recorded >= target_plies:
            break

    return recorded_position.as_array(), plies_recorded


def _trim_array(array: np.ndarray, row_count: int) -> np.ndarray:
    if row_count == len(array):
        return array
    return np.ascontiguousarray(array[:row_count])


def _trim_matrix(matrix: np.ndarray, row_count: int) -> np.ndarray:
    if row_count == len(matrix):
        return matrix
    return np.ascontiguousarray(matrix[:row_count, :])


def build_rapid_40ply_corpus(
    *,
    sample_dir: Path,
    outdir: Path,
    target_plies: int,
) -> dict[str, Path]:
    manifest = load_sample_manifest(sample_dir)
    chunk_files = manifest["chunk_files"]
    expected_rows = int(manifest["sample_size"])
    if expected_rows <= 0:
        raise ValueError("sample manifest sample_size must be positive")

    white_elo = np.empty(expected_rows, dtype=np.int16)
    black_elo = np.empty(expected_rows, dtype=np.int16)
    position_vectors = np.empty((expected_rows, BOARD_SIZE), dtype=np.uint8)
    plies_recorded = np.empty(expected_rows, dtype=np.uint16)

    row_count = 0
    reached_target_count = 0
    padded_short_game_count = 0

    for chunk_index, chunk_name in enumerate(chunk_files, start=1):
        chunk_path = sample_dir / chunk_name
        if not chunk_path.is_file():
            raise FileNotFoundError(f"sample chunk is missing: {chunk_path}")

        for game in iter_games_from_zst(chunk_path):
            if row_count >= expected_rows:
                raise ValueError(
                    f"sample manifest expected {expected_rows} rows, but more games were found"
                )

            headers = game.headers
            white_elo[row_count] = parse_rating(headers.get("WhiteElo", ""), field_name="WhiteElo")
            black_elo[row_count] = parse_rating(headers.get("BlackElo", ""), field_name="BlackElo")
            vector, recorded_plies = extract_position_vector(game, target_plies)
            position_vectors[row_count] = vector
            plies_recorded[row_count] = recorded_plies

            if recorded_plies >= target_plies:
                reached_target_count += 1
            else:
                padded_short_game_count += 1

            row_count += 1

        print(
            f"Processed chunk {chunk_index}/{len(chunk_files)} ({row_count:,} games total)",
            file=sys.stderr,
        )

    white_elo = _trim_array(white_elo, row_count)
    black_elo = _trim_array(black_elo, row_count)
    position_vectors = _trim_matrix(position_vectors, row_count)
    plies_recorded = _trim_array(plies_recorded, row_count)

    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "games_40ply_corpus.npz"
    manifest_path = outdir / "manifest.json"

    np.savez_compressed(
        npz_path,
        white_elo=white_elo,
        black_elo=black_elo,
        position_vectors=position_vectors,
        plies_recorded=plies_recorded,
    )

    output_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sample_dir": str(sample_dir.resolve()),
        "source_manifest_path": str((sample_dir / "manifest.json").resolve()),
        "source_chunk_files": list(chunk_files),
        "output_npz_path": str(npz_path.resolve()),
        "target_plies": target_plies,
        "row_count": row_count,
        "reached_target_plies_count": reached_target_count,
        "padded_short_game_count": padded_short_game_count,
        "arrays": {
            "white_elo": {
                "dtype": str(white_elo.dtype),
                "shape": list(white_elo.shape),
            },
            "black_elo": {
                "dtype": str(black_elo.dtype),
                "shape": list(black_elo.shape),
            },
            "position_vectors": {
                "dtype": str(position_vectors.dtype),
                "shape": list(position_vectors.shape),
            },
            "plies_recorded": {
                "dtype": str(plies_recorded.dtype),
                "shape": list(plies_recorded.shape),
            },
        },
        "notes": [
            "Each row stores WhiteElo, BlackElo, and a chess_distance-ready occupancy vector.",
            "Rows preserve the original order of games across sample chunk files.",
            "If a game ended before target_plies, the stored vector is the final position and plies_recorded is less than target_plies.",
            "position_vectors are directly comparable with chess_distance.ChessPosition.as_array().",
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(output_manifest, handle, indent=2)
        handle.write("\n")

    return {
        "npz": npz_path,
        "manifest": manifest_path,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_rapid_40ply_corpus(
        sample_dir=Path(args.sample_dir),
        outdir=Path(args.outdir),
        target_plies=args.target_plies,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
