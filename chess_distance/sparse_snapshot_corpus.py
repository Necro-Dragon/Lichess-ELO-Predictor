"""Build and load a sparse-snapshot rapid-game corpus for neural training."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import numpy as np
import chess.pgn
from tqdm.auto import tqdm

from .position import ChessPosition
from .rapid_corpus import iter_games_from_zst, load_sample_manifest, parse_rating


TRAIN_SPLIT_ID = 0
VAL_SPLIT_ID = 1
TEST_SPLIT_ID = 2
SPLIT_NAME_TO_ID = {
    "train": TRAIN_SPLIT_ID,
    "val": VAL_SPLIT_ID,
    "test": TEST_SPLIT_ID,
}
SPLIT_ID_TO_NAME = {value: key for key, value in SPLIT_NAME_TO_ID.items()}


@dataclass(frozen=True, slots=True)
class RatingBandSpec:
    """Train-derived fixed-width rating bands with underflow and overflow bins."""

    band_width: int
    lower_bound: int
    upper_bound: int

    @property
    def class_count(self) -> int:
        interior_band_count = (self.upper_bound - self.lower_bound) // self.band_width
        return interior_band_count + 2

    def encode_rating(self, rating: int) -> int:
        if rating < self.lower_bound:
            return 0
        if rating >= self.upper_bound:
            return self.class_count - 1
        return 1 + ((rating - self.lower_bound) // self.band_width)

    def encode_many(self, ratings: np.ndarray) -> np.ndarray:
        encoded = np.empty(len(ratings), dtype=np.int16)
        for index, rating in enumerate(ratings.tolist()):
            encoded[index] = self.encode_rating(int(rating))
        return encoded

    def band_label(self, band_id: int) -> str:
        if band_id == 0:
            return f"<{self.lower_bound}"
        if band_id == self.class_count - 1:
            return f">={self.upper_bound}"
        start = self.lower_bound + (band_id - 1) * self.band_width
        end = start + self.band_width - 1
        return f"{start}-{end}"

    def band_center(self, band_id: int) -> float:
        if band_id == 0:
            return float(self.lower_bound - (self.band_width / 2))
        if band_id == self.class_count - 1:
            return float(self.upper_bound + (self.band_width / 2))
        start = self.lower_bound + (band_id - 1) * self.band_width
        return float(start + (self.band_width / 2))

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_width": self.band_width,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "class_count": self.class_count,
            "labels": [self.band_label(band_id) for band_id in range(self.class_count)],
            "centers": [self.band_center(band_id) for band_id in range(self.class_count)],
        }

    @classmethod
    def from_train_ratings(cls, ratings: np.ndarray, band_width: int) -> "RatingBandSpec":
        if len(ratings) == 0:
            raise ValueError("cannot derive rating bands from an empty training set")
        lower_bound = (int(ratings.min()) // band_width) * band_width
        upper_bound = ((int(ratings.max()) + band_width - 1) // band_width) * band_width
        if upper_bound <= lower_bound:
            upper_bound = lower_bound + band_width
        return cls(
            band_width=band_width,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RatingBandSpec":
        return cls(
            band_width=int(payload["band_width"]),
            lower_bound=int(payload["lower_bound"]),
            upper_bound=int(payload["upper_bound"]),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-dir",
        default="data/rapid_2026-03_sample_1500000",
        help="Directory containing the sampled rapid PGN chunks and manifest.",
    )
    parser.add_argument(
        "--outdir",
        default="data/rapid_2026-03_sparse_snapshot_200000",
        help="Directory to write the sparse-snapshot corpus.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200_000,
        help="How many games to draw uniformly from the 1.5M rapid sample.",
    )
    parser.add_argument(
        "--snapshot-step",
        type=int,
        default=7,
        help="Store positions at ply 0, N, 2N, ... and always include the final position once.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260416,
        help="RNG seed used for subset sampling and split assignment.",
    )
    parser.add_argument(
        "--band-width",
        type=int,
        default=200,
        help="Width of the train-derived rating bands.",
    )
    parser.add_argument(
        "--buffer-games",
        type=int,
        default=2048,
        help="How many games to accumulate before flushing snapshot buffers.",
    )
    return parser.parse_args(argv)


def _snapshot_castling_rights(board: chess.Board) -> np.ndarray:
    return np.array(
        [
            int(board.has_kingside_castling_rights(chess.WHITE)),
            int(board.has_queenside_castling_rights(chess.WHITE)),
            int(board.has_kingside_castling_rights(chess.BLACK)),
            int(board.has_queenside_castling_rights(chess.BLACK)),
        ],
        dtype=np.uint8,
    )


def _snapshot_en_passant_file(board: chess.Board) -> int:
    if board.ep_square is None:
        return 0
    return chess.square_file(board.ep_square) + 1


def _record_snapshot(board: chess.Board) -> tuple[np.ndarray, int, np.ndarray, int]:
    position = ChessPosition.from_board(board).as_array().copy()
    side_to_move = int(board.turn)
    castling_rights = _snapshot_castling_rights(board)
    en_passant_file = _snapshot_en_passant_file(board)
    return position, side_to_move, castling_rights, en_passant_file


def extract_sparse_game_snapshots(
    game: chess.pgn.Game,
    snapshot_step: int,
) -> dict[str, np.ndarray | int]:
    if snapshot_step <= 0:
        raise ValueError("snapshot_step must be positive")

    board = game.board()
    board_codes: list[np.ndarray] = []
    side_to_move: list[int] = []
    castling_rights: list[np.ndarray] = []
    en_passant_file: list[int] = []
    snapshot_plies: list[int] = []

    initial_board, initial_turn, initial_castling, initial_ep = _record_snapshot(board)
    board_codes.append(initial_board)
    side_to_move.append(initial_turn)
    castling_rights.append(initial_castling)
    en_passant_file.append(initial_ep)
    snapshot_plies.append(0)

    total_plies = 0
    for move in game.mainline_moves():
        board.push(move)
        total_plies += 1
        if total_plies % snapshot_step == 0:
            board_state, turn, castling, ep_file = _record_snapshot(board)
            board_codes.append(board_state)
            side_to_move.append(turn)
            castling_rights.append(castling)
            en_passant_file.append(ep_file)
            snapshot_plies.append(total_plies)

    if snapshot_plies[-1] != total_plies:
        final_board, final_turn, final_castling, final_ep = _record_snapshot(board)
        board_codes.append(final_board)
        side_to_move.append(final_turn)
        castling_rights.append(final_castling)
        en_passant_file.append(final_ep)
        snapshot_plies.append(total_plies)

    return {
        "board_codes": np.ascontiguousarray(np.stack(board_codes, axis=0), dtype=np.uint8),
        "side_to_move": np.ascontiguousarray(np.array(side_to_move, dtype=np.uint8)),
        "castling_rights": np.ascontiguousarray(np.stack(castling_rights, axis=0), dtype=np.uint8),
        "en_passant_file": np.ascontiguousarray(np.array(en_passant_file, dtype=np.uint8)),
        "snapshot_plies": np.ascontiguousarray(np.array(snapshot_plies, dtype=np.uint16)),
        "final_plies": total_plies,
    }


def _allocate_stratified_counts(count: int) -> tuple[int, int, int]:
    train_count = int(round(count * 0.80))
    val_count = int(round(count * 0.10))
    if train_count + val_count > count:
        val_count = max(0, count - train_count)
    test_count = count - train_count - val_count
    return train_count, val_count, test_count


def stratified_split_ids(
    average_ratings: np.ndarray,
    *,
    band_width: int,
    seed: int,
) -> np.ndarray:
    rng = random.Random(seed)
    split_ids = np.empty(len(average_ratings), dtype=np.uint8)
    bucket_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, average_rating in enumerate(average_ratings.tolist()):
        bucket_id = int(average_rating) // band_width
        bucket_to_indices[bucket_id].append(index)

    for bucket_indices in bucket_to_indices.values():
        rng.shuffle(bucket_indices)
        train_count, val_count, _test_count = _allocate_stratified_counts(len(bucket_indices))
        for index in bucket_indices[:train_count]:
            split_ids[index] = TRAIN_SPLIT_ID
        for index in bucket_indices[train_count : train_count + val_count]:
            split_ids[index] = VAL_SPLIT_ID
        for index in bucket_indices[train_count + val_count :]:
            split_ids[index] = TEST_SPLIT_ID
    return split_ids


def resolve_corpus_dir(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.name == "manifest.json":
        return path.parent
    raise FileNotFoundError(f"corpus path must be a directory or manifest.json: {path}")


def load_sparse_snapshot_manifest(path: Path) -> dict[str, Any]:
    corpus_dir = resolve_corpus_dir(path)
    manifest_path = corpus_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_sparse_snapshot_arrays(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    corpus_dir = resolve_corpus_dir(path)
    manifest = load_sparse_snapshot_manifest(corpus_dir)
    npz_name = manifest.get("npz_file", "sparse_snapshot_corpus.npz")
    with np.load(corpus_dir / npz_name, allow_pickle=False) as bundle:
        arrays = {key: bundle[key] for key in bundle.files}
    return manifest, arrays


def _flush_game_buffers(
    board_chunks: list[np.ndarray],
    turn_chunks: list[np.ndarray],
    castling_chunks: list[np.ndarray],
    en_passant_chunks: list[np.ndarray],
    ply_chunks: list[np.ndarray],
    board_master: list[np.ndarray],
    turn_master: list[np.ndarray],
    castling_master: list[np.ndarray],
    en_passant_master: list[np.ndarray],
    ply_master: list[np.ndarray],
) -> None:
    if not board_chunks:
        return
    board_master.append(np.ascontiguousarray(np.concatenate(board_chunks, axis=0), dtype=np.uint8))
    turn_master.append(np.ascontiguousarray(np.concatenate(turn_chunks, axis=0), dtype=np.uint8))
    castling_master.append(np.ascontiguousarray(np.concatenate(castling_chunks, axis=0), dtype=np.uint8))
    en_passant_master.append(np.ascontiguousarray(np.concatenate(en_passant_chunks, axis=0), dtype=np.uint8))
    ply_master.append(np.ascontiguousarray(np.concatenate(ply_chunks, axis=0), dtype=np.uint16))
    board_chunks.clear()
    turn_chunks.clear()
    castling_chunks.clear()
    en_passant_chunks.clear()
    ply_chunks.clear()


def build_sparse_snapshot_corpus(
    *,
    sample_dir: Path,
    outdir: Path,
    sample_size: int,
    snapshot_step: int,
    seed: int,
    band_width: int,
    buffer_games: int,
) -> dict[str, Path]:
    manifest = load_sample_manifest(sample_dir)
    source_game_count = int(manifest["sample_size"])
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > source_game_count:
        raise ValueError(
            f"sample_size={sample_size:,} exceeds source sample size {source_game_count:,}"
        )

    rng = random.Random(seed)
    selected_source_indices = sorted(rng.sample(range(source_game_count), sample_size))

    white_elo = np.empty(sample_size, dtype=np.int16)
    black_elo = np.empty(sample_size, dtype=np.int16)
    offsets = np.empty(sample_size, dtype=np.int64)
    lengths = np.empty(sample_size, dtype=np.int32)
    final_plies = np.empty(sample_size, dtype=np.uint16)
    source_game_index = np.empty(sample_size, dtype=np.int32)

    sites: list[str] = [""] * sample_size
    openings: list[str] = [""] * sample_size
    ecos: list[str] = [""] * sample_size

    board_chunks: list[np.ndarray] = []
    turn_chunks: list[np.ndarray] = []
    castling_chunks: list[np.ndarray] = []
    en_passant_chunks: list[np.ndarray] = []
    ply_chunks: list[np.ndarray] = []

    board_master: list[np.ndarray] = []
    turn_master: list[np.ndarray] = []
    castling_master: list[np.ndarray] = []
    en_passant_master: list[np.ndarray] = []
    ply_master: list[np.ndarray] = []

    running_snapshot_offset = 0
    selected_pointer = 0
    selected_count = 0

    progress = tqdm(total=sample_size, desc="Extracting sparse snapshots", unit="game")
    try:
        for chunk_name in manifest["chunk_files"]:
            chunk_path = sample_dir / chunk_name
            for game in iter_games_from_zst(chunk_path):
                if selected_pointer >= len(selected_source_indices):
                    break

                current_source_index = selected_count
                selected_count += 1
                if current_source_index != selected_source_indices[selected_pointer]:
                    continue

                sparse = extract_sparse_game_snapshots(game, snapshot_step)
                local_index = selected_pointer
                sequence_length = int(len(sparse["snapshot_plies"]))

                offsets[local_index] = running_snapshot_offset
                lengths[local_index] = sequence_length
                final_plies[local_index] = int(sparse["final_plies"])
                source_game_index[local_index] = current_source_index
                running_snapshot_offset += sequence_length

                headers = game.headers
                white_elo[local_index] = parse_rating(headers.get("WhiteElo", ""), field_name="WhiteElo")
                black_elo[local_index] = parse_rating(headers.get("BlackElo", ""), field_name="BlackElo")
                sites[local_index] = headers.get("Site", "").strip()
                openings[local_index] = headers.get("Opening", "").strip()
                ecos[local_index] = headers.get("ECO", "").strip()

                board_chunks.append(sparse["board_codes"])
                turn_chunks.append(sparse["side_to_move"])
                castling_chunks.append(sparse["castling_rights"])
                en_passant_chunks.append(sparse["en_passant_file"])
                ply_chunks.append(sparse["snapshot_plies"])
                if len(board_chunks) >= buffer_games:
                    _flush_game_buffers(
                        board_chunks,
                        turn_chunks,
                        castling_chunks,
                        en_passant_chunks,
                        ply_chunks,
                        board_master,
                        turn_master,
                        castling_master,
                        en_passant_master,
                        ply_master,
                    )

                selected_pointer += 1
                progress.update(1)
            if selected_pointer >= len(selected_source_indices):
                break
    finally:
        progress.close()

    if selected_pointer != sample_size:
        raise RuntimeError(
            f"expected to extract {sample_size:,} selected games, only found {selected_pointer:,}"
        )

    _flush_game_buffers(
        board_chunks,
        turn_chunks,
        castling_chunks,
        en_passant_chunks,
        ply_chunks,
        board_master,
        turn_master,
        castling_master,
        en_passant_master,
        ply_master,
    )

    board_codes = np.ascontiguousarray(np.concatenate(board_master, axis=0), dtype=np.uint8)
    side_to_move = np.ascontiguousarray(np.concatenate(turn_master, axis=0), dtype=np.uint8)
    castling_rights = np.ascontiguousarray(np.concatenate(castling_master, axis=0), dtype=np.uint8)
    en_passant_file = np.ascontiguousarray(np.concatenate(en_passant_master, axis=0), dtype=np.uint8)
    snapshot_plies = np.ascontiguousarray(np.concatenate(ply_master, axis=0), dtype=np.uint16)

    average_ratings = np.rint((white_elo.astype(np.float32) + black_elo.astype(np.float32)) / 2.0).astype(np.int16)
    split_ids = stratified_split_ids(average_ratings, band_width=band_width, seed=seed)

    train_mask = split_ids == TRAIN_SPLIT_ID
    train_ratings = np.concatenate([white_elo[train_mask], black_elo[train_mask]]).astype(np.int16)
    band_spec = RatingBandSpec.from_train_ratings(train_ratings, band_width=band_width)
    white_band = band_spec.encode_many(white_elo)
    black_band = band_spec.encode_many(black_elo)

    max_snapshots_per_game = int(lengths.max()) if len(lengths) else 0
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "sparse_snapshot_corpus.npz"
    manifest_path = outdir / "manifest.json"

    np.savez_compressed(
        npz_path,
        board_codes=board_codes,
        side_to_move=side_to_move,
        castling_rights=castling_rights,
        en_passant_file=en_passant_file,
        snapshot_plies=snapshot_plies,
        offsets=offsets,
        lengths=lengths,
        final_plies=final_plies,
        source_game_index=source_game_index,
        white_elo=white_elo,
        black_elo=black_elo,
        average_elo=average_ratings,
        white_band=white_band,
        black_band=black_band,
        split_id=split_ids,
        site=np.asarray(sites),
        opening=np.asarray(openings),
        eco=np.asarray(ecos),
    )

    split_counts = {
        split_name: int(np.count_nonzero(split_ids == split_id))
        for split_name, split_id in SPLIT_NAME_TO_ID.items()
    }

    output_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sample_dir": str(sample_dir.resolve()),
        "source_manifest_path": str((sample_dir / "manifest.json").resolve()),
        "source_sample_size": source_game_count,
        "subset_seed": seed,
        "selected_game_count": sample_size,
        "selected_source_indices_are_uniform_sample": True,
        "snapshot_step": snapshot_step,
        "band_width": band_width,
        "total_snapshot_count": int(len(board_codes)),
        "max_snapshots_per_game": max_snapshots_per_game,
        "split_counts": split_counts,
        "split_ids": {name: split_id for name, split_id in SPLIT_NAME_TO_ID.items()},
        "rating_bands": band_spec.to_dict(),
        "npz_file": npz_path.name,
        "arrays": {
            "board_codes": {"dtype": str(board_codes.dtype), "shape": list(board_codes.shape)},
            "side_to_move": {"dtype": str(side_to_move.dtype), "shape": list(side_to_move.shape)},
            "castling_rights": {"dtype": str(castling_rights.dtype), "shape": list(castling_rights.shape)},
            "en_passant_file": {"dtype": str(en_passant_file.dtype), "shape": list(en_passant_file.shape)},
            "snapshot_plies": {"dtype": str(snapshot_plies.dtype), "shape": list(snapshot_plies.shape)},
            "offsets": {"dtype": str(offsets.dtype), "shape": list(offsets.shape)},
            "lengths": {"dtype": str(lengths.dtype), "shape": list(lengths.shape)},
            "final_plies": {"dtype": str(final_plies.dtype), "shape": list(final_plies.shape)},
            "source_game_index": {"dtype": str(source_game_index.dtype), "shape": list(source_game_index.shape)},
            "white_elo": {"dtype": str(white_elo.dtype), "shape": list(white_elo.shape)},
            "black_elo": {"dtype": str(black_elo.dtype), "shape": list(black_elo.shape)},
            "average_elo": {"dtype": str(average_ratings.dtype), "shape": list(average_ratings.shape)},
            "white_band": {"dtype": str(white_band.dtype), "shape": list(white_band.shape)},
            "black_band": {"dtype": str(black_band.dtype), "shape": list(black_band.shape)},
            "split_id": {"dtype": str(split_ids.dtype), "shape": list(split_ids.shape)},
            "site": {"dtype": str(np.asarray(sites).dtype), "shape": [len(sites)]},
            "opening": {"dtype": str(np.asarray(openings).dtype), "shape": [len(openings)]},
            "eco": {"dtype": str(np.asarray(ecos).dtype), "shape": [len(ecos)]},
        },
        "notes": [
            "Positions are stored at ply 0, snapshot_step, 2*snapshot_step, ... and the final position is appended once if not already present.",
            "split_ids use 0=train, 1=val, 2=test.",
            "Rating bands are derived from the training split only and reused unchanged for validation and test rows.",
            "board_codes are directly comparable with chess_distance.ChessPosition.as_array().",
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
    build_sparse_snapshot_corpus(
        sample_dir=Path(args.sample_dir),
        outdir=Path(args.outdir),
        sample_size=args.sample_size,
        snapshot_step=args.snapshot_step,
        seed=args.seed,
        band_width=args.band_width,
        buffer_games=args.buffer_games,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
