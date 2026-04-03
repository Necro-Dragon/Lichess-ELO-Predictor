#!/usr/bin/env python3
"""Create a uniform sample of March 2026 rapid games in a reduced PGN format."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


EVENT_PREFIX = '[Event "'
TIME_CONTROL_PREFIX = '[TimeControl "'
TERMINATION_PREFIX = '[Termination "'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the source .pgn.zst file.")
    parser.add_argument("--outdir", required=True, help="Output directory for compressed chunk files.")
    parser.add_argument("--sample-size", type=int, default=1_500_000, help="Uniform sample size.")
    parser.add_argument("--seed", type=int, default=202603, help="RNG seed for reproducibility.")
    parser.add_argument(
        "--chunk-games",
        type=int,
        default=250_000,
        help="How many sampled games to place in each compressed chunk.",
    )
    parser.add_argument(
        "--count-progress-every",
        type=int,
        default=5_000_000,
        help="Emit count-pass progress every N total games.",
    )
    parser.add_argument(
        "--extract-progress-every",
        type=int,
        default=250_000,
        help="Emit extraction progress every N sampled games written.",
    )
    return parser.parse_args()


def extract_tag_value(line: str, prefix: str) -> str:
    if not line.startswith(prefix) or not line.endswith('"]'):
        return ""
    return line[len(prefix) : -2]


def classify_speed(event_value: str, time_control_value: str) -> str:
    lower_value = event_value.lower()
    if "correspondence" in lower_value:
        return "Correspondence"
    if "ultrabullet" in lower_value:
        return "UltraBullet"
    if "bullet" in lower_value:
        return "Bullet"
    if "blitz" in lower_value:
        return "Blitz"
    if "rapid" in lower_value:
        return "Rapid"
    if "classical" in lower_value:
        return "Classical"

    if "+" in time_control_value:
        try:
            base_text, increment_text = time_control_value.split("+", 1)
            estimated_duration = int(base_text) + int(increment_text) * 40
        except ValueError:
            return "Unknown"
        if estimated_duration < 30:
            return "UltraBullet"
        if estimated_duration < 180:
            return "Bullet"
        if estimated_duration < 480:
            return "Blitz"
        if estimated_duration < 1_500:
            return "Rapid"
        return "Classical"
    return "Unknown"


def normalize_termination(raw_termination: str, result: str, moves_text: str) -> str | None:
    raw_lower = raw_termination.strip().lower()
    moves_text = moves_text.strip()

    if raw_lower == "abandoned":
        return None
    if raw_lower == "time forfeit":
        return "time forfeit"
    if raw_lower == "rules infraction":
        return "rules infraction"
    if raw_lower == "insufficient material":
        return "insufficient material"
    if raw_lower == "stalemate":
        return "stalemate"
    if raw_lower in {"outoftime", "timeout"}:
        return "time forfeit"

    if "#" in moves_text:
        return "checkmate"
    if result in {"1-0", "0-1"}:
        return "resignation"
    return "draw"


def iter_pgn_games(input_path: Path):
    process = subprocess.Popen(
        ["zstd", "-dc", str(input_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None

    headers: dict[str, str] = {}
    move_lines: list[str] = []
    in_moves = False

    def finalize():
        nonlocal headers, move_lines, in_moves
        if headers:
            game_headers = headers
            game_moves = move_lines
            headers = {}
            move_lines = []
            in_moves = False
            return game_headers, game_moves
        return None

    try:
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            if line.startswith("["):
                if in_moves and headers:
                    game = finalize()
                    if game is not None:
                        yield game
                if ' "' in line and line.endswith('"]'):
                    split_at = line.index(' "')
                    tag_name = line[1:split_at]
                    tag_value = line[split_at + 2 : -2]
                    headers[tag_name] = tag_value
                continue

            if line == "":
                if headers and not in_moves:
                    in_moves = True
                elif in_moves:
                    game = finalize()
                    if game is not None:
                        yield game
                continue

            if in_moves:
                move_lines.append(line)

        game = finalize()
        if game is not None:
            yield game
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()


def transformed_game_text(headers: dict[str, str], move_lines: list[str]) -> tuple[str | None, str]:
    moves_text = " ".join(line.strip() for line in move_lines if line.strip())
    normalized_termination = normalize_termination(
        headers.get("Termination", ""),
        headers.get("Result", ""),
        moves_text,
    )
    if normalized_termination is None:
        return None, moves_text

    game_text = (
        f'[Site "{headers.get("Site", "")}"]\n'
        f'[Result "{headers.get("Result", "")}"]\n'
        f'[Termination "{normalized_termination}"]\n'
        f'[WhiteElo "{headers.get("WhiteElo", "")}"]\n'
        f'[BlackElo "{headers.get("BlackElo", "")}"]\n'
        f'[ECO "{headers.get("ECO", "")}"]\n'
        f'[Opening "{headers.get("Opening", "")}"]\n\n'
        f"{moves_text}\n\n"
    )
    return game_text, normalized_termination


def write_manifest(outdir: Path, manifest: dict) -> None:
    manifest_path = outdir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def start_chunk_writer(base_name: str, outdir: Path, chunk_index: int) -> tuple[Path, subprocess.Popen]:
    chunk_path = outdir / f"{base_name}_part_{chunk_index:02d}.pgn.zst"
    process = subprocess.Popen(
        ["zstd", "-19", "-T0", "-q", "-o", str(chunk_path)],
        stdin=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdin is not None
    return chunk_path, process


def close_chunk_writer(process: subprocess.Popen) -> None:
    assert process.stdin is not None
    process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"zstd failed with exit code {return_code}")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    base_name = f"rapid_2026-03_sample_{args.sample_size}"
    outdir.mkdir(parents=True, exist_ok=True)

    print("Count pass: scanning March 2026 archive for rapid non-abandoned games...", file=sys.stderr)
    total_games = 0
    eligible_games = 0
    raw_termination_counts: Counter[str] = Counter()
    normalized_termination_counts: Counter[str] = Counter()
    count_started = time.monotonic()

    for headers, move_lines in iter_pgn_games(input_path):
        total_games += 1
        speed = classify_speed(headers.get("Event", ""), headers.get("TimeControl", ""))
        if speed == "Rapid":
            game_text, normalized = transformed_game_text(headers, move_lines)
            if game_text is not None:
                eligible_games += 1
                raw_termination_counts[headers.get("Termination", "")] += 1
                normalized_termination_counts[normalized] += 1

        if args.count_progress_every and total_games % args.count_progress_every == 0:
            elapsed = max(time.monotonic() - count_started, 1e-9)
            print(
                f"Counted {total_games:,} total games; {eligible_games:,} rapid non-abandoned "
                f"({total_games / elapsed:,.0f} total games/sec)",
                file=sys.stderr,
            )

    if args.sample_size > eligible_games:
        raise SystemExit(
            f"Requested sample of {args.sample_size:,} games, but only found "
            f"{eligible_games:,} rapid non-abandoned games."
        )

    print(
        f"Count pass complete: {eligible_games:,} rapid non-abandoned games eligible for sampling.",
        file=sys.stderr,
    )

    rng = random.Random(args.seed)
    sample_indices = sorted(rng.sample(range(eligible_games), args.sample_size))

    print(
        f"Extraction pass: writing {args.sample_size:,} sampled games in compressed chunks...",
        file=sys.stderr,
    )

    sampled_written = 0
    eligible_index = 0
    sample_pointer = 0
    chunk_index = 1
    chunk_games_written = 0
    chunk_paths: list[str] = []
    chunk_path, chunk_process = start_chunk_writer(base_name, outdir, chunk_index)
    chunk_paths.append(str(chunk_path.name))
    extract_started = time.monotonic()

    for headers, move_lines in iter_pgn_games(input_path):
        speed = classify_speed(headers.get("Event", ""), headers.get("TimeControl", ""))
        if speed != "Rapid":
            continue

        game_text, _normalized = transformed_game_text(headers, move_lines)
        if game_text is None:
            continue

        if sample_pointer >= len(sample_indices):
            break

        if eligible_index == sample_indices[sample_pointer]:
            assert chunk_process.stdin is not None
            chunk_process.stdin.write(game_text)
            sampled_written += 1
            chunk_games_written += 1
            sample_pointer += 1

            if args.extract_progress_every and sampled_written % args.extract_progress_every == 0:
                elapsed = max(time.monotonic() - extract_started, 1e-9)
                print(
                    f"Wrote {sampled_written:,} sampled games "
                    f"({sampled_written / elapsed:,.0f} sampled games/sec)",
                    file=sys.stderr,
                )

            if chunk_games_written >= args.chunk_games and sampled_written < args.sample_size:
                close_chunk_writer(chunk_process)
                chunk_index += 1
                chunk_games_written = 0
                chunk_path, chunk_process = start_chunk_writer(base_name, outdir, chunk_index)
                chunk_paths.append(str(chunk_path.name))

        eligible_index += 1

    close_chunk_writer(chunk_process)

    if sampled_written != args.sample_size:
        raise SystemExit(
            f"Extraction wrote {sampled_written:,} games, expected {args.sample_size:,}."
        )

    manifest = {
        "source_archive": str(input_path),
        "month": "2026-03",
        "speed": "Rapid",
        "sample_type": "uniform_without_replacement",
        "sample_size": args.sample_size,
        "eligible_games": eligible_games,
        "rng_seed": args.seed,
        "chunk_games": args.chunk_games,
        "chunk_files": chunk_paths,
        "fields_kept": [
            "Site",
            "Result",
            "Termination",
            "WhiteElo",
            "BlackElo",
            "ECO",
            "Opening",
            "MovesWithClocks",
        ],
        "population_raw_termination_counts": dict(sorted(raw_termination_counts.items())),
        "population_normalized_termination_counts": dict(sorted(normalized_termination_counts.items())),
        "notes": [
            "Abandoned games were excluded.",
            "Termination is a best-effort normalization from Lichess PGN metadata and movetext.",
            "Rapid draws in the monthly dump are not reliably separable into agreement, repetition, and 50-move rule from metadata alone.",
            "Additional observed endings include time forfeit, insufficient material, rules infraction, and stalemate.",
        ],
    }
    write_manifest(outdir, manifest)

    print(
        f"Sample build complete: wrote {sampled_written:,} games into {len(chunk_paths)} chunk files.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
