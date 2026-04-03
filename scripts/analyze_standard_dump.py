#!/usr/bin/env python3
"""Stream a decompressed Lichess PGN dump from stdin and summarize it."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EVENT_PREFIX = b'[Event "'
WHITE_ELO_PREFIX = b'[WhiteElo "'
BLACK_ELO_PREFIX = b'[BlackElo "'
TIME_CONTROL_PREFIX = b'[TimeControl "'

SPEED_ORDER = [
    "UltraBullet",
    "Bullet",
    "Blitz",
    "Rapid",
    "Classical",
    "Correspondence",
    "Unknown",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to write the JSON summary.")
    parser.add_argument(
        "--source-url",
        default="",
        help="Optional metadata string for the upstream dump URL.",
    )
    parser.add_argument(
        "--expected-games",
        type=int,
        default=0,
        help="Optional metadata value for the expected number of games in the dump.",
    )
    parser.add_argument(
        "--month",
        default="",
        help="Month identifier for metadata, for example 2026-03.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=2_000_000,
        help="Emit a progress update to stderr every N games.",
    )
    return parser.parse_args()


def extract_tag_value(line: bytes, prefix: bytes) -> bytes:
    line = line.rstrip(b"\r\n")
    if not line.startswith(prefix) or not line.endswith(b'"]'):
        return b""
    return line[len(prefix) : -2]


def classify_from_event(event_value: bytes) -> str:
    lower_value = event_value.lower()
    if b"correspondence" in lower_value:
        return "Correspondence"
    if b"ultrabullet" in lower_value:
        return "UltraBullet"
    if b"bullet" in lower_value:
        return "Bullet"
    if b"blitz" in lower_value:
        return "Blitz"
    if b"rapid" in lower_value:
        return "Rapid"
    if b"classical" in lower_value:
        return "Classical"
    return "Unknown"


def classify_from_time_control(time_control_value: bytes) -> str:
    value = time_control_value.rstrip(b"\r\n")
    if not value or value == b"-":
        return "Unknown"
    if b"+" not in value:
        return "Unknown"
    try:
        base_text, increment_text = value.split(b"+", 1)
        base_seconds = int(base_text)
        increment_seconds = int(increment_text)
    except ValueError:
        return "Unknown"

    if base_seconds >= 86_400:
        return "Correspondence"

    estimated_duration = base_seconds + increment_seconds * 40
    if estimated_duration < 30:
        return "UltraBullet"
    if estimated_duration < 180:
        return "Bullet"
    if estimated_duration < 480:
        return "Blitz"
    if estimated_duration < 1_500:
        return "Rapid"
    return "Classical"


def ensure_size(values: list[int], index: int) -> None:
    if index >= len(values):
        values.extend([0] * (index - len(values) + 1))


def parse_rating(value: bytes) -> int | None:
    value = value.strip()
    if not value.isdigit():
        return None
    rating = int(value)
    if rating < 0:
        return None
    return rating


def percentile_from_counts(counts: list[int], percentile: float) -> int | None:
    total = sum(counts)
    if total == 0:
        return None

    target = math.ceil(total * percentile)
    running_total = 0
    for rating, count in enumerate(counts):
        running_total += count
        if running_total >= target:
            return rating
    return len(counts) - 1


def finalize_game(
    event_value: bytes,
    time_control_value: bytes,
    white_elo_value: bytes,
    black_elo_value: bytes,
    speed_counts: Counter[str],
    raw_event_counts: Counter[str],
    player_rating_counts: list[int],
    speed_rating_sums: Counter[str],
    speed_rating_counts: Counter[str],
    metrics: dict[str, int | float | None],
) -> None:
    if not event_value and not time_control_value and not white_elo_value and not black_elo_value:
        return

    speed = classify_from_event(event_value)
    if speed == "Unknown":
        speed = classify_from_time_control(time_control_value)
    speed_counts[speed] += 1
    raw_event_counts[event_value.decode("utf-8", "replace") or ""] += 1

    metrics["games"] += 1

    white_rating = parse_rating(white_elo_value)
    black_rating = parse_rating(black_elo_value)
    for rating in (white_rating, black_rating):
        if rating is None:
            continue
        ensure_size(player_rating_counts, rating)
        player_rating_counts[rating] += 1
        speed_rating_sums[speed] += rating
        speed_rating_counts[speed] += 1

        metrics["player_slots"] += 1
        metrics["rating_sum"] += rating
        metrics["rating_sum_squares"] += rating * rating
        metrics["rating_min"] = rating if metrics["rating_min"] is None else min(metrics["rating_min"], rating)
        metrics["rating_max"] = rating if metrics["rating_max"] is None else max(metrics["rating_max"], rating)


def main() -> int:
    args = parse_args()
    output_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    speed_counts: Counter[str] = Counter()
    raw_event_counts: Counter[str] = Counter()
    speed_rating_sums: Counter[str] = Counter()
    speed_rating_counts: Counter[str] = Counter()
    player_rating_counts = [0] * 4001
    metrics: dict[str, int | float | None] = {
        "games": 0,
        "player_slots": 0,
        "rating_sum": 0,
        "rating_sum_squares": 0,
        "rating_min": None,
        "rating_max": None,
    }

    event_value = b""
    white_elo_value = b""
    black_elo_value = b""
    time_control_value = b""
    start_time = time.monotonic()

    for raw_line in sys.stdin.buffer:
        if raw_line.startswith(EVENT_PREFIX):
            if event_value or time_control_value or white_elo_value or black_elo_value:
                finalize_game(
                    event_value,
                    time_control_value,
                    white_elo_value,
                    black_elo_value,
                    speed_counts,
                    raw_event_counts,
                    player_rating_counts,
                    speed_rating_sums,
                    speed_rating_counts,
                    metrics,
                )

                if args.progress_every and metrics["games"] % args.progress_every == 0:
                    elapsed_seconds = max(time.monotonic() - start_time, 1e-9)
                    games_per_second = metrics["games"] / elapsed_seconds
                    print(
                        f"Processed {metrics['games']:,} games "
                        f"({games_per_second:,.0f} games/sec)",
                        file=sys.stderr,
                    )

            event_value = extract_tag_value(raw_line, EVENT_PREFIX)
            white_elo_value = b""
            black_elo_value = b""
            time_control_value = b""
            continue
        if raw_line.startswith(WHITE_ELO_PREFIX):
            white_elo_value = extract_tag_value(raw_line, WHITE_ELO_PREFIX)
            continue
        if raw_line.startswith(BLACK_ELO_PREFIX):
            black_elo_value = extract_tag_value(raw_line, BLACK_ELO_PREFIX)
            continue
        if raw_line.startswith(TIME_CONTROL_PREFIX):
            time_control_value = extract_tag_value(raw_line, TIME_CONTROL_PREFIX)
            continue

    finalize_game(
        event_value,
        time_control_value,
        white_elo_value,
        black_elo_value,
        speed_counts,
        raw_event_counts,
        player_rating_counts,
        speed_rating_sums,
        speed_rating_counts,
        metrics,
    )

    elapsed_seconds = time.monotonic() - start_time
    player_slots = int(metrics["player_slots"])
    mean_rating = (metrics["rating_sum"] / player_slots) if player_slots else None
    variance = None
    stddev = None
    if player_slots:
        variance = (metrics["rating_sum_squares"] / player_slots) - (mean_rating * mean_rating)
        variance = max(variance, 0.0)
        stddev = math.sqrt(variance)

    speed_shares = {
        speed: (speed_counts[speed] / metrics["games"]) if metrics["games"] else 0.0
        for speed in SPEED_ORDER
        if speed_counts[speed]
    }
    speed_average_player_elo = {
        speed: (speed_rating_sums[speed] / speed_rating_counts[speed])
        for speed in SPEED_ORDER
        if speed_rating_counts[speed]
    }

    summary = {
        "month": args.month,
        "source_url": args.source_url,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed_seconds,
        "expected_games": args.expected_games or None,
        "total_games": int(metrics["games"]),
        "total_player_slots": player_slots,
        "rating_stats": {
            "min": metrics["rating_min"],
            "p10": percentile_from_counts(player_rating_counts, 0.10),
            "p25": percentile_from_counts(player_rating_counts, 0.25),
            "median": percentile_from_counts(player_rating_counts, 0.50),
            "p75": percentile_from_counts(player_rating_counts, 0.75),
            "p90": percentile_from_counts(player_rating_counts, 0.90),
            "max": metrics["rating_max"],
            "mean": mean_rating,
            "stddev": stddev,
        },
        "speed_counts": {speed: speed_counts[speed] for speed in SPEED_ORDER if speed_counts[speed]},
        "speed_shares": speed_shares,
        "speed_average_player_elo": speed_average_player_elo,
        "top_raw_events": raw_event_counts.most_common(12),
        "player_rating_counts": player_rating_counts,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    mismatch = ""
    if args.expected_games and args.expected_games != metrics["games"]:
        mismatch = f" (expected {args.expected_games:,})"

    print(
        f"Wrote {output_path} with {metrics['games']:,} games and "
        f"{player_slots:,} player-slots in {elapsed_seconds / 60:.1f} minutes{mismatch}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
