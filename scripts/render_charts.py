#!/usr/bin/env python3
"""Render SVG charts from a Lichess summary JSON file."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

SPEED_ORDER = [
    "UltraBullet",
    "Bullet",
    "Blitz",
    "Rapid",
    "Classical",
    "Correspondence",
    "Unknown",
]

BACKGROUND = "#faf7f2"
SURFACE = "#fffdf9"
TEXT = "#18212f"
MUTED = "#667085"
GRID = "#d8d2c8"
ACCENT = "#d96b38"
ACCENT_DARK = "#a3481f"
ACCENT_ALT = "#2e6f95"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to the input summary JSON.")
    parser.add_argument("--outdir", required=True, help="Directory to write the SVG charts.")
    return parser.parse_args()


def fmt_int(value: int | None) -> str:
    return "n/a" if value is None else f"{value:,}"


def fmt_millions(value: float) -> str:
    if value >= 10_000_000:
        return f"{value / 1_000_000:.0f}M"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def fmt_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def aggregate_rating_bins(counts: list[int], bin_size: int = 50) -> list[dict[str, int]]:
    nonzero_ratings = [rating for rating, count in enumerate(counts) if count]
    if not nonzero_ratings:
        return []

    lower = (min(nonzero_ratings) // bin_size) * bin_size
    upper = ((max(nonzero_ratings) + bin_size - 1) // bin_size) * bin_size
    bins: list[dict[str, int]] = []
    for start in range(lower, upper + 1, bin_size):
        end = min(start + bin_size - 1, len(counts) - 1)
        count = sum(counts[start : end + 1])
        bins.append({"start": start, "end": end, "count": count})
    return bins


def svg_root(width: int, height: int, body: str, title: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{html.escape(title)}</title>
  <desc id="desc">{html.escape(title)}</desc>
  <defs>
    <linearGradient id="accentGradient" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{ACCENT}"/>
      <stop offset="100%" stop-color="{ACCENT_DARK}"/>
    </linearGradient>
    <linearGradient id="altGradient" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="{ACCENT_ALT}"/>
      <stop offset="100%" stop-color="#7ec8b2"/>
    </linearGradient>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#b8ab95" flood-opacity="0.18"/>
    </filter>
  </defs>
  <rect width="{width}" height="{height}" fill="{BACKGROUND}" />
{body}
</svg>
"""


def render_rating_distribution(summary: dict) -> str:
    width, height = 1400, 900
    chart_x = 120
    chart_y = 240
    chart_w = 1160
    chart_h = 500

    bins = aggregate_rating_bins(summary["player_rating_counts"], bin_size=50)
    max_bin_count = max(bin_info["count"] for bin_info in bins)
    y_tick_values = [max_bin_count * step / 5 for step in range(6)]

    bar_gap = 2
    bar_width = max((chart_w / max(len(bins), 1)) - bar_gap, 1)
    bars = []
    for index, bin_info in enumerate(bins):
        x = chart_x + index * (bar_width + bar_gap)
        height_ratio = bin_info["count"] / max_bin_count if max_bin_count else 0
        bar_height = chart_h * height_ratio
        y = chart_y + chart_h - bar_height
        bars.append(
            f'  <rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" '
            f'rx="3" fill="url(#accentGradient)" opacity="0.92"/>'
        )

    x_labels = []
    label_step = 200
    rating_min = bins[0]["start"]
    rating_max = bins[-1]["end"]
    for rating in range((rating_min // label_step) * label_step, rating_max + 1, label_step):
        position = chart_x + ((rating - rating_min) / max(rating_max - rating_min, 1)) * chart_w
        x_labels.append(
            f'  <line x1="{position:.2f}" y1="{chart_y + chart_h}" x2="{position:.2f}" y2="{chart_y + chart_h + 12}" stroke="{TEXT}" stroke-width="1"/>'
            f'  <text x="{position:.2f}" y="{chart_y + chart_h + 38}" text-anchor="middle" fill="{MUTED}" font-size="22" font-family="Georgia, serif">{rating}</text>'
        )

    y_grid = []
    for value in y_tick_values:
        y = chart_y + chart_h - (value / max_bin_count * chart_h if max_bin_count else 0)
        y_grid.append(
            f'  <line x1="{chart_x}" y1="{y:.2f}" x2="{chart_x + chart_w}" y2="{y:.2f}" stroke="{GRID}" stroke-width="1"/>'
            f'  <text x="{chart_x - 18}" y="{y + 8:.2f}" text-anchor="end" fill="{MUTED}" font-size="22" font-family="Georgia, serif">{fmt_millions(value)}</text>'
        )

    markers = []
    for key, label, color in [
        ("p10", "10th pct", "#7f8ea3"),
        ("median", "Median", ACCENT_ALT),
        ("p90", "90th pct", "#7f8ea3"),
    ]:
        rating = summary["rating_stats"][key]
        if rating is None:
            continue
        position = chart_x + ((rating - rating_min) / max(rating_max - rating_min, 1)) * chart_w
        markers.append(
            f'  <line x1="{position:.2f}" y1="{chart_y - 12}" x2="{position:.2f}" y2="{chart_y + chart_h}" stroke="{color}" stroke-width="4" stroke-dasharray="12 12"/>'
            f'  <text x="{position:.2f}" y="{chart_y - 28}" text-anchor="middle" fill="{color}" font-size="22" font-weight="700" font-family="Georgia, serif">{label}: {rating}</text>'
        )

    stat_cards = []
    card_specs = [
        ("Player-slots", fmt_int(summary["total_player_slots"])),
        ("Median Elo", fmt_int(summary["rating_stats"]["median"])),
        ("Middle 50%", f'{fmt_int(summary["rating_stats"]["p25"])} to {fmt_int(summary["rating_stats"]["p75"])}'),
        ("Mean Elo", f'{summary["rating_stats"]["mean"]:.1f}' if summary["rating_stats"]["mean"] is not None else "n/a"),
    ]
    for index, (label, value) in enumerate(card_specs):
        x = 120 + index * 285
        stat_cards.append(
            f'  <g filter="url(#shadow)">'
            f'    <rect x="{x}" y="72" width="245" height="108" rx="18" fill="{SURFACE}"/>'
            f'  </g>'
            f'  <text x="{x + 24}" y="114" fill="{MUTED}" font-size="22" font-family="Georgia, serif">{html.escape(label)}</text>'
            f'  <text x="{x + 24}" y="154" fill="{TEXT}" font-size="34" font-weight="700" font-family="Georgia, serif">{html.escape(value)}</text>'
        )

    body = "\n".join(
        [
            '  <g filter="url(#shadow)">',
            f'    <rect x="70" y="42" width="{width - 140}" height="{height - 84}" rx="28" fill="{SURFACE}"/>',
            "  </g>",
            f'  <text x="{chart_x}" y="110" fill="{TEXT}" font-size="42" font-weight="700" font-family="Georgia, serif">Lichess March 2026 player rating distribution</text>',
            f'  <text x="{chart_x}" y="148" fill="{MUTED}" font-size="24" font-family="Georgia, serif">All player-slots from the March 2026 rated standard dump, binned in 50-point Elo intervals.</text>',
            *stat_cards,
            *y_grid,
            f'  <line x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" stroke="{TEXT}" stroke-width="2"/>',
            f'  <line x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{chart_y + chart_h}" stroke="{TEXT}" stroke-width="2"/>',
            *bars,
            *markers,
            *x_labels,
            f'  <text x="{chart_x + chart_w / 2:.2f}" y="{height - 72}" text-anchor="middle" fill="{MUTED}" font-size="24" font-family="Georgia, serif">Elo rating</text>',
            f'  <text x="44" y="{chart_y + chart_h / 2:.2f}" transform="rotate(-90 44 {chart_y + chart_h / 2:.2f})" text-anchor="middle" fill="{MUTED}" font-size="24" font-family="Georgia, serif">Player-slots</text>',
        ]
    )
    return svg_root(width, height, body, "Lichess March 2026 player rating distribution")


def render_speed_distribution(summary: dict) -> str:
    width, height = 1400, 900
    chart_x = 250
    chart_y = 210
    chart_w = 1000
    row_h = 82
    bar_h = 40

    speeds = [speed for speed in SPEED_ORDER if speed in summary["speed_counts"]]
    max_count = max(summary["speed_counts"][speed] for speed in speeds)

    rows = []
    for index, speed in enumerate(speeds):
        y = chart_y + index * row_h
        count = summary["speed_counts"][speed]
        share = summary["speed_shares"][speed]
        avg_elo = summary["speed_average_player_elo"].get(speed)
        bar_width = chart_w * (count / max_count if max_count else 0)
        rows.append(
            f'  <text x="{chart_x - 26}" y="{y + 28}" text-anchor="end" fill="{TEXT}" font-size="28" font-weight="700" font-family="Georgia, serif">{html.escape(speed)}</text>'
            f'  <rect x="{chart_x}" y="{y}" width="{chart_w}" height="{bar_h}" rx="14" fill="#f0ebe3"/>'
            f'  <rect x="{chart_x}" y="{y}" width="{bar_width:.2f}" height="{bar_h}" rx="14" fill="url(#altGradient)"/>'
            f'  <text x="{chart_x + chart_w + 18}" y="{y + 28}" fill="{TEXT}" font-size="24" font-family="Georgia, serif">{fmt_percent(share)}  ·  {count:,} games</text>'
            f'  <text x="{chart_x}" y="{y + 68}" fill="{MUTED}" font-size="21" font-family="Georgia, serif">Average player Elo: {avg_elo:.1f}</text>'
        )

    insight_lines = [
        ("Largest bucket", max(speeds, key=lambda speed: summary["speed_counts"][speed])),
        ("Smallest bucket", min(speeds, key=lambda speed: summary["speed_counts"][speed])),
        ("Total games", f"{summary['total_games']:,}"),
    ]
    callouts = []
    for index, (label, value) in enumerate(insight_lines):
        x = 110 + index * 360
        callouts.append(
            f'  <g filter="url(#shadow)">'
            f'    <rect x="{x}" y="88" width="300" height="88" rx="18" fill="{SURFACE}"/>'
            f'  </g>'
            f'  <text x="{x + 22}" y="122" fill="{MUTED}" font-size="22" font-family="Georgia, serif">{html.escape(label)}</text>'
            f'  <text x="{x + 22}" y="154" fill="{TEXT}" font-size="30" font-weight="700" font-family="Georgia, serif">{html.escape(value)}</text>'
        )

    body = "\n".join(
        [
            '  <g filter="url(#shadow)">',
            f'    <rect x="70" y="42" width="{width - 140}" height="{height - 84}" rx="28" fill="{SURFACE}"/>',
            "  </g>",
            f'  <text x="{chart_x}" y="110" fill="{TEXT}" font-size="42" font-weight="700" font-family="Georgia, serif">Lichess March 2026 game speed mix</text>',
            f'  <text x="{chart_x}" y="148" fill="{MUTED}" font-size="24" font-family="Georgia, serif">Speed classes inferred from the PGN Event tags in the March 2026 rated standard dump.</text>',
            *callouts,
            *rows,
        ]
    )
    return svg_root(width, height, body, "Lichess March 2026 game speed mix")


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    rating_svg = render_rating_distribution(summary)
    speed_svg = render_speed_distribution(summary)

    (outdir / "march_2026_player_elo_distribution.svg").write_text(rating_svg, encoding="utf-8")
    (outdir / "march_2026_game_speed_distribution.svg").write_text(speed_svg, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
