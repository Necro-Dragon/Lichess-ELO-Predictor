"""Generate a UMAP embedding for the lichess opening taxonomy."""

from __future__ import annotations

import argparse
import colorsys
import csv
import html
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

import numpy as np
import chess.pgn
import umap

from .position import ChessPosition, stack_positions

OPENING_VOLUME_FILES = ("a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv")
RAW_BASE_URL = "https://raw.githubusercontent.com/lichess-org/chess-openings"

BACKGROUND = "#faf7f2"
SURFACE = "#fffdf9"
TEXT = "#18212f"
MUTED = "#667085"
GRID = "#d8d2c8"


@dataclass(slots=True)
class OpeningRecord:
    eco: str
    name: str
    pgn: str
    family: str
    variation: str
    position_fen: str
    position: ChessPosition
    color_hex: str = ""
    family_hue: float = 0.0
    duplicate_group_size: int = 1
    duplicate_group_rank: int = 1
    umap_x: float = 0.0
    umap_y: float = 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref", default="master", help="Git ref to fetch from lichess-org/chess-openings.")
    parser.add_argument("--graphics-dir", default="graphics", help="Directory to write SVG outputs.")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory to write CSV/JSON outputs.")
    parser.add_argument("--seed", type=int, default=20260406, help="UMAP random seed.")
    parser.add_argument("--n-neighbors", type=int, default=30, help="UMAP n_neighbors parameter.")
    parser.add_argument("--min-dist", type=float, default=0.10, help="UMAP min_dist parameter.")
    return parser.parse_args(argv)


def build_source_url(source_ref: str, volume_name: str) -> str:
    return f"{RAW_BASE_URL}/{source_ref}/{volume_name}"


def fetch_source_text(url: str) -> str:
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def parse_opening_tsv(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    expected = {"eco", "name", "pgn"}
    fieldnames = set(reader.fieldnames or [])
    if not expected.issubset(fieldnames):
        missing = ", ".join(sorted(expected - fieldnames))
        raise ValueError(f"opening TSV is missing required columns: {missing}")

    rows: list[dict[str, str]] = []
    for raw_row in reader:
        row = {key: (value or "").strip() for key, value in raw_row.items()}
        if not row["eco"] or not row["name"] or not row["pgn"]:
            raise ValueError(f"opening TSV row is incomplete: {raw_row}")
        rows.append(row)
    return rows


def parse_opening_taxonomy(name: str) -> tuple[str, str]:
    family, separator, remainder = name.partition(":")
    family = family.strip()
    if not family:
        raise ValueError(f"opening name is missing a family: {name!r}")

    if not separator or not remainder.strip():
        return family, "Root"

    variation = remainder.split(",", 1)[0].strip()
    return family, variation or "Root"


def replay_pgn_to_position(pgn: str) -> tuple[str, ChessPosition]:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError(f"could not parse PGN: {pgn!r}")
    if game.errors:
        raise ValueError(f"PGN contains parse errors: {pgn!r}")

    board = game.board()
    for move in game.mainline_moves():
        board.push(move)

    position_fen = board.fen()
    return position_fen, ChessPosition.from_fen(position_fen)


def fetch_opening_rows(
    source_ref: str,
    *,
    fetch_text: Callable[[str], str] = fetch_source_text,
    volume_files: tuple[str, ...] = OPENING_VOLUME_FILES,
) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    urls: list[str] = []
    for volume_name in volume_files:
        url = build_source_url(source_ref, volume_name)
        urls.append(url)
        rows.extend(parse_opening_tsv(fetch_text(url)))
    if not rows:
        raise ValueError("no openings were loaded")
    return rows, urls


def build_opening_records(rows: list[dict[str, str]]) -> list[OpeningRecord]:
    records: list[OpeningRecord] = []
    for row in rows:
        family, variation = parse_opening_taxonomy(row["name"])
        position_fen, position = replay_pgn_to_position(row["pgn"])
        records.append(
            OpeningRecord(
                eco=row["eco"],
                name=row["name"],
                pgn=row["pgn"],
                family=family,
                variation=variation,
                position_fen=position_fen,
                position=position,
            )
        )
    return records


def _rgb_to_hex(red: float, green: float, blue: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(round(max(0.0, min(1.0, red)) * 255)),
        int(round(max(0.0, min(1.0, green)) * 255)),
        int(round(max(0.0, min(1.0, blue)) * 255)),
    )


def _family_swatch_from_hue(hue: float) -> str:
    red, green, blue = colorsys.hls_to_rgb(hue / 360.0, 0.50, 0.72)
    return _rgb_to_hex(red, green, blue)


def _variation_swatch_from_hue(hue: float, index: int, total: int) -> str:
    if total <= 1:
        lightness = 0.50
        saturation = 0.72
    else:
        ratio = index / max(total - 1, 1)
        lightness = 0.35 + ratio * 0.38
        saturation = 0.82 - ((index % 4) * 0.06)
    red, green, blue = colorsys.hls_to_rgb(hue / 360.0, lightness, saturation)
    return _rgb_to_hex(red, green, blue)


def assign_family_colors(records: list[OpeningRecord]) -> tuple[dict[str, float], dict[str, str], dict[tuple[str, str], str]]:
    families = sorted({record.family for record in records})
    family_hues = {
        family: (index * 360.0 / max(len(families), 1))
        for index, family in enumerate(families)
    }
    family_colors = {
        family: _family_swatch_from_hue(family_hues[family])
        for family in families
    }

    variation_colors: dict[tuple[str, str], str] = {}
    variations_by_family: dict[str, list[str]] = {}
    for family in families:
        variations = sorted({record.variation for record in records if record.family == family})
        variations_by_family[family] = variations
        for variation_index, variation in enumerate(variations):
            variation_colors[(family, variation)] = _variation_swatch_from_hue(
                family_hues[family],
                variation_index,
                len(variations),
            )

    for record in records:
        record.family_hue = family_hues[record.family]
        record.color_hex = variation_colors[(record.family, record.variation)]

    return family_hues, family_colors, variation_colors


def annotate_duplicate_groups(records: list[OpeningRecord]) -> tuple[int, int]:
    counts = Counter(record.position.as_array().tobytes() for record in records)
    seen = Counter()
    for record in records:
        key = record.position.as_array().tobytes()
        seen[key] += 1
        record.duplicate_group_size = counts[key]
        record.duplicate_group_rank = seen[key]
    unique_positions = len(counts)
    duplicate_groups = sum(1 for count in counts.values() if count > 1)
    return unique_positions, duplicate_groups


def fit_umap_embedding(
    records: list[OpeningRecord],
    *,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    if len(records) < 3:
        raise ValueError("UMAP embedding requires at least 3 openings")
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2")
    if n_neighbors >= len(records):
        raise ValueError(f"n_neighbors must be less than the row count ({len(records)})")

    matrix = stack_positions([record.position for record in records])
    reducer = umap.UMAP(
        n_components=2,
        metric="hamming",
        init="spectral",
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    return reducer.fit_transform(matrix)


def apply_embedding(records: list[OpeningRecord], embedding: np.ndarray) -> None:
    for record, point in zip(records, embedding, strict=True):
        record.umap_x = float(point[0])
        record.umap_y = float(point[1])


def _svg_root(width: int, height: int, body: str, title: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{html.escape(title)}</title>
  <desc id="desc">{html.escape(title)}</desc>
  <defs>
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#b8ab95" flood-opacity="0.18"/>
    </filter>
  </defs>
  <rect width="{width}" height="{height}" fill="{BACKGROUND}" />
{body}
</svg>
"""


def _project_value(value: float, lower: float, upper: float, start: float, span: float) -> float:
    if math.isclose(lower, upper):
        return start + (span / 2.0)
    return start + ((value - lower) / (upper - lower)) * span


def render_scatter_svg(records: list[OpeningRecord], unique_positions: int) -> str:
    width, height = 1600, 1150
    chart_x = 110
    chart_y = 100
    chart_w = 1360
    chart_h = 900

    xs = [record.umap_x for record in records]
    ys = [record.umap_y for record in records]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    y_ticks = []
    x_ticks = []
    grid_lines = []
    for step in range(6):
        x_position = chart_x + (chart_w * step / 5)
        tick_value = min_x + ((max_x - min_x) * step / 5) if not math.isclose(min_x, max_x) else min_x
        x_ticks.append(
            f'  <line x1="{x_position:.2f}" y1="{chart_y + chart_h}" x2="{x_position:.2f}" y2="{chart_y + chart_h + 12}" stroke="{TEXT}" stroke-width="1"/>'
            f'  <text x="{x_position:.2f}" y="{chart_y + chart_h + 38}" text-anchor="middle" fill="{MUTED}" font-size="22" font-family="Georgia, serif">{tick_value:.2f}</text>'
        )

        y_position = chart_y + chart_h - (chart_h * step / 5)
        y_value = min_y + ((max_y - min_y) * step / 5) if not math.isclose(min_y, max_y) else min_y
        y_ticks.append(
            f'  <line x1="{chart_x - 12}" y1="{y_position:.2f}" x2="{chart_x}" y2="{y_position:.2f}" stroke="{TEXT}" stroke-width="1"/>'
            f'  <text x="{chart_x - 22}" y="{y_position + 8:.2f}" text-anchor="end" fill="{MUTED}" font-size="22" font-family="Georgia, serif">{y_value:.2f}</text>'
        )
        grid_lines.append(
            f'  <line x1="{chart_x}" y1="{y_position:.2f}" x2="{chart_x + chart_w}" y2="{y_position:.2f}" stroke="{GRID}" stroke-width="1"/>'
        )

    points = []
    for record in records:
        x_position = _project_value(record.umap_x, min_x, max_x, chart_x, chart_w)
        y_position = chart_y + chart_h - _project_value(record.umap_y, min_y, max_y, 0, chart_h)
        points.append(
            f'  <circle cx="{x_position:.2f}" cy="{y_position:.2f}" r="3.2" fill="{record.color_hex}" fill-opacity="0.72"/>'
        )

    body = "\n".join(
        [
            '  <g filter="url(#shadow)">',
            f'    <rect x="70" y="42" width="{width - 140}" height="{height - 84}" rx="28" fill="{SURFACE}"/>',
            "  </g>",
            *grid_lines,
            f'  <line x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" stroke="{TEXT}" stroke-width="2"/>',
            f'  <line x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{chart_y + chart_h}" stroke="{TEXT}" stroke-width="2"/>',
            *points,
            *x_ticks,
            *y_ticks,
            f'  <text x="{chart_x + chart_w / 2:.2f}" y="{height - 64}" text-anchor="middle" fill="{MUTED}" font-size="24" font-family="Georgia, serif">UMAP 1</text>',
            f'  <text x="42" y="{chart_y + chart_h / 2:.2f}" transform="rotate(-90 42 {chart_y + chart_h / 2:.2f})" text-anchor="middle" fill="{MUTED}" font-size="24" font-family="Georgia, serif">UMAP 2</text>',
        ]
    )
    return _svg_root(width, height, body, "Lichess opening UMAP embedding")


def build_legend_entries(
    records: list[OpeningRecord],
    family_colors: dict[str, str],
    variation_colors: dict[tuple[str, str], str],
) -> list[dict[str, object]]:
    variations_by_family: dict[str, set[str]] = defaultdict(set)
    for record in records:
        variations_by_family[record.family].add(record.variation)

    entries: list[dict[str, object]] = []
    for family in sorted(variations_by_family):
        variations = sorted(variations_by_family[family])
        entries.append(
            {
                "family": family,
                "family_color": family_colors[family],
                "variations": [
                    {
                        "name": variation,
                        "color": variation_colors[(family, variation)],
                    }
                    for variation in variations
                ],
            }
        )
    return entries


def render_legend_svg(legend_entries: list[dict[str, object]]) -> str:
    margin = 48
    column_gap = 34
    column_width = 360
    title_height = 124
    family_header_height = 34
    variation_row_height = 22
    family_gap = 18
    max_column_height = 4800

    section_heights = [
        family_header_height + (len(entry["variations"]) * variation_row_height) + family_gap  # type: ignore[arg-type]
        for entry in legend_entries
    ]
    total_section_height = sum(section_heights)
    column_count = max(1, math.ceil(total_section_height / max_column_height))

    columns: list[list[dict[str, object]]] = [[] for _ in range(column_count)]
    column_heights = [0] * column_count
    for entry, section_height in zip(legend_entries, section_heights, strict=True):
        column_index = min(range(column_count), key=lambda idx: column_heights[idx])
        columns[column_index].append(entry)
        column_heights[column_index] += section_height

    width = (margin * 2) + (column_width * column_count) + (column_gap * max(column_count - 1, 0))
    height = title_height + margin + max(column_heights, default=0)

    body_parts = [
        '  <g filter="url(#shadow)">',
        f'    <rect x="24" y="24" width="{width - 48}" height="{height - 48}" rx="28" fill="{SURFACE}"/>',
        "  </g>",
        f'  <text x="{margin}" y="78" fill="{TEXT}" font-size="42" font-weight="700" font-family="Georgia, serif">Lichess opening family and variation legend</text>',
        f'  <text x="{margin}" y="112" fill="{MUTED}" font-size="24" font-family="Georgia, serif">Family headers define the hue. Variation rows keep that hue and vary only by shade.</text>',
    ]

    for column_index, entries in enumerate(columns):
        x = margin + column_index * (column_width + column_gap)
        y = title_height
        for entry in entries:
            family = entry["family"]  # type: ignore[assignment]
            family_color = entry["family_color"]  # type: ignore[assignment]
            variations = entry["variations"]  # type: ignore[assignment]
            body_parts.append(
                f'  <rect x="{x}" y="{y}" width="20" height="20" rx="4" fill="{family_color}"/>'
                f'  <text x="{x + 30}" y="{y + 16}" fill="{TEXT}" font-size="22" font-weight="700" font-family="Georgia, serif">{html.escape(str(family))}</text>'
            )
            y += family_header_height
            for variation in variations:
                body_parts.append(
                    f'  <rect x="{x + 8}" y="{y - 14}" width="12" height="12" rx="3" fill="{variation["color"]}"/>'
                    f'  <text x="{x + 30}" y="{y - 2}" fill="{MUTED}" font-size="18" font-family="Georgia, serif">{html.escape(str(variation["name"]))}</text>'
                )
                y += variation_row_height
            y += family_gap

    body = "\n".join(body_parts)
    return _svg_root(width, height, body, "Lichess opening family and variation legend")


def build_output_rows(records: list[OpeningRecord]) -> list[dict[str, object]]:
    return [
        {
            "eco": record.eco,
            "name": record.name,
            "family": record.family,
            "variation": record.variation,
            "pgn": record.pgn,
            "position_fen": record.position_fen,
            "umap_x": record.umap_x,
            "umap_y": record.umap_y,
            "color_hex": record.color_hex,
            "family_hue": record.family_hue,
            "duplicate_group_size": record.duplicate_group_size,
            "duplicate_group_rank": record.duplicate_group_rank,
        }
        for record in records
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "eco",
        "name",
        "family",
        "variation",
        "pgn",
        "position_fen",
        "umap_x",
        "umap_y",
        "color_hex",
        "family_hue",
        "duplicate_group_size",
        "duplicate_group_rank",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    rows: list[dict[str, object]],
    *,
    source_ref: str,
    source_urls: list[str],
    seed: int,
    n_neighbors: int,
    min_dist: float,
    unique_positions: int,
    duplicate_group_count: int,
) -> None:
    families = {str(row["family"]) for row in rows}
    variations = {(str(row["family"]), str(row["variation"])) for row in rows}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_ref": source_ref,
        "source_urls": source_urls,
        "row_count": len(rows),
        "family_count": len(families),
        "variation_count": len(variations),
        "unique_occupancy_count": unique_positions,
        "duplicate_group_count": duplicate_group_count,
        "umap_parameters": {
            "metric": "hamming",
            "n_components": 2,
            "init": "spectral",
            "random_state": seed,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        },
        "openings": rows,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def generate_opening_umap(
    *,
    source_ref: str,
    graphics_dir: Path,
    artifacts_dir: Path,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    fetch_text: Callable[[str], str] = fetch_source_text,
    volume_files: tuple[str, ...] = OPENING_VOLUME_FILES,
) -> dict[str, Path]:
    rows, source_urls = fetch_opening_rows(
        source_ref,
        fetch_text=fetch_text,
        volume_files=volume_files,
    )
    records = build_opening_records(rows)
    family_hues, family_colors, variation_colors = assign_family_colors(records)
    unique_positions, duplicate_group_count = annotate_duplicate_groups(records)
    embedding = fit_umap_embedding(
        records,
        seed=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    apply_embedding(records, embedding)

    graphics_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    scatter_svg = render_scatter_svg(records, unique_positions)
    legend_entries = build_legend_entries(records, family_colors, variation_colors)
    legend_svg = render_legend_svg(legend_entries)
    output_rows = build_output_rows(records)

    scatter_path = graphics_dir / "lichess_openings_umap.svg"
    legend_path = graphics_dir / "lichess_openings_umap_legend.svg"
    csv_path = artifacts_dir / "lichess_openings_umap_key.csv"
    json_path = artifacts_dir / "lichess_openings_umap_embedding.json"

    scatter_path.write_text(scatter_svg, encoding="utf-8")
    legend_path.write_text(legend_svg, encoding="utf-8")
    write_csv(csv_path, output_rows)
    write_json(
        json_path,
        output_rows,
        source_ref=source_ref,
        source_urls=source_urls,
        seed=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        unique_positions=unique_positions,
        duplicate_group_count=duplicate_group_count,
    )

    return {
        "scatter_svg": scatter_path,
        "legend_svg": legend_path,
        "key_csv": csv_path,
        "embedding_json": json_path,
    }


def main(
    argv: list[str] | None = None,
    *,
    fetch_text: Callable[[str], str] = fetch_source_text,
    volume_files: tuple[str, ...] = OPENING_VOLUME_FILES,
) -> int:
    args = parse_args(argv)
    generate_opening_umap(
        source_ref=args.source_ref,
        graphics_dir=Path(args.graphics_dir),
        artifacts_dir=Path(args.artifacts_dir),
        seed=args.seed,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        fetch_text=fetch_text,
        volume_files=volume_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
