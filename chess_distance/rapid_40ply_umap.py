"""Generate a UMAP embedding for rapid-game positions at ply 40."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import umap

from .opening_umap import (
    BACKGROUND,
    GRID,
    MUTED,
    SURFACE,
    TEXT,
    assign_family_colors,
    build_legend_entries,
    parse_opening_taxonomy,
    render_legend_svg,
)
from .rapid_corpus import extract_position_vector, iter_games_from_zst, load_sample_manifest, parse_rating


@dataclass(slots=True)
class RapidGameRecord:
    game_index: int
    chunk_file: str
    site: str
    eco: str
    opening: str
    family: str
    variation: str
    white_elo: int
    black_elo: int
    plies_recorded: int
    position_vector: np.ndarray
    color_hex: str = ""
    family_hue: float = 0.0
    umap_x: float = 0.0
    umap_y: float = 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-dir",
        default="data/rapid_2026-03_sample_1500000",
        help="Directory containing the sampled rapid PGN chunks and manifest.",
    )
    parser.add_argument(
        "--graphics-dir",
        default="graphics",
        help="Directory to write SVG outputs.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory to write CSV/JSON outputs.",
    )
    parser.add_argument(
        "--target-plies",
        type=int,
        default=40,
        help="Ply depth to encode, using the final position when a game ends earlier.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=1,
        help="How many source PGN chunks to process. Defaults to the first chunk for runtime reasons.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260406,
        help="UMAP random seed.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
        help="UMAP n_neighbors parameter.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.10,
        help="UMAP min_dist parameter.",
    )
    return parser.parse_args(argv)


def load_rapid_game_records(
    sample_dir: Path,
    *,
    target_plies: int,
    max_chunks: int,
) -> tuple[list[RapidGameRecord], list[str], int, int]:
    manifest = load_sample_manifest(sample_dir)
    chunk_files = list(manifest["chunk_files"])
    if max_chunks <= 0:
        raise ValueError("max_chunks must be positive")
    selected_chunk_files = chunk_files[:max_chunks]

    records: list[RapidGameRecord] = []
    reached_target_count = 0
    padded_short_game_count = 0
    global_index = 0

    for chunk_index, chunk_name in enumerate(selected_chunk_files, start=1):
        chunk_path = sample_dir / chunk_name
        for game in iter_games_from_zst(chunk_path):
            headers = game.headers
            opening = headers.get("Opening", "").strip() or "Unknown"
            family, variation = parse_opening_taxonomy(opening)
            vector, plies_recorded = extract_position_vector(game, target_plies)
            if plies_recorded >= target_plies:
                reached_target_count += 1
            else:
                padded_short_game_count += 1

            records.append(
                RapidGameRecord(
                    game_index=global_index,
                    chunk_file=chunk_name,
                    site=headers.get("Site", "").strip(),
                    eco=headers.get("ECO", "").strip(),
                    opening=opening,
                    family=family,
                    variation=variation,
                    white_elo=parse_rating(headers.get("WhiteElo", ""), field_name="WhiteElo"),
                    black_elo=parse_rating(headers.get("BlackElo", ""), field_name="BlackElo"),
                    plies_recorded=plies_recorded,
                    position_vector=np.ascontiguousarray(vector, dtype=np.uint8),
                )
            )
            global_index += 1

        print(
            f"Loaded chunk {chunk_index}/{len(selected_chunk_files)} ({global_index:,} games total)",
            file=sys.stderr,
        )

    if not records:
        raise ValueError("no rapid games were loaded")
    return records, selected_chunk_files, reached_target_count, padded_short_game_count


def fit_umap_embedding(
    records: list[RapidGameRecord],
    *,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    row_count = len(records)
    if row_count < 3:
        raise ValueError("UMAP embedding requires at least 3 games")
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2")
    if n_neighbors >= row_count:
        raise ValueError(f"n_neighbors must be less than the row count ({row_count})")

    matrix = np.ascontiguousarray(np.stack([record.position_vector for record in records], axis=0), dtype=np.uint8)
    init = "random" if row_count <= 3 else "spectral"
    reducer = umap.UMAP(
        n_components=2,
        metric="hamming",
        init=init,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    return reducer.fit_transform(matrix)


def apply_embedding(records: list[RapidGameRecord], embedding: np.ndarray) -> None:
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


def render_scatter_svg(records: list[RapidGameRecord]) -> str:
    width, height = 1600, 1150
    chart_x = 110
    chart_y = 100
    chart_w = 1360
    chart_h = 900

    xs = [record.umap_x for record in records]
    ys = [record.umap_y for record in records]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    x_ticks = []
    y_ticks = []
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
            f'  <circle cx="{x_position:.2f}" cy="{y_position:.2f}" r="2.4" fill="{record.color_hex}" fill-opacity="0.50"/>'
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
    return _svg_root(width, height, body, "Rapid 40-ply UMAP embedding")


def build_output_rows(records: list[RapidGameRecord]) -> list[dict[str, object]]:
    return [
        {
            "game_index": record.game_index,
            "chunk_file": record.chunk_file,
            "site": record.site,
            "eco": record.eco,
            "opening": record.opening,
            "family": record.family,
            "variation": record.variation,
            "white_elo": record.white_elo,
            "black_elo": record.black_elo,
            "plies_recorded": record.plies_recorded,
            "umap_x": record.umap_x,
            "umap_y": record.umap_y,
            "color_hex": record.color_hex,
            "family_hue": record.family_hue,
        }
        for record in records
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "game_index",
        "chunk_file",
        "site",
        "eco",
        "opening",
        "family",
        "variation",
        "white_elo",
        "black_elo",
        "plies_recorded",
        "umap_x",
        "umap_y",
        "color_hex",
        "family_hue",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    rows: list[dict[str, object]],
    *,
    sample_dir: Path,
    source_chunk_files: list[str],
    target_plies: int,
    seed: int,
    n_neighbors: int,
    min_dist: float,
    reached_target_count: int,
    padded_short_game_count: int,
) -> None:
    families = {str(row["family"]) for row in rows}
    variations = {(str(row["family"]), str(row["variation"])) for row in rows}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sample_dir": str(sample_dir.resolve()),
        "source_chunk_files": source_chunk_files,
        "row_count": len(rows),
        "family_count": len(families),
        "variation_count": len(variations),
        "target_plies": target_plies,
        "reached_target_plies_count": reached_target_count,
        "padded_short_game_count": padded_short_game_count,
        "umap_parameters": {
            "metric": "hamming",
            "n_components": 2,
            "init": "spectral",
            "random_state": seed,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        },
        "games": rows,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def generate_rapid_40ply_umap(
    *,
    sample_dir: Path,
    graphics_dir: Path,
    artifacts_dir: Path,
    target_plies: int,
    max_chunks: int,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> dict[str, Path]:
    records, source_chunk_files, reached_target_count, padded_short_game_count = load_rapid_game_records(
        sample_dir,
        target_plies=target_plies,
        max_chunks=max_chunks,
    )
    _family_hues, family_colors, variation_colors = assign_family_colors(records)
    print(f"Fitting UMAP for {len(records):,} rapid games...", file=sys.stderr)
    embedding = fit_umap_embedding(
        records,
        seed=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    apply_embedding(records, embedding)

    graphics_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    scatter_svg = render_scatter_svg(records)
    legend_svg = render_legend_svg(build_legend_entries(records, family_colors, variation_colors))
    output_rows = build_output_rows(records)

    scatter_path = graphics_dir / "rapid_40ply_umap.svg"
    legend_path = graphics_dir / "rapid_40ply_umap_legend.svg"
    csv_path = artifacts_dir / "rapid_40ply_umap_key.csv"
    json_path = artifacts_dir / "rapid_40ply_umap_embedding.json"

    print("Writing rapid 40-ply UMAP artifacts...", file=sys.stderr)
    scatter_path.write_text(scatter_svg, encoding="utf-8")
    legend_path.write_text(legend_svg, encoding="utf-8")
    write_csv(csv_path, output_rows)
    write_json(
        json_path,
        output_rows,
        sample_dir=sample_dir,
        source_chunk_files=source_chunk_files,
        target_plies=target_plies,
        seed=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        reached_target_count=reached_target_count,
        padded_short_game_count=padded_short_game_count,
    )
    return {
        "scatter_svg": scatter_path,
        "legend_svg": legend_path,
        "key_csv": csv_path,
        "embedding_json": json_path,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generate_rapid_40ply_umap(
        sample_dir=Path(args.sample_dir),
        graphics_dir=Path(args.graphics_dir),
        artifacts_dir=Path(args.artifacts_dir),
        target_plies=args.target_plies,
        max_chunks=args.max_chunks,
        seed=args.seed,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
