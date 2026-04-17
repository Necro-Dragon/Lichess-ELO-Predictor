"""Generate a UMAP embedding for the lichess opening taxonomy using model embeddings."""

from __future__ import annotations

import sys
print("Module loading...", flush=True)
sys.stdout.flush()

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
import torch

from .position import ChessPosition, stack_positions
from .rating_band_training import SparseGameRatingBandModel

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
    board: chess.Board
    color_hex: str = ""
    family_hue: float = 0.0
    duplicate_group_size: int = 1
    duplicate_group_rank: int = 1
    umap_x: float = 0.0
    umap_y: float = 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ref", default="master", help="Git ref to fetch from lichess-org/chess-openings.")
    parser.add_argument("--checkpoint", default="artifacts/rating_band_training/best.pt", help="Path to the trained model checkpoint.")
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


def replay_pgn_to_position_and_board(pgn: str) -> tuple[str, ChessPosition, chess.Board]:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise ValueError(f"could not parse PGN: {pgn!r}")
    if game.errors:
        raise ValueError(f"PGN contains parse errors: {pgn!r}")

    board = game.board()
    for move in game.mainline_moves():
        board.push(move)

    position_fen = board.fen()
    return position_fen, ChessPosition.from_fen(position_fen), board


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
        position_fen, position, board = replay_pgn_to_position_and_board(row["pgn"])
        records.append(
            OpeningRecord(
                eco=row["eco"],
                name=row["name"],
                pgn=row["pgn"],
                family=family,
                variation=variation,
                position_fen=position_fen,
                position=position,
                board=board,
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


def compute_model_embeddings(records: list[OpeningRecord], model: SparseGameRatingBandModel, device: torch.device) -> np.ndarray:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for record in records:
            board = record.board
            board_codes = record.position.as_array().astype(np.int64).reshape(1, 1, 64)
            side_to_move = int(board.turn)
            castling_rights = [
                int(bool(board.castling_rights & chess.BB_H1)),
                int(bool(board.castling_rights & chess.BB_A1)),
                int(bool(board.castling_rights & chess.BB_H8)),
                int(bool(board.castling_rights & chess.BB_A8)),
            ]
            en_passant_file = chess.square_file(board.ep_square) + 1 if board.ep_square else 0
            snapshot_plies = 0
            final_plies = 1
            length = 1

            batch = {
                "board_codes": torch.tensor(board_codes, dtype=torch.int64, device=device),
                "side_to_move": torch.tensor([[side_to_move]], dtype=torch.int64, device=device),
                "castling_rights": torch.tensor([castling_rights], dtype=torch.int64, device=device).unsqueeze(0),
                "en_passant_file": torch.tensor([[en_passant_file]], dtype=torch.int64, device=device),
                "snapshot_plies": torch.tensor([[snapshot_plies]], dtype=torch.int64, device=device),
                "final_plies": torch.tensor([final_plies], dtype=torch.int64, device=device),
                "lengths": torch.tensor([length], dtype=torch.int64, device=device),
            }
            embedding = model.encode_games(batch).cpu().numpy().squeeze()
            embeddings.append(embedding)
    return np.array(embeddings)


def fit_umap_embedding(
    embeddings: np.ndarray,
    *,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    if len(embeddings) < 3:
        raise ValueError("UMAP embedding requires at least 3 openings")
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2")
    if n_neighbors >= len(embeddings):
        raise ValueError(f"n_neighbors must be less than the row count ({len(embeddings)})")

    reducer = umap.UMAP(
        n_components=2,
        metric="euclidean",
        init="spectral",
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    return reducer.fit_transform(embeddings)


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
      <feDropShadow dx="0" dy="0" stdDeviation="14" flood-color="#b8ab95" flood-opacity="0.18"/>
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
            f'  <line x1="{chart_x - 12}" y1="{y_position:.2f}" x2="{chart_x}" y2="{y_position:.2f}" stroke="{TEXT}" stroke-width="1"/>'            f'  <text x="{chart_x - 38}" y="{y_position:.2f}" text-anchor="middle" fill="{MUTED}" font-size="22" font-family="Georgia, serif" transform="rotate(-90 {chart_x - 38} {y_position:.2f})">{y_value:.2f}</text>'
        )

        grid_lines.append(
            f'  <line x1="{x_position:.2f}" y1="{chart_y}" x2="{x_position:.2f}" y2="{chart_y + chart_h}" stroke="{GRID}" stroke-width="1"/>'
            f'  <line x1="{chart_x}" y1="{y_position:.2f}" x2="{chart_x + chart_w}" y2="{y_position:.2f}" stroke="{GRID}" stroke-width="1"/>'
        )

    points = []
    for record in records:
        x = _project_value(record.umap_x, min_x, max_x, chart_x, chart_w)
        y = _project_value(record.umap_y, max_y, min_y, chart_y, chart_h)  # inverted
        points.append(
            f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{record.color_hex}" filter="url(#shadow)"/>'
        )

    body = "\n".join(grid_lines + x_ticks + y_ticks + points)
    title = f"Lichess Opening Taxonomy Model Embeddings UMAP (n={len(records)}, unique={unique_positions})"
    return _svg_root(width, height, body, title)


def build_legend_entries(family_colors: dict[str, str], variation_colors: dict[tuple[str, str], str]) -> list[tuple[str, str, str]]:
    entries = []
    for family, color in family_colors.items():
        entries.append((family, color, "family"))
    for (family, variation), color in variation_colors.items():
        if variation != "Root":
            entries.append((f"{family}: {variation}", color, "variation"))
    return sorted(entries, key=lambda x: x[0])


def render_legend_svg(entries: list[tuple[str, str, str]]) -> str:
    width, height = 400, 20 + 25 * len(entries)
    body = []
    for i, (label, color, kind) in enumerate(entries):
        y = 20 + i * 25
        body.append(f'  <circle cx="20" cy="{y}" r="8" fill="{color}"/>')
        body.append(f'  <text x="40" y="{y + 5}" fill="{TEXT}" font-size="16" font-family="Georgia, serif">{html.escape(label)}</text>')
    return _svg_root(width, height, "\n".join(body), "Legend")


def write_csv(records: list[OpeningRecord], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "eco", "name", "pgn", "family", "variation", "position_fen",
            "color_hex", "family_hue", "duplicate_group_size", "duplicate_group_rank",
            "umap_x", "umap_y"
        ])
        writer.writeheader()
        for record in records:
            writer.writerow({
                "eco": record.eco,
                "name": record.name,
                "pgn": record.pgn,
                "family": record.family,
                "variation": record.variation,
                "position_fen": record.position_fen,
                "color_hex": record.color_hex,
                "family_hue": record.family_hue,
                "duplicate_group_size": record.duplicate_group_size,
                "duplicate_group_rank": record.duplicate_group_rank,
                "umap_x": record.umap_x,
                "umap_y": record.umap_y,
            })


def write_json(records: list[OpeningRecord], path: Path) -> None:
    data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": [
            {
                "eco": record.eco,
                "name": record.name,
                "pgn": record.pgn,
                "family": record.family,
                "variation": record.variation,
                "position_fen": record.position_fen,
                "color_hex": record.color_hex,
                "family_hue": record.family_hue,
                "duplicate_group_size": record.duplicate_group_size,
                "duplicate_group_rank": record.duplicate_group_rank,
                "umap_x": record.umap_x,
                "umap_y": record.umap_y,
            }
            for record in records
        ]
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()

    print("Fetching opening data...")
    rows, urls = fetch_opening_rows(args.source_ref)
    print(f"Loaded {len(rows)} openings from {len(urls)} volumes.")

    print("Building opening records...")
    records = build_opening_records(rows)

    print("Assigning colors...")
    family_hues, family_colors, variation_colors = assign_family_colors(records)

    unique_positions, duplicate_groups = annotate_duplicate_groups(records)
    print(f"Found {unique_positions} unique positions, {duplicate_groups} duplicate groups.")

    print("Loading model...")
    device = torch.device("cpu")
    model = SparseGameRatingBandModel(band_count=16)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print("Computing model embeddings...")
    embeddings = compute_model_embeddings(records, model, device)
    print(f"Computed embeddings with shape {embeddings.shape}.")

    print("Fitting UMAP...")
    umap_embedding = fit_umap_embedding(
        embeddings,
        seed=args.seed,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )

    apply_embedding(records, umap_embedding)

    graphics_dir = Path(args.graphics_dir)
    graphics_dir.mkdir(parents=True, exist_ok=True)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Rendering scatter plot...")
    scatter_svg = render_scatter_svg(records, unique_positions)
    scatter_path = graphics_dir / "lichess_openings_model_umap.svg"
    scatter_path.write_text(scatter_svg, encoding="utf-8")
    print(f"Wrote {scatter_path}")

    print("Rendering legend...")
    legend_entries = build_legend_entries(family_colors, variation_colors)
    legend_svg = render_legend_svg(legend_entries)
    legend_path = graphics_dir / "lichess_openings_model_umap_legend.svg"
    legend_path.write_text(legend_svg, encoding="utf-8")
    print(f"Wrote {legend_path}")

    print("Writing CSV...")
    csv_path = artifacts_dir / "lichess_openings_model_umap.csv"
    write_csv(records, csv_path)
    print(f"Wrote {csv_path}")

    print("Writing JSON...")
    json_path = artifacts_dir / "lichess_openings_model_umap.json"
    write_json(records, json_path)
    print(f"Wrote {json_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())