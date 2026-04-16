from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from chess_distance.opening_umap import (
    assign_family_colors,
    build_opening_records,
    build_output_rows,
    main,
    parse_opening_taxonomy,
    parse_opening_tsv,
    render_legend_svg,
    render_scatter_svg,
    annotate_duplicate_groups,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "opening_sample.tsv"


def read_fixture_text() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


class OpeningUmapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = parse_opening_tsv(read_fixture_text())
        self.records = build_opening_records(self.rows)

    def test_taxonomy_parses_root_and_variation(self) -> None:
        self.assertEqual(parse_opening_taxonomy("Sicilian Defense"), ("Sicilian Defense", "Root"))
        self.assertEqual(
            parse_opening_taxonomy("Sicilian Defense: Najdorf Variation, English Attack"),
            ("Sicilian Defense", "Najdorf Variation"),
        )

    def test_family_colors_share_hue_and_variation_color(self) -> None:
        family_hues, _family_colors, variation_colors = assign_family_colors(self.records)
        self.assertEqual(family_hues["Amar Opening"], family_hues["Amar Opening"])
        self.assertEqual(
            variation_colors[("Sicilian Defense", "Najdorf Variation")],
            variation_colors[("Sicilian Defense", "Najdorf Variation")],
        )
        sicilian_records = [record for record in self.records if record.family == "Sicilian Defense"]
        najdorf_colors = {record.color_hex for record in sicilian_records if record.variation == "Najdorf Variation"}
        self.assertEqual(len(najdorf_colors), 1)
        root_colors = {record.color_hex for record in sicilian_records if record.variation == "Root"}
        self.assertEqual(len(root_colors), 1)
        self.assertNotEqual(next(iter(najdorf_colors)), next(iter(root_colors)))

    def test_replay_pgn_builds_final_board_position(self) -> None:
        amar = next(record for record in self.records if record.name == "Amar Opening")
        self.assertEqual(amar.position_fen, "rnbqkbnr/pppppppp/8/8/8/7N/PPPPPPPP/RNBQKB1R b KQkq - 1 1")

    def test_duplicate_rows_are_preserved(self) -> None:
        unique_positions, duplicate_group_count = annotate_duplicate_groups(self.records)
        duplicates = [record for record in self.records if record.duplicate_group_size > 1]
        self.assertEqual(unique_positions, len(self.records) - 1)
        self.assertEqual(duplicate_group_count, 1)
        self.assertEqual(len(duplicates), 2)
        self.assertEqual({record.duplicate_group_rank for record in duplicates}, {1, 2})

    def test_svg_renderers_include_expected_labels(self) -> None:
        unique_positions, _duplicate_group_count = annotate_duplicate_groups(self.records)
        assign_family_colors(self.records)
        for index, record in enumerate(self.records):
            record.umap_x = float(index)
            record.umap_y = float(index * 2)
        scatter_svg = render_scatter_svg(self.records, unique_positions)
        self.assertIn("Lichess opening UMAP embedding", scatter_svg)
        self.assertNotIn(">Openings<", scatter_svg)
        self.assertNotIn(">Families<", scatter_svg)
        self.assertNotIn(">Family + Variation<", scatter_svg)
        self.assertNotIn(">Unique Occupancies<", scatter_svg)

        _family_hues, family_colors, variation_colors = assign_family_colors(self.records)
        legend_svg = render_legend_svg(
            [
                {
                    "family": "Amar Opening",
                    "family_color": family_colors["Amar Opening"],
                    "variations": [
                        {"name": "Root", "color": variation_colors[("Amar Opening", "Root")]},
                        {"name": "Paris Gambit", "color": variation_colors[("Amar Opening", "Paris Gambit")]},
                    ],
                }
            ]
        )
        self.assertIn("Amar Opening", legend_svg)
        self.assertIn("Paris Gambit", legend_svg)

    def test_cli_smoke_writes_all_artifacts_and_counts(self) -> None:
        fixture_text = read_fixture_text()

        def fake_fetch_text(_url: str) -> str:
            return fixture_text

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            graphics_dir = root / "graphics"
            artifacts_dir = root / "artifacts"
            exit_code = main(
                [
                    "--source-ref",
                    "fixture",
                    "--graphics-dir",
                    str(graphics_dir),
                    "--artifacts-dir",
                    str(artifacts_dir),
                    "--seed",
                    "7",
                    "--n-neighbors",
                    "3",
                    "--min-dist",
                    "0.05",
                ],
                fetch_text=fake_fetch_text,
                volume_files=("fixture.tsv",),
            )
            self.assertEqual(exit_code, 0)

            scatter_path = graphics_dir / "lichess_openings_umap.svg"
            legend_path = graphics_dir / "lichess_openings_umap_legend.svg"
            csv_path = artifacts_dir / "lichess_openings_umap_key.csv"
            json_path = artifacts_dir / "lichess_openings_umap_embedding.json"

            self.assertTrue(scatter_path.exists())
            self.assertTrue(legend_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), len(self.rows))

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["row_count"], len(self.rows))
            self.assertEqual(len(payload["openings"]), len(self.rows))
            self.assertIn("Sicilian Defense", legend_path.read_text(encoding="utf-8"))
            scatter_svg = scatter_path.read_text(encoding="utf-8")
            self.assertIn("Lichess opening UMAP embedding", scatter_svg)
            self.assertNotIn(">Openings<", scatter_svg)

    def test_output_rows_match_record_count(self) -> None:
        assign_family_colors(self.records)
        annotate_duplicate_groups(self.records)
        for index, record in enumerate(self.records):
            record.umap_x = float(index)
            record.umap_y = float(-index)
        output_rows = build_output_rows(self.records)
        self.assertEqual(len(output_rows), len(self.records))
        self.assertEqual(output_rows[0]["family"], self.records[0].family)


if __name__ == "__main__":
    unittest.main()
