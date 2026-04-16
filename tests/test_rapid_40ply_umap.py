from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from chess_distance.rapid_40ply_umap import generate_rapid_40ply_umap, load_rapid_game_records, render_scatter_svg


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
CHUNK_01_TEXT = (FIXTURE_DIR / "rapid_chunk_01.pgn").read_text(encoding="utf-8")
CHUNK_02_TEXT = (FIXTURE_DIR / "rapid_chunk_02.pgn").read_text(encoding="utf-8")


def write_zst(path: Path, text: str) -> None:
    subprocess.run(
        ["zstd", "-19", "-q", "-o", str(path)],
        input=text,
        text=True,
        check=True,
    )


class Rapid40PlyUmapTests(unittest.TestCase):
    def test_load_records_extracts_opening_metadata_and_plies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_dir = root / "sample"
            sample_dir.mkdir()
            chunk_one = sample_dir / "sample_part_01.pgn.zst"
            chunk_two = sample_dir / "sample_part_02.pgn.zst"
            write_zst(chunk_one, CHUNK_01_TEXT)
            write_zst(chunk_two, CHUNK_02_TEXT)
            (sample_dir / "manifest.json").write_text(
                json.dumps({"sample_size": 3, "chunk_files": [chunk_one.name, chunk_two.name]}),
                encoding="utf-8",
            )

            records, chunk_files, reached_target_count, padded_short_game_count = load_rapid_game_records(
                sample_dir,
                target_plies=40,
                max_chunks=2,
            )

            self.assertEqual(len(records), 3)
            self.assertEqual(chunk_files, [chunk_one.name, chunk_two.name])
            self.assertEqual(reached_target_count, 2)
            self.assertEqual(padded_short_game_count, 1)
            self.assertEqual(records[0].family, "Bird Opening")
            self.assertEqual(records[0].variation, "Root")
            self.assertEqual(records[1].family, "Vienna Game")
            self.assertEqual(records[1].variation, "Anderssen Defense")
            self.assertEqual(records[1].plies_recorded, 18)
            self.assertEqual(records[2].plies_recorded, 40)

    def test_scatter_svg_has_no_visible_top_stat_cards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_dir = root / "sample"
            sample_dir.mkdir()
            chunk_one = sample_dir / "sample_part_01.pgn.zst"
            write_zst(chunk_one, CHUNK_01_TEXT)
            (sample_dir / "manifest.json").write_text(
                json.dumps({"sample_size": 2, "chunk_files": [chunk_one.name]}),
                encoding="utf-8",
            )

            records, _chunk_files, _reached_target_count, _padded_short_game_count = load_rapid_game_records(
                sample_dir,
                target_plies=40,
                max_chunks=1,
            )
            for index, record in enumerate(records):
                record.umap_x = float(index)
                record.umap_y = float(index * 2)
            scatter_svg = render_scatter_svg(records)
            self.assertIn("Rapid 40-ply UMAP embedding", scatter_svg)
            self.assertNotIn(">Openings<", scatter_svg)
            self.assertNotIn(">Families<", scatter_svg)

    def test_generate_umap_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_dir = root / "sample"
            graphics_dir = root / "graphics"
            artifacts_dir = root / "artifacts"
            sample_dir.mkdir()
            chunk_one = sample_dir / "sample_part_01.pgn.zst"
            chunk_two = sample_dir / "sample_part_02.pgn.zst"
            write_zst(chunk_one, CHUNK_01_TEXT)
            write_zst(chunk_two, CHUNK_02_TEXT)
            (sample_dir / "manifest.json").write_text(
                json.dumps({"sample_size": 3, "chunk_files": [chunk_one.name, chunk_two.name]}),
                encoding="utf-8",
            )

            outputs = generate_rapid_40ply_umap(
                sample_dir=sample_dir,
                graphics_dir=graphics_dir,
                artifacts_dir=artifacts_dir,
                target_plies=40,
                max_chunks=2,
                seed=7,
                n_neighbors=2,
                min_dist=0.05,
            )

            self.assertTrue(outputs["scatter_svg"].exists())
            self.assertTrue(outputs["legend_svg"].exists())
            self.assertTrue(outputs["key_csv"].exists())
            self.assertTrue(outputs["embedding_json"].exists())

            with outputs["key_csv"].open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertIn("plies_recorded", rows[0])

            payload = json.loads(outputs["embedding_json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["row_count"], 3)
            self.assertEqual(payload["reached_target_plies_count"], 2)
            self.assertEqual(payload["padded_short_game_count"], 1)
            self.assertIn("Vienna Game", outputs["legend_svg"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
