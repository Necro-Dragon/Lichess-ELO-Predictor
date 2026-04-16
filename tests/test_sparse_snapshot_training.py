from __future__ import annotations

import io
import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

import chess.pgn
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency in local env
    torch = None

from chess_distance.rating_band_training import (
    SparseGameRatingBandModel,
    collate_sparse_snapshot_batch,
    export_game_embeddings,
    train_rating_band_model,
)
from chess_distance.sparse_snapshot_corpus import (
    RatingBandSpec,
    build_sparse_snapshot_corpus,
    extract_sparse_game_snapshots,
    load_sparse_snapshot_arrays,
)


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


def read_games(text: str) -> list[chess.pgn.Game]:
    games = []
    stream = io.StringIO(text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        games.append(game)
    return games


def create_large_sample_dir(root: Path) -> Path:
    sample_dir = root / "sample"
    sample_dir.mkdir()
    combined_text = CHUNK_01_TEXT + "\n" + CHUNK_02_TEXT
    chunk_files = []
    for index in range(8):
        chunk_path = sample_dir / f"sample_part_{index + 1:02d}.pgn.zst"
        write_zst(chunk_path, combined_text)
        chunk_files.append(chunk_path.name)
    manifest = {
        "sample_size": 24,
        "chunk_files": chunk_files,
    }
    (sample_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return sample_dir


class SparseSnapshotCorpusTests(unittest.TestCase):
    def test_sparse_snapshot_extraction_uses_step_and_final_position(self) -> None:
        long_game = read_games(CHUNK_01_TEXT)[0]
        short_game = read_games(CHUNK_02_TEXT)[0]

        long_sparse = extract_sparse_game_snapshots(long_game, snapshot_step=7)
        short_sparse = extract_sparse_game_snapshots(short_game, snapshot_step=7)

        np.testing.assert_array_equal(
            long_sparse["snapshot_plies"],
            np.array([0, 7, 14, 21, 28, 35, 40], dtype=np.uint16),
        )
        np.testing.assert_array_equal(
            short_sparse["snapshot_plies"],
            np.array([0, 7, 14, 18], dtype=np.uint16),
        )
        self.assertEqual(int(long_sparse["final_plies"]), 40)
        self.assertEqual(int(short_sparse["final_plies"]), 18)
        self.assertEqual(long_sparse["board_codes"].shape, (7, 64))
        self.assertEqual(short_sparse["board_codes"].shape, (4, 64))

    def test_rating_band_edges_and_encoding(self) -> None:
        spec = RatingBandSpec.from_train_ratings(np.array([734, 1528], dtype=np.int16), band_width=200)
        ratings = np.array([599, 600, 799, 800, 1599, 1600], dtype=np.int16)
        encoded = spec.encode_many(ratings)
        np.testing.assert_array_equal(encoded, np.array([0, 1, 1, 2, 5, 6], dtype=np.int16))
        self.assertEqual(spec.to_dict()["labels"][0], "<600")
        self.assertEqual(spec.to_dict()["labels"][-1], ">=1600")

    def test_subset_selection_is_reproducible_and_not_first_slice(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_dir = create_large_sample_dir(root)
            out_one = root / "out_one"
            out_two = root / "out_two"

            outputs_one = build_sparse_snapshot_corpus(
                sample_dir=sample_dir,
                outdir=out_one,
                sample_size=6,
                snapshot_step=7,
                seed=17,
                band_width=200,
                buffer_games=2,
            )
            outputs_two = build_sparse_snapshot_corpus(
                sample_dir=sample_dir,
                outdir=out_two,
                sample_size=6,
                snapshot_step=7,
                seed=17,
                band_width=200,
                buffer_games=2,
            )

            _manifest_one, arrays_one = load_sparse_snapshot_arrays(outputs_one["manifest"])
            _manifest_two, arrays_two = load_sparse_snapshot_arrays(outputs_two["manifest"])
            expected = np.array(sorted(random.Random(17).sample(range(24), 6)), dtype=np.int32)

            np.testing.assert_array_equal(arrays_one["source_game_index"], expected)
            np.testing.assert_array_equal(arrays_two["source_game_index"], expected)
            self.assertFalse(np.array_equal(expected, np.arange(6, dtype=np.int32)))


@unittest.skipUnless(torch is not None, "PyTorch is required for neural training tests")
class SparseSnapshotTrainingTests(unittest.TestCase):
    def test_model_encode_games_returns_128_dimensional_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_dir = create_large_sample_dir(root)
            corpus_dir = root / "corpus"
            outputs = build_sparse_snapshot_corpus(
                sample_dir=sample_dir,
                outdir=corpus_dir,
                sample_size=24,
                snapshot_step=7,
                seed=11,
                band_width=200,
                buffer_games=4,
            )
            _manifest, arrays = load_sparse_snapshot_arrays(outputs["manifest"])
            samples = []
            for game_row in range(2):
                start = int(arrays["offsets"][game_row])
                stop = start + int(arrays["lengths"][game_row])
                samples.append(
                    {
                        "board_codes": arrays["board_codes"][start:stop],
                        "side_to_move": arrays["side_to_move"][start:stop],
                        "castling_rights": arrays["castling_rights"][start:stop],
                        "en_passant_file": arrays["en_passant_file"][start:stop],
                        "snapshot_plies": arrays["snapshot_plies"][start:stop],
                        "length": int(arrays["lengths"][game_row]),
                        "final_plies": int(arrays["final_plies"][game_row]),
                        "game_index": int(arrays["source_game_index"][game_row]),
                        "white_elo": int(arrays["white_elo"][game_row]),
                        "black_elo": int(arrays["black_elo"][game_row]),
                        "white_band": int(arrays["white_band"][game_row]),
                        "black_band": int(arrays["black_band"][game_row]),
                        "split_id": int(arrays["split_id"][game_row]),
                        "site": str(arrays["site"][game_row]),
                        "opening": str(arrays["opening"][game_row]),
                        "eco": str(arrays["eco"][game_row]),
                    }
                )
            batch = collate_sparse_snapshot_batch(samples)
            model = SparseGameRatingBandModel(band_count=7)
            outputs = model(batch)

            self.assertEqual(outputs["embedding"].shape, (2, 128))
            self.assertEqual(outputs["white_band_logits"].shape, (2, 7))
            self.assertEqual(outputs["black_band_logits"].shape, (2, 7))
            self.assertEqual(model.encode_games(batch).shape, (2, 128))

    def test_training_and_embedding_export_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_dir = create_large_sample_dir(root)
            corpus_dir = root / "corpus"
            checkpoint_dir = root / "checkpoints"
            graphics_dir = root / "graphics"
            report_path = root / "report" / "training_report.md"
            embeddings_path = root / "artifacts" / "test_embeddings.npz"

            build_sparse_snapshot_corpus(
                sample_dir=sample_dir,
                outdir=corpus_dir,
                sample_size=24,
                snapshot_step=7,
                seed=13,
                band_width=200,
                buffer_games=4,
            )
            outputs = train_rating_band_model(
                corpus_path=corpus_dir,
                checkpoint_dir=checkpoint_dir,
                graphics_dir=graphics_dir,
                report_path=report_path,
                epochs=2,
                patience=2,
                seed=19,
                learning_rate=1e-3,
                weight_decay=1e-4,
                batch_size_accelerator=4,
                batch_size_cpu=4,
            )

            self.assertTrue(outputs["history_json"].exists())
            self.assertTrue(outputs["metrics_summary_json"].exists())
            self.assertTrue(outputs["best_checkpoint"].exists())
            self.assertTrue(outputs["loss_chart_svg"].exists())
            self.assertTrue(outputs["accuracy_chart_svg"].exists())
            self.assertTrue(outputs["report_md"].exists())

            history_payload = json.loads(outputs["history_json"].read_text(encoding="utf-8"))
            self.assertTrue(history_payload["epochs"])
            self.assertIn("test", history_payload["epochs"][0])

            metrics_payload = json.loads(outputs["metrics_summary_json"].read_text(encoding="utf-8"))
            self.assertIn("best_test_metrics", metrics_payload)
            self.assertIn("elo_mae", metrics_payload["best_test_metrics"])

            report_text = outputs["report_md"].read_text(encoding="utf-8")
            self.assertIn("rating_band_training_loss.svg", report_text)
            self.assertIn("Test exact-Elo MAE", report_text)

            export_game_embeddings(
                checkpoint_path=outputs["best_checkpoint"],
                corpus_path=corpus_dir,
                split_name="test",
                out_path=embeddings_path,
            )
            self.assertTrue(embeddings_path.exists())
            bundle = np.load(embeddings_path)
            self.assertEqual(bundle["embeddings"].shape[1], 128)
            self.assertIn("white_elo", bundle.files)
            self.assertIn("opening", bundle.files)


if __name__ == "__main__":
    unittest.main()
