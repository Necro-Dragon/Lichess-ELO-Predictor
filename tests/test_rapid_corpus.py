from __future__ import annotations

import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from chess_distance import ChessPosition
from chess_distance.rapid_corpus import build_rapid_40ply_corpus


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


def position_after_target_or_final(text: str, game_index: int, target_plies: int) -> tuple[np.ndarray, int]:
    games = []
    text_stream = io.StringIO(text)
    while True:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            break
        games.append(game)

    game = games[game_index]
    board = game.board()
    recorded = ChessPosition.from_board(board).as_array()
    plies = 0
    for move in game.mainline_moves():
        board.push(move)
        plies += 1
        recorded = ChessPosition.from_board(board).as_array()
        if plies >= target_plies:
            break
    return recorded.copy(), plies


class RapidCorpusTests(unittest.TestCase):
    def test_from_board_matches_from_fen(self) -> None:
        board = chess.Board()
        board.push_san("e4")
        board.push_san("c5")
        board.push_san("Nf3")

        from_board = ChessPosition.from_board(board).as_array()
        from_fen = ChessPosition.from_fen(board.fen()).as_array()
        np.testing.assert_array_equal(from_board, from_fen)

    def test_build_corpus_writes_expected_vectors_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sample_dir = root / "sample"
            outdir = root / "out"
            sample_dir.mkdir()

            chunk_one = sample_dir / "sample_part_01.pgn.zst"
            chunk_two = sample_dir / "sample_part_02.pgn.zst"
            write_zst(chunk_one, CHUNK_01_TEXT)
            write_zst(chunk_two, CHUNK_02_TEXT)

            manifest = {
                "sample_size": 3,
                "chunk_files": [chunk_one.name, chunk_two.name],
            }
            (sample_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            outputs = build_rapid_40ply_corpus(
                sample_dir=sample_dir,
                outdir=outdir,
                target_plies=40,
            )

            self.assertTrue(outputs["npz"].exists())
            self.assertTrue(outputs["manifest"].exists())

            bundle = np.load(outputs["npz"])
            np.testing.assert_array_equal(bundle["white_elo"], np.array([734, 835, 1117], dtype=np.int16))
            np.testing.assert_array_equal(bundle["black_elo"], np.array([593, 1528, 1136], dtype=np.int16))
            np.testing.assert_array_equal(bundle["plies_recorded"], np.array([40, 18, 40], dtype=np.uint16))

            long_vector, long_plies = position_after_target_or_final(CHUNK_01_TEXT, 0, 40)
            short_vector, short_plies = position_after_target_or_final(CHUNK_01_TEXT, 1, 40)
            third_vector, third_plies = position_after_target_or_final(CHUNK_02_TEXT, 0, 40)

            self.assertEqual(long_plies, 40)
            self.assertEqual(short_plies, 18)
            self.assertEqual(third_plies, 40)
            np.testing.assert_array_equal(bundle["position_vectors"][0], long_vector)
            np.testing.assert_array_equal(bundle["position_vectors"][1], short_vector)
            np.testing.assert_array_equal(bundle["position_vectors"][2], third_vector)

            self.assertEqual(bundle["position_vectors"].dtype, np.uint8)
            self.assertEqual(bundle["white_elo"].dtype, np.int16)
            self.assertEqual(bundle["black_elo"].dtype, np.int16)
            self.assertEqual(bundle["plies_recorded"].dtype, np.uint16)
            self.assertEqual(bundle["position_vectors"].shape, (3, 64))

            payload = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(payload["row_count"], 3)
            self.assertEqual(payload["reached_target_plies_count"], 2)
            self.assertEqual(payload["padded_short_game_count"], 1)
            self.assertEqual(payload["arrays"]["position_vectors"]["shape"], [3, 64])
            self.assertIn("directly comparable with chess_distance", " ".join(payload["notes"]))


if __name__ == "__main__":
    unittest.main()
