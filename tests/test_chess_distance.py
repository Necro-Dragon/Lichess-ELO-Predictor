from __future__ import annotations

import unittest

import numpy as np

from chess_distance.position import (
    NUMBA_AVAILABLE,
    ChessPosition,
    _distance_many_numba,
    _distance_many_numpy,
    _pair_distance_numba,
    _pair_distance_numpy,
    stack_positions,
)


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


class ChessPositionTests(unittest.TestCase):
    def test_same_position_has_zero_distance(self) -> None:
        position = ChessPosition.from_fen(START_FEN)
        self.assertEqual(position - position, 0)

    def test_single_move_counts_source_and_destination(self) -> None:
        start = ChessPosition.from_fen(START_FEN)
        moved = ChessPosition.from_fen(E4_FEN)
        self.assertEqual(start - moved, 2)

    def test_piece_type_change_on_one_square_counts_once(self) -> None:
        pawn = ChessPosition.from_fen("8/8/8/8/8/8/8/P7 w - - 0 1")
        knight = ChessPosition.from_fen("8/8/8/8/8/8/8/N7 w - - 0 1")
        self.assertEqual(pawn - knight, 1)

    def test_metadata_fields_do_not_change_distance(self) -> None:
        first = ChessPosition.from_fen("8/8/8/8/8/8/8/K6k w KQkq e3 7 42")
        second = ChessPosition.from_fen("8/8/8/8/8/8/8/K6k b - - 0 1")
        self.assertEqual(first - second, 0)

    def test_invalid_piece_placement_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ChessPosition.from_fen("8/8/8/8/8/8/8/9 w - - 0 1")

        with self.assertRaises(ValueError):
            ChessPosition.from_fen("8/8/8/8/8/8/8/7X w - - 0 1")

    def test_distance_to_many_matches_pairwise_subtraction(self) -> None:
        anchor = ChessPosition.from_fen(START_FEN)
        positions = [
            ChessPosition.from_fen(START_FEN),
            ChessPosition.from_fen(E4_FEN),
            ChessPosition.from_fen("8/8/8/8/8/8/8/K6k w - - 0 1"),
        ]
        expected = np.array([anchor - position for position in positions], dtype=np.uint8)
        actual = anchor.distance_to_many(positions)
        np.testing.assert_array_equal(actual, expected)

    def test_stack_positions_shape_dtype_and_hamming_equivalence(self) -> None:
        positions = [
            ChessPosition.from_fen(START_FEN),
            ChessPosition.from_fen(E4_FEN),
        ]
        stacked = stack_positions(positions)
        self.assertEqual(stacked.shape, (2, 64))
        self.assertEqual(stacked.dtype, np.uint8)
        self.assertTrue(stacked.flags["C_CONTIGUOUS"])

        hamming_distance = np.mean(stacked[0] != stacked[1])
        raw_distance = positions[0] - positions[1]
        self.assertEqual(hamming_distance, raw_distance / 64)

    def test_as_array_is_read_only(self) -> None:
        position = ChessPosition.from_fen(START_FEN)
        board = position.as_array()
        self.assertFalse(board.flags["WRITEABLE"])
        with self.assertRaises(ValueError):
            board[0] = 99

    def test_distance_to_many_accepts_matrix(self) -> None:
        anchor = ChessPosition.from_fen(START_FEN)
        matrix = stack_positions(
            [
                ChessPosition.from_fen(START_FEN),
                ChessPosition.from_fen(E4_FEN),
            ]
        )
        np.testing.assert_array_equal(anchor.distance_to_many(matrix), np.array([0, 2], dtype=np.uint8))

    def test_numba_and_numpy_kernels_match(self) -> None:
        left = ChessPosition.from_fen(START_FEN).as_array()
        right = ChessPosition.from_fen(E4_FEN).as_array()
        many = stack_positions(
            [
                ChessPosition.from_fen(START_FEN),
                ChessPosition.from_fen(E4_FEN),
            ]
        )

        self.assertEqual(_pair_distance_numpy(left, right), 2)
        np.testing.assert_array_equal(_distance_many_numpy(left, many), np.array([0, 2], dtype=np.uint8))

        if not NUMBA_AVAILABLE:
            self.skipTest("Numba is not installed in this environment")

        self.assertEqual(_pair_distance_numba(left, right), _pair_distance_numpy(left, right))
        np.testing.assert_array_equal(_distance_many_numba(left, many), _distance_many_numpy(left, many))


if __name__ == "__main__":
    unittest.main()
