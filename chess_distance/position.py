"""Occupancy-only chess position distance backed by NumPy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    import chess
except Exception:  # pragma: no cover - optional dependency in non-board workflows
    chess = None

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None


PIECE_CODES = {
    "P": 1,
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": 7,
    "n": 8,
    "b": 9,
    "r": 10,
    "q": 11,
    "k": 12,
}

NUMBA_AVAILABLE = njit is not None
BOARD_WIDTH = 8
BOARD_SIZE = BOARD_WIDTH * BOARD_WIDTH


def _readonly_board(board: np.ndarray) -> np.ndarray:
    board = np.ascontiguousarray(board, dtype=np.uint8)
    if board.shape != (BOARD_SIZE,):
        raise ValueError(f"board array must have shape ({BOARD_SIZE},), got {board.shape}")
    board.setflags(write=False)
    return board


def _parse_piece_placement(piece_placement: str) -> np.ndarray:
    ranks = piece_placement.split("/")
    if len(ranks) != BOARD_WIDTH:
        raise ValueError("FEN piece placement must contain exactly 8 ranks")

    board = np.zeros(BOARD_SIZE, dtype=np.uint8)
    index = 0

    for rank in ranks:
        files_seen = 0
        for symbol in rank:
            if symbol.isdigit():
                empty_squares = int(symbol)
                if empty_squares < 1 or empty_squares > BOARD_WIDTH:
                    raise ValueError(f"invalid FEN digit: {symbol!r}")
                files_seen += empty_squares
                index += empty_squares
                continue

            piece_code = PIECE_CODES.get(symbol)
            if piece_code is None:
                raise ValueError(f"invalid FEN piece symbol: {symbol!r}")

            files_seen += 1
            if files_seen > BOARD_WIDTH:
                raise ValueError("FEN rank contains more than 8 files")
            board[index] = piece_code
            index += 1

        if files_seen != BOARD_WIDTH:
            raise ValueError("FEN rank must contain exactly 8 files")

    if index != BOARD_SIZE:
        raise ValueError("FEN piece placement must resolve to exactly 64 squares")

    return board


def _coerce_many_arrays(others: np.ndarray | Sequence["ChessPosition"]) -> np.ndarray:
    if isinstance(others, np.ndarray):
        array = np.ascontiguousarray(others, dtype=np.uint8)
        if array.ndim == 1:
            if array.shape[0] != BOARD_SIZE:
                raise ValueError(f"board array must have length {BOARD_SIZE}, got {array.shape[0]}")
            return array.reshape(1, BOARD_SIZE)
        if array.ndim != 2 or array.shape[1] != BOARD_SIZE:
            raise ValueError(f"board matrix must have shape (n, {BOARD_SIZE}), got {array.shape}")
        return array

    stacked = stack_positions(others)
    return stacked


def _pair_distance_numpy(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left != right))


def _distance_many_numpy(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.count_nonzero(right != left, axis=1).astype(np.uint8, copy=False)


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _pair_distance_numba(left: np.ndarray, right: np.ndarray) -> int:
        count = 0
        for index in range(BOARD_SIZE):
            if left[index] != right[index]:
                count += 1
        return count


    @njit(cache=True)
    def _distance_many_numba(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        rows = right.shape[0]
        distances = np.empty(rows, dtype=np.uint8)
        for row in range(rows):
            count = 0
            for column in range(BOARD_SIZE):
                if right[row, column] != left[column]:
                    count += 1
            distances[row] = count
        return distances

else:
    _pair_distance_numba = None
    _distance_many_numba = None


class ChessPosition:
    """Immutable occupancy-only chess position in FEN order from a8 to h1."""

    __slots__ = ("_board",)

    def __init__(self, board: np.ndarray):
        self._board = _readonly_board(board)

    @classmethod
    def from_fen(cls, fen: str) -> "ChessPosition":
        fen = fen.strip()
        if not fen:
            raise ValueError("FEN must not be empty")

        piece_placement = fen.split()[0]
        return cls(_parse_piece_placement(piece_placement))

    @classmethod
    def from_board(cls, board: "chess.Board") -> "ChessPosition":
        if chess is None:
            raise ImportError("python-chess is required for ChessPosition.from_board()")
        if not isinstance(board, chess.Board):
            raise TypeError("board must be a python-chess Board")

        encoded = np.zeros(BOARD_SIZE, dtype=np.uint8)
        for square, piece in board.piece_map().items():
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            index = (7 - rank) * BOARD_WIDTH + file
            encoded[index] = PIECE_CODES[piece.symbol()]
        return cls(encoded)

    def __repr__(self) -> str:
        return f"ChessPosition({self._board.tolist()!r})"

    def __hash__(self) -> int:
        return hash(self._board.tobytes())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChessPosition):
            return NotImplemented
        return bool(np.array_equal(self._board, other._board))

    def __sub__(self, other: object) -> int:
        if not isinstance(other, ChessPosition):
            return NotImplemented
        if NUMBA_AVAILABLE:
            return int(_pair_distance_numba(self._board, other._board))
        return _pair_distance_numpy(self._board, other._board)

    def as_array(self) -> np.ndarray:
        return self._board

    def distance_to_many(self, others: np.ndarray | Sequence["ChessPosition"]) -> np.ndarray:
        other_arrays = _coerce_many_arrays(others)
        if NUMBA_AVAILABLE:
            return _distance_many_numba(self._board, other_arrays)
        return _distance_many_numpy(self._board, other_arrays)


def stack_positions(positions: Sequence[ChessPosition]) -> np.ndarray:
    if not positions:
        return np.empty((0, BOARD_SIZE), dtype=np.uint8)

    arrays = []
    for position in positions:
        if not isinstance(position, ChessPosition):
            raise TypeError("stack_positions expects ChessPosition instances")
        arrays.append(position.as_array())

    return np.ascontiguousarray(np.stack(arrays, axis=0), dtype=np.uint8)


def distance_to_many(
    position: ChessPosition,
    others: np.ndarray | Sequence[ChessPosition],
) -> np.ndarray:
    if not isinstance(position, ChessPosition):
        raise TypeError("position must be a ChessPosition")
    return position.distance_to_many(others)
