#!/usr/bin/env python3
"""CLI wrapper for training the sparse-game rating-band model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chess_distance.rating_band_training import train_main


if __name__ == "__main__":
    raise SystemExit(train_main())
