#!/usr/bin/env python3
"""CLI wrapper for building the 40-ply rapid corpus."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chess_distance.rapid_corpus import main


if __name__ == "__main__":
    raise SystemExit(main())
