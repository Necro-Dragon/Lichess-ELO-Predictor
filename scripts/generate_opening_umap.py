#!/usr/bin/env python3
"""CLI wrapper for generating the lichess opening UMAP embedding."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chess_distance.opening_umap import main


if __name__ == "__main__":
    raise SystemExit(main())
