#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_URL="https://database.lichess.org/standard/lichess_db_standard_rated_2026-03.pgn.zst"
INPUT_SOURCE="${1:-$SOURCE_URL}"
SUMMARY_PATH="$ROOT_DIR/artifacts/march_2026_standard_summary.json"
GRAPHICS_DIR="$ROOT_DIR/graphics"

mkdir -p "$GRAPHICS_DIR" "$(dirname "$SUMMARY_PATH")"

if [[ -f "$INPUT_SOURCE" ]]; then
  zstd -dc "$INPUT_SOURCE" \
    | rg '^\[(Event|WhiteElo|BlackElo|TimeControl) "' \
    | python3 "$ROOT_DIR/scripts/analyze_standard_dump.py" \
        --month "2026-03" \
        --source-url "$SOURCE_URL" \
        --expected-games 90074196 \
        --progress-every 5000000 \
        --summary "$SUMMARY_PATH"
else
  curl -L --fail --silent --show-error "$INPUT_SOURCE" \
    | zstd -dc \
    | rg '^\[(Event|WhiteElo|BlackElo|TimeControl) "' \
    | python3 "$ROOT_DIR/scripts/analyze_standard_dump.py" \
        --month "2026-03" \
        --source-url "$SOURCE_URL" \
        --expected-games 90074196 \
        --progress-every 5000000 \
        --summary "$SUMMARY_PATH"
fi

python3 "$ROOT_DIR/scripts/render_charts.py" \
  --summary "$SUMMARY_PATH" \
  --outdir "$GRAPHICS_DIR"

printf '\nSummary: %s\nGraphics: %s\n' "$SUMMARY_PATH" "$GRAPHICS_DIR"
