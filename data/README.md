# Rapid Sample Data

This folder contains a uniform sample of `1,500,000` non-abandoned rapid games from the Lichess March 2026 rated standard dump.

- Source archive: `lichess_db_standard_rated_2026-03.pgn.zst`
- Population size sampled from: `14,569,156` rapid non-abandoned games
- Sampling method: uniform without replacement
- RNG seed: `202603`
- Storage format: split `zstd`-compressed PGN chunks
- Total compressed size: `473,809,423` bytes

Each sampled game keeps only these fields:

- `Site`
- `Result`
- `Termination`
- `WhiteElo`
- `BlackElo`
- `ECO`
- `Opening`
- movetext with Lichess clock comments

Notes:

- Abandoned games were excluded.
- `Termination` is best-effort normalization from the monthly PGN dump and movetext.
- Rapid draws in the dump cannot be cleanly separated into agreement, repetition, and 50-move rule from metadata alone.
- Additional observed endings in the source population include time forfeit, insufficient material, rules infraction, and stalemate.
