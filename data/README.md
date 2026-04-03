# Rapid Sample Data

This folder contains a uniform sample of `1,500,000` March 2026 rapid games from the Lichess rated standard dump, after excluding raw Lichess terminations `Abandoned` and `Rules infraction`.

- Source archive: `lichess_db_standard_rated_2026-03.pgn.zst`
- Population size sampled from: `14,567,559` eligible rapid games
- Sampling method: uniform without replacement
- RNG seed: `202603`
- Storage format: split `zstd`-compressed PGN chunks
- Total compressed size: `473,779,245` bytes

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

- Raw `Abandoned` and `Rules infraction` games were excluded before sampling.
- This is a uniform random sample of the filtered population, not of the full rapid population.
- `Termination` is best-effort normalization from the monthly PGN dump and movetext.
- Rapid draws in the dump cannot be cleanly separated into agreement, repetition, and 50-move rule from metadata alone.
- Observed normalized endings in the filtered population are checkmate, resignation, draw, time forfeit, and insufficient material.
