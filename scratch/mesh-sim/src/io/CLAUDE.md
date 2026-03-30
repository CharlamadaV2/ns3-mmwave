# io/

## Scope
Output writers -- in-simulation visualization snapshots and post-simulation metrics.
Output format is compatible with the mmwave-sim GUI.

## Files
- **viz-writer.h/cc** -- Writes `positions.csv` and `links.csv` each tick. Same CSV format as mmwave-sim with additional mesh columns (capacity_mbps, delivered_mbps, hop_count).
- **metrics-writer.h/cc** -- Accumulates per-tick metrics in-memory, writes `summary.json` after simulation completes. No ns-3 deps.
- **progress-logger.h** -- Stderr progress output during simulation. Header-only.
- **run-logger.h** -- Writes `run.log` with seeds, CLI overrides, and resolved config for reproducibility. Header-only.

## Dependencies
- Depends on: `domain/`, `util/`
- Depended on by: `sim.cc`

## Rules
- metrics-writer must stay ns-3-free (stdlib + nlohmann/json only).
- CSV and JSON schemas must remain compatible with mmwave-sim's GUI.
