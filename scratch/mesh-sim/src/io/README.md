@page src_io src/io

@brief Writes and logs simulation metric output summary.  

Handles all simulation output: per-tick CSV snapshots for visualisation,
a post-run JSON metrics summary, a human-readable run log, and a stderr
progress reporter. 

## Output

| File | Written by | When |
|------|------------|------|
| `positions.csv` | `VizWriter` | Every `viz_tick_ms` milliseconds during the step loop. |
| `links.csv` | `VizWriter` | Every `viz_tick_ms` milliseconds during the step loop. |
| `rx-power.csv` | `VizWriter` | Every `viz_tick_ms` milliseconds during the step loop. |
| `mcs.csv` | `VizWriter` | Every `viz_tick_ms` milliseconds during the step loop. |
| `flows.csv` | `VizWriter` | Every `viz_tick_ms` milliseconds during the step loop. |
| `routes.csv` | `VizWriter` | Every `viz_tick_ms` milliseconds during the step loop. |
| `summary.json` | `MetricsWriter` | Once, after the step loop completes. |
| `run.log` | `RunLogger` | Once, before the step loop starts. |

All files are written to `cfg.output_dir/<seed>/` per seed run.
`run.log` is written to the batch root (not per-seed).

## Module Layout

| File | Description |
|------|-------------|
| `run-logger.h` | Header-only: `WriteRunLog` writes `run.log` capturing seeds, CLI overrides, and resolved config before the step loop starts. |
| `progress-logger.h` | Header-only: `ProgressLogger` prints sim-time progress, wall elapsed, and ETA to stderr at configurable tick intervals. |
| `metrics-writer.h` | `MetricsWriter` class declaration and private accumulator structs. |
| `metrics-writer.cc` | Per-tick accumulation (`AccumulateTick`) and post-run JSON serialisation (`Write`). |
| `viz-writer.h` | `VizWriter` class declaration; documents all six CSV schemas. |
| `viz-writer.cc` | Per-tick CSV writing rate-limited by `viz_tick_ms`; edge-flow aggregation for `links.csv`. |

## Conventions

- **Two-phase pattern for `MetricsWriter`.** `AccumulateTick` is called every
  tick during the step loop; `Write` is called once after it completes. Calling
  `Write` before the loop ends produces a valid but incomplete summary.

- **SINR sentinel filtering.** `AccumulateTick` only counts links with
  `sinr_db > −900.0`. Links at the −999.0 sentinel (unevaluated or below
  noise floor) are excluded from all SINR accumulators.

- **`ProgressLogger` and `RunLogger` are header-only.** They have no `.cc`
  file and no ns-3 dependency — they can be included anywhere without
  affecting build times.

## Dependencies

| Dependency | Reason |
|------------|--------|
| `ns3/mobility-model.h` | `VizWriter::WritePositions` queries node positions via `MobilityModel::GetPosition`. |
| `third_party/json.hpp` | `MetricsWriter::Write` serialises `summary.json` via nlohmann/json. |