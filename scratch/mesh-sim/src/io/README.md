@page src_io src/io

@brief Writes the simulator's output files: per-tick CSVs, a per-seed JSON summary, a run log, and stderr progress.

`sim.cc` creates one `VizWriter` and one `MetricsWriter` per seed, calls
`WriteRunLog` once per batch, and calls `ProgressLogger::Tick` every tick.
The CSV formats are read by the mmwave-sim GUI and by the Python plotting and
validation tools. This directory has no standalone unit suite.

## Module Layout

| File | Role |
|------|------|
| `viz-writer.h` | `VizWriter` declaration; documents the six CSV schemas. |
| `viz-writer.cc` | Opens the CSVs, rate-limits writes to `viz_tick_ms`, aggregates per-edge flow load for `links.csv`. |
| `metrics-writer.h` | `MetricsWriter` declaration; documents the `summary.json` schema. |
| `metrics-writer.cc` | `AccumulateTick` (post-warmup running sums) and `Write` (time-averaged JSON). |
| `progress-logger.h` | Header-only `ProgressLogger`: sim time, wall time and ETA on stderr. |
| `run-logger.h` | Header-only `WriteRunLog`: seeds, CLI overrides and resolved config. |

## Output

| Step | Where |
|------|-------|
| `VizWriter::Open` creates `positions.csv`, `links.csv`, `rx-power.csv`, `mcs.csv`, `flows.csv`, `routes.csv` | `<output_dir>/seed-N/` |
| `VizWriter::WriteTick` appends rows every `viz_tick_ms` ms of sim time | same files |
| `MetricsWriter::Write` writes `summary.json` once after the step loop | `<output_dir>/seed-N/` |
| `WriteRunLog` writes `run.log` once before the seed loop | `<output_dir>/` (batch root, not per seed) |

### CSV columns

| File | Columns |
|------|---------|
| `positions.csv` | `time_s,node_id,x,y,z,node_type,active` after a `# key=value` metadata block (`scenario`, `frequency`, `txPower`, `numNodes`, `simDuration`, `tickMs`, `dimensions`, `rainRate`, `channelModel`, `flowTopology`, `trafficModel`) |
| `links.csv` | `time_s,node_a,node_b,dist_m,sinr_db,condition,condition_reason,capacity_mbps,delivered_mbps,hop_count` |
| `rx-power.csv` | `time_s,node_a,node_b,rx_power_dbm` |
| `mcs.csv` | `time_s,node_a,node_b,mcs_index,spectral_eff` |
| `flows.csv` | `time_s,src,dst,demand_mbps,delivered_mbps,latency_ms,hop_count,routable` |
| `routes.csv` | `time_s,src,dst,path,bottleneck_mbps,hop_count,routable` |

Node ids are zero-based indices. `links.csv`, `rx-power.csv` and `mcs.csv` have
one row per unordered pair (`node_a < node_b`). `path` is semicolon-separated
(for example `0;2;3`).

### summary.json keys

`scenario`, `seed`, `duration_s`, `warmup_s`, optional `wall_clock_start`,
`wall_clock_end`, `wall_elapsed_s`, then `per_node`, `per_flow` (keys like
`A->B`) and `network`. Field details are in `metrics-writer.h`.

## Conventions

- **Two-phase `MetricsWriter`.** `AccumulateTick` runs every tick; `Write` runs
  once. Ticks before `warmup_s` are skipped.
- **SINR sentinel.** Links with `sinr_db <= -900` (the -999.0 sentinel) are
  excluded from SINR statistics.
- **Connectivity.** `network.connectivity` uses `LinkTable::IsConnected` with its
  default -6.7 dB threshold.
- **Best-effort logging.** `WriteRunLog` returns silently if `run.log` cannot be
  opened; `MetricsWriter::Write` prints to stderr and returns; `VizWriter::Open`
  throws `std::runtime_error`.
- **Header-only.** `ProgressLogger` and `WriteRunLog` have no `.cc` file.
- **Comments.** Doxygen `/** ... */` blocks with `@fn`, `@brief`, `@param`, `@return`.

## Dependencies

| Dependency | Reason |
|------------|--------|
| `src/domain/`, `src/eval/`, `src/routing/` | `SimConfig`, `LinkTable`, `MCS_TABLE`, `FlowResult`. |
| `src/cli/cli-parser.h`, `src/util/string-utils.h` | `run-logger.h` reads `CliArgs`; `toIso8601`. |
| `ns3/mobility-model.h` | `VizWriter` reads node positions. |
| `third_party/json.hpp` | `MetricsWriter::Write` serialises JSON. |
