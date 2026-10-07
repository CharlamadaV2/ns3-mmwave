# io/

## Scope
Per-seed output writers (CSV snapshots, `summary.json`), the batch `run.log`,
and stderr progress. Writers take data from `LinkTable` / `FlowResult`; they
never compute physics or routing.

## Files
- **viz-writer.h/cc** -- `VizWriter`: `Open` / `WriteTick` / `Close`. Writes `positions.csv`, `links.csv`, `rx-power.csv`, `mcs.csv`, `flows.csv`, `routes.csv` into `seed-N/`. Includes ns-3 (`MobilityModel`).
- **metrics-writer.h/cc** -- `MetricsWriter`: `AccumulateTick`, `SetTiming`, `Write` -> `seed-N/summary.json`. ns-3-free (stdlib + nlohmann/json).
- **progress-logger.h** -- header-only `ProgressLogger::Tick` (stderr).
- **run-logger.h** -- header-only `WriteRunLog` -> `<base_output_dir>/run.log` (batch root).

## Behavior to preserve
- `viz-writer.cc` is in `CMakeLists.txt`; `progress-logger.h` and `run-logger.h` are header-only.
- `positions.csv` starts with `# key=value` lines consumed by the GUI's `parseMeta`. Python readers skip them (`comment="#"` in `scripts/plotting/loaders.py`; `#`-skipping in `scripts/validation/regression_snapshot.py`, `scenario_fidelity.py`).
- CSV column names and order are read by name by `scripts/plotting/` and `scripts/validation/sim_to_traces.py` (`time_s`, `node_a`, `node_b`, plus a value column). `summary.json` is parsed by `scripts/plotting/cli.py` and `scripts/validation/smoke_check.py` (`network.*` keys, `scenario`, `seed`); `scripts/rl/policy/evaluate.py` only checks it exists.
- `run.log` "Resolved config" is the source of truth for a run; keep `key = value` lines stable.
- `VizWriter` holds a `const SimConfig&` (must outlive it); `MetricsWriter` copies the config at construction, so `output_dir` must already be the `seed-N` path.
- `VizWriter::Open` throws `std::runtime_error`; other writers report to stderr or return silently. `WriteRunLog` can throw from `create_directories`.
- Node ids in CSVs are zero-based indices; `summary.json` uses `nodes.json` ids.
- `MetricsWriter` skips ticks before `warmup_s` and SINR samples `<= -900`.

## Changing an output file
Update together: the writer, the schema table in the header and `README.md`,
the Python readers above, and "Training or episode output" in the
`CONTRIBUTING.md` table. `summary.json` has no `manifest_version`. No unit
suite covers `io/`; `make integration` in `tests/` exercises it via the binary.

## Dependencies
- Depends on: `domain/`, `eval/` (`link-table.h`, `sinr-capacity.h`), `routing/` (`mesh-router.h`), `util/` (`string-utils.h`), `cli/` (`cli-parser.h`, `run-logger.h` only), ns-3 `MobilityModel` (`viz-writer` only)
- Depended on by: `sim.cc`
