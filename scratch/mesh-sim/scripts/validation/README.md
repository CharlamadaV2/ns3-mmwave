@page scripts_validation scripts/validation

@brief Checks mesh-sim output against ARPO field data and against stored regression results.

This directory does two jobs:

- **Sim-vs-field validation.** Builds sim scenarios from ARPO field data, runs
  mesh-sim over several seeds, and overlays the sim distributions of SNR, RCPI
  (received channel power indicator, dBm) and MCS (modulation and coding scheme
  index) on the field traces. It reports the two-sample K-S
  (Kolmogorov-Smirnov) D statistic and |Δmedian| / |Δmean| for each one.
- **Regression and smoke checks.** Confirms that a new simulator binary still
  reproduces stored results (`regression_check`), or that it is at least
  self-consistent on a tiny synthetic scenario (`smoke_check`).

## Module Layout

| File | Role |
|------|------|
| @ref build_config_files.py "build_config_files" | Writes per-run `nodes.json` and `run.ini` from field GPS and Silvus config CSVs. |
| @ref build_waypoints.py "build_waypoints" | Patches waypoint mobility into `nodes.json` from a field GPS track (`scenario` and `node` subcommands). |
| @ref run_batch.py "run_batch" | Runs the sim binary once per scenario over a set of seeds; writes `batch_manifest.json`. |
| @ref sim_to_traces.py "sim_to_traces" | Rewrites per-seed sim CSVs into arpo_data-style trace CSVs. |
| @ref compare.py "compare" | Pools sim seeds and field traces; draws histogram + KDE overlays; writes K-S CSVs and heatmaps. |
| @ref compare_runs.py "compare_runs" | Cross-batch heatmaps (scenario x batch) from `validation_summary.csv` files. |
| @ref compare_baseline.py "compare_baseline" | Signed per-node heatmaps of each scenario against one baseline scenario. |
| @ref scenario_fidelity.py "scenario_fidelity" | Audits sim vs field mobile/static classification and initial pairwise distances (prints tables). |
| @ref validate_days.py "validate_days" | Runs `sim_to_traces`, `compare` and `scenario_fidelity` per day, then `compare_runs` across days. |
| @ref make_jammers.py "make_jammers" | Builds `jammers.json` from the field EW (electronic warfare) trial CSV. |
| @ref regression_check.py "regression_check" | CLI with `capture`, `compare` and `verify-suite` subcommands for the baseline regression. |
| @ref regression_snapshot.py "regression_snapshot" | Library that builds and compares normalized run snapshots. |
| @ref smoke_check.py "smoke_check" | Data-free two-run smoke and repeatability check. |
| @ref tests/test_regression_check.py "tests/test_regression_check.py" | pytest tests for the snapshot and launcher contracts. |

Launcher paths and loader setup live in `scripts/sim_support.py`, one level up.

## Setup

Run every command from `scratch/mesh-sim/`. The scripts use package-relative
imports, so always call them as `python -m scripts.validation.<name>`.

1. Build the simulator binary (the user handles builds).
2. Install the Python packages:

   ```bash
   pip install pandas numpy matplotlib pytest
   ```

3. For sim-vs-field validation, generate the field trace CSVs once per dataset:

   ```bash
   python -m scripts.arpo_data.cli plot --all
   ```

4. For the full regression suite, get the approved reference bundle from the
   project team and unpack its JSON snapshots under
   `tests/fixtures/regression/p0/`. The smoke check and unit tests do not need it.

## Run

### Sim-vs-field validation (CalFEX, node mode)

Run the steps in order. Each step needs the output of the one before it.

```bash
# 1. Write run.ini + nodes.json per run (waypoints left empty)
python -m scripts.validation.build_config_files \
    --csv-dir <per-node-config-dir> \
    -i data/arpo_extracted/_plots/per_day \
    -o inputs/calfex \
    -b sub-6 -m node --all-days

# 2. Fill in waypoints from the field GPS track (edits nodes.json in place)
python -m scripts.validation.build_waypoints node \
    -i inputs/calfex/<run> -f <field dir or gps_all_nodes_trace.csv> \
    --all-nodes --time-mode scale --duration 60

# 3. Run the sim over every scenario (default seeds 1,2,3,4,5)
python -m scripts.validation.run_batch --scenarios-dir inputs/calfex

# 4. Convert per-seed sim CSVs to field-style trace CSVs
python -m scripts.validation.sim_to_traces -m node \
    outputs/<YYYY-MM>/<DD>/<HH-MM-SS>-validation/<scenario>

# 5. Overlays, K-S statistics, CSVs and heatmaps
python -m scripts.validation.compare -m node \
    --batch_root outputs/.../<scenario> --field-root <field dir for that run>
```

`--csv-dir` points at a directory with one subdirectory per node, each holding
`silvus/config.csv`. Use `--day <run>` instead of `--all-days` for one run.

### Scenario mode (spring_lake)

Same pipeline, with the scenario variants of steps 2, 4 and 5:

```bash
python -m scripts.validation.build_waypoints scenario --all --time-mode scale --duration 60
python -m scripts.validation.sim_to_traces -m scenario outputs/.../<HH-MM-SS>-validation
python -m scripts.validation.compare -m scenario --batch_root outputs/.../<HH-MM-SS>-validation
```

### Optional follow-ups

Run these after `compare`, because they read `validation_summary.csv` or the
sim outputs.

```bash
# Cross-batch heatmaps (auto-discovers outputs/**/validation_summary.csv)
python -m scripts.validation.compare_runs

# Layout / mobility audit (prints tables, writes nothing)
python -m scripts.validation.scenario_fidelity -m node \
    --field-gps <gps_all_nodes_trace.csv> outputs/.../<scenario>

# Signed per-node comparison against one baseline scenario
python -m scripts.validation.compare_baseline \
    --baseline outputs/.../<base> outputs/.../<other> ...
```

### Per-day wrapper

`validate_days` runs steps 4 and 5 plus `scenario_fidelity` for each day, then
`compare_runs` across days. It expects sim outputs laid out as
`outputs/calfex/<YYYY-MM-DD>/{inputs/,seed-*/}`.

```bash
python -m scripts.validation.validate_days --day 2026-05-13 --time 60
python -m scripts.validation.validate_days --all-days --time 60
```

### Jammer file

```bash
python -m scripts.validation.make_jammers --trials <ew_trials.csv> \
    --trace <gps_all_nodes_trace.csv> -o <scenario dir>/jammers.json \
    [--run-ini <scenario dir>/run.ini]
```

### Regression suite

Re-runs every case in the manifest against a freshly built binary and compares
the normalized results:

```bash
python3 -m scripts.validation.regression_check verify-suite \
    --sim-binary <BIN> \
    --manifest tests/fixtures/regression/p0/manifest.json \
    --out outputs/baseline-regression/<name>
```

- It checks the manifest and every snapshot hash before starting the simulator.
- It prints one `PASS` / `FAIL` / `SKIP` row per case.
- The optional Sherpa Spring Lake case is `SKIP` when its data is absent; the
  console names the missing paths. Add `--require-all` to make that a failure.
- `--out` must be under `outputs/` (not `outputs/` itself). Reusing it replaces
  only `suite-report.json` and the case subdirectories.
- Exit codes: `0` all passed, `1` mismatch or failure, `2` usage or I/O error.

Single-case subcommands:

```bash
python3 -m scripts.validation.regression_check capture --sim-binary <BIN> \
    --run-config <run.ini> --band {mmwave,sub-6} --seed <N> --family <F> --case <C> \
    --out <run dir> [--snapshot <file.json>] [--force]
python3 -m scripts.validation.regression_check compare --baseline <snapshot.json> \
    --candidate <run dir> [--atol 1e-9] [--max-diffs 20] [--report <file.json>]
```

### Smoke check

Runs `inputs/baselines/p0-jammer-smoke` twice with seed 1, checks the output
contracts, and checks that the two runs match. It needs no reference data.

```bash
python3 -m scripts.validation.smoke_check --sim-binary <BIN> --out outputs/smoke-check/<new-name>
```

`--out` must not exist yet. It prints `PASS` (exit 0) or `FAIL: <reason>` (exit 1).

### Unit tests

```bash
python -m pytest scripts/validation/tests
```

## Output

| Step | Where |
|------|-------|
| `build_config_files` | `<output>/<run>/nodes.json` and `run.ini` (default `<output>` is `inputs/calfex`) |
| `build_waypoints` | Edits `nodes.json` in place |
| `run_batch` | `outputs/<YYYY-MM>/<DD>/<HH-MM-SS>-validation/<scenario>/{console.log, seed-N/...}` and `batch_manifest.json` |
| `sim_to_traces` | `<scenario>/sim_traces/seed-N/csvs/<src>/` trace CSVs (`bh2_*` in scenario mode, `IH_*` in node mode) |
| `compare` | `<scenario>/validation/pngs/<src>/hist_*.png`, `<scenario>/validation/metrics.csv`, `<batch_root>/validation_summary.csv`, `<batch_root>/summary/heatmap_*.png` |
| `compare_runs` | `outputs/cross_batch_summary/heatmap_*.png` (or `--out`) |
| `compare_baseline` | `<baseline>/baseline_comparison/` heatmaps and `baseline_comparison.csv` (or `--out`) |
| `validate_days` | Outputs of the wrapped steps, plus `<out-root>/cross_day_summary/` |
| `make_jammers` | The `-o` JSON file |
| `regression_check verify-suite` | `<out>/suite-report.json` and `<out>/<case>/` with `console.log`, run output and `comparison.json` |
| `smoke_check` | `<out>/run-1/` and `<out>/run-2/`, each with `console.log` |

`validation_summary.csv` has one row per (scenario, link, metric) with means,
medians, IQRs, sample counts, `abs_diff_*` and `ks_statistic`.

## Conventions

### Comparison method

- **Distributions, not time series.** Sim and field samples are compared as
  marginal distributions with no time alignment. `--window N` keeps only the
  first N seconds of both.
- **Seed pooling.** If every seed's trace has the same length, `compare`
  averages them row by row; otherwise it concatenates them.
- **Density-normalized histograms** (area = 1), so sim and field compare
  directly when their sample counts differ. There is no bootstrap band and no
  K-S p-value.
- **MCS cap.** Sim and field MCS are both clipped at 12, the field firmware
  ceiling.
- **Metrics:** `snr` (dB), `rcpi` (dBm), `mcs`.

### Naming

- Sim scenario names map to field names by stripping `arpo-`, joining the
  middle tokens with `_`, and upper-casing the minor token:
  `arpo-1-x-misc-04172026` becomes `1-X_misc_04172026`.
- Sim trace CSVs leave `tag_interface` blank and fill the MAC columns with node
  names. Node mode mirrors each link so every node appears as a source.

### Inputs and data handling

- `build_config_files` supports only `-m node -b sub-6`. It writes empty
  waypoints, so always run `build_waypoints` afterwards.
- `build_waypoints` edits `nodes.json` in place. Review with `git diff inputs/`
  and revert with `git checkout inputs/`. A node whose track fits in a 20 m
  box becomes `fixed`.
- NaN rows in `validation_summary.csv` mean the field collect had no telemetry
  for that link. This is expected for some scenarios.
- Regression snapshots stay local and untracked. Never recapture them from
  changed code.

### Known issues

- `run_batch --auto-waypoints` fails with a `TypeError`.
- `build_waypoints --anchor` defaults to no alignment, despite the `--help` text.
- `validate_days --time` is required but unused.

## Dependencies

| Scope | Needs |
|-------|-------|
| Field-comparison scripts (`build_*`, `compare*`, `scenario_fidelity`, `sim_to_traces`, `validate_days`, `make_jammers`) | Python 3.10+ (`build_config_files` uses `match`), `pandas`, `numpy`, `matplotlib` |
| `run_batch`, `regression_check`, `smoke_check` | A built simulator binary (`run_batch` auto-detects `build/scratch/mesh-sim/ns3*-sim-*`; the others take `--sim-binary`) and `scripts/sim_support.py` |
| `regression_check`, `regression_snapshot`, `smoke_check` | Standard library only |
| Tests | `pytest`, plus `scripts.sweep` (imported by one test) |

Data that must exist beforehand:

- Field traces under `data/arpo_extracted/_plots/per_day/`, from `scripts.arpo_data.cli plot`.
- Per-node Silvus config CSVs (`--csv-dir`).
- Spring Lake inputs under `inputs/custom/sherpa/spring_lake/` (scenario mode default).
- Regression reference snapshots from the project team.
