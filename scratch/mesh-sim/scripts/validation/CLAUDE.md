# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# scripts/validation

Two independent toolchains share this package:

- **Sim-vs-field pipeline** -- turns ARPO field data into scenarios, batch-runs
  the sim, and compares sim vs field SNR / RCPI / MCS distributions.
- **Regression / smoke checks** -- replays stored cases against a new sim
  binary and compares normalized snapshots.

Newcomer-facing usage, output paths and dependencies are in `README.md`; keep
it in sync when CLI flags or outputs change.

## Commands

Run everything from `scratch/mesh-sim/` as a module (package-relative imports
break direct `python file.py` calls):

```bash
python -m scripts.validation.<name> --help
.venv/bin/python -m pytest scripts/validation/tests -q
.venv/bin/python -m pytest scripts/validation/tests/test_regression_check.py::test_missing_reference_explains_cloud_bundle
python3 -m scripts.validation.smoke_check --sim-binary <BIN> --out outputs/smoke-check/<new-name>
```

CI (`.github/workflows/mesh-sim.yml`) runs these pytest tests and `smoke_check`
against the freshly built binary. Never run `./ns3 build`/`./ns3 run`; use an
existing binary via `--sim-binary`.

## Architecture

### Sim-vs-field pipeline

```
build_config_files -> build_waypoints -> run_batch -> sim_to_traces -> compare
                                                                        |-> compare_runs / compare_baseline
validate_days = sim_to_traces + compare + scenario_fidelity per day, then compare_runs
```

- `sim_to_traces` rewrites sim CSVs into the **same trace CSV layout that
  `scripts.arpo_data` produces for field data**, so `compare` can load both
  sides with one reader. Changing either trace format breaks `compare`.
- `compare.sim_to_field_scenario` is the single sim-name -> field-name mapping;
  `build_waypoints` and `scenario_fidelity` import it. Don't duplicate it.
- `run_batch --auto-waypoints` calls `build_waypoints.patch_scenario_waypoints`.
- Two modes throughout: `-m node` (CalFEX, per-node `IH_*` traces) and
  `-m scenario` (spring_lake, `bh2_*` traces).
- `build_config_files` and `build_waypoints` write into `inputs/` in place.

### Regression / smoke

- `regression_snapshot.py` is a **standard-library-only** library (no pandas /
  numpy) so `smoke_check` runs on a bare CI Python. Keep it that way.
- `regression_check` (CLI: `capture`, `compare`, `verify-suite`) and
  `smoke_check` both build on it; simulator launching goes through
  `scripts/sim_support.py` (`find_mesh_root`, `simulator_env`, `find_sim_binary`).
- The tracked `tests/fixtures/regression/p0/manifest.json` pins cases and
  reference hashes; the reference snapshots themselves are untracked external
  artifacts. Never regenerate them from changed code to make a check pass.
- `verify-suite --out` must resolve under `outputs/`; exit codes are
  `0` pass, `1` mismatch, `2` usage / I/O error.
- `tests/test_regression_check.py` also covers `run_batch` and
  `scripts.sweep` launcher behavior (stdin closed, loader paths cleaned), so
  changes to launchers there can fail this suite.

## Conventions

- Public functions use Doxygen `##` / `@fn` / `@brief` comment blocks above the
  `def` (see `compare.py`). `smoke_check.py`'s module docstring doubles as its
  argparse description, so extra header text goes in `#` comments there.
- Comparisons are distributional (no time alignment); histograms are
  density-normalized with a KDE overlay and K-S D statistic only (no p-value).
- MCS is clipped at 12 on both sides before comparison.

## Known issues (left unfixed)

- `run_batch --auto-waypoints` raises `TypeError`: it omits the required
  keyword-only `anchor` argument to `patch_scenario_waypoints`.
- `build_waypoints --anchor` help text claims a default; the real default is
  `None` (no alignment). `--dry-run` still rewrites nodes whose track fits in 20 m.
- `validate_days --time` is required but unused.
- `sim_to_traces` always exits 0; `scenario_fidelity` scenario mode exits 0
  even when scenarios are flagged.
