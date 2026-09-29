# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# scripts/plotting

Post-sim plotting: reads a finished run's `seed-N/` folders, aggregates across
seeds with 95% CIs, and writes one figure per enabled plot to
`<data_dir>/figures/`. It never modifies sim output. Plot toggles, the
toggle -> input file -> output file table, and INI keys are in `README.md`;
keep it in sync when plots or keys change.

## Commands

Run from `scratch/mesh-sim/` as a module (package-relative imports):

```bash
.venv/bin/python -m scripts.plotting.cli --config scripts/plotting/plot.example.ini
```

`plot.example.ini` has a hardcoded `data_dir` pointing at an old run; copy it
or edit `data_dir` before running. There is no test directory here; the only
test touching this package is
`scripts/rl/tests/test_policy_comparison.py::test_plotting_uses_the_shared_statistics`.

## Architecture

`cli.main` -> `loaders` (one loader per sim output file, `None` if missing)
-> `aggregation` -> `plots_timeseries` / `plots_summary` (each returns a
`Figure`) -> `cli._save`.

- **Two aggregation paths.** `aggregate_timeseries` pools one CSV across
  seeds and groups by `(time_s, *series keys)`, rounding time to 1 ms first.
  It emits `<col>_mean` / `<col>_ci95`; plot functions depend on those
  suffixes. `aggregate_summaries` does the same for `summary.json` scalars,
  but only metrics in `_CI_METRICS` get a `ci95`.
- **Input schema is owned by the C++ `src/io/` writers.** Column names
  (`node_a`, `node_b`, `sinr_db`, `capacity_mbps`, `src`, `dst`, ...) and the
  `summary.json` keys are hardcoded in `cli.py` and the loaders. A seed that
  is missing a required column is skipped with a warning, not an error, so a
  renamed output column can silently empty a plot.
- **Shared stats.** `sample_stats` / `t_critical_95` come from
  `scripts/stats.py`, which `scripts/rl` also uses. The test above asserts
  plotting imports those exact objects; don't reimplement CI math here.
- **Callers.** `scripts/sweep/runner.py` (`auto_plot`) runs this CLI as a
  subprocess with a generated `plot.ini`. If that INI has no `[plots]`
  section, sweep enables its own subset of seven plots. Toggles absent from
  `[plots]` default to `true` here.

## Conventions

- Plot functions never call `savefig` / `plt.show`; only `cli._save` saves
  and closes figures.
- Missing input files skip their plot silently. The only hard failure is a
  `data_dir` with no `seed-N/` folders (exit 1).
- SINR / Rx power values at or below -900 are sim placeholders and are
  dropped before plotting.
- Doxygen `##` / `@fn` / `@brief` blocks above functions.
- Dependencies: `pandas`, `numpy`, `matplotlib`; no ns-3.
