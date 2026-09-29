# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# scripts/arpo_data

Read-only tooling that turns raw ARPO field-test CSVs into per-day plots and
trace CSVs. Those traces feed `scripts/validation` (sim-vs-field comparison
and waypoint building). Usage, flags, and output paths are in `README.md`;
keep it in sync when CLI flags or outputs change.

## Commands

Run from `scratch/mesh-sim/` as modules (package-relative imports):

```bash
python -m scripts.arpo_data.cli {extract,plot,multi-day,split-day,split-scenario} --help
python -m scripts.arpo_data.multiday_variance --help
python -m scripts.arpo_data.topology_audit
```

There are no tests for this package. Data lives under `scratch/mesh-sim/data/`
(untracked); `paths.py` resolves it relative to the repo root.

## Architecture

### Two data formats, one plot pipeline

- **bh2** (older Spring Lake format): per-node `bh2.csv`, loaded by
  `load_bh2_scenario`, plotted by `plot_bh2_*`, traces named `bh2_<metric>__*`.
  Its calls in `cli.py` are commented out, not deleted.
- **Silvus** (current format): per-node `gps/` + `silvus/{network_status,local_stats,config}.csv`,
  loaded by `load_rf_scenario`, plotted by `plot_silvus_*`, traces named `IH_<metric>__*`.
- `plots/bh2.py::_silvus_to_bh2_format` maps Silvus columns onto the bh2
  column layout (`__session__`, `__peer__`, `tag_local_mac`, `tag_sta_mac`,
  `field_*`) so both formats share `_per_radio_metric` / `_build_per_radio_figure`.
  Changing those column names breaks both paths.

### Data flow

```
extract -> csv/<scenario>/<node>/
plot --nodes -> per_day/<YYYY-MM-DD>/<node>/{pngs,csvs}/IH_*  + gps_all_nodes_trace.csv
             -> split-day / split-scenario (slice GPS traces by day or wall-clock window)
plot --scenario|--all -> per_day/<scenario>/{pngs,csvs}/...
multi-day (reads bh2_* traces under per_day/) -> _pairwise_ks.csv -> multiday_variance
```

- Plot functions never save; `cli._save` / `cli._save_pairs` own all file
  writes and derive the `<prefix>__<src>_to_<peer>[_trace].csv` names.
- Trace CSV names and columns are an interface: `scripts/validation`
  (`compare`, `validate_days`, `build_waypoints`, and `sim_to_traces`, which
  writes sim output in the same layout) reads them. `-m node` there means
  `IH_*` traces, `-m scenario` means `bh2_*`.
- `multi_day` groups scenario dirs into families by stripping the
  `_[baseline_N_]MMDDYYYY` suffix and skips `paths.KNOWN_BAD_SCENARIOS`.
- `topology.py` builds MAC -> `rabN` / `rabN.M` / `extN` labels from the global
  union of MACs across every scenario under the CSV root, cached per root in
  module-level dicts. Only rab2 has an antenna-orientation table.

## Conventions

- Doxygen comments (`##`, `##<`, `@brief`, `@fn`) above definitions; the
  README is a Doxygen `@page`.
- Loaders return `pd.DataFrame | None`; internal columns use `__name__` style
  (`__node__`, `__t__`, `__sec__`). Compare across nodes on `__sec__`
  (session-relative), since node clocks are skewed.
- No scipy: K-S and KDE are hand-rolled in numpy in `multi_day.py`.
  matplotlib is forced to the Agg backend.
- `split-scenario` times are `Hour.Minute` floats in UTC (`9.05` = 09:05).

## Known issues (left unfixed)

- `plot --scenario` / `--all` raise `TypeError`: `_plot_one` calls
  `load_rf_scenario(node_dir)` without the required `valid_nodes` and
  `node_id_to_name`. Only `plot --nodes` works on Silvus data.
- `multi-day` only matches `bh2_*` trace names (`multi_day._TRACE_RE`), so it
  finds nothing in `IH_*` output from the current Silvus flow.
