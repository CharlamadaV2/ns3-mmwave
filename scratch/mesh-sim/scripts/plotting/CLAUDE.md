# scripts/plotting

Post-simulation Python plotting module. Reads CSV and JSON outputs produced
by the C++ sim and generates matplotlib figures.

## Scope

- Loading sim output files (CSV time-series, summary JSON)
- Multi-seed aggregation with 95% confidence intervals
- Time-series plots (SINR, capacity, Rx power, MCS, throughput, latency, geometry)
- Summary bar charts and tables (per-node, per-flow, network-level)
- INI-driven CLI that selects which plots to generate

This module is read-only with respect to sim data -- it never modifies outputs.

## File layout

| File | Role |
|---|---|
| `cli.py` | Entry point: parses INI config, loads data, generates and saves figures |
| `loaders.py` | Discovers seed directories, loads each CSV/JSON file type |
| `aggregation.py` | Computes per-group mean and 95% CI across seeds (t-distribution) |
| `plots_timeseries.py` | One function per time-series plot; each returns a `Figure` |
| `plots_summary.py` | One function per summary plot; each returns a `Figure` |
| `plot.example.ini` | Example config showing all available plot toggles |

## Conventions

- Every plot function returns a `matplotlib.Figure` -- never calls `savefig` or `plt.show`.
- `cli.py` owns saving and closing figures.
- Aggregation is seed-aware: single-seed runs skip CI, multi-seed runs add CI bands/error bars.
- CSV loaders return `pd.DataFrame | None`; callers must handle `None`.

## Running

```
python -m scripts.plotting.cli --config scripts/plotting/plot.example.ini
```

## Dependencies

Python-only: `pandas`, `numpy`, `matplotlib`. No ns-3 dependency.
