@page scripts_plotting scripts/plotting

@brief Post-simulation plotting: turns mesh-sim CSV/JSON outputs into figures.

Reads the `seed-N/` folders of a finished run, averages across seeds with 95% confidence intervals (CI), and saves one image per enabled plot. It only reads sim data and never modifies it.

## Setup

From `scratch/mesh-sim/`:

```bash
python3 scripts/rl/bootstrap_venv.py   # creates .venv (includes pandas, numpy, matplotlib)
```

Edit `scripts/plotting/plot.example.ini` (or copy it) and set `data_dir` to your run output folder.

## Run

Working directory: `scratch/mesh-sim/`.

```bash
.venv/bin/python -m scripts.plotting.cli --config scripts/plotting/plot.example.ini
```

- `--config <path>` (required): plot INI file.
- Produces `<data_dir>/figures/*.<format>`. Exits with status 1 if `data_dir` has no `seed-N/` folders.
- Plots whose input file is missing are skipped silently.

## Module Layout

| File | Role |
|---|---|
| `cli.py` | Entry point: reads the INI, loads and aggregates data, saves figures |
| `loaders.py` | Finds `seed-N/` folders and loads each CSV/JSON file (returns `None` if missing) |
| `aggregation.py` | Mean and 95% CI across seeds for time series and `summary.json` |
| `plots_timeseries.py` | One function per time-series plot; each returns a `Figure` |
| `plots_summary.py` | Bar charts and tables from `summary.json`; each returns a `Figure` |
| `__init__.py` | Re-exports the loaders, aggregators and plot functions |
| `plot.example.ini` | Example config with every plot toggle |
| `CLAUDE.md` | Scope notes for AI assistants |

## Configuration

| Section / key | Default | Meaning |
|---|---|---|
| `[output] data_dir` | required | Run folder containing `seed-N/` |
| `[output] format` | `png` | Image file extension |
| `[output] dpi` | `150` | Resolution (the example file sets 300) |
| `[plots] <name>` | `true` | Set `false` to skip a plot |

Plot toggles: `sinr_timeseries`, `rx_power_timeseries`, `capacity_timeseries`, `mcs_timeseries`, `throughput_timeseries`, `latency_timeseries`, `geometry`, `per_node_sinr`, `per_node_throughput`, `per_flow_throughput`, `per_flow_latency`, `network_summary`, `sim_runtime`.

## Output

All files go to `<data_dir>/figures/`.

| Step (toggle) | Where (file, without extension) | Input |
|---|---|---|
| `sinr_timeseries` | `sinr_timeseries` | `links.csv` |
| `rx_power_timeseries` | `rx_power_timeseries` | `rx-power.csv` |
| `capacity_timeseries` | `capacity_timeseries` | `links.csv` |
| `mcs_timeseries` | `mcs_timeseries` | `mcs.csv` |
| `throughput_timeseries` | `flow_throughput_timeseries`, `link_throughput_timeseries` | `flows.csv`, `links.csv` |
| `latency_timeseries` | `latency_timeseries` | `flows.csv` |
| `geometry` | `geometry` | `positions.csv` |
| `per_node_sinr` | `per_node_sinr` | `summary.json` |
| `per_node_throughput` | `per_node_tx_throughput`, `per_node_rx_throughput` | `summary.json` |
| `per_flow_throughput` | `per_flow_throughput` | `summary.json` |
| `per_flow_latency` | `per_flow_latency` | `summary.json` |
| `network_summary` | `network_summary` | `summary.json` |
| `sim_runtime` | `sim_runtime` | `summary.json` |

## Conventions

- Plot functions return a `matplotlib.Figure` and never call `savefig` or `plt.show`; `cli.py` saves and closes figures.
- Single-seed runs have no CI; multi-seed runs add CI bands or error bars.
- Loaders return `pd.DataFrame | None` (or `dict | None`); callers handle `None`.
- SINR and Rx power values at or below -900 are treated as placeholders and dropped before plotting.
- Time is rounded to 1 ms before grouping across seeds.
- t critical values come from `scripts/stats.py`.
- Comments use Doxygen `##` blocks above each function.

## Dependencies

- Python packages: `pandas`, `numpy`, `matplotlib`.
- `scripts/stats.py` (in this repo) for `sample_stats` and `t_critical_95`.
- A finished sim run with `seed-N/` folders containing `summary.json` and the CSVs listed above.
- No ns-3 needed.
