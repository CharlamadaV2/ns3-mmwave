# scripts/arpo_data

Exploration tool for the **ARPO Spring Lake** mmWave-mesh field-test
**dataset** (CSV exports from outdoor radio deployments). Unzips the
data, reports anomalies, renders quicklook plots, and compares SNR
across scenarios.

This module is read-only with respect to the dataset -- it never
modifies the source CSVs. All outputs land under `data/arpo_extracted/`.

## What's in the dataset

`data/arpo_spring_lake_data.zip` contains CSV exports from outdoor mesh
deployments. Each *scenario* (e.g. `lab_static`, `field_run_1`) has up to
four node directories (`rab1`, `rab2`, `rab3`, `sdwan`), each with:

| File             | Contents                                                                  |
|------------------|---------------------------------------------------------------------------|
| `bh2.csv`        | Per-link backhaul stats: SNR, MCS, throughput, peer MAC                   |
| `gps.csv`        | Local GPS receiver readings                                               |
| `geotak_gps.csv` | TAK-formatted GPS positions                                               |
| `ping.csv`       | Ping latency to test targets                                              |
| `mesh.csv`       | Mesh-layer events (associations, etc.)                                    |
| `system.csv`     | Host load / uptime                                                        |
| `chrony.csv`     | NTP/chrony clock-discipline (offset, skew, stratum) — contextualizes bh2 timestamp skew |
| `mcm.csv`        | Mesh-client-manager radio state: SSID, BSSID, channel, beacon RSSI, peer count |

Each node also produces ~25 host-telemetry CSVs (`cpu.csv`, `mem.csv`,
`disk.csv`, etc.) that this tool ignores -- they're useful for incident
debugging but not for radio-link analysis.

## Running

All commands run **from `scratch/mesh-sim/`** so the `scripts.arpo_data`
package import resolves.

```bash
# 1. Unzip the bundle (idempotent, skips macOS metadata)
python -m scripts.arpo_data.cli extract

# 2. Report file presence + flag suspicious data
python -m scripts.arpo_data.cli summarize

# 3. Plot a single scenario
python -m scripts.arpo_data.cli plot --scenario lab_static

# 4. Plot every scenario; --gallery adds 4 alternate SNR views per scenario
python -m scripts.arpo_data.cli plot --all --gallery

# 5. Cross-scenario SNR heatmap (rows = scenarios, columns = nodes)
python -m scripts.arpo_data.cli compare
```

For *how to interpret* what `plot` produces, see [`PLOTS.md`](PLOTS.md)
-- a per-figure reading guide (axes, what good/bad looks like, common
gotchas, suggested cross-plot workflow).

## Where output lands

| What                 | Where                                                   |
|----------------------|---------------------------------------------------------|
| Extracted CSVs       | `data/arpo_extracted/csv/<scenario>/<node>/`            |
| Per-scenario figures | `data/arpo_extracted/_plots/<scenario>/`                |
| Gallery figures      | `data/arpo_extracted/_plots/<scenario>/snr_gallery/`    |
| Cross-scenario       | `data/arpo_extracted/_plots/_compare/`                  |

## What `summarize` flags

The `summarize` subcommand surfaces issues, not column statistics. It
prints a per-scenario block (file presence, row counts, time range)
followed by an `ISSUES` section listing:

- Missing priority files per (scenario, node)
- Clock skew > 1h between any two nodes in a scenario (cross-checked against
  chrony's recorded offsets — large bh2 skew + small chrony offset = nodes
  booted on different days; both large = real ongoing drift)
- Chrony median clock offset > 100 ms (poorly disciplined clock)
- GPS lock rate < 50% (rows where lat = lon = 0)
- SNR readings outside the plausible range [-40, 60] dB

If the `ISSUES` section is empty, the data passes the basic sanity checks.

## Module layout

| File           | Role                                                              |
|----------------|-------------------------------------------------------------------|
| `cli.py`       | Subcommand parser; loads data, calls plot fns, saves figures      |
| `paths.py`     | Filesystem constants (zip, extract dir, plots dir, priority files)|
| `extract.py`   | `extract` subcommand                                              |
| `summarize.py` | `summarize` subcommand: presence + anomaly report                 |
| `topology.py`  | MAC -> physical-node label resolution + node color table          |
| `loaders.py`   | Per-scenario `bh2`/`ping`/`gps` loaders, session-relative time    |
| `plots.py`     | Quicklook plot functions (snr, mcs, throughput, ping, gps)        |
| `gallery.py`   | 4-style SNR comparison gallery + topology MAC card                |
| `compare.py`   | `compare` subcommand: cross-scenario SNR heatmap                  |

## Conventions

- Plot functions return a `matplotlib.Figure` and never call `savefig`
  or `plt.show`. `cli.py` is the only place that saves and closes figures.
- Loaders return `pd.DataFrame | None` (`None` means no usable data).
- Field-node clocks are skewed across nodes -- comparisons use
  **session-relative seconds** (the `__sec__` column, zeroed at each
  node's first timestamp), not raw UTC.
- The 4-MAC clustering in `topology.py` reflects that each radio carries
  four sequential MAC addresses on its interfaces.

## Dependencies

Python-only: `pandas`, `numpy`, `matplotlib`. No ns-3 dependency.
