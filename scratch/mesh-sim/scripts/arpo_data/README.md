# scripts/arpo_data

Quicklook tooling for the **ARPO Spring Lake** mmWave-mesh field-test
dataset (CSV exports from outdoor radio deployments). Read-only with
respect to the source CSVs.

## Setup

Place the bundle here:

```
scratch/mesh-sim/data/arpo_spring_lake_data.zip
```

Each scenario inside has up to four node dirs (`rab1`, `rab2`, `rab3`,
`sdwan`). The tool reads `bh2.csv` (per-link SNR/MCS/throughput/peer MAC),
`gps.csv` / `geotak_gps.csv`, and ignores the ~25 host-telemetry CSVs.

## Run

From `scratch/mesh-sim/`:

```bash
python -m scripts.arpo_data.cli extract                       # unzip
python -m scripts.arpo_data.cli plot --scenario <name>        # one scenario
python -m scripts.arpo_data.cli plot --all                    # every scenario
python -m scripts.arpo_data.cli multi-day                     # day-vs-day ECDFs + K-S
```

## Output

| Step      | Where                                                              |
|-----------|--------------------------------------------------------------------|
| extract   | `data/arpo_extracted/csv/<scenario>/<node>/`                       |
| plot      | `data/arpo_extracted/_plots/per_day/<scenario>/{pngs,csvs}/<src>/` |
| multi-day | `data/arpo_extracted/_plots/multi_day/<family>/` + two CSVs        |

`plot` emits per-radio bh2 metric figures (SNR, RCPI, MCS, PER, throughput)
plus GPS tracks; each figure has a matching trace CSV. `multi-day` pools
those traces by scenario family and writes ECDF overlays + `_per_day_stats.csv`
and `_pairwise_ks.csv`.

## Module layout

| File           | Role                                                          |
|----------------|---------------------------------------------------------------|
| `cli.py`       | Subcommand parser; saves figures returned by plot fns         |
| `paths.py`     | Filesystem constants                                          |
| `extract.py`   | `extract` subcommand                                          |
| `topology.py`  | MAC -> rab label resolution + plot colors                     |
| `loaders.py`   | Per-scenario `bh2` / `gps` / `mcm` loaders                    |
| `plots/`       | Per-scenario plot functions (bh2 metrics, GPS)                |
| `multi_day.py` | `multi-day` subcommand                                        |

## Conventions

- Plot fns return a `matplotlib.Figure`; only `cli.py` calls `savefig`.
- Loaders return `pd.DataFrame | None`.
- Node clocks are skewed -- comparisons use `__sec__` (session-relative),
  not raw UTC.

## Dependencies

`pandas`, `numpy`, `matplotlib`. No ns-3.
