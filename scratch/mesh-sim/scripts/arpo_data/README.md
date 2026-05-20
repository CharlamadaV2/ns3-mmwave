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
python -m scripts.arpo_data.multiday_variance                 # variance table from _pairwise_ks.csv
```

`multiday_variance` is a read-only summary over the CSV produced by `multi-day`.
It prints three views — per-family rollup, per-link rollup, and the detail
table sorted by `|Δmed|` descending — so unstable day-to-day links surface at
the top without opening any PNGs. Flags: `--metric {snr,rcpi,mcs,per,throughput}`,
`--family <name>`, `--top N`, `--csv <out>`.

## Output

| Step      | Where                                                              |
|-----------|--------------------------------------------------------------------|
| `extract`   | `data/arpo_extracted/csv/<scenario>/<node>/`                       |
| `plot`      | `data/arpo_extracted/_plots/per_day/<scenario>/{pngs,csvs}/<src>/` |
| `multi-day` | `data/arpo_extracted/_plots/multi_day/<family>/` + two CSVs        |

`plot` emits per-radio bh2 metric figures (SNR, RCPI, MCS, PER, throughput)
plus GPS tracks; each figure has a matching trace CSV. `multi-day` pools
those traces by scenario family and writes ECDF overlays + `_per_day_stats.csv`
and `_pairwise_ks.csv`.

## Module layout

| File           | Role                                                          |
|----------------|---------------------------------------------------------------|
| @ref cli.py "cli"  | Subcommand parser; saves figures returned by plot fns         |
| @ref paths.py "paths"     | Filesystem constants                                          |
| @ref extract.py "extract"  | `extract` subcommand                                          |
| @ref topology.py "topology"  | MAC -> rab label resolution + plot colors                     |
| @ref loaders.py "loaders"   | Per-scenario `bh2` / `gps` / `mcm` loaders                    |
| @ref scripts.arpo_data.plots "plots/"     | Per-scenario plot functions (bh2 metrics, GPS)                |
| @ref multi_day.py "multi_day" | `multi-day` subcommand                                        |
| @ref multiday_variance.py "multiday_variance" | Variance table over `_pairwise_ks.csv`                |

## Conventions

- Plot fns return a `matplotlib.Figure`; only `cli.py` calls `savefig`.
- Loaders return `pd.DataFrame | None`.
- Node clocks are skewed -- comparisons use `__sec__` (session-relative),
  not raw UTC.

## Dependencies

`pandas`, `numpy`, and `matplotlib`