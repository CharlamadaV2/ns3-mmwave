# scripts/validation

Sim-vs-field comparison: runs mesh-sim across a directory of scenarios
(multi-seed), then overlays the pooled sim distribution against the ARPO
field traces as normalized histograms with a bootstrap CI band on the sim
curve, plus a two-sample K-S statistic and |Δmedian|/|Δmean| in the
chart subtitle and CSV.

Scenario inputs come from `inputs/custom/sherpa/spring_lake/` (one dir
per scenario, each holding `run.ini` + `nodes.json`). Field traces come
from `data/arpo_extracted/_plots/per_day/<scenario>/csvs/<src>/` and
must be generated up-front via `scripts.arpo_data.cli plot`.

## Run

From `scratch/mesh-sim/`:

```bash
# 1. produce field trace CSVs (one-time per dataset; skip if already done)
python -m scripts.arpo_data.cli plot --all

# 2. run the sim across all scenarios (default: 5 seeds each)
python -m scripts.validation.run_batch

# 3. convert per-seed sim outputs to arpo_data-style trace CSVs
python -m scripts.validation.sim_to_traces outputs/<YYYY-MM>/<DD>/<HH-MM-SS>-validation

# 4. histogram overlay + bootstrap CI + K-S vs field
python -m scripts.validation.compare      outputs/<YYYY-MM>/<DD>/<HH-MM-SS>-validation

# 5. cross-batch summary (overall + per-scenario)
python -m scripts.validation.compare_runs

# 6. scenario fidelity audit (sim vs field layout/mobility)
python -m scripts.validation.scenario_fidelity outputs/<YYYY-MM>/<DD>/<HH-MM-SS>-validation
```

Flags worth knowing:

```bash
python -m scripts.validation.run_batch --seeds 1,2,3,4,5,6,7,8,9,10
python -m scripts.validation.run_batch --only arpo-1-1-static-04172026 --dry-run
python -m scripts.validation.compare   <batch> --metrics snr,rcpi --ci 95
python -m scripts.validation.compare_runs <batch1> <batch2> --metric snr
python -m scripts.validation.compare_runs --no-per-scenario       # overall tables only
python -m scripts.validation.compare_runs --csv outputs/cmp.csv --per-scenario-csv outputs/cmp_long.csv
python -m scripts.validation.scenario_fidelity <batch> --only arpo-2-2-los-obstruction-04172026
python -m scripts.validation.scenario_fidelity <batch> --tol-m 10
python -m scripts.validation.build_waypoints arpo-2-2-los-obstruction-04172026 --dry-run
python -m scripts.validation.build_waypoints arpo-2-2-los-obstruction-04172026 --time-mode scale --duration 120
python -m scripts.validation.build_waypoints --all --time-mode scale --duration 120
python -m scripts.validation.run_batch --auto-waypoints
```

`--auto-waypoints` on `run_batch` patches each scenario's `nodes.json` in-place
from its field GPS trace before invoking the sim (skips static-rab2 scenarios
automatically). The patch is permanent — `git diff inputs/` to see what
changed, `git checkout inputs/` to revert.

`build_waypoints` reads the field `gps_track_trace.csv` for a scenario, translates
to the sim frame using a stationary anchor node (default `rab1`), downsamples the
moving node's track to N waypoints, and patches them into the scenario's
`nodes.json` with `mobility = "waypoint"`. Time modes: `raw` (use field clock as-is),
`clip` (cap at `--duration`), `scale` (stretch/compress to `--duration`). After
patching, rebuild the sim and re-run.

`scenario_fidelity` reads each scenario's snapshotted `nodes.json` + `seed-1/positions.csv`
and the field's `gps_track_trace.csv` (produced by `arpo_data.cli plot`), then for each rab
reports field-vs-sim bounding box, path length, and mobility class (static / mobile), plus
pairwise t=0 distances between rabs. It surfaces mismatches like "rab2 drives a 1.7 km loop
in the field but the sim has it pinned `fixed`." The mobility classifier is bbox-based
(threshold 20 m max-dim) so cumulative GPS jitter doesn't trip it.

`compare_runs` auto-discovers every batch under `outputs/` with a
`validation_summary.csv` (or takes explicit batch dirs) and writes cross-batch
heatmap PNGs to `outputs/cross_batch_summary/`. One PNG per (metric, score) —
rows = scenarios with date, cols = batches with timestamp, cells = mean across
rab links. For multi-day variance in the **field** data itself (independent of
any sim run), see `python -m scripts.arpo_data.multiday_variance`.

## How to read the chart

One axis: normalized histograms (densities) of sim and field samples,
overlaid. Density normalization (∫=1) lets the curves compare directly even
though sim N (tens of thousands) is much larger than field N. The shaded
band on the sim curve is a pointwise bootstrap CI (default 90%, B=1000).

Subtitle annotates the K-S D-statistic and |Δmedian|/|Δmean| in the metric's
units. K-S is unitless (0–1) — useful for ranking similarity across runs;
|Δmed| / |Δmean| are in the metric's own units. Median is the robust
central-tendency (skewed wireless metrics + bursty fades drag the mean
around); mean is reported alongside it so you can see when they diverge.

The K-S p-value is intentionally not reported: with N in the tens of
thousands per pool, p ≈ 0 even for operationally trivial differences.

## Output

| Step           | Where                                                              |
|----------------|--------------------------------------------------------------------|
| `run_batch`    | `outputs/.../<HH-MM-SS>-validation/<scenario>/seed-N/{links,mcs,rx-power}.csv` + `batch_manifest.json` |
| `sim_to_traces`| `outputs/.../<scenario>/sim_traces/seed-N/csvs/<src>/bh2_<metric>__<src>_to_<peer>_trace.csv` |
| `compare`      | `outputs/.../<scenario>/validation/pngs/<src>/hist_<metric>__<src>_to_<peer>.png` + per-scenario `metrics.csv` + top-level `validation_summary.csv` |

Each figure plots sim density with a shaded pointwise bootstrap CI band
(default 90%, B=1000) and field density as a line; subtitle annotates K-S,
|Δmed|, and |Δmean|. The summary CSV has one row per (scenario, link, metric)
with means, medians, IQRs, sample counts, |Δmedians|, |Δmeans|, and K-S.

## Module layout

| File               | Role                                                              |
|--------------------|-------------------------------------------------------------------|
| `run_batch.py`     | One sim binary call per scenario, fanning out seeds via `--seeds=` |
| `sim_to_traces.py` | Rewrites per-seed sim CSVs into arpo_data-compatible trace CSVs   |
| `compare.py`       | Pools sim seeds + field directions, renders histogram overlays + K-S |
| `compare_runs.py`  | Cross-batch summary table (overall + per-scenario)                |
| `scenario_fidelity.py` | Per-scenario sim-vs-field layout/mobility audit               |
| `build_waypoints.py` | Generate waypoint mobility for a node from its field GPS trace  |

## Conventions

- Sim and field samples are pooled raw (no smoothing, no time alignment) --
  the comparison is distributional, not temporal.
- Sim MCS is post-capped at 12 to match the field firmware limit.
- Sim trace CSVs leave `tag_local_mac` / `tag_sta_mac` / `tag_interface`
  empty (sim has no per-antenna identity).
- Scenario names map by stripping `arpo-`, joining the middle tokens with
  `_`, and upper-casing single-letter family tokens
  (`arpo-1-x-misc-04172026` → `1-X_misc_04172026`).
- `NaN` rows in `validation_summary.csv` mean the field collect didn't
  capture telemetry for that link (e.g. rab2 and rab3 weren't beam-paired
  in `2-7_throughput_stationary` or `2-X_blocking`). Not a pipeline bug --
  there is nothing to compare against.

## Dependencies

`pandas`, `numpy`, `matplotlib`. No scipy, no ns-3.
