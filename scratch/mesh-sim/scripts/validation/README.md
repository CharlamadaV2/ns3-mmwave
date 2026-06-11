@page scripts_validation scripts/validation

@brief Module for comparing simulation results to arpo_data

Sim-vs-field comparison: runs mesh-sim across a directory of scenarios
(multi-seed), then overlays the pooled sim distribution against the ARPO
field traces as normalized histograms with a bootstrap CI band on the sim
curve, plus a two-sample K-S statistic and |Δmedian|/|Δmean| in the
chart subtitle and CSV.

**Scenario inputs** come from `inputs/custom/sherpa/spring_lake/` — one
subdirectory per scenario, each containing `run.ini` and `nodes.json`.
**Field traces** come from
`data/arpo_extracted/_plots/per_day/<scenario>/csvs/<src>/` and must be
generated up-front via `scripts.arpo_data.cli plot`.


## Run

From `scratch/mesh-sim/`:

```bash
# 1. Produce field trace CSVs (one-time per dataset; skip if already done)
python -m scripts.arpo_data.cli plot --all

# 2. Generate sim config files from calfex field data
python scripts/validation/build_config_files.py \
    -i data/arpo_extracted \
    -o inputs/calfex \
    -b sub-6 \
    -m node

# 3. Patch waypoints from GPS tracks into nodes.json
python -m scripts.validation.build_waypoints --all --time-mode scale --duration 60

# 4. Run the sim across all scenarios (default: 5 seeds each)
python -m scripts.validation.run_batch

# 5. Convert per-seed sim outputs to arpo_data-style trace CSVs
python -m scripts.validation.sim_to_traces outputs/<batch>

# 6. histogram overlay + bootstrap CI + K-S vs field
python -m scripts.validation.compare      outputs/<YYYY-MM>/<DD>/<HH-MM-SS>-validation

# 7. Cross-batch summary (overall + per-scenario)
python -m scripts.validation.compare_runs

# 8. Scenario fidelity audit (sim vs field layout / mobility)
python -m scripts.validation.scenario_fidelity outputs/<batch>
```

`build_config_files` reads the calfex GPS and silvus CSVs and writes
`nodes.json` and `run.ini` directly from field measurements. Channel
parameters come from `silvus/config.csv` on each IH node. Node starting
positions are the first GPS fix per node converted to local ENU metres
relative to the centroid of all nodes. Gateway name and traffic demand are
read from the sibling `calfex_sdwan/` directory; if not found both default
to safe placeholders. All IH nodes are written with `"mobility": "waypoint"`
and empty waypoints — always run `build_waypoints.py` afterwards before
invoking the sim.

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

| Step                 | Where                                                              |
|----------------------|--------------------------------------------------------------------|
| `build_config_files` | `inputs/calfex/nodes.json` + `inputs/calfex/run.ini`              |
| `run_batch`          | `outputs/.../<HH-MM-SS>-validation/<scenario>/seed-N/{links,mcs,rx-power}.csv` + `batch_manifest.json` |
| `sim_to_traces`      | `outputs/.../<scenario>/sim_traces/seed-N/csvs/<src>/bh2_<metric>__<src>_to_<peer>_trace.csv` |
| `compare`            | `outputs/.../<scenario>/validation/pngs/<src>/hist_<metric>__<src>_to_<peer>.png` + per-scenario `metrics.csv` + top-level `validation_summary.csv` |

Each figure plots sim density with a shaded pointwise bootstrap CI band
(default 90%, B=1000) and field density as a line; subtitle annotates K-S,
|Δmed|, and |Δmean|. The summary CSV has one row per (scenario, link, metric)
with means, medians, IQRs, sample counts, |Δmedians|, |Δmeans|, and K-S.

## Module layout

| File               | Role                                                              |
|--------------------|-------------------------------------------------------------------|
| @ref build_config_files.py "build_config_files" | Generates `nodes.json` and `run.ini` from calfex GPS and silvus field data. |
| @ref run_batch.py "run_batch"     | One sim binary call per scenario, fanning out seeds via `--seeds=` |
| @ref sim_to_traces.py "sim_to_traces" | Rewrites per-seed sim CSVs into arpo_data-compatible trace CSVs   |
| @ref compare.py "compare"       | Pools sim seeds + field directions, renders ECDF overlays + K-S   |
| @ref compare_runs.py "compare_runs"  | Cross-batch summary table (overall + per-scenario)                |
| @ref scenario_fidelity.py "scenario_fidelity" | Per-scenario sim-vs-field layout/mobility audit               |
| @ref build_waypoints.py "build_waypoints" | Generate waypoint mobility for a node from its field GPS trace  |


## Conventions

- **Distributional comparison, not temporal.** Sim and field samples are
  pooled raw (no smoothing, no time alignment). The comparison is over the
  marginal distribution of each metric.
- **MCS cap.** Sim MCS is post-capped at 12 to match the field firmware
  limit before any comparison is made.
- **Sim trace MAC columns are empty.** `sim_to_traces` leaves
  `tag_local_mac`, `tag_sta_mac`, and `tag_interface` blank — the sim has
  no per-antenna identity.
- **Scenario name mapping.** Sim names translate to field names by stripping
  `arpo-`, joining the middle tokens with `_`, and upper-casing
  single-letter family tokens:
  `arpo-1-x-misc-04172026` → `1-X_misc_04172026`.
- **NaN rows in `validation_summary.csv`** mean the field collect did not
  capture telemetry for that link (e.g. rab2 and rab3 were not beam-paired
  during the scenario). This is expected for some scenarios and is not a
  pipeline error.
- **`--auto-waypoints` mutates `inputs/`.** The patch is permanent for the
  duration of the run. Use `git diff inputs/` to review and
  `git checkout inputs/` to revert.
- **`build_waypoints` mobility classifier is bbox-based** (threshold: 20 m
  max bounding-box dimension). Path length alone is GPS-noise-prone and
  would misclassify jittery static nodes as mobile.
- **`build_config_files` writes empty waypoints.** Always run
  `build_waypoints.py` after `build_config_files.py` before running the sim.


## Dependencies

`pandas`, `numpy`, and `matplotlib`