# scripts/validation

Sim-vs-field comparison pipeline. Runs mesh-sim across a directory of
scenarios (multi-seed), then overlays the pooled sim distribution against
ARPO field traces as an ECDF with a bootstrap 90% CI band and a two-sample
K-S statistic.

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

# 2. Run the sim across all scenarios (default: 5 seeds each)
python -m scripts.validation.run_batch

# 3. Convert per-seed sim outputs to arpo_data-style trace CSVs
python -m scripts.validation.sim_to_traces outputs/<batch>

# 4. ECDF + bootstrap CI + K-S vs field
python -m scripts.validation.compare outputs/<batch>

# 5. Cross-batch summary (overall + per-scenario)
python -m scripts.validation.compare_runs

# 6. Scenario fidelity audit (sim vs field layout / mobility)
python -m scripts.validation.scenario_fidelity outputs/<batch>
```


### Flags

```bash
# Run with more seeds
python -m scripts.validation.run_batch --seeds 1,2,3,4,5,6,7,8,9,10

# Dry-run a single scenario
python -m scripts.validation.run_batch --only arpo-1-1-static-04172026 --dry-run

# Compare only SNR and RCPI with a tighter CI
python -m scripts.validation.compare <batch> --metrics snr,rcpi --ci 95

# Compare two specific batches
python -m scripts.validation.compare_runs <batch1> <batch2> --metric snr

# Overall tables only (suppress per-scenario matrix)
python -m scripts.validation.compare_runs --no-per-scenario

# Export comparison results to CSV
python -m scripts.validation.compare_runs \
    --csv outputs/cmp.csv \
    --per-scenario-csv outputs/cmp_long.csv

# Fidelity check for a single scenario with a looser tolerance
python -m scripts.validation.scenario_fidelity <batch> \
    --only arpo-2-2-los-obstruction-04172026 --tol-m 10

# Preview waypoint patch without writing
python -m scripts.validation.build_waypoints \
    arpo-2-2-los-obstruction-04172026 --dry-run

# Patch waypoints scaled to a 120 s sim window, then run
python -m scripts.validation.build_waypoints --all --time-mode scale --duration 120
python -m scripts.validation.run_batch

# Same thing in one step
python -m scripts.validation.run_batch --auto-waypoints
```


## Output

| Step | Location |
|------|----------|
| `run_batch` | `outputs/.../<scenario>/seed-N/{links,mcs,rx-power}.csv` + `batch_manifest.json` |
| `sim_to_traces` | `outputs/.../<scenario>/sim_traces/seed-N/csvs/<src>/bh2_<metric>__<src>_to_<peer>_trace.csv` |
| `compare` | `outputs/.../<scenario>/validation/pngs/<src>/ecdf_<metric>__<src>_to_<peer>.png` + per-scenario `metrics.csv` + top-level `validation_summary.csv` |


## Module layout

| File               | Role                                                              |
|--------------------|-------------------------------------------------------------------|
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


## Dependencies

`pandas`, `numpy`, and `matplotlib`