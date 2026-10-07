@page scripts_baselines scripts/baselines

@brief Plans gateway-free geometric and optimization node placements before a run, scored by the simulator channel.

`scripts/baselines/` plans the gateway-free `geometric` and `optimization`
placement baselines once, before a simulation, and hands the simulator an
ordinary scenario whose selected nodes start at the planned positions. Every
candidate layout is scored by the simulator's own channel through its
`--channel-query` mode ([contract](@ref src_query)); there is no
separate RF model, RF file or extra Python dependency beyond `numpy`.

| File | Owns |
| --- | --- |
| `config.py` | Strict `[baseline]` parsing, the optional mapping file, explicit `[rl]` bounds. |
| `adapter.py` | Node records (roles `movable`/`fixed`, platforms), penalty/grid/probe resolution, datum and result validation, `prepare()`. |
| `solver.py` | `PlanRequest`/`PlanResult`; runs one strategy with a `ChannelScorer` and packages diagnostics, query statistics and timing. |
| `planners/` | The query client, shared objective and the two strategies; see [planners/README.md](@ref scripts_baselines_planners). |
| `effective_inputs.py` | Source snapshots, line-preserving INI edits, two-step staging of `effective-inputs/`, the `nodes.json` rewrite. |
| `artifacts.py` | Manifest/plan v2 schemas, status transitions, fingerprint, adapted-code and channel-runtime identity. |
| `runner.py` | The standalone command and the simulator child lifecycle. |

No module here imports `scripts.rl.*`, Gymnasium, Torch, or Stable-Baselines3.
With `algorithm = none`, neither `solver.py` nor any `planners/` module is
imported. The copy in `third_party/arpo_placement/` is an unused reference;
its `PROVENANCE.md` lists what was adapted and every deviation.

## Commands

Standalone: plan once, then run the ordinary (non-RL) simulator on the plan.

```bash
.venv/bin/python -m scripts.baselines.runner \
  --sim-binary <BIN> --run-config <INI> \
  [--seeds 1,2,3] [--algorithm none|geometric|optimization] [--output-dir DIR] \
  [--band mmwave|sub-6]
```

| Option | Default |
| --- | --- |
| `--algorithm` | `[baseline] algorithm`, else `none` |
| `--seeds` | `[scenario] seed` (the simulator's default, 42, when the key is absent). Accepts `A-B` ranges; the simulator receives the expanded comma list. |
| `--output-dir` | `outputs/YYYY-MM/DD/HH-MM-SS-baseline[-N]` under the mesh-sim root |
| `--band` | the scenario's band |

The same `--sim-binary` serves the channel queries while planning and then the
run. The output directory must be absent or empty; otherwise the runner exits
1 and writes nothing. Exit codes: `0` only when the run is `complete`; `1` for
an argument or preparation error, or when a seed has no `summary.json`; the
simulator's own exit code when it exits non-zero; `128 + signal` (130 for
SIGINT, 143 for SIGTERM) when interrupted.

Evaluation: the placement methods are two more policy names in
`scripts.rl.evaluate`, next to `model`, `hold`, and `random_valid`:

```bash
.venv/bin/python -m scripts.rl.evaluate \
  --sim-binary <BIN> --run-config <INI> \
  --seeds 1,2 --policies hold,random_valid,geometric,optimization
```

The `--policies` default is still `model,hold,random_valid`. Every requested
placement method is prepared before any episode runs; a preparation failure
exits 1, runs no episode, and writes no `eval_manifest.json`. In evaluation the
policy name is the method; `[baseline] algorithm` is only recorded as
`requested_algorithm`. Placement policies need `[baseline] movable_nodes` to
name the same nodes as `[rl] controlled_nodes` (see
[the hold executor contract](#hold-executor-contract-evaluation)).

## The [baseline] section

The C++ binary reads only `algorithm`. Everything else is read by
`config.py`, which rejects unknown keys, unknown values, duplicate sections or
keys, `:` assignments, and indented continuation lines. See also
[`src/config/run-ini-reference.md`](@ref src_config_run_ini_reference).

| Key | Values | Default | Required when active |
| --- | --- | --- | --- |
| `algorithm` | `none`, `geometric`, `optimization` | `none` (also for an absent section or blank value) | — |
| `objective` | `coverage`, `balanced`, `resilience` | none | yes |
| `application` | `initial_positions` | `initial_positions` | no |
| `movable_nodes` | distinct comma list of node ids, or `all` | none | yes |
| `seed` | integer ≥ 0; the optimizer's RNG seed, never a simulation seed | none | `optimization` |
| `max_iterations` | integer ≥ 1; the optimizer's only termination rule | none | `optimization` |
| `waypoint_policy` | `reject`, `translate` | `reject` | no |
| `mapping_file` | mapping v2 path, relative to the INI | none (see below) | no |
| `aerial_fixed_cost_m2`, `ground_fixed_cost_m2` | number ≥ 0, m² per move | `150000` | no |
| `aerial_cost_m2_per_m`, `ground_cost_m2_per_m` | number ≥ 0, m² per metre | `500` / `100` | no |
| `aerial_max_displacement_m`, `ground_max_displacement_m` | number > 0, metres | none (unlimited) | no |
| `candidate_grid_cells`, `coverage_grid_cells` | integer ≥ 1 | `400` | no |
| `grid_min_resolution_m` | number > 0 | `5.0` | no |
| `coverage_probe_height_m` | number ≥ 0 | `1.5` | no |
| `coverage_probe_rx_gain_dbi` | finite number ≥ 0 | the query's `[channel] rx_array_gain_dbi` | no |
| `coverage_sinr_db` | finite number | `-6.7` | no |
| `balanced_core_fraction` | number in [0, 1] | `0.5` | no |

`geometric` accepts `seed` and `max_iterations` so one INI can serve both
methods; the manifest records them as `null` with status `unused` /
`not_applicable`. Movable nodes are never inferred from `[traffic]` or `[rl]`.
Zero is not a valid displacement cap: omit the key for no cap.

## Mapping file (optional)

Without `mapping_file` the geofence is `rl_bounds` and each node's platform
comes from its `node_type`. A mapping file has exactly these top-level keys
(`platforms` is optional):

```json
{
  "baseline_mapping_version": 2,
  "geofence": {"source": "rl_bounds"},
  "platforms": {"nodes": {"<node-id>": "aerial"}}
}
```

| Field | Rule |
| --- | --- |
| `geofence` | `{"source": "rl_bounds"}` uses `[rl] x_min, x_max, y_min, y_max`, all four written in the INI (loader defaults are not used). `{"source": "rectangle_xy_m", "x_min": …, "x_max": …, "y_min": …, "y_max": …}` gives the rectangle directly. |
| `platforms` | Optional per-node `ground` or `aerial`. Otherwise `drone` is `aerial`, `vehicle` and `pedestrian` are `ground`; another `node_type` needs an entry here. |

z is metres above flat ground at z = 0; a node with z < 0 fails preparation.

## Objective

All terms are m² so movement penalties and coverage trade off directly. The
mesh link table at t = 0 comes from the query; `connected` means
`sinr_db ≥ -6.7` dB as computed by the simulator. Full definitions and the two
search strategies are in [planners/README.md](@ref scripts_baselines_planners).

- **Core**: the largest connected component of all mesh nodes (ties: the one
  holding the smallest roster index). There is no gateway or root.
- **Coverage**: the clipped-cell area of coverage-grid probes (at
  `coverage_probe_height_m`, receive gain `coverage_probe_rx_gain_dbi`)
  whose node-to-probe SINR is at least `coverage_sinr_db` for some core node.
  The probe areas sum to the geofence area (AOI).
- **Connectivity**: `2 x AOI` per selected node outside the core.
- **Resilience**: vulnerability pairs — for each selected core node, the
  unordered core pairs its removal disconnects; charged `AOI` per pair for
  `resilience`, `0.02 x AOI` for `balanced`, nothing for `coverage`.
- **Separation**: a tie-break below one coverage cell.
- **Movement**: per selected node, `0` at zero displacement, else
  `fixed_cost_m2 + cost_m2_per_m x d` from the node's original start, using its
  platform's keys; `max_displacement_m` is a hard cap.

Score = coverage + separation − movement − connectivity − resilience. Staying
put is always a candidate and a valid result (zero displacement, not a
failure).

### Penalty scale

The defaults are the Desktop reference values: 150000 m² to move at all for
either platform, then 500 m²/m (aerial) or 100 m²/m (ground). On a 400 m x
400 m AOI (160000 m²) the fixed cost alone is about 94 % of the area, so most
nodes stay put. `planner.log` and the manifest record `aoi_m2` and
`fixed_cost_m2 / aoi_m2` per platform; a ratio ≥ 1 is logged as a warning (no
coverage gain can pay for a move), never an error. For a zero-cost comparison
set the four `*_fixed_cost_m2` / `*_cost_m2_per_m` keys to `0`; keep the cap
keys absent, or at the same positive value, across compared runs.

## Channel scoring

`adapter.prepare` stages the effective INI first and points the query at it,
with the source `nodes.json`, so the query builds the same nodes, mobility
models, buildings and jammers as the later run; candidate positions come from
each request. In evaluation the query gets `--rl-mode`, matching the episode.
The planning seed is the first simulation seed (`--seeds` / `simulation_seeds`),
else `[scenario] seed`; `run_id` is `[scenario] run_id`. The jammer seed equals
the planning seed, as in an ordinary run. One plan is reused for every
simulation seed: it is optimised for the first seed's realization and only
evaluated on the others. Any query failure aborts preparation; there is no
partial plan.

Parity, at two levels (see [the query contract](@ref src_query)):
mechanics parity — every candidate goes through `LinkEvaluator::Evaluate` with
the run's resolved band, gains, buildings and jammers — holds by construction.
An identical t = 0 realization with the first seed of an ordinary run on the
effective inputs is expected but is claimed only after the human-run
real-binary parity checks pass, and never for later seeds of a multi-seed run.
Scores are deterministic per layout, not a stable per-pair random stream across
layouts.

Jammer limitation (inherited from the simulator): with sub-6 jammers a finite
jammed SINR is clamped to ≥ 0 dB, so a jammer cannot disconnect a mesh link or
uncover a probe at the default −6.7 dB thresholds. A positive custom
`coverage_sinr_db` can change probe coverage. Capacity can drop but is not in
the objective. The baselines do not claim jammer avoidance.

Each layout costs one full `EvaluateAll` plus N x G probe links (G =
coverage probes) in a forked child; `channel_scoring.queries` in the manifest
records the totals.

## TODO: polygon geofences

The adapter supports **only an axis-aligned rectangle** in scenario metres. The
reference planner in `third_party/` handles polygon geofences, but the adapted
planners do not: they accept no polygon vertices and do not derive a polygon
from buildings.

A polygon-shaped mapping request is rejected, not approximated. `config.py`
fails when `geofence` is a JSON list, when it has any of the keys `vertices`,
`polygon`, `polygons`, `points`, `coordinates`, `rings`, `holes`, `geojson`,
`exterior`, or `interiors`, or when an unknown `source` mentions `poly`,
`geojson`, `vert`, or `point`. The error ends with:

> only axis-aligned rectangle geofences are supported (geofence.source
> 'rl_bounds' or 'rectangle_xy_m'); polygon geofences are a documented TODO,
> and a bounding rectangle is never substituted

Do not work around this by writing the bounding rectangle of an irregular
permitted area. The planner would then be free to place nodes in the excluded
parts of that rectangle (a no-fly zone, a building footprint, water), and every
recorded check would still pass, so the run would appear to enforce a geofence
that it does not. Polygon support needs its own mapping schema, coordinate
conversion, containment checks on the result, boundary and hole semantics, and
tests.

## Direct-binary guard

A direct simulator run whose INI has `algorithm = geometric` or
`optimization` exits 1 with a message pointing at this runner or at
`scripts.rl.evaluate --policies`. `[rl] enabled = true` alone does not bypass
the guard; only the explicit `--rl-mode` flag does, and then the binary prints
one stderr notice and runs the **original** layout. `algorithm = none` or an
absent section behaves exactly as before. Prepared effective INIs always carry
`algorithm = none`, so they pass the guard. The binary does not see misspelled
`[baseline]` keys; the Python launchers reject them.

## Hold executor contract (evaluation)

Placement policies run `hold` on every RL slot of a layout whose selected nodes
start at the plan, so preparation requires:

1. `[rl] controlled_nodes` is present (legacy single-node control is rejected);
2. the resolved `movable_nodes` set **equals** the resolved `controlled_nodes`
   set. `all` on either side means the whole roster and order does not matter;
   RL slots keep the `controlled_nodes` order. A mismatch fails before any
   query and names both differences (movable but not controlled, controlled
   but not movable) with the fix: set both keys to the same ids, or `all`;
3. every candidate and planned position lies in the evaluation region: the
   geofence rectangle intersected with the `[rl]` x/y bounds (an empty
   intersection fails preparation).

Equality means no RL slot outside the plan is frozen by `hold`, and every node
that is not RL-controlled is unselected and follows its scenario mobility under
every policy. Letting some RL slots follow scenario mobility would need new
simulator slot semantics and is not supported. Saved models are untouched: a
model still needs the roster it was trained on, so pick `movable_nodes` to
match its `controlled_nodes`. Standalone runs have no ownership rule: a moved
node starts at the plan and then follows its configured mobility; unselected
nodes are untouched. `ownership` in the manifest records both resolved lists.

## What preparation does

1. Parse and validate `[baseline]`, the mapping file (or its defaults), and
   `nodes.json`; in evaluation check ownership and the evaluation region (see
   above); resolve penalties, grids, probe settings, the planning seed and
   `run_id`; record the binary, channel-runtime and adapted-code hashes.
2. Copy the INI and every file it references into `source-inputs/`.
3. Stage `effective-inputs.partial/`: the INI with `[baseline] algorithm =
   none`, `[scenario] nodes_file = nodes.json`, referenced files rebased to
   copied basenames, and, standalone only, `[rl] enabled = false`; every other
   byte is kept. The source `nodes.json` is copied unchanged.
4. For an active method, run the strategy against a query worker on the staged
   INI and validate the result. Validation fails, and never clamps, on: missing
   or extra ids; non-finite values; a selected node outside the rectangle (the
   boundary counts as inside); a planned z that differs from the source z by
   more than 1e-6; displacement beyond the platform's `*_max_displacement_m`;
   an unselected node moved by more than 0.01 m; in evaluation, a position
   outside the `[rl]` x/y bounds; for a `random_walk` node, a position outside its
   own `random_walk.bounds`.
5. Rewrite selected nodes' start x/y in the staged `nodes.json` (z and all other
   nodes are written from the source; unmoved plans keep the source bytes), add
   `baseline-plan.json`, rename the directory to `effective-inputs/` once, and
   hash the result. A selected `waypoint` node fails under `waypoint_policy =
   reject`; under `translate` its whole path shifts by the planned offset. Any
   failure removes the staging directory.

## Outputs {#scripts_baselines_outputs}

Standalone (`<run>` is `--output-dir`):

```
<run>/
  baseline_manifest.json
  planner.log               query command, scale check, planner notes and timing,
                            worker stderr, or "method none: no planner was run"
  sim.log                   simulator stdout and stderr
  source-inputs/            byte copies of the requested INI and its files
  effective-inputs/         run.ini nodes.json [assets] baseline-plan.json
  run.log  inputs/  seed-N/ written by the simulator, unchanged
```

No `rl_episode.json` or `steps.jsonl` is written. The simulator archives every
file beside the effective INI, so `inputs/` also holds `baseline-plan.json`.

Evaluation adds `<eval-out>/<method>/baseline/` with the same preparation files
(no `sim.log`, no `seed-N/`), beside that policy's usual `episode-NNNN/`
directories. `eval_manifest.json` gains `policies[<method>].baseline` for
placement policies only: `baseline_manifest_version`, method, requested
algorithm, objective, executor, planner seed, `max_iterations`,
`planning_seed`, `ownership`, fingerprint, `initial_displacement_m_total`,
effective input hashes, the mapping hash, and paths to the baseline manifest
and plan. `eval_manifest_version` stays 2.

`baseline_manifest.json` (`baseline_manifest_version: 2`):

| Field | Meaning |
| --- | --- |
| `status` | `preparing` → `prepared` → `running` → `complete`, or `failed` / `interrupted` at any step. Evaluation stops at `prepared`; episode results live in `eval_manifest.json`. |
| `error` | Failure reason; for a simulator failure, its exit code and the tail of `sim.log`. |
| `seeds` | Standalone, one `{seed, status, summary}` per simulation seed: `complete` (with `seed-N/summary.json`), `missing`, or `failed` (no summary after a non-zero exit). |
| `method`, `requested_algorithm` | Effective method and the INI's `algorithm`. |
| `planner_seed`, `max_iterations` (+ `_status`) | Optimizer settings, separate from `simulation_seeds` and the planning seed. |
| `mapping_sha256`, `geofence` | Mapping file hash (`null` without one) and the resolved rectangle. |
| `sim_binary_sha256` | The binary that scored candidates; required for an active method. |
| `channel_runtime_sha256`, `channel_runtime_files` | Aggregate hash of the binary plus the ns-3 core/network/mobility/propagation/buildings libraries it loads (the binary alone if none are dynamic), and the files hashed. |
| `planner_code_sha256` | Aggregate hash of the adapted modules (`solver`, `config`, `adapter`, `effective_inputs`, `artifacts`, `planners/{objective,geometric,optimization,channel}`). |
| `channel_scoring` | Query contract and isolation, planning seed, `planning_run_id`, `jammer_seed`, `mode_flag` (`--rl-mode` or `null`), band and source, link and coverage thresholds, probe (height, resolved receive gain, cells, cell size, count), candidate grid, and query totals. |
| `penalties` | Per-platform fixed, per-metre and cap values, `aoi_m2`, and `fixed_cost_over_aoi`. |
| `planner_settings` | The strategy's constants (temperatures, move weights, anchor schedule, …). |
| `ownership` | `mode`, `movable_resolved`, and in evaluation `controlled_resolved` (slot order). |
| `fingerprint` | Hash (`fingerprint_version` 2) of method, objective, planner seed, `max_iterations`, waypoint policy, mapping hash, penalty/grid/probe/threshold settings, planning seed and `run_id`, mode flag, band, the four effective-input hashes, binary, channel-runtime and adapted-code hashes, `planner_settings`, and ownership; never timing or paths. `scripts.rl.compare` rejects a `--label` group whose placement fingerprints differ. |
| `initial_displacement_m_total` | Sum of planned x/y displacement. `travel_m_total` and `displacement_m_final` in evaluation keep measuring motion after decision 0. |

All paths are relative to the run directory, so a moved run stays readable;
`source_run_config_abs` and `channel_runtime_files` are informational absolute
paths. `baseline-plan.json` (`baseline_plan_version: 2`) lists, per node, id,
roster index, RL slot (evaluation), role (`movable` or `fixed`), platform,
`selected`, `original` and `planned` x/y/z, and `displacement_m`, plus the
planner's diagnostics and score breakdown as `planner_predictions`.

## Migrating from version 1

- Delete `[baseline] gateway_node_id` and `rf_config`; both are now rejected
  as unknown keys with a hint. `[traffic] gateway_node_id` is unrelated and
  unchanged.
- Movement costs and caps move from the RF file into the `[baseline]` cost
  keys above.
- Mapping files: set `baseline_mapping_version` to 2 and delete `origin`,
  `ground_datum` and `radios`; or drop `mapping_file` to use `rl_bounds` and
  platforms by node type.
- `baseline_manifest.json` and `baseline-plan.json` are version 2: `rf`,
  `planner_source`, `origin`, `ground_datum` and per-node `radios` are gone.
  `--planner-source`, `$MESH_SIM_ARPO_PATH` and `requirements-baselines.txt`
  no longer exist.
- Evaluation: version 1 accepted `movable_nodes` as a subset of `[rl]
  controlled_nodes`. Now the two sets must be equal; widen `movable_nodes` (or
  set both to `all`) or narrow `controlled_nodes`. Narrowing `controlled_nodes`
  changes the RL roster, so a saved model trained on the wider roster no
  longer applies. Standalone runs are unaffected.

## Planner limits

- Rectangle geofences only (see the TODO above); flat ground at z = 0 only.
- Placement is 2-D: each node keeps its z, and a changed z fails validation.
- No periodic replanning, no 3D placement, no target following.
- Optimizer determinism per seed is tested against the fake query worker on
  one machine; determinism across platforms or numpy versions is not claimed.
- `planner_predictions` are the planner's t = 0 scores on the planning seed,
  not measurements of the run.

## What may not be committed

Only the nine reference files and `PROVENANCE.md` under
`third_party/arpo_placement/` are authorized. Never commit the supplied
`rf_config.yaml`, datasets, examples, payloads, weights, archives, generated
plans, or run outputs, and nothing from `data/` or `outputs/`. Tests use only
hand-written synthetic scenarios (`tests/conftest.py` and the inline fixtures
of `tests/test_real_binary.py`); no scenario is added under `inputs/`.

## Tests

```bash
.venv/bin/python -m pytest scripts/baselines/tests -q -rs
MESH_SIM_BIN=<BIN> .venv/bin/python -m pytest scripts/baselines/tests/test_real_binary.py -q
```

`test_runner.py` drives the runner against `tests/fake_child.py`, a stand-in
that checks argv and writes `seed-N/summary.json`, with failure and hang modes
selected by `FAKE_CHILD_MODE`. Planner and preparation tests score layouts
with `tests/fake_query.py`, a stand-in channel-query worker with a
distance-threshold channel and fault modes. `test_provenance.py` checks the
reference hashes and the plan identity. `test_real_binary.py` covers the
guard, the runner, evaluation (ownership equality, `--rl-mode` query parity,
the gateway-free all-movable and partial-selection acceptance rows) and
channel-query parity on a real binary;
without `MESH_SIM_BIN` each test is skipped with a reason starting `BLOCKED:`,
which is missing evidence, not a pass.
