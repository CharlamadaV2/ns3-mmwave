@page scripts_rl scripts/rl
# RL guide and code map

Python side of mesh-sim RL: a Gymnasium environment that drives the simulator
over a stdin/stdout JSON bridge, MaskablePPO training, saved-policy evaluation
and comparison, and experiment matrices. The wire contract lives in
[`src/rl/README.md`](@ref src_rl); this page covers how to run and
configure RL and where things are.

All commands run from `scratch/mesh-sim/` and need a built simulator binary
(`--sim-binary <BIN>`), except `inspect_model`.

## Setup

Create the virtual environment (see [Python setup](@ref python_environment)):

```bash
python3 scripts/rl/bootstrap_venv.py
```

## Smoke run

Global options go before the `m-ppo` subcommand:

```bash
.venv/bin/python -m scripts.rl.train \
  --sim-binary <BIN> \
  --run-config inputs/baselines/p0-smoke/run.ini \
  --output-dir outputs/rl-smoke-verification/rl \
  m-ppo --total-timesteps 16 --n-steps 16 --seed 1
```

- `--band sub-6` may be added before `m-ppo`.
- Omitting `--seed` falls back to `[scenario] seed` in the `run.ini`. Either
  way every episode reuses that one seed (a multi-seed training policy is a
  later TODO).
- Training refuses to start if the output directory already contains
  `train_manifest.json` or `maskable_ppo_mesh.zip`.

## Control modes

Chosen in `run.ini`, not on the command line.

- **Legacy single-node**: one `controlled_node_id`, `Discrete(7)` action (`6:Stay`),
  no `init` message. The jammer is part of the scenario, not an agent.
- **Centralized multi-node**: `controlled_nodes`, `MultiDiscrete([5]*M)` action
  (`4:hold`). C++ owns masks and clamping; Python builds observations and
  rewards from the raw facts.

C++ owns the simulation because ns-3 mobility, propagation, link evaluation, and
routing already live there and must stay deterministic and testable without
Python. Python provides the Gymnasium/SB3 integration. In centralized mode,
movement limits and action validity are decided once, in C++; legacy mode builds
its `Discrete(7)` mask in Python from the `[rl]` bounds.

## Scenario keys

### Reward type

`[rl] reward_type` is the reward the simulator computes:

- `throughput`: sum of `delivered_mbps` across flows (default).
- `all_links_los`: `+1` when every peer link of the controlled node is LOS,
  `-1` otherwise.

`mean_sinr` is a deprecated alias for `all_links_los`. It still runs, prints one
warning on stderr, and is recorded in `run.log` as `rl.reward_alias`.
`[rl] z_min`/`z_max` are validated like the x and y bounds (`min < max`).
In centralized mode Python can instead compose the reward from named
components (see [Selection keys](@ref scripts_rl_selection_keys)).

### Centralized control keys {#scripts_rl_centralized_control_keys}

Setting `[rl] controlled_nodes` switches the bridge from legacy to centralized
mode. The keys below apply only when `[rl] enabled = true`; other `[rl]` keys
keep their meaning.

| Key | Type / unit | Default | Mode |
|---|---|---|---|
| `controlled_nodes` | `all` or comma-separated node ids | absent (legacy mode) | centralized selector |
| `max_controlled_nodes` | int, slot count `M` | `0` (auto-size to the resolved count), max `64` | centralized |
| `action_profile` | enum | `move_2d` (only accepted value) | centralized |
| `decision_interval_s` | seconds | `0` (means `tick_s`); must be an integer multiple of `tick_s` | centralized |

- `controlled_nodes` and the legacy `controlled_node_id` are mutually exclusive;
  setting both is a configuration error.
- `all` means every node in `nodes.json`, in file order. Jammers are never mesh
  nodes and can never be controlled, even if a jammer `id` equals a node `id`.
- With `reward_type = all_links_los`, the centralized reward is `+1` only if
  every controlled node has at least one peer link and all of them are LOS. The
  reward per decision is the mean of the per-tick rewards in that window.
- `action_set` and `dimensions` are **not** accepted keys. The loader ignores
  unknown keys silently; `dimensions` is derived metadata reported in `init`.

Train on the bundled centralized fixture:

```bash
.venv/bin/python -m scripts.rl.train \
  --sim-binary <BIN> \
  --run-config inputs/baselines/centralized-multi-smoke/run.ini \
  --output-dir outputs/rl-multi \
  m-ppo --total-timesteps 16 --n-steps 16 --seed 1
```

### Selection keys {#scripts_rl_selection_keys}

These `[rl]` keys are read by the Python env and `train.py`; the simulator
ignores them. They apply to centralized mode only.

| Key | CLI flag | Default | Meaning |
|---|---|---|---|
| `observation_preset` | `--observation-preset` | `raw_links_v1` | which observation the policy sees |
| `reward_components` | `--reward-components` | absent (the C++ reward) | comma-separated components Python composes into the reward |
| `reward_weights` | `--reward-weights` | `1.0` per component | one float per component, same order |
| `telemetry` | `--telemetry` | `none` | `none` or `steps`; `steps` writes `<episode-dir>/steps.jsonl` |
| `telemetry_every` | `--telemetry-every` | `1` | positive int; save every kth decision (requires `telemetry = steps`) |

CLI flags go before the `m-ppo` subcommand:

```bash
.venv/bin/python -m scripts.rl.train \
  --sim-binary <BIN> \
  --run-config inputs/baselines/centralized-multi-smoke/run.ini \
  --output-dir outputs/rl-multi-custom \
  --observation-preset local_links_v1 \
  --reward-components delivery_ratio,connectivity \
  --telemetry steps --telemetry-every 2 \
  m-ppo --total-timesteps 16 --n-steps 16 --seed 1
```

- Each key resolves independently with precedence CLI > `run.ini` > default. Both
  manifests record the resolved value and its source.
- An unknown preset or component, a weight count that does not match the
  components, or any non-default value in legacy mode fails before the simulator
  starts.
- Naming any component makes Python the reward authority: `step()` returns the
  weighted total, `info["reward"]` holds per-component values, validity, and
  weights, and the C++ value stays in `info["reward"]["legacy"]`.
- Presets, components, formulas, and telemetry records are in the
  [policy-input guide](@ref src_rl_policy_inputs).

Telemetry details:

- Records are buffered and flushed every 32 records and on stop, so a hard kill
  can lose up to 32 records.
- `rl_episode.json` is updated every decision regardless of `telemetry_every`.
- On `centralized-multi-smoke` a record is about 0.9 KB and the header about 3.5 KB.
- `scripts.rl.env.telemetry.replay_file` rebuilds every saved observation and
  recomputes Python-composed rewards from stored facts and schema. For the
  default C++ reward it checks the observation only. It does not verify
  simulator physics, the next state, or unsaved decisions.

Both manifests carry SHA-256 fingerprints of the observation and reward schema
descriptions. They detect changed declared layouts, normalization, components,
or weights; they do not detect every code or physics change and are not evidence
that a policy transfers between scenarios. `manifest_version` labels the saved
JSON format, not the model or simulator version. Centralized runs need a
simulator binary that exports per-decision facts: an `init` without
`facts_schema` is rejected before training starts.

## Lifecycle {#scripts_rl_lifecycle}

Each stage writes a manifest that the next one reads. The
[lifecycle test map](@ref src_rl_policy_lifecycle_tests) gives the purpose
and expected result of each focused check.

1. [`validate_config.py`](validate_config.py): pre-flight check of scenario,
   selection, and binary; `--launch` adds one real reset.
2. [`train.py`](train.py): trains MaskablePPO; writes `train_manifest.json`,
   the model, checkpoints, and per-episode `rl_episode.json`.
3. [`inspect_model.py`](inspect_model.py): prints manifest provenance and
   verifies model file digests (`--run-dir`).
4. [`evaluate.py`](evaluate.py): loads the training run as a verified bundle,
   runs the model and baselines (for example hold) on `--seeds`; writes
   `eval_manifest.json`.
5. [`compare.py`](compare.py): paired model-minus-baseline statistics from
   `--eval-dirs` or `--plan`; writes `comparison.json` and `episodes.csv`.

### Validate

Without `--launch` every check is static (no simulator process). `--launch`
starts the simulator under `<output-dir>/validate/`, resets once, and reports
the live contract. It does not complete an episode.

```bash
.venv/bin/python -m scripts.rl.validate_config \
  --sim-binary <BIN> \
  --run-config inputs/baselines/building-bypass-smoke/run.ini \
  --output-dir outputs/bypass-validate --launch --json
```

### Train

```bash
.venv/bin/python -m scripts.rl.train \
  --sim-binary <BIN> \
  --run-config inputs/baselines/building-bypass-smoke/run.ini \
  --output-dir outputs/bypass-train \
  m-ppo --total-timesteps 1024 --n-steps 128 --seed 1 \
  --checkpoint-every-steps 512 --keep-checkpoints 2 \
  --eval-every-steps 256 --eval-episodes 1 --eval-seed 2
```

- `--checkpoint-every-steps 0` (default) disables checkpoints.
- `--eval-every-steps 0` (default) disables the evaluation callback.
- `--keep-checkpoints` (default 3) prunes the oldest checkpoints.
- `--eval-seed` defaults to the training seed + 1.
- Units are SB3 timesteps, which equal policy decisions here (one environment).
- [`agents/callbacks.py`](agents/callbacks.py) wires SB3's checkpoint callback
  and `MaskableEvalCallback` (masked, deterministic evaluation on a separate
  environment and seed), which saves `best_model.zip` when mean evaluation
  reward improves. Neither changes the training reward or action rules.

`train_manifest.json` is version 4. It records `status`/`error`, `algorithm`,
`seed` and `seed_source`, `control_mode`, the live `contract`, the resolved
`selection`, `observation_schema` and `reward_schema` (with SHA-256),
`scenario_identity` (SHA-256 of `run.ini`, `nodes.json`, and when configured
`buildings.json` and `jammers.json`), `model_path`/`model_sha256`,
`best_model_path`/`best_model_sha256`/`best_mean_reward`, a `checkpoints` list of
`{path, sha256, num_timesteps}`, the `evaluation` block (or `null`),
`hyperparameters`, `package_versions`, `python_version`, and `platform`. Models
trained with older tooling carry an older `manifest_version` and are not
loadable: retrain.

### Inspect

```bash
.venv/bin/python -m scripts.rl.inspect_model --run-dir outputs/bypass-train
```

Add `--json` for machine-readable output. It reads the manifest and saved files
without loading the policy, and prints status, seed and source, control mode,
contract shape, selection, schema and scenario digests, each model file with
`exists`/`digest_ok`, evaluation settings, and recorded versus installed package
versions.

| Exit | Meaning |
|---|---|
| 0 | manifest readable; every recorded model file present with matching digest |
| 1 | manifest missing or unreadable |
| 2 | a model file is missing or its digest differs |

### Evaluate

Evaluate a saved model against the `hold` and seeded `random_valid` baselines:

```bash
.venv/bin/python -m scripts.rl.evaluate \
  --sim-binary <BIN> \
  --run-dir outputs/bypass-train \
  --output-dir outputs/bypass-eval \
  --seeds 11,12,13 --policies model,hold,random_valid
```

- `model` uses deterministic MaskablePPO predictions under the live action mask.
  `hold` stops every controlled node. `random_valid` picks uniformly among each
  position's valid actions, restarting its generator from each episode seed.
- `--model` selects `final` (default), `best`, or `checkpoints/<file>.zip`.
  With `--run-dir`, run config, band, and selection come from the training
  manifest; `--run-config`/`--band` may override them (the scenario check still
  runs and may mismatch).
- `--seeds` accepts comma-separated seeds and inclusive ranges:
  `301-303,310` equals `301,302,303,310`.
- `--label <name>` names the experiment row an evaluation belongs to; `compare`
  uses it to group evaluations of independently trained models.
- [`policy/bundle.py`](policy/bundle.py) treats the training manifest plus a
  chosen final, best, or checkpoint ZIP as a saved model bundle. It checks run
  status and the ZIP's recorded digest before loading;
  [`policy/compat.py`](policy/compat.py) then checks the live scenario and
  policy contract.
- Output: `eval_manifest.json` plus one `<policy>/episode-NNNN/` per policy and
  seed. An `--output-dir` inside the training run is refused.
- The manifest is rewritten after every episode. A seed whose simulator dies
  mid-episode keeps its episode directory and partial return but reports every
  metric as `null`.
- Manifest `status`: `completed` (all episodes completed), `partial` (some did),
  `failed` (aborted or none completed).

| Exit | Meaning |
|---|---|
| 0 | every episode completed, no revalidated slots, no masked actions |
| 1 | error, or an episode did not complete |
| 2 | episodes completed but one of those counters is non-zero |

`eval_manifest.json` is version 2. It records `label`; `seed_roles` (training
seed, model-selection seed, held-out seeds, overlap, whether overlap was allowed,
whether the result is held out); `training` (copied from the training manifest);
`metric_source`; and `episodes_expected`/`episodes_completed`. Each policy holds
one record per requested seed in seed order, with `status` `completed`, `failed`,
or `not_run`, plus `error`, `metrics.unroutable_fraction`, and a `summary_json`
path (a pointer only, never parsed). Each episode's `steps.jsonl` has the
decision trace.

### Seed discipline

Keep the training seed, the during-training evaluation seed, and the standalone
evaluation seeds disjoint.

- With `model` among `--policies`, `evaluate` refuses to start when a requested
  seed is the training seed or the recorded model-selection seed, naming each.
- `--allow-seed-overlap` downgrades that to a stderr `WARNING:` and records
  `seed_roles.held_out: false`, which excludes the evaluation from across-run
  aggregates.
- The during-training evaluation has its own environment, seed, and output
  directory (`<output-dir>/eval/`) and contributes no gradient steps. Only it
  writes `best_model.zip`; standalone evaluation never informs model selection.

### Compatibility checks

Loading a model checks, in order, and stops at the first failure: structural
contract fields, observation schema, reward schema, scenario identity.
`--allow-different-scenario` relaxes only the last step (including differing
scenario-input digests). It never bypasses the ZIP digest, structural,
observation-schema, or reward checks, and the evaluation manifest records the
run as `overridden`. Passing checks mean shapes and declared meanings match, not
that a policy transfers to another scenario. Details are in the
[compatibility envelope](@ref src_rl_compatibility_envelope).

### Bypass fixture

`inputs/baselines/building-bypass-smoke/` is a diagnostic fixture: two drones
with one building between them. Going north or south around it is the fastest
route to LOS. Action masks check bounds, not buildings, so moving west through
the building also reaches LOS once x < 90. Its `run.ini` header records the
geometry and the expected hold and north-moving numbers. It is a smoke fixture
for the lifecycle tools, not a benchmark.

## Comparing and experiment matrices

`compare.py` turns finished evaluations into paired model-minus-baseline
statistics. [`experiment.py`](experiment.py) expands a named matrix
(`inputs/experiments/*.json`) into `experiment_plan.json`, then runs the train,
evaluate, and compare steps. Subcommands: `plan`, `run`, `status`.

```bash
.venv/bin/python -m scripts.rl.compare \
  --eval-dirs outputs/bypass-eval --output-dir outputs/bypass-compare

.venv/bin/python -m scripts.rl.experiment plan \
  --matrix inputs/experiments/bypass-smoke-matrix.json \
  --output-root outputs/bypass-matrix --sim-binary <BIN>

.venv/bin/python -m scripts.rl.experiment run \
  --matrix inputs/experiments/bypass-smoke-matrix.json \
  --output-root outputs/bypass-matrix --sim-binary <BIN>

.venv/bin/python -m scripts.rl.experiment status --output-root outputs/bypass-matrix
```

### compare

- Takes exactly one of `--eval-dirs DIR [DIR ...]` or `--plan experiment_plan.json`,
  plus a required `--output-dir` that must not already hold a training or
  evaluation manifest.
- `--baselines a,b` restricts baselines (default: every non-`model` policy
  present). `--json` prints `comparison.json` instead of summary lines.
- Reads only `eval_manifest.json` files and refuses version-1 manifests. Writes
  `episodes.csv` (one row per evaluation directory, policy, and seed) and
  `comparison.json`. Neither has a timestamp, so repeating a command gives
  byte-identical output.
- A listed directory without a readable manifest becomes a
  `missing_evaluations` entry, not a crash.

| Exit | Meaning |
|---|---|
| 0 | every expected evaluation found, complete, held out, and free of health counters |
| 1 | comparison incomplete (outputs still written) or refused (nothing written) |
| 2 | outputs complete but a health counter is non-zero or an evaluation is not held out |

### experiment

- The matrix names the scenario in `run_config`, lists independent training,
  model-selection, and held-out seeds under `seeds`, and sets training and
  evaluation budgets. Each `rows` entry picks one observation preset, action
  profile, and weighted reward; rows are explicit combinations, not a Cartesian
  product.
- `--rows local-delivery` selects named rows; `plan` and `run` must use the same
  filter for one output root.
- Layout under `--output-root`: `experiment_plan.json`,
  `train/<row>/train-seed-<S>/`, `eval/<row>/train-seed-<S>/`, and
  `comparison/{episodes.csv,comparison.json}`.
- `run` executes pending steps sequentially in fresh processes and always runs
  `compare` last, so a missing run is reported rather than hidden.
- Step state comes only from its manifest: `pending`, `done`, `partial`, or
  `blocked`. A blocked step is never re-run automatically; move or delete its
  directory. Re-planning the same root with a different matrix, binary, or
  `--rows` is refused; use a new root.
- `inputs/experiments/bypass-smoke-matrix.json` is diagnostic: four explicit
  rows around one anchor on the bypass fixture with smoke-sized budgets. It
  exercises the harness and is not evidence that a policy learns.

### Reading the statistics

- Two kinds of interval never share a label. *Across evaluation seeds* (one
  saved model): model and baseline are paired seed by seed, and the interval
  shows variation over held-out scenario seeds with the model fixed. *Across
  independently trained models*: evaluations sharing a `--label` contribute one
  mean difference each over their common seeds, and the interval shows run-to-run
  variation. The training seed currently sets both PPO initialization and the
  training scenario seed, so the two cannot be separated (TODO-RL-SEEDS-1).
- A single evaluation seed or training run yields a value and no interval;
  `interval_omitted` says why.
- Every comparison reports `n_expected` beside `n_used` (pairs both policies
  completed) and lists each dropped seed with a per-side reason such as
  `model:failed`, `hold:metric_null`, or `random_valid:not_run`. A `null` metric
  is never read as zero. `zero_variance` marks a zero-width interval.
- `delivery_ratio` is the primary metric; `connectivity`, `los_fraction`, and
  `unroutable_fraction` follow. `return` is compared only inside one evaluation
  and is flagged `comparable_across_reward_definitions: false`, since different
  reward components give returns on different scales.
- All statistics come from the RL telemetry window (`metric_source`). Nothing is
  read from `summary.json`, which excludes warmup ticks while the RL window does
  not (see the RL-reward warmup entry in `TODO.md`).
- Intervals are t intervals assuming approximately normal paired differences;
  on a bounded ratio with few seeds they are approximate.

The purpose and expected output of each comparison and matrix test are in
[Policy comparison tests](@ref src_rl_policy_comparison_tests).

## Benchmark, tuning, and cluster runs {#scripts_rl_benchmark}

`scripts.rl.ops` measures one `(row, training seed)` task, runs a small Optuna
search over the wired PPO knobs, submits a planned matrix to SLURM, and fetches
results. Commands, schemas, and refusal rules are in
[`ops/README.md`](@ref scripts_rl_ops). Cluster commands are validated against a fake
scheduler only, and `fetch` against a fake `rsync`.

`requirements-tuning.txt` (Optuna) is installed by hand, only for `ops.tune`:

```bash
.venv/bin/python -m pip install -r requirements-tuning.txt
```

It is kept out of `requirements.txt` because that file defines the direct
dependency set recorded in every manifest and checked by
`bootstrap_venv.py --check`.

## Output {#scripts_rl_output}

| Step | Where |
|---|---|
| Training | `<output-dir>/train_manifest.json`, `<output-dir>/maskable_ppo_mesh.zip`, and one `episode-NNNN/` per episode containing `run.log`, `inputs/`, `sim_stderr.log`, `rl_episode.json`, `steps.jsonl` (optional), and `seed-<seed>/...` |
| Training with cadence flags | also `checkpoints/checkpoint_<N>_steps.zip`, `best_model.zip`, `evaluations.npz`, and `eval/episode-NNNN/` for the during-training evaluation |
| Evaluation | `<output-dir>/eval_manifest.json` and one `<policy>/episode-NNNN/` per policy and seed, with the same episode contents as training |

SB3 resets also leave zero-step interrupted episodes in `eval/`; filter by
`rl_episode.json` `status` when aggregating.

## Module Layout

| File | Role |
| --- | --- |
| [`__init__.py`](__init__.py) | Registers `mesh_sim/MeshEnv-v0` with Gymnasium |
| [`bootstrap_venv.py`](bootstrap_venv.py) | Creates `.venv` from `requirements.txt` and checks pins; does not build the simulator |
| [`cli_common.py`](cli_common.py) | Shared CLI options, atomic JSON writes, hashes, package versions |
| [`validate_config.py`](validate_config.py) | Pre-flight scenario check |
| [`train.py`](train.py) | MaskablePPO training CLI |
| [`inspect_model.py`](inspect_model.py) | Training-run summary |
| [`evaluate.py`](evaluate.py) | Policy and baseline evaluation CLI |
| [`compare.py`](compare.py) | Paired comparison CLI |
| [`experiment.py`](experiment.py) | Experiment matrix planner and runner |

## Subdirectories

| Directory | Contents |
| --- | --- |
| @ref scripts_rl_agents | MaskablePPO glue; exploratory `q_learning.py` (not on the training path) |
| @ref scripts_rl_env | Gymnasium env, simulator subprocess, protocol checks, selection, observations, rewards, telemetry |
| @ref scripts_rl_policy  | Bundle loading, compatibility checks, evaluation, comparison |
| @ref scripts_rl_ops | SLURM, benchmark, and tuning wrappers around an existing plan |
| @ref scripts_rl_tests | Test map; `fake_sim.py` exercises the Python exchange without ns-3 |

The C++ end of the bridge is [`src/rl/rl-bridge.cc`](@ref rl-bridge.cc).
For saved files, see [Where output lands](@ref where_output_lands).

## Dependencies

Python packages are pinned in `requirements.txt` (installed by
`bootstrap_venv.py`); Optuna for `ops/tune.py` is in `requirements-tuning.txt`.
An already-built simulator binary is passed with `--sim-binary`.
