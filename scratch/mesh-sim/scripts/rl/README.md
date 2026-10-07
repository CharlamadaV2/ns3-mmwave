@page scripts_rl scripts/rl

@brief Code map and command guide for the Python side of mesh-sim RL: environment, training, evaluation, comparison, and experiment matrices.

Python side of mesh-sim RL: a Gymnasium environment that drives the simulator
over a stdin/stdout JSON bridge, MaskablePPO training, saved-policy evaluation
and comparison, and experiment matrices. The wire contract lives in
[src/rl](@ref src_rl); the `run.ini` keys are in the
[run.ini reference](@ref src_config_run_ini_reference).

## Module Layout

| File | Role |
| --- | --- |
| `__init__.py` | Registers `mesh_sim/MeshEnv-v0` with Gymnasium |
| [bootstrap_venv.py](@ref scripts/rl/bootstrap_venv.py) | Creates `.venv` from `requirements.txt` and checks pins; does not build the simulator |
| [cli_common.py](@ref scripts/rl/cli_common.py) | Shared CLI options, atomic JSON writes, hashes, package versions |
| [validate_config.py](@ref scripts/rl/validate_config.py) | Pre-flight scenario check |
| [train.py](@ref scripts/rl/train.py) | MaskablePPO training CLI; requires the `m-ppo` subcommand (`qr-dqn` is disabled and exits 1) |
| [inspect_model.py](@ref scripts/rl/inspect_model.py) | Training-run summary |
| [evaluate.py](@ref scripts/rl/evaluate.py) | Policy and baseline evaluation CLI |
| [compare.py](@ref scripts/rl/compare.py) | Paired comparison CLI |
| [experiment.py](@ref scripts/rl/experiment.py) | Experiment matrix planner and runner |

### Subdirectories

| Directory | Contents |
| --- | --- |
| @ref scripts_rl_agents | MaskablePPO glue; exploratory `q_learning.py` (not on the training path) |
| @ref scripts_rl_env | Gymnasium env, simulator subprocess, protocol checks, selection, observations, rewards, telemetry |
| @ref scripts_rl_policy | Bundle loading, compatibility checks, evaluation, comparison |
| @ref scripts_rl_ops | SLURM, benchmark, tuning, and fetch wrappers around an existing plan |
| @ref scripts_rl_tests | Test map; `fake_sim.py` exercises the Python exchange without ns-3 |

The C++ end of the bridge is [rl-bridge.cc](@ref src/rl/rl-bridge.cc).

## Control Modes

Chosen in `run.ini`, not on the command line.

- **Legacy single-node**: one `controlled_node_id`, `Discrete(7)` action, no
  `init` message. The jammer is part of the scenario, not an agent.
- **Centralized multi-node**: `controlled_nodes`, `MultiDiscrete([5]*M)` action.
  C++ owns masks and clamping; Python builds observations and rewards from the
  raw facts. The two selectors are mutually exclusive. Keys are in the
  [run.ini reference](@ref src_config_run_ini_reference); observation and reward
  selection is in [Selecting observations, rewards, and telemetry](@ref src_rl_policy_inputs_selection).

C++ owns the simulation because ns-3 mobility, propagation, link evaluation, and
routing already live there and must stay deterministic and testable without
Python. Python provides the Gymnasium and Stable-Baselines3 integration. In
centralized mode, movement limits and action validity are decided once, in C++;
Python never re-derives them. Legacy mode builds its `Discrete(7)` mask in
Python from the `[rl]` bounds.

## Setup

Run from `scratch/mesh-sim/`. See the [Overview](@ref index) for the one-command
environment setup.

```bash
python3 scripts/rl/bootstrap_venv.py          # creates .venv from requirements.txt
python3 scripts/rl/bootstrap_venv.py --check  # verify installed versions only
.venv/bin/python -m pytest scripts/rl/tests -q
```

Tuning needs Optuna, which is installed by hand:

```bash
.venv/bin/python -m pip install -r requirements-tuning.txt
```

## Run

Run everything from `scratch/mesh-sim/`. Replace `<BIN>` with an already-built
simulator binary. Global options go before the `m-ppo` subcommand.

### MaskablePPO smoke run {#scripts_rl_smoke_run}

```bash
.venv/bin/python -m scripts.rl.train \
  --sim-binary <BIN> \
  --run-config inputs/baselines/p0-smoke/run.ini \
  --output-dir outputs/rl-smoke-verification/rl \
  m-ppo --total-timesteps 16 --n-steps 16 --seed 1
```

- `--band sub-6` may be added before `m-ppo`.
- Omitting `--seed` falls back to `[scenario] seed` in the `run.ini`. Every
  episode reuses that one seed (multi-seed training is a later TODO).
- Training refuses to start if the output directory already contains
  `train_manifest.json` or `maskable_ppo_mesh.zip`.

For centralized control, point `--run-config` at
`inputs/baselines/centralized-multi-smoke/run.ini` and use `--output-dir
outputs/rl-multi`. Observation, reward, and telemetry flags are in
[Selecting observations, rewards, and telemetry](@ref src_rl_policy_inputs_selection).

### Model lifecycle {#scripts_rl_model_lifecycle}

Four commands cover a centralized run from configuration to evaluation. All but
`inspect_model` need a built simulator binary. Each stage writes a manifest that
the next one reads. The [lifecycle test map](@ref src_rl_policy_lifecycle_tests)
gives the purpose and expected result of each focused check.

#### 1. Validate the configuration

Without `--launch` every check is static (no simulator process). `--launch`
starts the simulator under `<output-dir>/validate/`, resets once, and reports the
live contract. It does not complete an episode.

```bash
.venv/bin/python -m scripts.rl.validate_config \
  --sim-binary <BIN> \
  --run-config inputs/baselines/building-bypass-smoke/run.ini \
  --output-dir outputs/bypass-validate --launch --json
```

#### 2. Train

```bash
.venv/bin/python -m scripts.rl.train \
  --sim-binary <BIN> \
  --run-config inputs/baselines/building-bypass-smoke/run.ini \
  --output-dir outputs/bypass-train \
  m-ppo --total-timesteps 1024 --n-steps 128 --seed 1 \
  --checkpoint-every-steps 512 --keep-checkpoints 2 \
  --eval-every-steps 256 --eval-episodes 1 --eval-seed 2
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint-every-steps` | `0` (off) | Save a checkpoint every N SB3 timesteps. |
| `--keep-checkpoints` | `3` | Prune the oldest checkpoints beyond this count. |
| `--eval-every-steps` | `0` (off) | Run the masked during-training evaluation every N timesteps. |
| `--eval-episodes` | `1` | Episodes per during-training evaluation. |
| `--eval-seed` | training seed + 1 | Seed for the during-training evaluation. |

Timesteps equal policy decisions here because training uses one environment.
[callbacks.py](@ref scripts/rl/agents/callbacks.py) wires SB3's checkpoint
callback and `MaskableEvalCallback` (masked, deterministic evaluation on a
separate environment and seed). The latter saves `best_model.zip` when mean
evaluation reward improves. Neither callback changes the training reward or
action rules.

#### 3. Inspect

Summarize a finished or failed run without loading the model. Add `--json` for
machine-readable output.

```bash
.venv/bin/python -m scripts.rl.inspect_model --run-dir outputs/bypass-train
```

It reads the manifest and saved files and prints status, seed and seed source,
control mode, the contract shape, selection, schema digests, scenario digests,
every model file with `exists`/`digest_ok`, the evaluation settings, and
recorded versus installed package versions.

| Exit | Meaning |
| --- | --- |
| `0` | Manifest readable; every recorded model file present with a matching digest. |
| `2` | A model file is missing or its digest differs. |
| `1` | Manifest missing or unreadable. |

#### 4. Evaluate

Evaluate a saved model against the `hold` and seeded `random_valid` baselines:

```bash
.venv/bin/python -m scripts.rl.evaluate \
  --sim-binary <BIN> \
  --run-dir outputs/bypass-train \
  --output-dir outputs/bypass-eval \
  --seeds 11,12,13 --policies model,hold,random_valid
```

| Flag | Meaning |
| --- | --- |
| `--model` | `final` (default), `best`, or `checkpoints/<file>.zip`. |
| `--run-dir` | Takes the run config, band, and selection from the training manifest. `--run-config` and `--band` may override them; the scenario check still runs, so an override may cause a mismatch. |
| `--seeds` | Comma-separated seeds and inclusive ranges: `301-303,310` equals `301,302,303,310`. |
| `--policies` | Default `model,hold,random_valid`. `geometric` and `optimization` are placement baselines (see [placement-baseline guide](@ref scripts_baselines)); they are not in the default list. |
| `--label` | Names the experiment row an evaluation belongs to; [compare.py](@ref scripts/rl/compare.py) uses it to group evaluations of independently trained models. |
| `--allow-seed-overlap` | Downgrades the seed-overlap refusal to a warning (see seed discipline below). |
| `--allow-different-scenario` | Relaxes only the scenario-identity compatibility step. |

Policies: `model` uses deterministic MaskablePPO predictions under the live
mask; `hold` stops every controlled node; `random_valid` picks uniformly among
each position's valid actions, restarting its generator from each episode seed.
Placement policies need `[baseline] movable_nodes` and `[rl] controlled_nodes`
to name the same nodes (or both `all`); otherwise preparation fails and no
episode runs.

[bundle.py](@ref scripts/rl/policy/bundle.py) treats the training manifest plus a
chosen final, best, or checkpoint ZIP as a saved model bundle and checks run
status and the ZIP's digest before loading.
[compat.py](@ref scripts/rl/policy/compat.py) then checks the live scenario and
policy contract, in the order given in [policy package](@ref scripts_rl_policy).
`--allow-different-scenario` never bypasses the digest, structural,
observation-schema, or reward checks; the evaluation manifest records the run as
`overridden`. Passing checks mean shapes and declared meanings match, never that
a policy transfers to another scenario.

| Exit | Meaning |
| --- | --- |
| `0` | Every episode completed, with no revalidated slots and no masked actions. |
| `2` | Episodes completed, but one of those counters is non-zero. |
| `1` | An error, or an episode that did not complete. |

#### Seed discipline

Keep the training seed, the during-training evaluation seed, and the standalone
evaluation seeds disjoint. With `model` in `--policies`, `evaluate` refuses to
start when a requested seed is the training seed or the recorded model-selection
seed, naming each seed and its role. `--allow-seed-overlap` turns that into a
stderr `WARNING:` line, records `seed_roles.held_out: false`, and excludes the
evaluation from across-run aggregates. The during-training evaluation runs in
its own environment, seed, and `<output-dir>/eval/` directory and contributes no
gradient steps. Only that callback writes `best_model.zip`; standalone
evaluation never informs model selection.

#### Manifests

| File | Version | Contents |
| --- | --- | --- |
| `train_manifest.json` | 4 | `status`/`error`, `algorithm`, `seed` and `seed_source`, `control_mode`, the live `contract`, resolved `selection`, `observation_schema` and `reward_schema` (with SHA-256), `scenario_identity` (SHA-256 of `run.ini`, `nodes.json`, and when configured `buildings.json` and `jammers.json`), `model_path`/`model_sha256`, `best_model_path`/`best_model_sha256`/`best_mean_reward`, a `checkpoints` list of `{path, sha256, num_timesteps}`, the `evaluation` block (or `null`), `hyperparameters`, `package_versions`, `python_version`, `platform`. |
| `eval_manifest.json` | 2 | Version-1 fields plus `label`; `seed_roles` (training, model-selection, held-out seeds, any overlap, whether overlap was allowed, whether held out); `training` (copied from the training manifest); `metric_source`; `episodes_expected` and `episodes_completed`. Every policy holds one record per requested seed, in seed order, with `status` `completed`, `failed`, or `not_run`; each record adds `error`, `metrics.unroutable_fraction`, and a `summary_json` pointer that is never parsed. |

Models trained with older tooling carry an older `manifest_version` and cannot
be loaded; retrain with the current tooling. The evaluation manifest is
rewritten after every episode. If a seed's simulator dies mid-episode, its
record keeps the episode directory and the partial return but reports every
metric as `null`, and the other seeds stay intact. Manifest `status` is
`completed` (all expected episodes completed), `partial` (some did), or `failed`
(aborted or nothing completed). `evaluate` refuses an `--output-dir` inside the
training run. Per-seed metrics and action-validity counts are in
`eval_manifest.json`; each episode's `steps.jsonl` has the decision trace.

#### Diagnostic fixture

`inputs/baselines/building-bypass-smoke/` is a diagnostic fixture: two drones
with one building between them. Going north or south around it is the fastest
route to line of sight. Action masks check bounds, not buildings, so moving west
through the building also reaches line of sight once x < 90. Its `run.ini`
header records the geometry and expected hold and north-moving numbers. It is a
smoke fixture for the lifecycle tools, not a benchmark or a training campaign.

### Comparing policies and running an experiment matrix {#scripts_rl_comparing_policies}

[compare.py](@ref scripts/rl/compare.py) turns finished evaluations into paired
model-minus-baseline statistics. [experiment.py](@ref scripts/rl/experiment.py)
expands a named matrix into the train, evaluate, and compare steps that produce
them. The statistics and exit codes are explained in the
[policy package](@ref scripts_rl_policy).

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

#### compare

| Flag | Meaning |
| --- | --- |
| `--eval-dirs DIR [DIR ...]` or `--plan experiment_plan.json` | Exactly one is required. |
| `--output-dir` | Required; must not already contain a training or evaluation manifest. |
| `--baselines a,b` | Restrict baselines; default is every non-`model` policy present. |
| `--json` | Print `comparison.json` instead of the summary lines. |

It reads only `eval_manifest.json` files, refuses version-1 manifests, and writes
`episodes.csv` (one row per evaluation directory, policy, and seed) and
`comparison.json`. Neither file carries a timestamp, so repeating the command
reproduces byte-identical output. A listed directory without a readable manifest
becomes a `missing_evaluations` entry instead of a crash.

#### experiment

- The matrix names the scenario in `run_config`, lists independent training,
  model-selection, and held-out evaluation seeds under `seeds`, and sets
  training and evaluation budgets.
- Each `rows` entry chooses one observation preset, action profile, and weighted
  reward. Rows are explicit combinations, not a Cartesian product. Add `--rows
  local-delivery` to both `plan` and `run` to select one row; use the same filter
  for one output root.
- Layout under `--output-root`: `experiment_plan.json`,
  `train/<row>/train-seed-<S>/`, `eval/<row>/train-seed-<S>/`, and
  `comparison/{episodes.csv,comparison.json}`.
- `run` executes pending steps sequentially in fresh processes and always runs
  `compare` last, so a missing run is reported rather than hidden.
- Step state comes from its manifest alone: `pending`, `done`, `partial`, or
  `blocked`. A blocked step is never re-run automatically; move or delete its
  directory. Re-planning the same root with a different matrix, binary, or
  `--rows` is refused; use a new root.
- `inputs/experiments/bypass-smoke-matrix.json` is diagnostic, not a benchmark:
  four explicit rows around one anchor on the bypass fixture with smoke-sized
  budgets. It exercises the harness and is not evidence that a policy learns.
  See [Policy comparison tests](@ref src_rl_policy_comparison_tests).

### Benchmarking, tuning, cluster runs, and fetch

[scripts/rl/ops](@ref scripts_rl_ops) wraps an existing plan: it measures one
`(row, training seed)` task (`benchmark`), searches the wired PPO knobs with
Optuna (`tune`), submits the plan to SLURM (`cluster`), and copies results back
(`fetch`). Commands, schemas, and refusals are all documented there. The cluster
commands are validated against a fake scheduler only, and `fetch` against a fake
`rsync`.

## Output {#scripts_rl_output}

| Step | Where |
| --- | --- |
| RL training | `<output-dir>/train_manifest.json`, `<output-dir>/maskable_ppo_mesh.zip`, and one `episode-NNNN/` per episode (below). With the cadence flags also `checkpoints/checkpoint_<N>_steps.zip`, `best_model.zip`, `evaluations.npz`, and `eval/episode-NNNN/` for the during-training evaluation. |
| RL evaluation | `<output-dir>/eval_manifest.json` and one `<policy>/episode-NNNN/` per evaluated policy and seed, with the same episode contents as training. |
| Compare | `<output-dir>/episodes.csv` and `<output-dir>/comparison.json`. |
| Experiment | `experiment_plan.json`, `train/`, `eval/`, and `comparison/` under `--output-root`. |

Each `episode-NNNN/` holds `run.log`, `inputs/`, `sim_stderr.log`,
`rl_episode.json`, optional `steps.jsonl`, optional `policy_decisions.jsonl` and
`policy_decisions_manifest.json` (`--decision-records`), and `seed-<seed>/...`.
Startup and evaluation resets can leave zero-step interrupted episodes; filter
by `rl_episode.json` `status` when aggregating. For a file-by-file reading path,
see [Reading an RL output directory](@ref src_rl_reading_output). The direct-run
layout is in the [Overview](@ref readme_where_output_lands).

## Dependencies

- Python packages are pinned in `requirements.txt` (installed by
  `bootstrap_venv.py`); its direct dependencies are recorded in every training
  and evaluation manifest and checked by `bootstrap_venv.py --check`.
- `requirements-tuning.txt` pins Optuna separately, for `ops/tune.py` only.
  Adding a tuner to `requirements.txt` would change every manifest and force a
  tuning-only package on machines that only train, evaluate, or compare.
- An already-built simulator binary is passed with `--sim-binary`.
