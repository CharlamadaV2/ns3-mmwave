@page scripts_rl scripts/rl
# RL code map

Python side of mesh-sim RL: a Gymnasium environment that drives the simulator
over a stdin/stdout JSON bridge, MaskablePPO training, saved-policy evaluation
and comparison, and experiment matrices. The wire contract lives in
[`src/rl/README.md`](../../src/rl/README.md); this page only says where things are.

Start with [Python setup](../../README.md#python-environment) and the
[smoke run](../../README.md#maskableppo-smoke-run) in the main README.

## Control modes

Chosen in `run.ini`, not on the command line.

- **Legacy single-node**: one `controlled_node_id`, `Discrete(7)` action, no
  `init` message. The jammer is part of the scenario, not an agent.
- **Centralized multi-node**: `controlled_nodes`, `MultiDiscrete([5]*M)` action.
  C++ owns masks and clamping; Python builds observations and rewards from the
  raw facts. See [Centralized multi-node control](../../README.md#centralized-multi-node-control)
  and [Selecting observations, rewards, and telemetry](../../README.md#selecting-observations-rewards-and-telemetry).

## Lifecycle

Each stage writes a manifest that the next one reads. Details are in
[Model lifecycle](../../README.md#model-lifecycle).

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

`evaluate` and `compare` exit `2` for warnings (for example seed overlap); see
[Comparing policies](../../README.md#comparing-policies-and-running-an-experiment-matrix).

## Experiment matrices

[`experiment.py`](experiment.py) expands an explicit matrix
(`inputs/experiments/*.json`) into `experiment_plan.json`, then runs the
train, evaluate, and compare steps. Subcommands: `plan`, `run`, `status`.
Step state comes only from each step's output directory and manifest, so a
changed matrix, binary, or `--rows` needs a new `--output-root`.

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

The C++ end of the bridge is [`src/rl/rl-bridge.cc`](../../src/rl/rl-bridge.cc).
For saved files, see [Where output lands](../../README.md#where-output-lands).

## Dependencies

Python packages are pinned in `requirements.txt` (installed by
`bootstrap_venv.py`); Optuna for `ops/tune.py` is in `requirements-tuning.txt`.
An already-built simulator binary is passed with `--sim-binary`.
