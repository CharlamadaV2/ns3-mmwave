# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# scripts/rl

Python side of mesh-sim RL: the Gymnasium env that drives the simulator over
a stdin/stdout JSON bridge, MaskablePPO training, saved-policy evaluation and
comparison, experiment matrices, and cluster/tuning ops. Docs to read first:

- `src/rl/README.md` is the **wire contract** (modes, masks, cadence, reward
  window). `src/rl/policy-inputs.md` covers observation presets and reward
  components.
- `README.md` here is the code map. Each subdirectory (`agents/`, `env/`,
  `ops/`, `policy/`, `tests/`) has its own CLAUDE.md with local invariants.
  Test maps are split between `tests/README.md` and
  `src/rl/policy-{input,lifecycle,comparison}-tests.md`. `ops/README.md` is the full spec for
  `ops/` (SLURM receipts, reconcile states, fetch, benchmark, tune).
- `CONTRIBUTING.md` (mesh-sim root) has the "update together" table for
  changes to action/obs/reward semantics, outputs, or scenario identity.

## Commands

From `scratch/mesh-sim/`. Never build the simulator; pass an existing binary.

```bash
python3 scripts/rl/bootstrap_venv.py [--check] [--venv <path>]
.venv/bin/python -m pytest scripts/rl/tests -q
.venv/bin/python -m pytest scripts/rl/tests/test_mesh_env.py::<test_name>
MESH_SIM_BIN=/abs/path/to/bin .venv/bin/python -m pytest scripts/rl/tests/test_real_binary.py -q
.venv/bin/python -m pip install -r requirements-tuning.txt   # only for ops.tune / its tests

.venv/bin/python -m scripts.rl.validate_config ...   # pre-flight a scenario; --launch does one reset
.venv/bin/python -m scripts.rl.train | evaluate | compare | inspect_model ...
.venv/bin/python -m scripts.rl.experiment {plan,run} --matrix inputs/experiments/<m>.json --output-root R --sim-binary <BIN>
```

- Tests run against `tests/fake_sim.py`, a scripted protocol peer with no
  radio model, and `tests/fake_slurm.py` shims. `test_real_binary.py` skips
  without `MESH_SIM_BIN`. Tests needing `sb3_contrib`, Optuna, or pandas skip
  when those are missing.
- For any protocol or simulator change, also run the C++ config tests
  (`make -C tests/unit/config test`) and `make -C tests integration` against a
  fresh build.

## Architecture

### One step through the stack

`train.py` -> `agents/mask_ppo.py` (`ActionMasker`) -> `env/mesh_env.py`
(`MeshRlEnv`) -> `env/episode.py` (one simulator subprocess per `reset`, plus
a stdout reader thread) -> `src/rl/rl-bridge.cc`.

- **C++ owns** masks, clamping, speed caps, action revalidation, and the base
  per-tick reward. **Python never re-derives masks or clamps** in centralized
  mode; the legacy `Discrete(7)` mask is the one exception (built from `[rl]`
  bounds in `MeshRlEnv.action_masks`).
- `env/protocol.py` validates each message against the contract
  (`mesh_move_2d_v1`, facts schema `mesh_facts_v1`) and the action shape.
- Two modes, chosen in `run.ini`: legacy (`controlled_node_id`,
  `Discrete(7)`, no `init`) and centralized (`controlled_nodes`,
  `MultiDiscrete([5]*M)`). Action `4` means `-Z` in legacy mode but hold in
  centralized mode.
- In centralized mode, Python turns the raw `facts` into the policy's inputs.
  `env/selection.py` resolves observation preset, reward components, and
  telemetry (CLI > `run.ini` > default). `observations.py` builds the preset
  and a schema hash. `rewards.py` composes rewards from window sums.
  `telemetry.py` writes and replays `steps.jsonl`.

### Policy lifecycle

`train` writes `train_manifest.json` (+ `maskable_ppo_mesh.zip`, checkpoints,
per-episode `rl_episode.json`) -> `evaluate` reads it as a verified bundle
(`policy/bundle.py`), runs `policy/compat.py` checks (structural fields,
schema hash, scenario-identity digests), and evaluates the model against
baselines such as hold -> `eval_manifest.json` -> `compare` computes paired
model-minus-baseline stats (`policy/compare.py`, primary metric
`delivery_ratio`) -> `comparison.json` + `episodes.csv`.

- Scenario identity (`env/config.py::read_scenario_identity`) is SHA-256 of
  the `run.ini` / nodes / buildings / jammers files (the last two `null` when
  unconfigured). Compat and benchmark compare these
  digests, not paths.
- Exit code `2` from `evaluate` / `compare` is a warning (health counters,
  seed overlap, not held out), not a failure. `experiment` and `ops` tolerate
  it.

### Experiments and ops

- `experiment.py` expands an explicit matrix (rows are listed, not a product)
  into an immutable `experiment_plan.json` of train/evaluate/compare steps.
  Step state comes **only from the step's output dir and manifest**
  (`step_state`): missing dir = `pending`, completed manifest = `done`, anything
  else = `blocked` ("move or delete <dir> to retry"). Nothing ever deletes or
  overwrites a step dir. A changed matrix, binary, or `--rows` needs a new
  `--output-root`.
- Seed roles are disjoint: `training`, `model_selection`, `held_out`. Held-out
  seeds must never reach training or tuning; `load_matrix` and `ops.tune`
  enforce this.
- `ops/` wraps an existing plan: it never adds training capability. Only
  `n_steps`, `gamma`, and `ent_coef` reach `MaskablePPO`, so only those are
  tunable. `ops/tasks.py` and `ops/reconcile.py` are pure; subprocess and
  scheduler calls live in `slurm.py` / `cluster.py`. Scheduler query failures
  must degrade to `unknown`, which blocks resubmission.

## Conventions

- Saved JSON files carry a `*_version` constant next to their writer
  (`train.MANIFEST_VERSION`, mirrored in `policy/bundle.py`;
  `EVAL_MANIFEST_VERSION`, mirrored as `REQUIRED_MANIFEST_VERSION` in
  `compare_inputs.py`; `PLAN_VERSION`, `COMPARISON_VERSION`, ops versions).
  Bump the relevant one and update its reader when fields change.
  `CONTRIBUTING.md` still says `train_manifest.json` is version 2; the code is
  at 4.
- `requirements.txt` direct deps (`bootstrap_venv.DIRECT_DEPS`) are recorded
  in every manifest. Keep Optuna in `requirements-tuning.txt` only;
  `ops/tune.py` imports it lazily and checks the pin.
- Shared CI math is in `scripts/stats.py`; shared launch helpers
  (`find_mesh_root`, `simulator_env`, `parse_seed_spec`) are in
  `scripts/sim_support.py`.
- `agents/q_learning.py` is exploratory and not on the training path.
- Importing `scripts.rl` registers `mesh_sim/MeshEnv-v0` with Gymnasium (idempotent).
