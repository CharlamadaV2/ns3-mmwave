# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# scripts/rl/tests

Pytest suite for `scripts/rl`. Run from `scratch/mesh-sim/`; tests write only to
pytest temp dirs, never to `inputs/`.

```bash
.venv/bin/python -m pytest scripts/rl/tests -q
.venv/bin/python -m pytest scripts/rl/tests/test_mesh_env.py::<test_name>
MESH_SIM_BIN=/abs/path/to/bin .venv/bin/python -m pytest scripts/rl/tests/test_real_binary.py -q
```

## Fakes

- `fake_sim.py` -- stand-in simulator speaking the same flags and RL protocol
  (legacy unless `[rl] controlled_nodes` is set). `FAKE_SIM_MODE` selects fault
  modes; `FAKE_SIM_FAIL_SEEDS` kills chosen seeds mid-episode. It has no radio
  model, so it proves protocol handling, never movement or reward correctness.
  Protocol changes in `src/rl/rl-bridge.cc` must be mirrored here.
- `fake_slurm.py` -- `sbatch`/`squeue`/`sacct`/`scancel` shims sharing one JSON
  state file (`FAKE_SLURM_STATE`), installed via `fake_slurm.install(tmp_path)`.
- `conftest.py` -- shared `sim_binary` (shell shim that execs `fake_sim.py`) and
  `multi_run_config` (centralized 3-node scenario) fixtures. Older modules keep
  local copies; new tests should use the shared fixtures.

## Skips

- `test_real_binary.py` skips entirely without `MESH_SIM_BIN`; it uses
  `inputs/baselines/p1-multi-smoke/`. Run it against a fresh build after any
  protocol or `mesh_env` change.
- Tests needing `sb3_contrib`, Optuna (`requirements-tuning.txt`), or pandas
  skip when those are missing, so a green run may hide skipped coverage; use
  `-rs` to see skips.

## Test maps

Each test's input and expected output is documented, and new tests should be
added to the matching map:

- `README.md` here -- `test_mesh_env.py`, `test_real_binary.py`, the C++
  config/CLI RL tests, `test_ops_*.py`, and `test_bootstrap_venv.py`.
- `src/rl/policy-input-tests.md` -- `test_observations_rewards.py` and the
  policy-input cases in `test_mesh_env.py` / `test_real_binary.py`.
- `src/rl/policy-lifecycle-tests.md` -- `test_lifecycle_cli.py`,
  `test_policy_lifecycle.py`.
- `src/rl/policy-comparison-tests.md` -- `test_evaluation_pipeline.py`,
  `test_experiment_matrix.py`, `test_policy_comparison.py`.