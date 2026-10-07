# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# scripts/rl/agents

Thin SB3 wrappers used by `train.py` and `policy/bundle.py`.

- `mask_ppo.py` -- `MaskablePPOConfig` + `MaskablePpoTrainer`, which wraps the
  env in `ActionMasker`. Only `seed`, `n_steps`, `gamma`, and `ent_coef` reach
  the `MaskablePPO` constructor; `ops/tune.py` and experiment matrices can only
  vary those. Wiring a new hyperparameter means updating `train.py` CLI, the
  train manifest, `policy/compare.py::_TRAINING_SETTINGS`, and `ops/tune.py`'s
  `_SEARCHABLE` together.
- `MaskablePpoTrainer.load` forces `device="cpu"` so evaluation replay is
  deterministic; keep it that way.
- `callbacks.py` -- `BoundedCheckpointCallback` (keeps the newest `keep_last`
  `checkpoints/checkpoint_<N>_steps.zip`) and `MaskableEvalCallback` wiring.
  All cadences are SB3 timesteps, i.e. one RL decision each. `policy/bundle.py`
  depends on the checkpoint naming via `list_checkpoints`.
- `q_learning.py` -- exploratory tabular agent for the legacy single-node mode;
  not on the training, evaluation, or experiment path.

`agents/__init__.py` imports `sb3_contrib` eagerly, so importing this package
needs the full `requirements.txt`. Code that must work without it (e.g.
`policy/bundle.py::load_model`) imports `mask_ppo` lazily.
