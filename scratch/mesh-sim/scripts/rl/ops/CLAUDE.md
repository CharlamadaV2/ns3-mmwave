# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# scripts/rl/ops

Operations around an existing `experiment_plan.json`: in-job task runner,
benchmark/estimate, Optuna smoke, SLURM front end, and rsync fetch.
`README.md` here is the full spec (task unit, exit-code mapping, receipt
protocol, the eight reported states, resume and cancel rules, fetch layout);
read the relevant section before editing.

## Commands

From `scratch/mesh-sim/`:

```bash
.venv/bin/python -m scripts.rl.ops.run_task --output-root R (--task-index I | --compare [--allow-incomplete])
.venv/bin/python -m scripts.rl.ops.benchmark ...
.venv/bin/python -m scripts.rl.ops.tune --study inputs/experiments/<s>.json --output-root R --sim-binary <BIN> [--dry-run]
.venv/bin/python -m scripts.rl.ops.cluster {plan,submit,status,resume,cancel,compare} ...
.venv/bin/python -m scripts.rl.ops.fetch ...
.venv/bin/python -m pytest scripts/rl/tests/test_ops_*.py -q
```

`tune` needs `requirements-tuning.txt`; its tests skip without Optuna.

## Boundaries

- Never modify `train.py`, `evaluate.py`, `compare.py`, `mask_ppo.py`, or
  `experiment.py` from here; they are executed or imported. Only the public
  experiment names (`load_matrix`, `build_plan`, `load_plan`, `step_state`,
  `PLAN_NAME`) may be imported, and only by `tasks`, `tune`, `benchmark`.
- Import direction (no cycles): `tasks` is the base; `reconcile` imports only
  `tasks`; `slurm` and `receipts` import neither each other nor `reconcile`;
  `cluster` is the only module that combines them. `tune` and `fetch` import
  no scheduler module.
- `tasks.py` and `reconcile.py` are pure (no subprocess, receipts and scheduler
  snapshots passed as dicts). Subprocess calls live in `slurm.py`, `fetch.py`,
  `benchmark.py`, `run_task.py`, `tune.py`.
- No cluster constants in code: partition, account, time, memory, venv, etc.
  come from the cluster config (`cluster-config.example.json`).

## Safety invariants

- Nothing deletes or renames a step directory; blocked dirs report
  `move or delete <dir> to retry`.
- Any scheduler query failure degrades to `unknown`, and `unknown` is never
  resubmitted without a recorded human assertion (`resume --inactive-job`).
- Receipts are written as an intent *before* `sbatch` and finalized after, under
  `cluster/submit.lock`; cancellations are recorded only after a successful
  `scancel`, and only exact `<array_id>_<index>` element ids are cancelled.
- There is only ever one writer of `comparison.json`; the compare job uses
  `afterany` and correctness comes from `tasks.compare_prerequisites`.
- Tuning never touches held-out seeds; a trial is one configuration on one
  model-selection seed, not evidence of learning.
- Everything here is tested only against `tests/fake_slurm.py` and a fake
  `rsync`; live-cluster findings go in `TODO-RL-OPS-1`.
