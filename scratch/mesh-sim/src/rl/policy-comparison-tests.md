@page src_rl_policy_comparison_tests src/rl/policy-comparison-tests

@brief Map of the policy comparison, experiment-matrix, and evaluation-pipeline tests.

Run these from `scratch/mesh-sim/`. The first three files use temporary
manifests, a stub executor, or `fake_sim.py`; they do not need an ns-3 build:

```bash
.venv/bin/python -m pytest -q scripts/rl/tests/test_policy_comparison.py \
  scripts/rl/tests/test_experiment_matrix.py \
  scripts/rl/tests/test_evaluation_pipeline.py
```

| Test file | Input | What it checks | Expected result |
| --- | --- | --- | --- |
| `test_policy_comparison.py` | Hand-built `eval_manifest.json` files with controlled per-seed metrics | Shared t statistics; paired model-minus-baseline differences; missing, failed, and null episodes; held-out seed and compatibility refusals; grouping across trained models; deterministic CSV/JSON and CLI flags | A valid comparison writes `episodes.csv` and `comparison.json` under the chosen output directory. Incomplete inputs are reported without turning missing values into zero; incompatible inputs write nothing. |
| `test_experiment_matrix.py` | Temporary matrices and the tracked `inputs/experiments/bypass-smoke-matrix.json` | Matrix validation, explicit row filtering, seed roles, generated CLI arguments, reproducible plan, manifest-based step status, failure/blocked handling, and resume | `experiment_plan.json` contains the expected train/evaluate/compare steps. Invalid matrices fail before execution; a second run executes only pending steps. |
| `test_evaluation_pipeline.py` | Tiny training jobs against `fake_sim.py`, then evaluation and comparison | A failed evaluation seed is retained and excluded from paired statistics; two trained models form an across-run group | Temporary train/evaluation manifests and comparison files agree on completed seeds, excluded seeds, and interval type. Requires `sb3_contrib`. |

For the built simulator, run the focused fresh-process smoke test after setting
`MESH_SIM_BIN` to an absolute path:

```bash
MESH_SIM_BIN=/absolute/path/to/ns3.42-sim-debug \
  .venv/bin/python -m pytest -q \
  scripts/rl/tests/test_real_binary.py::test_experiment_matrix_smoke_in_fresh_processes
```

That test uses one temporary matrix with two training seeds and two held-out
seeds. It checks completed evaluation manifests, real summary files, and the
paired and across-run comparison under `tmp_path/matrix-root/`. Without
`MESH_SIM_BIN`, the real-binary module skips instead of testing the simulator.
All test outputs stay in pytest's temporary directory; they are not reference
results or evidence of RL improvement.
