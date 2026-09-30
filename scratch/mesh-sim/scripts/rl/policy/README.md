@page scripts_rl_policy scripts/rl/policy
@brief Library behind `evaluate.py` and `compare.py`: verify a saved training run, run policies, compute paired statistics.

This package holds the logic. The two command-line programs, [`../evaluate.py`](@ref rl/evaluate.py)
and [`../compare.py`](@ref rl/compare.py), own argument parsing, printing, and process exit codes.
For the wider RL code map see [`../README.md`](@ref scripts_rl); for the four-command
walkthrough see [Model lifecycle](@ref scripts_rl_lifecycle).

## Module Layout

| File | Role |
| --- | --- |
| [`bundle.py`](bundle.py) | Reads `train_manifest.json` and the model file, verifies both, and rebuilds the saved observation/reward selection. |
| [`compat.py`](compat.py) | Ordered checks that a saved model fits the live environment; defines the error types. |
| [`evaluate.py`](evaluate.py) | Baseline and model policies, the per-episode loop, per-episode metrics, and `eval_manifest.json` writing. |
| [`compare_inputs.py`](compare_inputs.py) | Loads `eval_manifest.json` files and flattens them into per-episode rows. |
| [`compare.py`](compare.py) | Paired model-minus-baseline statistics, run grouping, and `exit_code`. |

## Lifecycle

Train, evaluate, and compare are three separate commands. This package serves the last two.

1. Training (`../train.py`) writes `train_manifest.json` and the model files.
2. Evaluation loads that run with `read_bundle`, checks it with `check_compatibility`,
   runs policies with `evaluate`, and writes `eval_manifest.json`.
3. Comparison reads one or more `eval_manifest.json` files and writes `episodes.csv` and
   `comparison.json`.

Test maps: [lifecycle](@ref src_rl_policy_lifecycle_tests) and
[comparison](@ref src_rl_policy_comparison_tests).

## Bundle Verification

`read_bundle` refuses a run unless all of these hold:

- `manifest_version` equals `MANIFEST_VERSION` in `bundle.py` (currently 4).
- `status` is `completed`.
- `control_mode` is `centralized`.
- The chosen model file exists and its SHA-256 matches the digest in the manifest.

`--model` picks `final`, `best`, or `checkpoints/<name>.zip`; a checkpoint must be listed
in the manifest. `best` exists only if training used `--eval-every-steps > 0`.
No flag skips a digest check. Every failure is a `BundleError`.

## Compatibility Checks

`check_compatibility` must run before `load_model`. The checks run in this order and stop
at the first failure:

1. Structural contract (`contract`, `dimensions`, `obs_dim`, `mask_dim`, and so on).
   Always fatal (`StructuralMismatchError`).
2. Observation schema (`check_schema`). Structural differences raise
   `SchemaMismatchError` and are always fatal; scenario-level differences it returns as
   warnings are folded into step 4.
3. Reward schema SHA-256, plus `reward_type` and `reward_window` when the reward is
   authored in the simulator. Always fatal (`RewardMismatchError`).
4. Scenario identity: node slots, the four input-file SHA-256s, band, and any
   observation-schema findings. This is the only overridable step
   (`--allow-different-scenario`); the report then says `scenario: overridden`.
   Otherwise it raises `ScenarioMismatchError`.

A passing report still carries the note that matching checks do not imply transfer.

## Evaluation

`evaluate.py` offers three policies:

| Policy | Behavior |
| --- | --- |
| `hold` | Every slot takes the hold action. |
| `random_valid` | Uniform choice among valid actions, reseeded per episode. |
| `model` | Deterministic MaskablePPO prediction under the live mask. |

Each policy gets its own environment and runs every seed. `eval_selection` forces
`telemetry=steps` so metrics come from the per-decision telemetry file.
Metrics (`delivery_ratio`, `connectivity`, `los_fraction`, `unroutable_fraction`,
`first_all_los_decision`) are summed over telemetry windows; warm-up is not excluded.
Each episode's return is checked against `rl_episode.json`, and a mismatch is an error.

`eval_manifest.json` is written before the first episode and rewritten after every
episode. A crash therefore leaves a manifest with status `failed` or `partial`, never
none. Seeds that were never attempted are recorded as `not_run`.

## Comparison

`build_comparison` pairs `model` with each baseline on the same evaluation seed and
computes `model - baseline` differences.

- **Primary metric**: `delivery_ratio` (`PRIMARY_METRIC`). Higher is better.
- **Excluded pairs**: a seed is dropped if either side is missing, not `completed`,
  or has a null metric.
- **Interval**: 95% t interval when at least 2 pairs exist; otherwise omitted.
- **Groups**: evaluations sharing a `--label` are grouped, and a t interval is taken across
  training runs. Runs whose settings or scenario digests differ raise `ComparisonError`.
  Runs with seed overlap or no usable pairs are excluded from the group.

`exit_code` returns:

| Code | Meaning |
| --- | --- |
| 0 | Comparison is complete and clean. |
| 1 | Comparison is incomplete, or inputs were refused (`ComparisonError`). |
| 2 | Some evaluation overlapped a training or model-selection seed, or a policy has non-zero `mask_violations` or `revalidated_slots`. |

## Output

| Step | Where |
| --- | --- |
| Evaluate | `<output-dir>/eval_manifest.json` (`EVAL_MANIFEST_VERSION` 2) plus per-episode simulator output. |
| Compare | `<output-dir>/episodes.csv` and `<output-dir>/comparison.json` (`COMPARISON_VERSION` 1). |

## Conventions

- `REQUIRED_MANIFEST_VERSION` in `compare_inputs.py` must track `EVAL_MANIFEST_VERSION`
  in `evaluate.py`; change both together.
- Comparisons group by scenario digests and training settings, not by path.
- Docstrings are one line.

## Dependencies

- `numpy`; MaskablePPO (via [`../agents/mask_ppo.py`](@ref agents/mask_ppo.py)) only when loading a model.
- `../cli_common.py`, `../env/`, and `scripts/stats.py` (`t_critical_95`) from this repository.
- Run the CLIs as `python -m scripts.rl.evaluate` / `python -m scripts.rl.compare` from
  `scratch/mesh-sim/`; evaluation needs a built simulator binary.
