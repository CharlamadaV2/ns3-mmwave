@page scripts_rl_policy scripts/rl/policy
@brief Library behind `evaluate.py` and `compare.py`: verify a saved training run, run policies, compute paired statistics.

This package holds the logic. The two command-line programs, [`../evaluate.py`](@ref scripts/rl/evaluate.py)
and [`../compare.py`](@ref scripts/rl/compare.py), own argument parsing, printing, and process exit codes.
For the wider RL code map see [scripts/rl README](@ref scripts_rl); for the four-command
walkthrough see [Model lifecycle](@ref scripts_rl_model_lifecycle).

## Module Layout

| File | Role |
| --- | --- |
| [`bundle.py`](@ref scripts/rl/policy/bundle.py) | Reads `train_manifest.json` and the model file, verifies both, and rebuilds the saved observation/reward selection. |
| [`compat.py`](@ref scripts/rl/policy/compat.py) | Ordered checks that a saved model fits the live environment; defines the error types. |
| [`evaluate.py`](@ref scripts/rl/policy/evaluate.py) | Baseline and model policies, the per-episode loop, per-episode metrics, and `eval_manifest.json` writing. |
| [`compare_inputs.py`](@ref scripts/rl/policy/compare_inputs.py) | Loads `eval_manifest.json` files and flattens them into per-episode rows. |
| [`compare.py`](@ref scripts/rl/policy/compare.py) | Paired model-minus-baseline statistics, run grouping, and `exit_code`. |

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

`scripts/rl/evaluate.py` offers these policies (default: `model`, `hold`, `random_valid`):

| Policy | Behavior |
| --- | --- |
| `hold` | Every slot takes the hold action. |
| `random_valid` | Uniform choice among valid actions, reseeded per episode. |
| `model` | Deterministic MaskablePPO prediction under the live mask. |
| `geometric`, `optimization` | Placement baselines: the plan is prepared first by `scripts.baselines.adapter`, then every slot holds. See [baselines README](@ref scripts_baselines). |

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

### Two kinds of interval

They never share a label.

- **Across evaluation seeds, one saved model.** The model and a baseline are
  paired seed by seed inside one evaluation. The interval shows how the paired
  difference varies over held-out scenario seeds with the model fixed.
- **Across independently trained models.** Evaluations sharing a `--label`
  contribute one mean difference each, over the seeds common to all of them. The
  interval shows run-to-run variation. The training seed currently sets both the
  PPO initialization and the training scenario seed, so the two cannot be
  separated (TODO-RL-SEEDS-1).

A single evaluation seed or a single training run yields a value and no
interval; `interval_omitted` says why. The intervals are t intervals that assume
approximately normal paired differences, so on a bounded ratio with few seeds
they are approximate.

### Reading the output

- Every comparison reports `n_expected` (requested seeds) beside `n_used` (pairs
  both policies completed), and lists each dropped seed with a reason per side,
  such as `model:failed`, `hold:metric_null`, or `random_valid:not_run`.
- A `null` metric is never read as zero. `zero_variance` marks a zero-width
  interval so a deterministic scenario is not mistaken for certainty.
- Metrics after `delivery_ratio`: `connectivity`, `los_fraction`,
  `unroutable_fraction`. `return` is compared only inside one evaluation, where
  every policy shares one reward definition, and is flagged
  `comparable_across_reward_definitions: false`.
- All statistics come from the RL telemetry window, stated as `metric_source`.
  Nothing is read from a run's `summary.json`, because that file excludes warmup
  ticks while the RL window does not. Mixing them would be wrong whenever
  `warmup_s > 0` (see the RL-reward warmup entry in `TODO.md`).

`exit_code` returns:

| Code | Meaning |
| --- | --- |
| 0 | Comparison is complete and clean: every expected evaluation found, complete, held out, and free of health counters. |
| 1 | Comparison is incomplete (outputs are still written) or was refused (nothing is written; `ComparisonError`). |
| 2 | Outputs are complete, but an evaluation overlapped a training or model-selection seed (not held out), or a policy has non-zero `mask_violations` or `revalidated_slots`. |

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

- `numpy`; MaskablePPO (via [`../agents/mask_ppo.py`](@ref scripts/rl/agents/mask_ppo.py)) only when loading a model.
- `../cli_common.py`, `../env/`, and `scripts/stats.py` (`t_critical_95`) from this repository.
- Run the CLIs as `python -m scripts.rl.evaluate` / `python -m scripts.rl.compare` from
  `scratch/mesh-sim/`; evaluation needs a built simulator binary.
