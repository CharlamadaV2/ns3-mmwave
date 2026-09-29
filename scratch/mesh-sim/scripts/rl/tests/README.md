@page scripts_rl_tests scripts/rl/tests

## RL control test map

Run these from `scratch/mesh-sim/`. The tests use temporary output directories;
they do not write to scenario inputs. `fake_sim.py` supplies protocol messages
for Python tests and does not model radio propagation. Set `MESH_SIM_BIN` to a
freshly built simulator for the real-binary tests; without it, that module
skips. Install the Python requirements first as described in the [main
README](../../../README.md#python-environment).

```bash
.venv/bin/python -m pytest scripts/rl/tests/test_mesh_env.py -q
MESH_SIM_BIN=/absolute/path/to/mesh-sim-binary .venv/bin/python -m pytest scripts/rl/tests/test_real_binary.py -q
make -C tests/unit/config test
MESH_SIM_BIN=/absolute/path/to/mesh-sim-binary make -C tests integration
```

## Test helpers

| File | Role |
| --- | --- |
| `fake_sim.py` | Stand-in simulator with the real flags and RL protocol; `FAKE_SIM_MODE` selects fault modes, `FAKE_SIM_FAIL_SEEDS` kills chosen seeds. No radio model. |
| `fake_slurm.py` | `sbatch`/`squeue`/`sacct`/`scancel` shims over one JSON state file (`FAKE_SLURM_STATE`); `fake_slurm.install(tmp_path)` returns a handle to inspect submissions and inject failures. |
| `conftest.py` | Shared `sim_binary` and `multi_run_config` fixtures. |

## Other test maps

| Modules | Map |
| --- | --- |
| `test_observations_rewards.py` and policy-input cases in `test_mesh_env.py` / `test_real_binary.py` | [policy-input-tests.md](../../../src/rl/policy-input-tests.md) |
| `test_lifecycle_cli.py`, `test_policy_lifecycle.py` | [policy-lifecycle-tests.md](../../../src/rl/policy-lifecycle-tests.md) |
| `test_evaluation_pipeline.py`, `test_experiment_matrix.py`, `test_policy_comparison.py` | [policy-comparison-tests.md](../../../src/rl/policy-comparison-tests.md) |

## Python adapter: `test_mesh_env.py`

Unless noted, input is an in-process scenario with `fake_sim.py`; expected
output is the stated Gymnasium result or error plus a correctly finalized
`rl_episode.json`. Earlier single-node tests remain in this file.

| Test | Input → expected output |
| --- | --- |
| `test_centralized_spaces_contract_and_padding` | Two controlled nodes, three positions → `MultiDiscrete([5,5,5])`, 24 observations, padded position masked to hold. |
| `test_centralized_single_slot_spaces` | One controlled node → one five-action position and eight observations. |
| `test_centralized_joint_action_is_forwarded` | `[2,1,4]` → correct positions after five ticks, wall mask, and decision metadata. |
| `test_centralized_rejects_a_scalar_action` | Scalar or short action → `ValueError` before forwarding. |
| `test_centralized_masked_random_run_completes` | Valid masked random actions → two decisions and completed episode metadata. |
| `test_centralized_faults_are_rejected` | Six malformed init/step variants → protocol error, failed episode, cleaned-up process. |
| `test_reset_signature_drift_fails_before_replacing_spaces` | Change slot capacity between resets → error; original Gym spaces remain. |
| `test_reset_mode_drift_is_rejected` | Switch centralized to legacy between resets → error; spaces remain. |
| `test_long_stdout_is_drained_on_close` | Long simulator output, then close → process and reader thread exit; episode interrupted. |
| `test_close_is_idempotent_and_reset_still_works` | Close twice, then reset → new episode allocated without leaked reader. |
| `test_completed_episode_leaves_no_process_or_reader` | Two valid actions → completed episode and no child/reader left running. |
| `test_scenario_identity_uses_relative_nodes_and_inline_comments` | Relative nodes path with comment → absolute `run_config` and exact file hashes. |
| `test_scenario_identity_accepts_absolute_nodes_path` | Absolute nodes path → hash of that file. |
| `test_scenario_identity_reports_missing_nodes_file` | Missing nodes file → `FileNotFoundError`. |
| `test_centralized_tiny_training_run` | 16-step MaskablePPO run → saved model, training manifest, episode metadata; skipped without `sb3_contrib`. |
| `test_training_failure_closes_the_environment` | Fake simulator exits with error → failed manifest and no leaked process; skipped without `sb3_contrib`. |
| `test_episode_allocation_scans_existing_directories_once` | Existing episode directories → next two names allocated with one directory scan. |

## Real simulator: `test_real_binary.py`

Input is `inputs/baselines/p1-multi-smoke/` unless the test creates a small
temporary variant. Outputs are parsed simulator messages, metrics, and episode
manifests under pytest's temporary directory.

| Test | Input → expected output |
| --- | --- |
| `test_live_spaces_and_metadata` | Reset → real `init`, spaces, slot mapping, reset info, and padding match contract. |
| `test_scripted_positions_and_masks` | South/east then hold/west → exact positions, clipping, masks, and completed manifest. |
| `test_repeated_resets_are_deterministic` | Same scenario, seed, and actions twice → identical observations/rewards. |
| `test_structural_malformed_actions_hold_everything` | Wrong-length joint action → one warning and all nodes hold. |
| `test_malformed_action_stops_a_moving_node` | Valid move then malformed action → earlier movement does not persist. |
| `test_semantic_errors_only_affect_invalid_slots` | Masked direction and non-hold padding → only invalid positions hold; reported in `revalidated_slots`. |
| `test_close_mid_episode_is_interrupted` | Close after one decision → interrupted manifest, no child/reader leak. |
| `test_partial_windows_clipping_and_reward_mean` | Same scripted path at coarse/fine cadence → clipped trajectory, final short window, coarse reward equals mean of fine tick rewards (excluding reset). |

## C++ configuration and CLI

`tests/unit/config/config-validator-test.cc` uses in-memory configurations
and checks accepted values or precise errors; it writes no simulation output.

| Test | Input → expected output |
| --- | --- |
| `test_rl_inverted_z_bounds` | Inverted z bounds with RL enabled → error. |
| `test_rl_valid_z_bounds` | Valid z bounds → accepted. |
| `test_rl_disabled_is_legacy` | Disabled RL → legacy mode with no controlled selection. |
| `test_rl_selection_order` | Explicit node list → list order preserved. |
| `test_rl_selection_all` | `all` → every mesh node in file order. |
| `test_rl_all_with_ids_rejected` | `all` mixed with IDs → error. |
| `test_rl_jammer_id_rejected` | Jammer ID in selection → jammer-specific error. |
| `test_rl_unknown_id_rejected` | Unknown mesh ID → error. |
| `test_rl_duplicate_token_rejected` | Repeated selected ID → error. |
| `test_rl_both_selectors_rejected` | Legacy and centralized selectors together → error. |
| `test_rl_empty_selector_rejected` | Present but empty selector → error. |
| `test_rl_continuous_rejected` | Continuous action in centralized mode → error. |
| `test_rl_action_profile` | Unsupported 3D or unknown profile → error. |
| `test_rl_max_controlled_nodes` | Invalid/valid capacities → error or correctly padded position count. |
| `test_rl_decision_interval` | Default, valid, fractional, excessive, invalid intervals → correct tick count or error. |
| `test_rl_tick_counts` | Fractional duration/tick ratios → expected centralized and legacy tick counts. |
| `test_rl_legacy_selection` | Absent centralized selector → existing single-node selection behavior. |
| `test_rl_duplicate_node_ids` | Duplicate node IDs → centralized error; legacy behavior unchanged. |
| `test_rl_start_outside_bounds` | Controlled start outside bounds → error. |
| `test_rl_validator_reports_resolver_errors` | Invalid selection → validator reports it only with RL enabled. |
| `test_rl_safety_cases` | Invalid time, bounds, step, nodes, or waypoints → error, never an uncaught exception. |
| `test_controlled_start_position` | Fixed/waypoint/empty waypoint node → correct start or exception. |
| `test_apply_rl_control` | Resolved settings or invalid resolution → copied fields or exception. |

`tests/integration/cli-integration-test.sh` Test 9 runs the legacy smoke
scenario with closed stdin. It expects five `step` lines, no `init`, and a
`controlled_pos` observation on the first line; other CLI checks are described
in the [main verification section](../../../README.md#verify).

## Ops and tooling tests

These use stub executors, `fake_slurm.py`, or a fake `rsync`; no cluster or
simulator is needed. Behavior is specified in `scripts/rl/ops/README.md`.
Parametrized tests cover several inputs each; only the pattern is listed.

### `test_ops_tasks.py`

| Test | Input → expected output |
| --- | --- |
| `test_tracked_matrix_maps_to_eight_paired_tasks` | Tracked experiment matrix → eight paired train/evaluate tasks. |
| `test_malformed_plans_are_refused` | Malformed plan variants → error. |
| `test_plan_root_requires_a_plan_argument` | Missing plan argument → error. |
| `test_tolerated_table`, `test_tolerated_refuses_an_unknown_kind` | Step kind and exit code → tolerated or not; unknown kind → error. |
| `test_task_state_reports_both_steps_and_the_train_manifest_status` | Manifests on disk → per-step state and train status. |
| `test_failed_training_skips_the_evaluation` | Training exits 1 → evaluation not run. |
| `test_tolerated_evaluation_exit_is_recorded_but_not_a_failure` | Tolerated evaluation exit code → recorded, run succeeds. |
| `test_a_finished_training_is_skipped` | Completed train manifest → training not re-run. |
| `test_a_dirty_directory_is_refused_and_left_untouched` | Step directory with stray content → refusal, contents unchanged. |
| `test_an_out_of_range_task_index_is_refused` | Index beyond the task table → error. |
| `test_compare_*` (five tests) | Unfinished evaluations → refused unless incomplete is allowed; tolerated compare exit is not a failure; exactly one selector required. |
| `test_comparison_outcome_reads_the_comparison_status` | `comparison.json` status → outcome string. |

### `test_ops_benchmark.py`

| Test | Input → expected output |
| --- | --- |
| `test_ps_text_maps_to_one_process_tree`, `test_ps_parsing_skips_unusable_lines` | `ps` output text → one process tree; bad lines skipped. |
| `test_a_real_process_tree_is_measured` | Real child process → memory and process-count peaks recorded. |
| `test_a_ps_failure_nulls_memory_but_keeps_timing` | `ps` unavailable → memory fields `null`, wall time kept. |
| `test_effective_timesteps_round_up_to_whole_rollouts` | Requested steps and `n_steps` → next whole rollout, or `None` if invalid. |
| `test_seconds_format_as_hms` | Seconds → `HH:MM:SS`. |
| `test_a_measured_task_records_both_steps` | Fake launcher → record with train and evaluate steps. |
| `test_a_failed_training_stops_the_benchmark` | Training fails → benchmark stops. |
| `test_an_existing_record_is_refused`, `test_a_dirty_step_directory_is_refused` | Existing record or dirty step directory → refusal. |
| `test_estimate_arithmetic_scales_the_measured_seconds` | Benchmark record, matrix, safety factor → per-task and serial totals. |
| `test_the_estimate_takes_the_largest_step_peak` | Train/evaluate memory peaks → the larger one, or `None`. |
| `test_a_changed_selection_still_estimates_but_warns` | Differing selection → estimate with warning. |
| `test_a_mismatched_row_gets_no_estimate` | Row not matching the benchmark → no estimate, reason given. |
| `test_an_unmeasurable_benchmark_is_refused` | Benchmark without measurements → refused. |
| `test_estimate_writes_a_file_and_records_the_benchmark_hash` | Estimate run → estimate file with benchmark hash. |
| `test_a_missing_safety_factor_is_an_argparse_error`, `test_a_non_positive_safety_factor_is_refused` | Missing or non-positive safety factor → error. |

### `test_ops_cluster.py`

| Test | Input → expected output |
| --- | --- |
| `test_the_example_config_is_refused_unedited`, `test_cluster_config_refusals`, `test_a_venv_without_bin_python_is_refused` | Unedited example, bad values, or bad venv → refused. |
| `test_a_valid_config_keeps_every_value` | Valid config → all values preserved. |
| `test_array_spec_compression` | Task indices and concurrency → compact `--array` spec. |
| `test_sbatch_argv_*`, `test_the_job_script_quotes_paths_and_exports_threads` | Config → `sbatch` argv (null values omitted, placement and dependency carried) and a quoted job script. |
| `test_parse_squeue_*`, `test_parse_sacct_*`, `test_parse_job_id` | Scheduler text → parsed rows and job IDs. |
| `test_a_missing_scheduler_binary_is_reported_not_raised` | No scheduler on PATH → reported, no exception. |
| `test_reconcile_state_table` and the pending, active, and compare-row tests | Filesystem, receipts, and queue snapshot → task state; active elements never resume-eligible; compare row follows `comparison.json`. |
| `test_plan_*`, `test_the_submit_resume_sequence`, `test_submit_refuses_*`, lock tests | `plan` writes the task table only; `submit`/`resume` submit the right elements; blocked directories, non-unsubmitted tasks, and a held lock → refused. |
| Intent and assertion tests (`test_sbatch_dying_*`, `test_an_unmatched_intent_*`, `test_abandoning_*`, `test_one_queue_match_*`, `test_a_known_job_*`, `test_an_assertion_*`, `test_no_second_job_*`) | sbatch failure or ambiguity → recoverable intent; no duplicate job for a covered task; human assertions cannot override active or populated tasks. |
| `test_cancel_*`, `test_a_failed_scancel_*` | Selectors → exact element IDs cancelled, only from receipts; failed `scancel` not recorded. |
| `test_dry_runs_change_nothing`, `test_a_submit_dry_run_previews_the_argv_and_script` | `--dry-run` → preview, no writes. |
| `test_a_plan_from_another_root_is_refused`, `test_a_non_executable_binary_is_refused` | Foreign plan root or non-executable binary → refused. |
| `test_status_*`, `test_ops_never_writes_the_plan_or_a_step_directory` | `status` works without a scheduler, lists every task and the comparison, never writes. |
| `test_manual_compare_*`, `test_evidence_that_a_compare_job_ended_*`, `test_a_compare_job_without_*` | Manual compare reports each outcome and is refused while tasks or compare jobs are active or unresolved. |
| `test_a_new_compare_job_cancels_and_verifies_the_earlier_one`, `test_a_failed_cancellation_blocks_a_second_compare_job` | Resubmitted compare → earlier job cancelled first; failed cancel blocks it. |

### `test_ops_fetch.py`

| Test | Input → expected output |
| --- | --- |
| `test_each_category_adds_its_include_rules`, `test_the_always_included_files_need_no_selection`, `test_two_categories_are_combined_in_order` | Categories → `rsync` include rules. |
| `test_an_unknown_category_is_refused` | Unknown category → error. |
| `test_ignore_existing_is_always_passed` | Any run → `--ignore-existing` in the argv. |
| `test_a_non_empty_destination_is_refused_without_update` | Populated destination → refused unless update. |
| `test_dangerous_destinations_are_refused`, `test_a_destination_symlink_cannot_bypass_the_checks`, `test_a_local_source_inside_the_destination_is_refused` | Root, home, mesh root, symlinked, or overlapping destination → refused. |
| `test_a_missing_rsync_is_reported_not_crashed` | No `rsync` → reported error. |
| `test_task_states_come_from_the_rebased_remote_paths`, `test_unfetched_manifests_are_never_reported_as_missing_tasks` | Fetched manifests → task states and incompleteness. |
| `test_the_comparison_state_is_read_when_selected` | Fetched `comparison.json` → comparison state. |
| `test_the_manifest_records_remote_argv_and_file_digests`, `test_update_keeps_the_earlier_manifest` | Fetch → provenance manifest with argv and digests; update keeps the earlier one. |
| `test_a_failed_rsync_writes_no_manifest`, `test_dry_run_prints_the_argv_and_writes_nothing` | Failed rsync or dry run → no manifest. |

### `test_ops_tune.py`

| Test | Input → expected output |
| --- | --- |
| `test_bad_specs_are_refused_before_anything_runs`, `test_refusal_messages_name_the_offending_setting` | Invalid study specs (unknown key, fixed budget, bad ranges, seeds, cadence) → refused with a message naming the setting. |
| `test_an_existing_study_directory_is_refused` | Existing study directory → refused. |
| `test_a_missing_binary_is_refused_unless_the_run_is_dry` | Missing binary → refused; dry run allowed. |
| `test_trial_arguments_carry_the_sampled_values_and_the_right_seeds` | Sampled values → trial arguments with correct seeds. |
| `test_a_sampled_value_equal_to_a_held_out_seed_is_not_refused`, `test_a_held_out_seed_in_a_seed_flag_is_refused` | Held-out seed as a sampled value → allowed; in a seed flag → refused. |
| `test_no_plan_file_is_written_for_a_trial` | Trial → no plan file. |
| `test_a_completed_study_records_objectives_and_the_best_training_block` | Stub trainer → study record with objectives and best block. |
| `test_a_failed_trial_is_recorded_and_the_study_continues` | Trial failure → recorded; later trials still run. |
| `test_a_dry_run_prints_commands_without_creating_anything` | `--dry-run` → commands printed, nothing created. |
| `test_the_install_hint_names_the_pin_file`, `test_the_pin_file_holds_one_exact_optuna_pin`, `test_a_version_other_than_the_pin_is_refused` | Optuna missing or wrong version → refusal naming the pin file. |
| `test_the_seeded_sampler_matches_its_dry_run_preview`, `test_two_runs_with_the_same_sampler_seed_agree` | Same sampler seed → same trials as the preview and across runs. |

### `test_bootstrap_venv.py`

| Test | Input → expected output |
| --- | --- |
| `test_direct_pins_match_probe_dependencies` | `requirements.txt` → pinned names equal `DIRECT_DEPS`. |
| `test_version_probe_rejects_mismatch` | Wrong expected version → probe exits 1 and reports `expected 0.0.invalid`. |
