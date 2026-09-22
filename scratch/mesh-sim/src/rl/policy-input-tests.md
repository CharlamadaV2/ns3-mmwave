# Centralized RL and policy-input tests

These checks cover the centralized action contract and the configurable
observations, rewards, and telemetry described in the
[policy-input guide](policy-inputs.md). The Python-only tests use
`fake_sim.py` and temporary directories; they do not validate radio physics.
`test_real_binary.py` runs the bundled `centralized-multi-smoke` scenario in
pytest's temporary directory and skips unless `MESH_SIM_BIN` points to a built
simulator. No reference snapshots are created or changed by these tests.

| Test file and checks | Input → expected output |
| --- | --- |
| `test_mesh_env.py`: `test_centralized_spaces_contract_and_padding`, `test_centralized_single_slot_spaces` | Fake `init` with two active/one padded position or one position → fixed action/observation shapes, zero padding, hold-only padded mask. |
| `test_mesh_env.py`: `test_centralized_joint_action_is_forwarded`, `test_centralized_rejects_a_scalar_action`, `test_centralized_masked_random_run_completes` | Joint action/masked samples → expected movement and completed manifest; scalar or wrong-length action → error. |
| `test_mesh_env.py`: `test_centralized_faults_are_rejected`, `test_reset_signature_drift_fails_before_replacing_spaces`, `test_reset_mode_drift_is_rejected` | Malformed or changed simulator contract → failed episode/error, without silently changing established Gymnasium spaces. |
| `test_mesh_env.py`: `test_long_stdout_is_drained_on_close`, `test_close_is_idempotent_and_reset_still_works`, `test_completed_episode_leaves_no_process_or_reader` | Long/finished fake run or repeated close → child and reader thread cleaned up, correct episode status. |
| `test_real_binary.py`: `test_live_spaces_and_metadata`, `test_scripted_positions_and_masks`, `test_repeated_resets_are_deterministic` | Real three-node fixture and scripted actions → contract fields, clipped positions/masks, identical repeated trajectories. |
| `test_real_binary.py`: `test_structural_malformed_actions_hold_everything`, `test_malformed_action_stops_a_moving_node`, `test_semantic_errors_only_affect_invalid_slots` | Bad joint actions → all hold for a malformed list, or only invalid positions hold and appear in `revalidated_slots`. |
| `test_real_binary.py`: `test_partial_windows_clipping_and_reward_mean` | Coarse versus one-tick decisions → matching shared positions and mean reward over each full/partial window. |
| `test_observations_rewards.py`: `test_local_links_v1_exact_slot_vector`, `test_local_links_v1_padded_slot_is_zero`, `test_local_links_v1_clips_extreme_links`, `test_raw_links_matches_cpp_layout` | Raw facts → exact preset values, bounded local features, zero padding, unchanged raw layout. |
| `test_observations_rewards.py`: `test_schema_hash_is_deterministic_and_strict_json`, `test_raw_links_schema_uses_null_bounds_but_infinite_box`, `test_check_schema_rejects_structural_change`, `test_check_schema_reports_identity_change_as_warning` | Saved/live schemas → stable fingerprint, correct Box bounds, structural rejection and identity warnings. |
| `test_observations_rewards.py`: `test_delivery_ratio_masks_zero_demand`, `test_other_components`, `test_composer_total_is_weighted_sum_over_valid_components`, `test_legacy_component_reproduces_cpp_reward`, `test_composer_rejects_bad_configuration`, `test_reward_schema_authorities` | Synthetic window sums/configuration → component formulas, validity, weighted total, C++ alias, schema authority, and errors for invalid weights/names. |
| `test_observations_rewards.py`: `test_resolve_selection_precedence`, `test_resolve_selection_rejects_invalid_keys`, `test_resolve_selection_rejects_p2_selection_in_legacy_mode` | CLI/INI/default choices → recorded precedence or a pre-launch validation error. |
| `test_observations_rewards.py`: `test_telemetry_records_replay_without_mismatch`, `test_should_save_sampling` | Synthetic steps → header and sampled records; replay reports zero observation/reward mismatches. |
| `test_mesh_env.py`: `test_custom_preset_and_reward_block`, `test_negative_legacy_reward_window`, `test_zero_demand_masks_delivery_ratio` | Fake decisions → local observation shape, inspectable reward parts, preserved negative reward, zero-demand validity flag. |
| `test_mesh_env.py`: `test_telemetry_file_only_when_selected`, `test_telemetry_default_cadence_replays`, `test_telemetry_stride_keeps_full_manifest_totals`, `test_default_selection_telemetry_replays_via_raw_links` | Telemetry off/on/strided → optional `steps.jsonl`, successful replay, complete manifest totals even for unsaved decisions. |
| `test_mesh_env.py`: `test_binary_without_facts_is_rejected`, `test_training_records_selection_and_schema_hashes` | Old fake protocol or tiny training run → clear missing-facts failure or matching saved selection/schema fingerprints; training check skips without `sb3_contrib`. |
| `test_real_binary.py`: `test_facts_rows_window_and_raw_links_rebuild`, `test_window_sums_match_the_run_summary` | Real simulator facts → correct row/window sizes, C++ reward mean, rebuilt raw observation, summary totals. |
| `test_real_binary.py`: `test_custom_selection_observations_rewards_and_replay`, `test_telemetry_is_reproducible_and_cadence_bounded`, `test_training_run_writes_matching_schema_hashes` | Real run with custom selection → finite policy inputs, saved reward/trace replay, repeatable/strided telemetry, matching manifest fingerprints; training check skips without `sb3_contrib`. |
