# RL control test map

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
