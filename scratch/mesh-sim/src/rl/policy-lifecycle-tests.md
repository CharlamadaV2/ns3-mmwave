# Policy lifecycle test map

The [four-command walkthrough](../../README.md#model-lifecycle) shows how to
validate, train, inspect, and evaluate. These tests check that path without
adding reference snapshots. The Python contract tests use a fake simulator
and temporary directories; they do not establish radio accuracy. Run them
from `scratch/mesh-sim/`:

```bash
.venv/bin/python -m pytest -q scripts/rl/tests/test_lifecycle_cli.py scripts/rl/tests/test_policy_lifecycle.py
```

The two real-binary checks need a built simulator:

```bash
MESH_SIM_BIN=<BIN> .venv/bin/python -m pytest -q \
  scripts/rl/tests/test_real_binary.py \
  -k 'building_bypass_fixture_geometry or lifecycle_train_inspect_evaluate_in_fresh_processes'
```

Without `MESH_SIM_BIN`, that module skips. Tests that train/load MaskablePPO skip when
`sb3_contrib` is unavailable. Each check below passes through assertions or
fails with pytest output; its files are confined to pytest's temporary paths.

| Test file and checks | Input → expected output |
| --- | --- |
| `test_lifecycle_cli.py`: `test_static_validation_passes`, `test_unknown_preset_is_rejected`, `test_missing_seed_is_rejected`, `test_non_executable_binary_is_rejected`, `test_launch_requires_output_dir` | Valid or broken config/arguments → structured validation status or a named pre-launch error. |
| `test_lifecycle_cli.py`: `test_launch_reports_the_live_contract` | Launch with fake simulator → `validate/` episode and live contract dimensions in the report. |
| `test_lifecycle_cli.py`: `test_missing_manifest_is_reported`, `test_missing_model_file_exits_two`, `test_inspect_a_training_run` | Missing or complete training directory → exit 1/2 as appropriate, or a readable report of saved models and digests. |
| `test_policy_lifecycle.py`: `test_matching_bundle_and_env_are_compatible`, `test_structural_field_change_is_refused`, `test_different_observation_preset_is_refused`, `test_different_reward_composition_is_refused` | Saved versus live contracts → compatible result only for matching structural, observation, and reward meanings. |
| `test_policy_lifecycle.py`: `test_legacy_component_requires_the_same_cpp_reward`, `test_python_reward_without_legacy_ignores_the_cpp_reward_type` | Changed C++ reward → refusal only when the selected policy reward depends on it. |
| `test_policy_lifecycle.py`: `test_scenario_identity_change_is_refused`, `test_scenario_identity_change_can_be_overridden`, `test_structural_mismatch_is_reported_before_scenario_identity` | Changed input fingerprints/node identity → refusal by default, recorded override when requested; structural errors still take precedence. |
| `test_policy_lifecycle.py`: `test_read_bundle_accepts_final_and_checkpoint`, `test_read_bundle_refuses_unusable_runs`, `test_read_bundle_refuses_a_missing_or_invalid_manifest`, `test_read_bundle_refuses_a_missing_file_or_digest_mismatch` | Manifest plus chosen ZIP → verified bundle or a specific status/version/file/digest error. |
| `test_policy_lifecycle.py`: `test_selection_from_manifest_round_trips_and_eval_forces_telemetry`, `test_selection_from_manifest_revalidates` | Saved settings → restored selection and full evaluation telemetry; malformed saved settings → error. |
| `test_policy_lifecycle.py`: `test_hold_policy_holds_every_slot`, `test_random_valid_policy_never_picks_a_masked_action`, `test_random_valid_policy_is_seeded_per_episode` | Live action masks and seeds → all-hold or reproducible mask-valid random actions. |
| `test_policy_lifecycle.py`: `test_baseline_evaluation_writes_a_completed_manifest`, `test_json_output_is_the_eval_manifest`, `test_evaluation_refuses_to_write_into_a_training_run` | Fake baseline episodes → completed `eval_manifest.json`/JSON stdout; training-directory target → refusal. |
| `test_policy_lifecycle.py`: `test_model_policy_requires_a_run_dir`, `test_invalid_seed_and_policy_lists_are_refused`, `test_a_masked_action_is_counted_and_reported_as_exit_code_two` | Invalid evaluation request or masked action → input error or recorded violation with exit 2. |
| `test_policy_lifecycle.py`: `test_model_evaluation_reloads_a_trained_policy`, `test_model_evaluation_refuses_a_different_node_count` | Tiny saved model → evaluated policy; incompatible live node count → structural refusal. |
| `test_real_binary.py`: `test_building_bypass_fixture_geometry` | Built binary and building fixture → scripted hold/movement produces expected LOS and positions. |
| `test_real_binary.py`: `test_lifecycle_train_inspect_evaluate_in_fresh_processes` | Built binary and fixture → train, inspect, and evaluate in separate processes with verified artifacts and completed episodes. |
