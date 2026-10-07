@page src_rl_decision_records src/rl/decision-records

@brief Opt-in per-episode records of what a centralized policy saw, requested, and got back.

This file records what the policy saw and chose; it does not explain why. No influence or attribution output exists.
Each record pairs one centralized decision's pre-action observation identity and
joint mask with the requested action, its revalidation, and the outcome window that
followed. Recording never changes actions, observations, rewards, `rl_episode.json`
totals, or `steps.jsonl`. Legacy control mode is refused.

## Files

| File in `episode-NNNN/` | Writer rule |
| --- | --- |
| `policy_decisions_manifest.json` | All identity (episode, settings, schemas, contract, model, coverage, status). Written atomically at open with `status = writing` and again at close. |
| `policy_decisions.jsonl` | Append-only, one compact sorted-key JSON record per line, flushed per line: a `reset` record (decision 0), then one `decision` record per saved decision. |

When recording is on, `rl_episode.json` gains `"decision_records": {"manifest": ..., "file": ...}`;
when off, the key is absent and no file is written.

## Enable

The flags exist on `train.py` and `evaluate.py` (also with `--run-dir`); there is no `run.ini` key.

```bash
.venv/bin/python -m scripts.rl.evaluate --sim-binary <BIN> --run-dir <run> \
  --output-dir <out> --seeds 11 --policies model,hold --decision-records
.venv/bin/python -m scripts.rl.train --sim-binary <BIN> --run-config <run.ini> \
  --output-dir <out> --decision-records --decision-records-every 10 m-ppo
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--decision-records` | off | Opt in. Every other flag requires it. |
| `--decision-records-every N` | `1` | Save decision n when `n % N == 0` or it is terminal; reset is always saved. |
| `--decision-records-max-bytes B` | `67108864` | Per-episode cap on `policy_decisions.jsonl`; `0` = no cap. |
| `--decision-records-obs-vector` | `auto` | `auto` stores the vector only when `steps_ref` is null; `always` stores it on every record. |
| `--decision-records-preferences` | `auto` | `auto` captures preferences on evaluation `model` episodes; `off` never does. |

`settings.source` marks each key `cli` or `default`. `train.py` records training
episodes (`mode training`, `source train`) and callback evaluation episodes
(`mode evaluation`, `source train_eval`, `policy model`, no model identity or preferences).
The cap bounds the optional on-disk file only; it is not a simulator or Python
memory limit, since each record is flushed to disk as it is written.

## Join rules

- Record n's `input` is the state *before* action n: decision n−1's message
  (`input.source_decision == n − 1`). Never place `steps.jsonl` record n's mask beside action n.
- Join on integers: `input.tick == outcome.tick − outcome.ticks_in_step`.
  Compare `time_s` only with a tolerance. The outcome window is `(input.time_s, outcome.time_s]`.
- `input.steps_ref` names `steps.jsonl` record n−1 only when that record was saved;
  otherwise it is null and `input.obs_vector` carries the exact vector. A sampled row is never relabelled.
- Decision 0 has no action or outcome; the terminal observation is never an input.
- Node IDs are strings from `contract_identity.slot_node_ids`; padded slots are `null`.
  `targets` lists every active slot because the action is joint.

## Masks and hashes

`input.mask` is the full joint pre-action mask as integers 0/1. `mask_sha256` is SHA-256
of `json.dumps(mask, separators=(",", ":"))` over those integers; a boolean JSON form gives
a different digest. `obs_sha256` hashes the vector in the observation schema dtype, as in
`steps.jsonl`. `model_input_sha256` hashes the float32 cast the model consumes; the two
coincide only for float32 presets (`model_input.lossless_from_obs`).

## Requested, revalidated, applied

`action.requested` is the joint action Python sent. `revalidated_slots` are the
slot indexes the simulator replaced by hold (`hold_index = 4`). `applied` is
`requested` with 4 at those slots; `applied_status` is always `derived`. An empty
`revalidated_slots` proves `applied == requested` only for `MeshRlEnv` senders.

## Sampling, caps, coverage, status

Records are skipped only at whole-line boundaries. When the next line would exceed a
positive cap, writing stops (`capped`) but decisions are still counted. A positive cap
smaller than the reset line fails the sidecar before writing (`failed`, zero bytes and records, `decision_0_retained false`).

`coverage.records` is the line count; `saved_ranges` and `gaps` partition
`[1, decisions_expected]`; `terminal_decision` is the last decision whose outcome
was seen. With `record_every = 1`, `complete` has no gaps and `capped` has one trailing gap.

| `status` | Meaning |
| --- | --- |
| `writing` | Process still running or died; treat as incomplete. |
| `complete` | Episode reached `done`; every sampled record written. |
| `capped` | The cap stopped writing; records outside `saved_ranges` do not exist. |
| `interrupted` | Episode reset, closed, or failed before `done`; see `episode.status`, `error`. |
| `failed` | The writer itself failed; see `error`. |

## Failure isolation

On any recorder exception the sidecar becomes `failed` with `error`, the file is
closed, the manifest rewritten if possible, one `RuntimeWarning` emitted, and the
recorder dropped. Simulation, the policy, `steps.jsonl`, and `rl_episode.json` continue.

## Model identity

`model` is set only for `evaluate.py` episodes of the `model` policy: `model_sha256`
(the zip file as evaluated), `policy_weights_sha256` (the `policy.pth` zip entry; stable
across re-saves of identical weights), `model_selection`, `model_path_recorded` (relative
to the run directory when inside it), `num_timesteps`, `train_manifest_sha256`, and
`inference` versions. Training and callback episodes have `model = null`.

## Preferences and parity

On evaluation `model` episodes, a forward hook on `policy.action_net` copies the
float32 per-slot outputs computed inside the same `predict` call. Records store
them as `per_slot` (exact float32 values) plus `masked_probs` (softmax over
unmasked entries, masked entries exactly 0, 6 decimal places). The hook adds no
forward pass, RNG draw, or parameter access, so recording on or off yields the
same actions and identical `steps.jsonl` bytes (tests 16–17). Other policies record `preferences: null`.

## Consumer rules

- Locate: `eval_manifest.json` `policies.<name>.episodes[].episode_dir` (or a
  training `episode-NNNN/`) → `policy_decisions_manifest.json`. No file means
  "Decision records not enabled for this episode".
- Identity is episode-relative (`episode.dir_name`, `index`, `seed`) plus `decision`;
  apply the join rules above.
- Render unavailable states: `writing`, `failed`, `interrupted`, `capped` (show `gaps`),
  `record_every > 1` (gaps by design), `steps_ref == null` (use `input.obs_vector`),
  `preferences == null`, `model == null`.
- Influence unavailable: show "Influence unavailable"; no v1 field carries or implies it.

## Schema and pin

[`docs/schemas/decision_record.v1.schema.json`](../../docs/schemas/decision_record.v1.schema.json)
(JSON Schema 2020-12) validates the manifest and each JSONL line. Its SHA-256 is

```text
2855e03eacb33acdb654d7115ee77b23500428e5916f1942459350def5a12f79
```

A consumer vendors the file byte-for-byte and checks this digest in its own test;
`test_schema_v1_is_frozen` pins the same value. v1 never changes: any edit
is `decision_record.v2.schema.json` with `"version": 2`.

## Test map

All in `scripts/rl/tests/test_decision_records.py` unless noted; 16, 17, and 20 need `sb3_contrib`.

| Tests | Protects |
| --- | --- |
| 1 `test_settings_resolution_and_dependent_flags`, 22 `test_cli_dependent_flag_without_enable_exits_one` | Defaults, `cli`/`default` sources, rejection of bad values and dependent flags without opt-in. |
| 2 `test_disabled_by_default_writes_nothing` | Disabled runs write no sidecar, no `decision_records` key, identical `steps.jsonl`. |
| 3 `test_records_join_previous_step_at_every_one` | n−1 join, integer tick rule, coverage, file hash, `complete`. |
| 4 `test_sampled_steps_are_never_relabelled`, 5 `test_no_telemetry_stores_exact_vector`, 6 `test_obs_vector_always` | `steps_ref` only for saved rows; exact vector otherwise or on request. |
| 7 `test_float32_preset_hashes_coincide`, 8 `test_requested_revalidated_applied`, 9 `test_mask_hash_is_over_int_json` | Observation versus model-input hashes; requested/revalidated/applied; integer mask hash. |
| 10 `test_cap_stops_at_record_boundary`, 11 `test_record_every_two_partition`, 12 `test_zero_cap_means_no_cap`, 12a `test_cap_smaller_than_reset` | Cap boundary, sampling partition, unlimited cap, sub-reset cap failure. |
| 13 `test_writer_failure_is_isolated`, 14 `test_interrupted_episode_status` | Failure isolation and `interrupted` status. |
| 15 `test_string_ids_and_null_padding`, 20 `test_training_context_records`, 21 `test_legacy_mode_is_refused` | String IDs with `null` padding; training/callback labelling; legacy refusal. |
| 16 `test_model_evaluation_records_preferences_and_identity`, 17 `test_capture_on_off_parity` | Model identity, preference capture, on/off action and trace parity. |
| 18 `test_schema_well_formed_and_documents_validate`, 19 `test_schema_v1_is_frozen` | Produced documents validate; schema digest pinned. |
| `test_real_binary.py`: `test_decision_records_join_real_steps` | n−1 join and revalidation on the real bridge; skips without `MESH_SIM_BIN`. |

## TODO

Recording is CLI-only. A future `run.ini` control for on/off and size must decide
how it interacts with `run_ini_sha256` scenario identity before `source` gains a `run.ini` value.
