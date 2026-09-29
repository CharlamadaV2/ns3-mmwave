# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# scripts/rl/env

The Gymnasium adapter over the simulator's stdin/stdout JSON protocol. The wire
contract is `src/rl/README.md`; observation presets and reward components are
specified in `src/rl/policy-inputs.md`. Read those before changing anything here.

## Files

- `mesh_env.py` -- `MeshRlEnv`. Spaces are unknown until the first `reset`
  (they come from the simulator's `init` / first `step`). A later reset whose
  contract signature differs is a protocol error, and the original spaces stay.
- `episode.py` -- `EpisodeSession`: one simulator subprocess per `reset`
  (`--rl-mode --seed --output-dir=<root>/episode-NNN`), a stderr tail, a stdout
  drain thread on close, and ownership of `rl_episode.json` (status
  `completed` / `interrupted` / `failed`). Every exit path must leave no child
  process or reader thread; tests assert this.
- `protocol.py` -- `CentralizedProtocol` / `LegacyProtocol` validate every
  message (contract `mesh_move_2d_v1`, facts `mesh_facts_v1`, window sums,
  tick monotonicity) and the outgoing joint action shape. Raise `ProtocolError`;
  never coerce.
- `observations.py` -- presets `raw_links_v1` (default, float64) and
  `local_links_v1` (float32, normalized to RL bounds); `observation_schema` +
  `schema_sha256`; `check_schema` raises on structural fields, warns on
  `node_ids` / `slot_node_ids`.
- `rewards.py` -- components `delivery_ratio`, `connectivity`,
  `throughput_mbps`, `legacy`; `RewardComposer` sums only components whose
  window is valid.
- `selection.py` -- `RlSelection`, resolved CLI > `run.ini [rl]` > default, with
  a per-key `source`. Non-default selections are rejected in legacy mode.
- `telemetry.py` -- optional `steps.jsonl` writer and replay (`TELEMETRY_VERSION`).
- `config.py` -- `run.ini` readers (seed, bounds, action profile, control mode,
  scenario identity SHA-256s). Must follow the C++ INI conventions, including
  inline comments and `nodes_file` resolved relative to `run.ini`.

## Invariants

- Centralized mode: masks, clamping, speed caps, and the base per-tick reward
  come from C++. `action_masks()` returns the simulator mask as-is.
- Legacy mode is the exception: `action_masks()` derives the `Discrete(7)` mask
  from `[rl]` bounds in Python, and `action_type = continuous` is also allowed.
- The default preset and an empty reward selection reproduce the simulator's own
  obs/reward exactly; custom presets/rewards are built from `facts` only.
- Changing a preset's features, dtype, or normalization changes the schema hash
  and breaks saved-model compatibility (`policy/compat.py`). Add a new `_vN`
  preset instead of editing an existing one.
- `mesh_env` / `episode` / `protocol` changes need `tests/test_mesh_env.py` and,
  against a fresh build, `tests/test_real_binary.py` (see `../tests/CLAUDE.md`).
