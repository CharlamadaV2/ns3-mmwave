@page scripts_rl_env scripts/rl/env

@brief Gymnasium adapter that runs the mesh-sim simulator as a subprocess and talks to it over the RL JSON protocol.

`MeshRlEnv` starts one simulator process per episode, validates every message
it sends back, and turns the reply into a Gymnasium observation, reward, and
`info` dict. It also writes the per-episode manifest and optional step
telemetry. The wire contract lives in [`src/rl/README.md`](@ref src_rl);
observation presets and reward components are specified in
[`src/rl/policy-inputs.md`](@ref src_rl_policy_inputs). This page does
not repeat them.

## Module Layout

| File | Role |
| --- | --- |
| [`mesh_env.py`](mesh_env.py) | `MeshRlEnv`, the Gymnasium environment; owns reset/step flow, spaces, and action masks. |
| [`episode.py`](episode.py) | `EpisodeSession`: simulator subprocess, stderr tail, cleanup, and `rl_episode.json`. |
| [`protocol.py`](protocol.py) | `CentralizedProtocol` and `LegacyProtocol`: message validation and joint-action checks; raises `ProtocolError`. |
| [`observations.py`](observations.py) | Observation presets (`raw_links_v1`, `local_links_v1`), observation schema, and schema hashing/compatibility check. |
| [`rewards.py`](rewards.py) | Reward components (`delivery_ratio`, `connectivity`, `throughput_mbps`, `legacy`), `RewardComposer`, and reward schema. |
| [`selection.py`](selection.py) | `RlSelection`: preset, reward, and telemetry choices resolved as CLI > `run.ini [rl]` > default. |
| [`telemetry.py`](telemetry.py) | Optional `steps.jsonl` writer and offline replay. |
| [`config.py`](config.py) | `run.ini` readers: seed, movement bounds, action profile, control mode, scenario file hashes. |
| [`__init__.py`](__init__.py) | Re-exports `MeshRlEnv`. |

## How One Episode Flows

### Reset

1. `MeshRlEnv.reset` stops any earlier episode (status `interrupted`).
2. `EpisodeSession.start` creates `episode-NNNN/` under the output root and
   launches the simulator with `--run-config`, `--rl-mode`, `--seed`,
   `--output-dir`, and `--band` when one was given.
3. The first message decides the mode: `init` means centralized, `step` means
   legacy. Anything else is a protocol error.
4. Centralized: `CentralizedProtocol` validates `init`, spaces are built, the
   observation and reward schemas are recorded in the manifest, and the first
   `step` message becomes the reset observation.

### Step

1. `step(action)` checks the action shape (`CentralizedProtocol.joint_action`
   for centralized) and sends `{"action": ...}` to the simulator.
2. The reply is read and validated by the protocol object (contract, window
   sums, tick monotonicity in centralized mode).
3. The observation is the simulator's own vector, or a preset rebuilt from
   `facts` when a non-default preset is selected.
4. The reward is the simulator's, or a `RewardComposer` weighted sum when
   `reward_components` is set. Components whose window is invalid are left out.
5. `EpisodeSession.record_step` updates the manifest and, if telemetry is on,
   appends a record to `steps.jsonl`.
6. When the reply has `done`, the session stops the simulator and marks the
   episode `completed`.

### Failure and close

Any protocol error, bad JSON, or early process exit marks the episode `failed`
and raises `RuntimeError` with the command, exit code, and stderr tail.
`close()` marks it `interrupted`. Every path stops the child process and the
stdout drain thread.

## Control Modes

| Mode | Selected by | Action space | Observation | Masks |
| --- | --- | --- | --- | --- |
| Centralized | `[rl] controlled_nodes` present | `MultiDiscrete([5] * max_controlled_nodes)` | `Box`, size from `init` (or the preset) | Simulator mask, returned unchanged. |
| Legacy | `[rl] controlled_nodes` absent | `Discrete(7)`, or `Box(2)` if the simulator reports `continuous` | `Box` of position plus link SINR/capacity pairs | Derived in Python from the `[rl]` bounds. |

- Non-default preset, reward, and telemetry selections are rejected in legacy
  mode, because legacy messages carry no `facts`.
- Spaces are unknown until the first `reset`. A later reset whose contract
  differs from the first is a protocol error and the original spaces stay.

## Output

Written under `<output_dir>/episode-NNNN/`, one directory per reset.

| File | Where it comes from |
| --- | --- |
| `rl_episode.json` | `EpisodeSession`; status `running`, `completed`, `interrupted`, or `failed`, plus totals and (centralized) contract, selection, and schema hashes. |
| `sim_stderr.log` | Simulator stderr; its tail is shown in error messages. |
| `steps.jsonl` | `telemetry.StepRecorder`, only when `telemetry = steps`; replay with `telemetry.replay_file`. |
| Simulator outputs | Everything the simulator itself writes into `--output-dir`. |

## Conventions

- Validate, never coerce: a malformed message raises `ProtocolError`.
- Saved presets are immutable. A change to features, dtype, or normalization
  needs a new `_vN` preset, because it changes the schema hash.
- `config.py` follows the C++ INI rules (inline comments stripped, `nodes_file`
  relative to `run.ini`).
- Docstrings are one line, per the project `CLAUDE.md`.

## Dependencies

- Python packages: `gymnasium` and `numpy`; install as described in the
  [main README](@ref python_environment).
- `scripts/sim_support.py` for the simulator environment and helpers.
- A built simulator binary passed as `sim_binary`; this package does not build it.
- Tests: [`../tests/README.md`](@ref scripts_rl_tests).
