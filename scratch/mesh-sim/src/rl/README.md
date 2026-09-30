@page src_rl src/rl

@brief Wire contract for the stdin/stdout JSON bridge between the C++ simulator and the Python Gymnasium environment.


## RL bridge contract

At each decision, the simulator sends Python what the policy can see (`obs`),
which moves are allowed (`mask`), its reward, and raw measurements (`facts`).
The Gymnasium environment validates that message, selects the configured
observation and reward, and passes them to MaskablePPO. The chosen joint action
returns through the bridge; the simulator applies it until the next decision.
A decision window is the simulator ticks between those chances to change
direction. Each tick still updates movement, links, traffic, and reward.

C++ owns movement limits, action validity, and base reward accumulation. Python
converts and validates messages; it never re-derives masks or clamps.
For a code walkthrough, start with the tick loop in `sim.cc`, then
`src/config/rl-control.cc` for the selected nodes and timing, then
`src/rl/rl-bridge.cc` for the messages, actions, and tick reward. Follow one
`step` into `scripts/rl/env/mesh_env.py` to see what Gymnasium returns.

The Python side is documented in [`scripts/rl/README.md`](@ref scripts_rl),
[`scripts/rl/env/README.md`](@ref scripts_rl_env),
[`scripts/rl/policy/README.md`](@ref scripts_rl_policy), and
[`scripts/rl/tests/README.md`](@ref scripts_rl_tests). In short,
`scripts/rl/env/mesh_env.py` owns the Gymnasium API,
`protocol.py` validates actions and messages, and `episode.py` owns the
simulator process, episode directories, diagnostics, and manifest.
`selection.py` resolves the observation/reward/telemetry selection and its
precedence, and validates it. `observations.py` owns the named observation
presets and their schema identity. `rewards.py` owns the reward components and
the composer. `telemetry.py` owns `steps.jsonl` records and their replay.
For training, follow [`train.py`](@ref rl/train.py) (CLI and model
output) into [`agents/mask_ppo.py`](@ref agents/mask_ppo.py)
(MaskablePPO setup), then into the environment. Within `env/`, `config.py`
reads the seed and movement bounds.
For a worked configuration, formulas, normalization, and trace inspection, see
the [policy-input guide](@ref src_rl_policy_inputs).

## Modes

| Mode | Selected by | Action space | Stream |
| --- | --- | --- | --- |
| legacy | `[rl] controlled_node_id` (or neither selector) | `Discrete(7)`: `0:-X 1:+X 2:-Y 3:+Y 4:-Z 5:+Z 6:Stay`, absolute clamped target | one `step` line per tick, no `init` |
| centralized | presence of `[rl] controlled_nodes` | `MultiDiscrete([5]*M)` | one `init` line, then one `step` line per decision |

`controlled_node_id` and `controlled_nodes` are mutually exclusive. There is no
automatic migration: legacy action `4` is `-Z`, centralized action `4` is hold.
`action_set` and `dimensions` are not configuration keys.

## Centralized contract (`mesh_move_2d_v1`) {#src_rl_centralized_contract}

Here, **contract** means the agreed C++/Python message and action format—not a
radio link or a training objective. `mesh_move_2d_v1` identifies the current
two-dimensional, five-action-per-slot protocol. Python rejects an unknown
contract name; it is reported by the simulator, not chosen in `run.ini`.

| Item | Value |
| --- | --- |
| Slot order | resolved `controlled_nodes` order; `all` means every `nodes.json` entry in file order. Jammers are never controllable. |
| Slots | A slot is a zero-based place in the agent's fixed-length action, mask, and observation arrays—not another network node. An active slot maps to one controlled node; an unused slot is padding and must hold. `M = max_controlled_nodes` (`0` auto-sizes). |
| Action meanings | `0 west (-x)`, `1 east (+x)`, `2 south (-y)`, `3 north (+y)`, `4 hold`; z is never changed |
| Hold | action `4` commands zero velocity until the next decision. The node stops moving; it does not repeat the previous direction. A node can also stop at a wall while a direction remains commanded. |
| Mask order | flat `5*M` array of 0/1, `[slot0 W,E,S,N,H, slot1 …]`; padded slot is `[0,0,0,0,1]`; hold is always valid |
| Mask rule (clip-at-wall) | a direction is valid iff there is room > `1e-6` m to that bound, so a direction is valid exactly when realized displacement would be > 0 |
| Clamp | every tick, before the clock advances, each axis velocity is reduced so the node lands exactly on the bound instead of crossing it |
| Speed cap | `v_i = min(step_size_m / tick_s, MaxSpeedForType(node_type))`, fixed at construction and reported in `init.slot_speed_mps`. `step_size_m` is nominal movement **per simulator tick**, not per RL decision. |
| Cadence | decisions at ticks `0, k, 2k, …` with `k = decision_interval_ticks`; velocities persist between decisions; physics still advances every `tick_s` |
| Partial windows | `num_decisions = ceil(num_ticks / k)`; the last window may be shorter, and every `step` reports its `ticks_in_step` |
| Reward window | `mean`: one policy reward is the arithmetic mean of the per-tick rewards since the previous decision; the last window can be shorter. |
| Terminal message | emitted at `tick == num_ticks` with `done: true`; no action is read after it |

### Reward and wall behavior {#src_rl_reward_and_wall}

At tick 0, the simulator evaluates links and traffic and sends an initial
reward in the reset message; Gymnasium's `reset()` does **not** return that
reward as a learning step. After the agent chooses an action, every simulator
tick recomputes links, traffic, and the selected `[rl] reward_type`:
`throughput` sums delivered Mbps across flows; `all_links_los` is +1 only if
each controlled node has at least one other mesh node and every evaluated peer
path is line-of-sight, otherwise -1. At the next decision,
`reward_window = mean` sends their arithmetic mean as **one** RL step reward.
For three tick rewards `+1, -1, +1`,
the decision reward is `+1/3`, not `+1` and not three separate agent steps.
`ticks_in_step` gives the number averaged: usually `decision_interval_ticks`,
but only the remaining ticks in a short final window.

For `tick_s = 0.1`, `decision_interval_s = 0.5` (`k = 5`), and a 10-tick run:

| Simulator ticks | What Python receives |
| --- | --- |
| `0` | Initial observation and reward in `reset()`; this reward is not a training step. |
| `1–5` | First action stays in effect; `step()` at tick 5 returns `(r1 + r2 + r3 + r4 + r5) / 5` and `ticks_in_step = 5`. |
| `6–10` | Second action stays in effect; final `step()` returns `(r6 + r7 + r8 + r9 + r10) / 5` with `done = true`. |

The per-tick `r` is the selected network reward, not a reward per controlled
node or per slot. If the episode ended at tick 7 instead, the final mean would
use ticks 6–7 and report `ticks_in_step = 2`.

`wall_policy = clip` means that before each tick the simulator shortens an x/y
velocity that would cross a configured bound, landing exactly on the wall.
It does not bounce, wrap, or move beyond the bound; an outward direction is
masked at the next decision. A node can stop at a wall mid-window while other
ticks still contribute to its network reward. Hold and clipping have no
separate reward or penalty, and 2D actions do not change z.

The agent cannot interrupt a command halfway through a decision window. To
change direction more often, reduce `decision_interval_s` to an integer multiple
of `tick_s` (down to one tick). The ratio check allows a `1e-6`-tick tolerance
for floating-point rounding; it does not allow arbitrary fractions. A
wall-clipped or held node still participates in
link, traffic, and reward calculations every tick; there is no separate wall or
hold reward. `revalidated_slots` in the *next* `step` lists only action positions
whose invalid command C++ replaced with hold. It does not list deliberate hold
actions or every node that happens to be stationary.
With `all_links_los`, the reward still depends on the current links' LOS;
being at a boundary or choosing hold has no bonus or penalty by itself.

### Worked three-node example

`inputs/baselines/centralized-multi-smoke/run.ini` controls `node-b,node-c` and sets
`max_controlled_nodes = 3`, so action positions 0 and 1 belong to those nodes;
position 2 is padding. Node-a follows its own random walk. With `tick_s = 0.1`,
`decision_interval_s = 0.5`, and `step_size_m = 1`, each active node can move up
to 1 m per tick, or 5 m over a five-tick decision window (subject to its speed
cap and the arena wall).

| Time | Joint action just chosen | Result at next decision |
| --- | --- | --- |
| `0.0 s` | `[2,1,4]`: b south, c east, padding hold | At `0.5 s`, b moves `(100,0)` → `(100,-5)`; c moves `(97,50)` → `(100,50)` and clips at `x_max`. |
| `0.5 s` | `[4,0,4]`: b hold, c west, padding hold | At `1.0 s`, b stays `(100,-5)`; c reaches `(95,50)`. |

`[0,0,0]` would command **west**, not stop. Its padding action would be
revalidated to hold. `[4,4,4]` commands both active nodes to stop. Each decision
reward is the mean of the per-tick rewards since the prior decision. The
time-zero reward is shown in the reset message but is not a policy-step reward.

Observation layout per slot (`L = 4 + 2*(N-1)`, `obs_dim = M*L`):
`[active, x, y, z, sinr(i,j0), cap(i,j0), sinr(i,j1), cap(i,j1), …]` with peers
in node-index order skipping the slot's own node; a padded slot is all zeros.
The `-999` SINR sentinel is still passed through unchanged in this `obs`; the
Python observation presets treat `sinr_db < -900` or a non-finite value as invalid.

## Message fields

Centralized `init` is the first line, sent once per simulator process. Python
keeps the following values as an episode **signature** and compares them on
every later reset. A change is an error because the same Gymnasium environment
must not silently change its action/observation layout or behavior. This is an
ordinary dictionary, not a SHA hash or another configuration file.

| Signature field | Meaning |
| --- | --- |
| `control_mode` | `centralized`; identifies the multi-node protocol (set by Python, not sent in `init`). |
| `contract` | C++/Python protocol identifier; see [Centralized contract](@ref src_rl_centralized_contract). |
| `dimensions` | `2`: actions move horizontally; z can still appear in observations. |
| `action_meanings` | Ordered meanings of actions 0–4: west, east, south, north, hold. |
| `num_mesh_nodes` | Total mesh nodes `N` in `nodes.json`, including uncontrolled nodes but not jammers. |
| `max_controlled_nodes` | `M` fixed action/observation positions; includes padding, not just active controlled nodes. |
| `obs_dim` | Observation length, `M * (4 + 2*(N-1))` for this contract. |
| `mask_dim` | Flat action-mask length, `5*M`. |
| `nvec` | Python's `MultiDiscrete` sizes: one 5-action choice per position, `(5,)*M`. |
| `slot_node_ids` | Node ID assigned to each position, in order; `null` for padding. |
| `slot_speed_mps` | Movement speed cap for each position, in m/s; `null` for padding. |
| `tick_s` | Simulated seconds advanced by one simulator tick. |
| `decision_interval_s` | Simulated seconds between chances to choose a new joint action. |
| `decision_interval_ticks` | Number of simulator ticks per decision window, `k = decision_interval_s / tick_s`. |
| `num_ticks` | Total simulator ticks in the episode. |
| `num_decisions` | Number of decision windows, `ceil(num_ticks / k)`. |
| `reward_type` | C++ reward selected by `[rl] reward_type`. |
| `reward_window` | `mean`: one arithmetic mean over this decision's ticks; see [Reward and wall behavior](@ref src_rl_reward_and_wall). |
| `wall_policy` | `clip`: x/y movement lands at the bounds rather than crossing them; see [Reward and wall behavior](@ref src_rl_reward_and_wall). |

The `init` message also has `type: "init"` and `num_controlled`, the actual
number of active nodes (`num_controlled <= max_controlled_nodes`).
`contract`, `reward_window`, and `wall_policy` are simulator-reported protocol
facts, not additional `run.ini` settings.
The signature also includes `type`, `facts_schema`, `facts_columns`,
`node_ids`, `num_links`, `bounds`, `band`, `jammer_path_enabled`, and `warmup_s`
(see [Per-decision facts](@ref src_rl_per_decision_facts)). Padded entries
in `slot_node_ids` and `slot_speed_mps` are `null`.

Each centralized `step` is a message from C++ to Python:

| Field | Meaning |
| --- | --- |
| `type` | Always `step`. |
| `tick`, `time_s` | Current tick number and simulated time in seconds; the reset message is tick 0. |
| `decision` | Zero-based message/decision index; reset is 0, first action's result is 1. |
| `ticks_in_step` | Number of tick rewards averaged into this message's `reward`; reset has 1. |
| `obs` | Flat observation array in the slot layout above, length `obs_dim`. |
| `mask` | Flat 0/1 valid-action array, five entries per slot, length `mask_dim`. |
| `reward` | Mean reward across this message's ticks, as described above. |
| `done` | Whether the episode reached its final tick; no action follows a true value. |
| `revalidated_slots` | Zero-based positions from the previous action that C++ changed to hold; see [Invalid or missing actions](@ref src_rl_invalid_actions). |
| `facts` | Raw node, link, and window measurements described below. |

Python replies with `{"action":[…]}`: exactly `M` integers in `[0,4]`, with
`4` for every padded position.

Legacy `step` remains unchanged: `type`, `tick`, `time_s`,
`obs.{controlled_pos,link_sinrs,link_capacities}`, `reward`, `done`,
`action_type`; its reply is `{"action":<int>}`.

`time_s` is simulated time, never wall clock, so identical inputs give identical
stdout.

## Per-decision facts (`mesh_facts_v1`) {#src_rl_per_decision_facts}

Every centralized `step` carries a `facts` object of raw simulator values; C++
does no clipping, scaling, or feature selection. Legacy mode emits no facts and
its stream is unchanged. `init.facts_schema = "mesh_facts_v1"` names this
schema, and `init.facts_columns` names the column order of both tables:

| Table | Rows | Columns | Units |
| --- | --- | --- | --- |
| `facts.nodes` | `N`, in `nodes.json` order | `x, y, z, vx, vy, vz, slot` | m, m/s; `slot` is the controlled slot index or `-1` |
| `facts.links` | `N(N-1)/2`, pairs `i<j` with `i` outer | `sinr_db, capacity_mbps, is_los` | dB, Mbps, 0/1 |

Node rows are instantaneous at the decision tick, the same instant as `obs`:
position from the mobility model and the velocity in effect after the per-tick
bounds clamp. Link rows come straight from the symmetric link table, so a pair
appears once; the `-999` SINR sentinel is possible in principle and Python
treats it as invalid. `init` also carries `node_ids` (the `N` ids in file
order), `num_links`, the movement `bounds` (`x_min`…`z_max`), `band`,
`jammer_path_enabled` (`band = sub-6` with at least one enabled jammer, exported
as metadata only), and `warmup_s` (metadata: the RL reward does not skip warmup).

`facts.window` holds sums over exactly the ticks of this decision window, the
same window as `reward`:

| Field | Meaning |
| --- | --- |
| `ticks` | ticks in the window; equals `ticks_in_step` |
| `demand_mbps_sum`, `delivered_mbps_sum` | Σ over ticks of Σ over flows (Mbps·tick) |
| `flow_ticks_with_demand` | count of (tick, flow) pairs with `demand_mbps > 0` |
| `unroutable_flow_ticks` | those of the above that were not routable; zero-demand flows are excluded because the router marks them unroutable |
| `connected_pairs_sum` | Σ of the connected-pair count per tick |
| `los_pairs_sum` | Σ over ticks of the LOS pair count over `i<j` |
| `legacy_reward_sum` | Σ of the per-tick `reward_type` value |

Invariant: `reward == legacy_reward_sum / ticks` (`0.0` when `ticks == 0`), so
the emitted `reward` is unchanged by the facts export. The reset message covers
tick 0 only, so its window has `ticks = 1`; that reward is observable but is not
a policy reward. C++ never emits NaN or infinity in `facts`.

## Invalid or missing actions {#src_rl_invalid_actions}

| Case | Centralized | Legacy |
| --- | --- | --- |
| EOF (stdin closed) | all slots hold for the rest of the episode, one `Warning: RL action stream closed; all controlled nodes hold.`; every tick and message still runs | Stay, one `Warning: RL action stream closed; holding position (Stay).` |
| Structural error (bad JSON, not a list, wrong length, non-integer, out of range) | all slots hold, one `Warning: malformed RL joint action; all controlled nodes hold.` | Stay, one `Warning: malformed RL action JSON; holding position (Stay).` (a non-integer `action`, e.g. a list, is malformed rather than fatal) |
| Semantic error (masked-out direction, non-hold in a padded slot) | only that slot holds, one `Warning: RL joint action revalidated; invalid slot actions replaced by hold.`, and the slot index appears in the next message's `revalidated_slots` | n/a |

`revalidated_slots` means **zero-based action positions**, not node IDs or a
new set of controlled nodes. C++ replaces each invalid position with hold (`4`)
before movement; valid positions keep their commands. For example, with two
controlled nodes and a padded third position, `[2,1,0]` is executed as
`[2,1,4]`, and the *next* `step` reports `"revalidated_slots":[2]`. The list
is empty for a deliberate hold, wall clipping during movement, or a malformed
whole action (which makes all positions hold instead).

A centralized run never accepts a scalar action; a legacy run never accepts a list.

## Compatibility envelope {#src_rl_compatibility_envelope}

A policy is structurally compatible with a run only when `contract`,
`action_meanings`, `max_controlled_nodes` (→ `nvec`) and `num_mesh_nodes`
(→ `obs_dim`) all match. Padding stabilizes shapes across the number of *active*
slots within one N-node scenario; it does not stabilize shapes across scenarios
with different `N`, and matching shapes are not evidence that a policy transfers.

`scripts/rl/policy/compat.py` checks a saved model against the live run in this
order, each with its own error class, stopping at the first failure:

1. **Structural** — manifest `contract.contract`, `dimensions`,
   `action_meanings`, `max_controlled_nodes`, `num_mesh_nodes`, `obs_dim`,
   `mask_dim`, `facts_schema` equal the live `init`. Space equality alone cannot
   detect same-shape/different-meaning models.
2. **Observation schema** — the saved schema against the live one; structural
   fields are errors, scenario identity fields become warnings.
3. **Reward** — the saved `reward_schema` SHA-256 must equal the live one, and a
   C++-authored or `legacy`-containing reward must also match `reward_type` and
   `reward_window`.
4. **Scenario identity** — `slot_node_ids`, `node_ids`, the `run.ini`,
   `nodes.json`, `buildings.json`, and `jammers.json` SHA-256 digests, and the
   band. This is the only step an explicit override may relax.
5. **Transfer** — never inferred from 1–4.

## Random streams

Replacing a controlled node's mobility model with a `ConstantVelocityMobilityModel`
can renumber the automatic RNG streams of other random walkers. Repeated scripted
resets stay deterministic, but that alone does not make an RL-versus-baseline
comparison fair: confirm uncontrolled and jammer randomness is paired before any
such campaign.

## Saved scenario identity {#src_rl_saved_scenario_identity}

Training writes `scenario_identity` in `train_manifest.json` with five fields
(`read_scenario_identity` in `scripts/rl/env/config.py`): `run_config` is the
absolute path to the supplied `run.ini`; `run_ini_sha256`, `nodes_json_sha256`,
`buildings_json_sha256`, and `jammers_json_sha256` are fingerprints of the exact
file contents (`null` when the optional buildings or jammers file is not
configured). The nodes file comes from `[scenario] nodes_file`, resolved
relative to `run.ini` unless already absolute, and defaults to `nodes.json`.
These fingerprints do not change the simulation or the reward. The record does
**not** fingerprint waypoints, a CLI band override, or the simulator binary
(the band is compared separately in step 4 of the
[compatibility envelope](@ref src_rl_compatibility_envelope)). A matching identity is not
proof that a model transfers. The selected node order and protocol settings are
saved separately in the manifest's `contract` field.

For the checklist when changing the protocol or saved-file format, see
@ref contribution.

## Tests {#src_rl_tests}

Test maps (inputs and expected outputs):

- @ref src_rl_policy_input_tests -- centralized action contract, observations, rewards, telemetry.
- @ref src_rl_policy_lifecycle_tests -- validate, train, inspect, evaluate.
- @ref src_rl_policy_comparison_tests -- policy comparison, experiment matrix, evaluation pipeline.

The test files live in [`scripts/rl/tests/`](@ref scripts_rl_tests).
