@page src_rl_policy_inputs src/rl/policy-inputs

@brief Observation presets, reward components, and telemetry for centralized RL policies.

This page covers centralized RL. The simulator computes positions, links,
traffic, and per-tick reward in C++; every decision message includes those raw
measurements as `facts`. Python's `selection.py` chooses what the policy sees,
what reward Gymnasium returns, and whether to save decision records. The
simulator's physics and action masks do not change when these options change.
For actions and timing, see the [bridge contract](@ref src_rl); for the Python modules, see the [environment README](@ref scripts_rl_env).

## Try one selection

Add these lines to the `[rl]` section of a centralized scenario's `run.ini`
(one with `controlled_nodes`):

```ini
observation_preset = local_links_v1
reward_components = delivery_ratio, connectivity
reward_weights = 1.0, 0.5
telemetry = steps
telemetry_every = 2
```

Then train as shown in [Selection keys and flags](@ref src_rl_policy_inputs_selection).
Five matching CLI flags can override these keys independently; resolution
is CLI > `run.ini` > default. The resolved values and their sources go into
`train_manifest.json` and each `episode-NNNN/rl_episode.json`. Omitting
`reward_components` returns the C++ reward; omitting `telemetry` writes no
`steps.jsonl`. These choices require centralized mode because legacy mode does
not export `facts`.

Follow one policy step:

```text
sim.cc tick loop -> rl-bridge.cc step {obs, mask, reward, facts}
    -> mesh_env.py builds selected observation and reward
    -> MaskablePPO chooses one masked action per controlled position
    -> rl-bridge.cc applies those actions until the next decision
    -> episode.py optionally saves the decision in steps.jsonl
```

`observations.py` defines the policy input layouts; `rewards.py` defines reward
components and combines them; `telemetry.py` writes and replays optional
decision records. `mesh_env.py` connects those modules to Gymnasium.

## Selection keys and flags {#src_rl_policy_inputs_selection}

These `[rl]` keys are read by the Python env and `train.py`; the simulator
ignores them. They apply to centralized mode only.

| Key | CLI flag | Type | Default | Meaning |
|---|---|---|---|---|
| `observation_preset` | `--observation-preset` | preset name | `raw_links_v1` | Which observation the policy sees. |
| `reward_components` | `--reward-components` | comma-separated names | absent (the C++ reward) | Components Python composes into the returned reward. |
| `reward_weights` | `--reward-weights` | comma-separated floats | `1.0` per component | One weight per component, in the same order. |
| `telemetry` | `--telemetry` | `none` or `steps` | `none` | `steps` writes `<episode-dir>/steps.jsonl`. |
| `telemetry_every` | `--telemetry-every` | positive int | `1` | Save every kth policy decision (requires `telemetry = steps`). |

Run from `scratch/mesh-sim/`. The flags go before the `m-ppo` subcommand:

```bash
.venv/bin/python -m scripts.rl.train \
  --sim-binary <BIN> \
  --run-config inputs/baselines/centralized-multi-smoke/run.ini \
  --output-dir outputs/rl-multi-custom \
  --observation-preset local_links_v1 \
  --reward-components delivery_ratio,connectivity \
  --telemetry steps --telemetry-every 2 \
  m-ppo --total-timesteps 16 --n-steps 16 --seed 1
```

Each key resolves independently (CLI > `run.ini` > default), and both manifests
record the resolved value and its source. An unknown preset or component, a
weight count that does not match the components, or any non-default value of
these keys in legacy mode fails before the simulator starts. Centralized runs
need a simulator binary that exports per-decision facts: an `init` without
`facts_schema` is rejected before training starts.

### Observation presets and reward components

- `raw_links_v1` and `local_links_v1` are described in
  [What the observation contains](@ref src_rl_policy_inputs_observation).
- `delivery_ratio`, `connectivity`, `throughput_mbps`, and `legacy` are defined
  in [One action, several ticks, one reward](@ref src_rl_policy_inputs_reward).
  `throughput_mbps` is unnormalized: its scale grows with node count and demand.
- Naming any component makes Python the reward authority: `step()` returns the
  weighted total, `info["reward"]` holds the per-component values, validity, and
  weights, and the C++ value stays available as `info["reward"]["legacy"]`.

### Schema fingerprints

Both manifests carry SHA-256 fingerprints of the observation and reward schema
descriptions. They let a loader detect changed declared layouts, normalization,
components, or weights. They do not detect every code or physics change and are
not evidence that a policy transfers between scenarios. A compiled-binary hash,
when recorded during verification, answers a different question: which
executable was tested. `manifest_version` labels the saved JSON format, not the
model or simulator version.

## What the observation contains {#src_rl_policy_inputs_observation}

An observation is the numeric vector passed to the policy. Both presets make
one block per `max_controlled_nodes` position; a padded position is all zeros
and does not represent a node. `N` below is the total number of mesh nodes,
including uncontrolled nodes but not jammers. The vector uses the
instantaneous `facts.nodes` and `facts.links` at the decision boundary; the
reward instead summarizes the preceding ticks in `facts.window`.

| Preset | Per-position values | Type and size |
| --- | --- | --- |
| `raw_links_v1` (default) | `active, x, y, z`, then `SINR dB, capacity Mbps` for each other mesh node, in node-file order | `float64`; `M × [4 + 2(N-1)]` |
| `local_links_v1` | `active, x_n, y_n, z_n`, then `present, sinr_valid, sinr_n, cap_n` for each other mesh node | `float32`; `M × [4 + 4(N-1)]` |

For `local_links_v1`, each coordinate is mapped from its `[rl]` min/max bound
to `[-1, 1]` and clamped there. This makes coordinate scales comparable across
axes. `present=1` means the peer occupies that position in the node list; it
does **not** say the radio link works. `sinr_valid=0` for a non-finite SINR or
capacity, the simulator's `-999` SINR sentinel, or any SINR at or below
`-900` dB. An invalid link's `sinr_n` and `cap_n` are zero.

For a valid link, `sinr_n = clamp((SINR_dB + 20) / 60, 0, 1)`: values below
`-20` dB map to 0 and above `40` dB map to 1. Capacity uses
`cap_n = clamp(log10(1 + capacity_Mbps) / 4, 0, 1)`. The bounded inputs keep
extreme measurements from dominating this preset's policy vector; the
`-20`/`40` limits are feature-design choices, **not** physical SINR limits.
Neither clipping nor scaling changes C++ link calculations, the raw `facts`,
or `raw_links_v1`. Rewards are not normalized by this observation preset.

For example, with three mesh nodes and `max_controlled_nodes = 3`, the raw
vector has 24 values and the local vector has 36. If only two nodes are
controlled, the final 8 or 12 values, respectively, are padding. A node at
`x_max` has `x_n = 1` in the local preset even if its raw x coordinate is 100 m.

`observations.py` also declares a *schema*: ordered feature names, vector
length, numeric type, normalization rules, and a `low`/`high` bound for each
value in Gymnasium's `Box`. `low` and `high` describe allowed feature values;
they are not extra normalization operations. Raw features are unbounded
(`-∞`/`+∞` in the Box, recorded as JSON `null`); local position features are
`[-1,1]` and its flags/link features are `[0,1]`. The schema fingerprint in
the manifest helps reject a saved policy whose input layout changed. For the
local preset, changing movement bounds also changes the interpretation of
coordinates and fails the structural compatibility check.

## One action, several ticks, one reward {#src_rl_policy_inputs_reward}

If `tick_s = 0.1` and `decision_interval_s = 0.5`, the agent chooses an action
at tick 0 and the simulator runs ticks 1–5 with it. At tick 5 the agent sees
the resulting state and one reward computed from those five ticks; ticks 6–10
form the next window. The reset observation at tick 0 contains a one-tick
measurement but contributes **no** policy-step reward. The last window can be
shorter, and `ticks_in_step` reports its actual length.

Every tick, C++ adds the current flow demands and deliveries, connected and
line-of-sight link counts, and its own per-tick reward to `facts.window`.
`demand_mbps_sum` and `delivered_mbps_sum` are sums of Mbps readings over
flows and ticks (Mbps·tick), not transferred bytes. `flow_ticks_with_demand`
counts flow/tick pairs with positive demand; `unroutable_flow_ticks` counts
those that could not be routed. `connected_pairs_sum` and `los_pairs_sum` sum
counts of node pairs over ticks. `legacy_reward_sum` sums C++'s per-tick
`reward_type` values. The C++ message reward is always
`legacy_reward_sum / window.ticks`.

When `reward_components` is set, Python computes these components from the
*same* window, then returns the weighted sum to Gymnasium:

| Component | Value for one window |
| --- | --- |
| `delivery_ratio` | `delivered_mbps_sum / demand_mbps_sum`; marked invalid and contributes zero if demand sum ≤ `1e-9` |
| `connectivity` | `connected_pairs_sum / (window.ticks × num_links)` |
| `throughput_mbps` | `delivered_mbps_sum / window.ticks`; not scaled to `[0,1]` |
| `legacy` | `legacy_reward_sum / window.ticks`; C++ reward, **not** legacy control mode |

For five ticks, suppose demand sums to 150, delivery to 120, and 12 of the
possible `5 × 3 = 15` pair/tick observations are connected. Then delivery
ratio is `120/150 = 0.8`, connectivity is `12/15 = 0.8`, and the example
weights return `1.0×0.8 + 0.5×0.8 = 1.2`. There is no separate reward per
controlled node. `info["reward"]` shows the component values, validity flags,
weights, total, and the original C++ reward for diagnosis. A zero-demand
window is not treated as successful delivery: its ratio is invalid and has
zero contribution.

The reward schema's `authority` says which value Gymnasium returns:
`cpp` when no components were selected, `python` for the composed sum. C++
still calculates and exports its base reward in both cases. `reward_window =
mean` names the C++ averaging rule; Python components use the window sums
shown above. Neither setting changes how many ticks elapse per decision.

## Save and inspect decisions

With `telemetry = steps`, each episode gains `steps.jsonl`. Its first line is
a header with the run contract, chosen settings, and observation/reward
schemas. Later lines contain the decision number, tick, action that produced
the just-finished window, current mask,
revalidated positions, raw facts/window, returned reward, C++ reward, and an
observation fingerprint. `telemetry_every = 2` saves reset (decision 0), every
second policy decision, and the terminal decision even if it falls between
samples. Unsaved decisions still affect training, `rl_episode.json` totals,
and the final outcome; this option only reduces the detailed trace.
`rl_episode.json` is updated every decision regardless of `telemetry_every`.

Records are buffered and flushed every 32 records and on stop, so a hard kill
can lose up to 32 records. On the `centralized-multi-smoke` fixture a record is
about 0.9 KB and the header about 3.5 KB (measured by an earlier pass; not
re-measured here).

From the mesh-sim directory, inspect or replay one episode with:

```bash
head -n 2 outputs/rl-multi-custom/episode-0000/steps.jsonl
.venv/bin/python -c 'from scripts.rl.env.telemetry import replay_file; import sys; print(replay_file(sys.argv[1]))' outputs/rl-multi-custom/episode-0000/steps.jsonl
```

Replay rebuilds saved observations and Python-composed rewards from each
record's raw facts and checks for mismatches. It does not rerun radio physics,
verify unsaved decisions, or recheck a C++-authored reward. See the
[test map](@ref src_rl_policy_input_tests) for
the small fake-simulator and real-binary checks behind this contract.

Decision records are a separate opt-in (`--decision-records` on `train.py` and
`evaluate.py`) that joins each action to the preceding `steps.jsonl` record and
its outcome window; see [decision records](@ref src_rl_decision_records).

## Reading an RL output directory {#src_rl_reading_output}

Start with the [output-location table](@ref scripts_rl_output),
then read these files in order. The ordinary simulator CSV columns are covered
by the [I/O guide](@ref src_io) and are separate from the RL policy input.

| File | First question it answers |
| --- | --- |
| `train_manifest.json` or `eval_manifest.json` | Which scenario, seed roles, model, observation/reward selection, and schema did this run use? Did the run complete? |
| `maskable_ppo_mesh.zip`, `best_model.zip`, or `checkpoints/*.zip` (when produced) | Which policy artifact was saved? Use the manifest's path and digest to identify the intended file. |
| `episode-NNNN/rl_episode.json` | Did this particular episode complete, how many policy steps occurred, and what was its cumulative reward? Filter incomplete or zero-step episodes before aggregating results. |
| `episode-NNNN/steps.jsonl` (when telemetry is enabled) | What raw facts, action, mask, and reward were recorded at each saved decision? |
| `episode-NNNN/run.log`, `inputs/`, and `seed-N/` | What resolved simulator configuration and archived inputs produced the raw CSV outputs? |
| `episode-NNNN/sim_stderr.log` | What did the simulator report on stderr, including launch or runtime diagnostics? |

### Decode the trace header before reading a step

The header's `contract.obs_dim` is the length of the **raw C++ `obs` message**.
The header's `observation_schema.obs_dim` is the length of the **selected vector
passed to the policy** after Python builds it from `facts`. These can differ:
with three mesh nodes (`N=3`) and three control slots (`M=3`), the raw
`raw_links_v1` layout has `M × [4 + 2(N-1)] = 24` values, while
`local_links_v1` has `M × [4 + 4(N-1)] = 36`. A `local_links_v1` policy sees
36 values even if the same header says `contract.obs_dim: 24`. Changing the
selected preset, `N`, or `M` can change the policy dimension; changing the
positions or seed does not. `slot_node_ids` maps slots to real nodes, with
`null` for padding; it does not shorten the vector.

Within `observation_schema`, `feature_names[i]`, `low[i]`, and `high[i]`
describe the name and declared Gymnasium `Box` bounds of policy feature `i`.
Those bounds are **not** observed minima/maxima and do not themselves perform
normalization. They are saved with the preset, normalization settings, physical
movement `bounds`, and a schema fingerprint so a saved run remains
interpretable and incompatible model inputs can be detected. The raw preset
records unbounded Box entries as JSON `null`. For `local_links_v1`:

| Feature | How Python derives it from raw facts | Declared range |
| --- | --- | --- |
| `active`, `present`, `sinr_valid` | 0/1 indicators for slot occupancy, peer presence, and usable SINR/capacity | `[0,1]` |
| `x_n`, `y_n`, `z_n` | `clamp(2 × (position - axis_min) / (axis_max - axis_min) - 1, -1, 1)` using header `bounds` in metres | `[-1,1]` |
| `sinr_n` | `clamp((SINR_dB + 20) / 60, 0, 1)` for a valid link | `[0,1]` |
| `cap_n` | `clamp(log10(1 + capacity_Mbps) / 4, 0, 1)` for a valid link | `[0,1]` |

`sinr_clip_db: [-20, 40]` names the **raw dB endpoints** used in that SINR
formula: -20 dB becomes 0, 10 dB becomes 0.5, and 40 dB or higher becomes 1.
It does not clamp simulator physics, raw `facts.links`, or the C++ reward.
Invalid links have `sinr_valid=0` and zero normalized SINR/capacity. An unused
slot is all zeros. The `normalization` object records which conversions were
selected; changing the formulas requires changing the preset/schema, not just
the displayed `low`/`high` arrays.

### Decode the raw facts and decision records

`facts` is **source data**, not the policy vector. The header's
`contract.facts_columns` gives the row order. `facts.nodes` has one row per
`node_ids` entry, in `nodes.json` order:
`[x, y, z, vx, vy, vz, slot]`, with positions in metres, velocities in m/s,
and `slot=-1` for an uncontrolled node. `facts.links` has one row for each
unordered mesh-node pair in `i<j` order (first index outer):
`[sinr_db, capacity_mbps, is_los]`. With nodes A, B, C the three rows are
`(A,B)`, `(A,C)`, `(B,C)`; `is_los` is 0/1. Jammers are not mesh-node rows.

For a concrete decoding example, suppose the header maps slot 0 to node B and
sets x bounds `0..100`, y bounds `-50..100`, and z bounds `0..50`. A node-B row
`[100,-5,10,0,-10,0,0]` means position `(100,-5,10)` m, y velocity -10 m/s,
and control slot 0. Its policy position is `(x_n,y_n,z_n)=(1,-0.4,-0.6)`.
If the `(A,B)` link row is `[40.62,5397.59,1]`, the raw SINR is 40.62 dB,
capacity is 5397.59 Mbps, and the path is LOS. The corresponding local policy
features are `present=1`, `sinr_valid=1`, `sinr_n=1` (clipped at 40 dB), and
`cap_n≈0.933`. The actual observation vector is not copied into each trace
record: `obs_sha256` fingerprints it, and replay reconstructs it from facts.

`facts.window` is different from the instantaneous node/link tables: it sums
measurements over the ticks since the previous decision. For example,
`ticks=5` and `connected_pairs_sum=15` means three connected pairs in each of
five ticks, **not** 15 distinct links. Demand/delivery sums have units of
Mbps·tick, not transferred bytes; the reward formulas above use these sums.

Decision 0 is the reset observation. No action was sent yet, so
`action_sent=null` and the **policy** `reward=null`. A nonzero `legacy_reward`
there is the C++ tick-0 diagnostic, not training reward. Decision 1 records
the action sent after reset and the reward for the resulting window; its action
has one entry per control slot, including a padded slot's required hold (`4`).
Training may first reset solely to learn the simulator-dependent spaces before
PPO starts; that probe can be saved as `episode-0000` with `status=interrupted`,
`stop_reason=reset`, and zero policy steps. It is not a lost training action.
