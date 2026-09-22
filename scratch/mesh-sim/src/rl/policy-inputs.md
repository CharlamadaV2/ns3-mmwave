# Policy observations, rewards, and telemetry

This page covers centralized RL. The simulator computes positions, links,
traffic, and per-tick reward in C++; every decision message includes those raw
measurements as `facts`. Python's `selection.py` chooses what the policy sees,
what reward Gymnasium returns, and whether to save decision records. The
simulator's physics and action masks do not change when these options change.
For actions and timing, see the [bridge contract](README.md).

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

Then run the training command in the [RL setup guide](../../README.md#selecting-observations-rewards-and-telemetry).
Its five matching CLI flags can override these keys independently; resolution
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

## What the observation contains

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

## One action, several ticks, one reward

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

From the mesh-sim directory, inspect or replay one episode with:

```bash
head -n 2 outputs/rl-multi-custom/episode-0000/steps.jsonl
.venv/bin/python -c 'from scripts.rl.env.telemetry import replay_file; import sys; print(replay_file(sys.argv[1]))' outputs/rl-multi-custom/episode-0000/steps.jsonl
```

Replay rebuilds saved observations and Python-composed rewards from each
record's raw facts and checks for mismatches. It does not rerun radio physics,
verify unsaved decisions, or recheck a C++-authored reward. See the
[test map](policy-input-tests.md) for
the small fake-simulator and real-binary checks behind this contract.
