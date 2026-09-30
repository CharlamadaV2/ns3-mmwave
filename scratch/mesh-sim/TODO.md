# TODO — Future Possibilities

## RL Enhancements

### Multi-node control
`[rl] controlled_nodes` gives one MaskablePPO
policy a fixed set of mesh nodes with the 2-D `move_2d` profile. Remaining:

- **3-D `move_3d` profile.** `nvec = [7]*M` with `down`/`up` added and hold at
  index 6, contract id `mesh_move_3d_v1`. Rejected today with an explicit
  "reserved and not implemented" error. Owner: team — status: open.
- **Coupled and collision constraints.** Per-slot masks cannot express
  minimum separation, collision avoidance, or any joint constraint between
  controlled nodes. Owner: team — status: open.
- **Transfer and evaluation loader.** Landed. `scripts/rl/policy/compat.py`
  checks a saved model against the live `init` in a fixed order — structural
  contract fields, observation schema, reward schema, then scenario identity —
  and `scripts/rl/evaluate.py` loads and evaluates a bundle only after those
  checks pass. Transfer across scenarios is still never implied by matching
  shapes. Owner: team — status: landed.

### Centralized-control questions (temporary assumptions in effect)
Each item below records the assumption the landed code implements. Confirm or
change it; do not infer the answer from the code.

- **TODO-RL-CONTROL-1 — Waypoint start.** Should an RL-controlled waypoint node start
  at its first waypoint and ignore the rest? Assumed yes: a controlled
  waypoint node is placed at `waypoints.front()` and RL then owns its motion.
  Owner: team — status: open.
- **TODO-RL-CONTROL-2 — `all` eligibility.** Should `all` exclude any node class (for
  example a traffic gateway)? Assumed no exclusions: `all` is every entry of
  `nodes.json`. Owner: team — status: open.
- **TODO-RL-CONTROL-3 — Per-slot speed.** Is `step_size_m / tick_s` capped by node
  type the right per-slot speed, or is a `speed_mps` key wanted? Assumed the
  former, which keeps current movement for existing configs.
  Owner: team — status: open.
- **TODO-RL-CONTROL-4 — 3-D ground types.** In a 3-D profile, should vehicles and
  pedestrians have down/up masked off? Deferred with 3-D.
  Owner: team — status: open.
- **TODO-RL-CONTROL-5 — Observing uncontrolled nodes.** Should uncontrolled-node
  positions be observable by the agent? Currently excluded from
  `local_links_v1`; tracked with the other candidate features in TODO-P2-4.
  Owner: team — status: open.
- **TODO-TIME-1 — Tick-count unification.** Legacy and non-RL runs keep the
  truncating `duration_s / tick_s` count; centralized runs use a robust count
  (snap to the nearest integer within 1e-6, else truncate). Unifying them
  would change existing scenarios' tick counts, so it needs a separately
  approved regression-fixture review. Owner: team — status: open.

### P2 observation/reward/telemetry follow-ups

- **TODO-P2-1 — Jammer 0 dB SINR clamp.** the jammed branch of
  `LinkEvaluator::Evaluate` in `src/eval/link-evaluator.cc` clamps SINR to at least 0 dB whenever a jammer contributes power, so a jammed
  link can never fall below the connectivity threshold and a weak link can even
  become *connected* when a jammer turns on. Jamming enters SINR only when
  `band = sub-6`. No jamming-aware claim, reward experiment, or jammer feature
  should be made until this is decided; removing the clamp needs its own
  regression review. Owner: team — status: open.
- **TODO-P2-2 — RL reward ignores `warmup_s`.** `MetricsWriter::AccumulateTick`
  skips ticks before `warmup_s`, but `RlBridge::AccumulateTick` accumulates
  every tick, so RL rewards and facts windows include warmup ticks while
  `summary.json` does not. `init.warmup_s` is exported as metadata only.
  Aligning them would change existing RL rewards. Owner: team — status: open.
- **TODO-P2-3 — Peer padding across node counts.** `local_links_v1` sizes its
  peer block from the live `num_mesh_nodes`, so an observation from one N does
  not fit another. The `present` bits are already reserved for a padded preset
  keyed on a `max_mesh_nodes` bound; until then `check_schema` rejects a
  different N rather than implying transfer. Owner: team — status: open.
- **TODO-P2-4 — Candidate observation features.** The facts already carry every
  node's velocity and position, including uncontrolled nodes, and jammer
  interference could be added; none of them is in `local_links_v1`. Each needs
  a leakage review (what a real node could actually know) before becoming a
  preset feature, and any jammer feature also depends on TODO-P2-1.
  Owner: team — status: open.

### Continuous Desired-Position Actions with SB3
The discrete left/right/stay action space is a v0 simplification. The
continuous action type (`action_type = continuous`) is already implemented
and outputs absolute target (x, y) coordinates. Next step: integrate
Stable Baselines 3 (PPO/SAC) using the Gymnasium env which already supports
both discrete and continuous spaces.

### Richer Reward Shaping
Currently supports `throughput` (sum delivered_mbps) and `all_links_los`
(`mean_sinr` is a deprecated alias for it). Future: weighted combinations of
throughput, fairness (min-link capacity), latency, coverage area, or energy
cost.

### 3D Movement for Drones
The discrete action space already includes +/-Z alongside +/-X, +/-Y and Stay;
the continuous action type still emits a 2-D target only. Remaining work:
per-node-type constraints so aerial nodes (drones) control altitude while
ground nodes (vehicles, pedestrians) stay 2-D constrained, and a verified 3-D
continuous action space.

### Larger Discrete Action Spaces
Add 4-direction (up/down in y-axis) and 8-direction (diagonals) as
additional `action_type` options.

### TODO-RL-SEEDS-1 — Multi-seed training policy
Training currently uses one fixed seed (`m-ppo --seed`, else `[scenario] seed`) reused by
every episode. Deciding whether episodes should vary the seed, and how the
model/manifest should record that, is deferred.
That one integer also sets both the PPO initialization and the training scenario,
so the across-training-runs interval in `scripts.rl.compare` conflates the two
sources of randomness and cannot attribute run-to-run spread to either.
Separately, `--eval-episodes > 1` appears to replay the same evaluation seed
rather than sampling new ones — unverified.
Owner: team — status: open.

### TODO-RL-RESUME-1 — Resume from checkpoint
Training always starts a fresh model. `checkpoints/` and `best_model.zip` are
provenance and evaluation inputs only; there is no `--resume`. Adding one needs
`MaskablePPO.load` plus `learn(reset_num_timesteps=False)` handling and a
decision about manifest continuity (one manifest extended across runs, or a new
manifest that links to its parent). Owner: team — status: open.

### TODO-RL-LEARNING-1 — Learning signal beyond the diagnostic smoke
`inputs/baselines/building-bypass-smoke/` is a diagnostic fixture: it makes a
directional reward difference visible on a tiny run, and it is not evidence that
training converges or that a policy is useful. A real learning-signal study —
scenario set, seeds, budgets, baselines, and success criteria — belongs to a
separately approved campaign phase. Owner: team — status: open.

### TODO-RL-BASELINE-1 — Greedy baseline
`scripts.rl.evaluate` offers `hold` and seeded `random_valid`, which bracket "do
nothing" and "valid noise" but not "a sensible hand-written controller". A greedy
baseline cannot be added as just another policy: a policy sees only `obs`, whose
layout differs per observation preset (`raw_links_v1` is raw, `local_links_v1` is
normalized), so it needs either a facts-access contract of its own or a decoder
per preset, plus its own validation, because a greedy baseline becomes the de
facto benchmark everything else is judged against. `scripts.rl.compare` accepts
any baseline name present in a manifest, so adding one later needs no change to
the comparison harness. Owner: team — status: open.

### TODO-RL-EVAL-1 — Simulator-summary metrics in policy comparisons
Every statistic in `scripts.rl.compare` comes from the RL telemetry window, and
each output states `metric_source` accordingly. A run's `summary.json` carries
richer per-seed metrics but excludes warmup ticks while the RL window does not,
so the two cannot be mixed in one table whenever `warmup_s > 0`. Using
`summary.json` metrics requires resolving that warmup mismatch first (see the
RL-reward warmup entry above). Owner: team — status: open.

### TODO-RL-EVAL-2 — Merging evaluations of one model across output directories
An evaluation is compared as a single `eval_manifest.json`, so baselines are
re-run inside every evaluation and a per-seed cluster array (each seed in its own
output directory) cannot be assembled into one comparison. A merge needs an
explicit rule set, sketched as: a `completed` record supersedes a `failed` one
for the same seed, and two `completed` records for one seed must agree on
`actions_sha256` or the merge is refused. Owner: team — status: open.

### TODO-RL-OPS-1 — Live-cluster validation of the scheduler adapter
`scripts/rl/ops/cluster.py` and its `squeue`/`sacct`/`sbatch`/`scancel` parsers
are exercised against a fake scheduler only, so their state names, array-element
id forms, and accepted flags are unproven on a real site. Whether the site
exposes a meaningful start estimate or a queue position at all is likewise
unknown; the local benchmark supports no completion ETA, and a laptop
measurement is never a cluster estimate. Record each live finding here rather
than inferring site behavior from the code. Verify `sacct --array` explicitly;
the fake scheduler cannot reject unsupported flags. Owner: team — status: open.

### TODO-RL-OPS-2 — Oversized arrays and blocked-directory archiving
A plan with more tasks than the configured `max_array_size` is refused rather
than chunked across several arrays. A task blocked by a populated step directory
is reported with `move or delete <dir> to retry` and is never moved, renamed, or
deleted by the tooling; an archive helper that does it safely is not
implemented. Submitting a compare-only SLURM job after task submission succeeds
but compare submission is refused is also deferred; `resume` with no pending
tasks does not queue one. Owner: team — status: open.

### TODO-RL-TUNE-1 — Further PPO knobs, budget search, and study resume
`scripts/rl/ops/tune.py` can search only `n_steps`, `gamma`, and `ent_coef`,
because those are the only values that reach `MaskablePPO`. Widening the search
needs the whole chain first — constructor argument, `train.py` CLI flag,
manifest field, and comparison group key — otherwise a searched value would not
be recorded or checked anywhere. Searching `total_timesteps` is also deferred,
since the budget is what makes trials comparable, and study resume is deferred
because a seeded sampler restarts its sequence on reload.
Owner: team — status: open.

## Jammer model decisions (unresolved)

The jammer baseline work deliberately froze the current jammer behavior and changed no physics. The
following questions must be answered by the team before any change; do not
infer answers from the code.

- **TODO-JAM-1 — Link failure / 0 dB floor.** Jammed SINR is currently floored
  at 0 dB. Should 0 dB make the link unusable, or should the floor be removed so
  SINR can fall below the -6.7 dB link threshold?
  Owner: team/Kyle/jammer developer — status: open.
- **TODO-BAND-1 — Default band.** Should newly generated scenarios default to
  `sub-6` while existing scenarios keep `mmwave`, or should one global default
  apply?
  Owner: team/Kyle/jammer developer — status: open.
- **TODO-BAND-2 — Band versus frequency.** Should `band` stay an explicit
  scenario choice, be renamed to describe its actual jammer-interference role,
  or eventually be derived from / validated against `frequency_ghz`?
  Owner: team/Kyle/jammer developer — status: open.
- **TODO-JAM-2 — Endpoint rule.** Interference is computed at both link
  endpoints and the worse value is used. Is that correct?
  Owner: team/Kyle/jammer developer — status: open.
- **TODO-JAM-3 — Frequency overlap.** Should a jammer affect a channel whenever
  the frequency ranges overlap, instead of the current carrier-center test?
  Owner: team/Kyle/jammer developer — status: open.
- **TODO-JAM-4 — Recorded motion.** Which Sherpa file is the authoritative
  jammer trajectory, and should it become waypoints or another recorded-motion
  form?
  Owner: team/Kyle/jammer developer — status: open.
- **TODO-JAM-6 — Duty cycle.** Should constant and random jammers keep
  interpreting duty cycle differently?
  Owner: team/Kyle/jammer developer — status: open.

## Beam Codebook Model

mesh-sim currently uses ideal beamforming (fixed gain = 2 x 10*log10(N) dB).
In ns3-mmwave, beam management is handled by `MmWaveBeamforming` which uses a
discrete beam codebook, performs beam sweeping, selects optimal TX/RX beam
pairs, and triggers beam switches.

Adding a simplified codebook model would enable:
- **Beam index** (TX/RX beam pair per link per tick)
- **Beam switch events** (timestamp + old/new beam pair)
- **Per-beam RSRP** (received power per beam direction)
- Beam tracking lag and angular dead zone detection

Approach: define N beam directions per node (e.g., 64-element codebook),
compute array factor gain per direction, pick best pair per link based on
geometry, track indices over time.

## Packet-Level Loss

mesh-sim uses flow-level traffic abstraction (demand_mbps / delivered_mbps).
There are no packets, so true packet loss percentage cannot be computed.

Possible proxies:
- **Undelivered fraction**: `1 - (delivered_mbps / demand_mbps)` per flow
- **Link outage fraction**: fraction of ticks where a link's SINR is below
  the minimum MCS threshold (-6.7 dB)

Adding a packet-level model would require replacing the traffic matrix with
a packet generator and tracking per-packet delivery, which is a fundamental
architecture change.

## Interference Modeling

Apart from configured jammers (sub-6 only), assumes orthogonal channels
between all node pairs (no inter-node interference). This is reasonable for narrow mmWave beams with
high spatial isolation, but underestimates interference in dense deployments.

Full interference modeling would require:
- Joint scheduling + routing layer
- Per-link interference calculation from all active concurrent transmissions
- SINR = signal / (interference + noise) instead of signal / noise

A simpler alternative: configurable interference margin (e.g., subtract 3 dB
from SINR) as a conservative estimate.

## IMU / Attitude Modeling

Nodes are modeled as dimensionless points with position and velocity.
Physical UAV attitude (roll, pitch, yaw) is not tracked. This matters for
real radios because antenna patterns depend on platform orientation.

Would become relevant if beam codebook model is added, since physical tilt
changes the effective beam direction.

## LOS/NLOS Condition Tracking

Currently `condition_reason` in links.csv reports whether the channel
condition model is building-based (deterministic geometry) or probabilistic
(3GPP/NYU statistical model). This is a per-run property.

Future enhancement: per-link per-tick tracking of which specific building
caused NLOS, or the probability value from the statistical model.
