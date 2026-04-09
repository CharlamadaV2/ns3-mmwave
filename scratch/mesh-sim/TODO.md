# TODO — Future Possibilities

## RL Enhancements

### Multi-Node Control
RL currently controls one node. Future: control all node positions via
multi-agent RL or a centralized controller. Each node already has its own
max speed from `node_type`.

### Continuous Desired-Position Actions with SB3
The discrete left/right/stay action space is a v0 simplification. The
continuous action type (`action_type = continuous`) is already implemented
and outputs absolute target (x, y) coordinates. Next step: integrate
Stable Baselines 3 (PPO/SAC) using the Gymnasium env which already supports
both discrete and continuous spaces.

### Richer Reward Shaping
Currently supports `throughput` (sum delivered_mbps) and `mean_sinr`.
Future: weighted combinations of throughput, fairness (min-link capacity),
latency, coverage area, or energy cost.

### 3D Movement for Drones
Currently RL only adjusts x/y position. Aerial nodes (drones) should be
able to control altitude (z-axis) while ground nodes (vehicles, pedestrians)
remain 2D-constrained.

### Larger Discrete Action Spaces
Add 4-direction (up/down in y-axis) and 8-direction (diagonals) as
additional `action_type` options.

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

Currently assumes orthogonal channels between all node pairs (no
inter-node interference). This is reasonable for narrow mmWave beams with
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
