# Input File Schema

Three files define a scenario.  They live together in a directory under
`inputs/scenarios/<name>/` and are referenced by relative path in `run.ini`.

| File | Format | Purpose |
|---|---|---|
| `run.ini` | INI (sections + `key = value`) | Simulation parameters, channel, traffic, output path |
| `nodes.json` | JSON array | Node definitions: roles, positions, mobility |
| `buildings.json` | JSON array | Building and obstacle geometry |

See `run.ini` in each scenario directory for line-by-line commentary on every
option.  This document is the authoritative reference for the JSON formats.

---

## Simulation area

There is **no fixed simulation boundary**.  The scenario area is implicitly
defined by the geometry you specify in `nodes.json` and `buildings.json`.
What matters is the relative arrangement of nodes and obstacles.

- **For mobility models**, the bounding box is set per-node in `random_walk.bounds`.
  `constant_velocity` nodes will leave any implicit boundary if not stopped by
  a finite `duration_s`.
- **The coordinate origin** is arbitrary, place it wherever is most convenient
  (e.g., at the eNB (evolved Node B — the base station), at the center of the
  deployment, or at a building corner).
- **Distances between nodes** drive path loss; only relative positions matter.

For reference: at 28 GHz with UMi (Urban Micro) parameters and 30 dBm transmit
power, LOS (Line of Sight) connectivity typically extends to ~200–400 m before
SINR (Signal-to-Interference-plus-Noise Ratio) falls below viable levels.
NLOS (Non-Line of Sight) links have roughly 10–20 dB additional loss, so the
effective NLOS range is shorter.

---

## Channel scenario vs buildings

These are two independent, complementary mechanisms that affect propagation.
Both can be active simultaneously and they interact.

### Channel scenario (`[channel] scenario`)

Sets the 3GPP large-scale propagation model and the statistical LOS/NLOS
probability function used for every link:

| Value | Description |
|---|---|
| `UMi` or `UMi-StreetCanyon` | Street-level urban micro. eNB ~10 m, UE (User Equipment) ~1.6 m. Higher NLOS path-loss exponent, tighter LOS probability. |
| `UMa` | Urban Macro. eNB ~25 m. Larger cells, lower path-loss exponent. |

The channel scenario governs:
1. **Path-loss formula** — the coefficients for LOS and NLOS curves.
2. **Statistical LOS probability** — for a given link distance, what fraction
   of links are expected to be LOS at that range (probabilistic, not
   geometry-based).

### Buildings (`buildings.json`)

Buildings add **geometry-aware**, **deterministic** LOS/NLOS classification
on top of the statistical model above:

- Each link is tested against every building bounding box at every simulation
  step.
- If the straight-line path between the two nodes intersects a building, the
  link is classified NLOS (regardless of what the statistical model would say).
- NLOS links get additional penetration loss based on the `ext_walls` material.
- This updates dynamically as nodes move.

### How they interact

When buildings are present, `BuildingsChannelConditionModel` overrides the
purely statistical LOS/NLOS from the channel scenario with geometry-based
classification.  The path-loss formula (UMi vs UMa coefficients) still applies,
but the LOS/NLOS *decision* is now determined by actual building geometry.

**When to use each:**

| Goal | Recommendation |
|---|---|
| Model a street canyon without specific obstacles | Set `scenario = UMi-StreetCanyon`, leave `buildings_file` empty |
| Model specific obstacles (known building locations) | Add buildings to `buildings.json`; any scenario works |
| Most realistic urban simulation | Both: choose UMi or UMa to match the deployment scale, AND add buildings |
| Quick baseline / no blockage | `buildings_file =` (empty), `blockage_enabled = false` |

---

## nodes.json

A JSON array of node objects.  Order matters: UE nodes are assigned IMSI
(International Mobile Subscriber Identity) values 1, 2, 3, … in the order
they appear (eNB nodes are skipped in IMSI numbering).

### Terminology

**eNB (evolved Node B)**: the base station.  It provides the radio
air interface and is the fixed infrastructure side of the link.  A simulation
can have multiple eNBs; each UE attaches to its closest one at startup.  The
`relay` pattern (see `two-enb-relay` scenario) places a second eNB at a
position that can be controlled by an RL agent to
maximise coverage.

**UE (User Equipment)** — any mobile endpoint: smartphone, IoT sensor,
vehicle-mounted radio, drone.  UEs are the traffic endpoints.

### Node object fields

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Unique identifier.  Referenced in `positions_override.json` and appears in `summary.json`. |
| `role` | string | yes | `"enb"` — base station.  `"ue"` — user equipment. |
| `mobility` | string | yes | Movement model (see below).  **Ignored for eNBs** — they are always stationary. |
| `position` | object | yes | Initial position `{"x": …, "y": …, "z": …}` in meters. |
| `velocity` | object | if `mobility == "constant_velocity"` | Velocity vector `{"vx": …, "vy": …, "vz": …}` in m/s. |
| `random_walk` | object | if `mobility == "random_walk"` | Bounding box and speed (see below). |

### `mobility` options

| Value | ns-3 model | Behaviour |
|---|---|---|
| `"fixed"` | `ConstantPositionMobilityModel` | Node stays at `position` for the entire simulation. |
| `"constant_velocity"` | `ConstantVelocityMobilityModel` | Moves at the constant vector given by `velocity`.  No bounds — will leave the area if not stopped by `duration_s`. |
| `"random_walk"` | `RandomWalk2dMobilityModel` | Moves at constant speed `speed_mps` in a randomly chosen direction; bounces off the bounding box walls.  z stays fixed at `position.z`. |

### `random_walk` object

| Field | Default | Description |
|---|---|---|
| `bounds.x_min` | `-100.0` | Bounding box minimum X in meters. |
| `bounds.x_max` | `100.0` | Bounding box maximum X in meters. |
| `bounds.y_min` | `-100.0` | Bounding box minimum Y in meters. |
| `bounds.y_max` | `100.0` | Bounding box maximum Y in meters. |
| `speed_mps` | `1.5` | Constant walk speed in m/s.  Typical: 0.5 (slow pedestrian), 1.5 (normal walk), 5.0 (running/cyclist). |

### Position conventions

Positions are stored as `(x, y, z)` triples in meters.  The simulator
supports any spatial arrangement:

- **3D simulation** (default): all three coordinates are meaningful.
- **2D simulation**: set all z values to constant heights.
- **1D simulation** (e.g., highway, single street): set y = 0 for all nodes,
  vary only x.

Typical height conventions (needs a solid reference):
- eNBs: z = 10 m (lamppost / low rooftop)
- Pedestrian UEs: z = 1.6 m
- Vehicle-mounted UEs: z ≈ 1.5–2.0 m
- Drone / UAV relays: z = 20–100 m

The coordinate origin is arbitrary; place it wherever is convenient.

### RL extension point — `positions_override.json`

> **TODO (RL):** The current `positions_override.json` mechanism only sets node
> positions **before the simulation starts**.  This means an RL agent can only
> pick a starting position, not react to channel conditions during a run.

To inject node positions from an external controller at simulation start:

```json
{"relay0": [120.0, 45.0, 5.0]}
```

Pass via `--positions-override`.  Only the listed nodes are repositioned;
their role, mobility type, and velocity fields remain from `nodes.json`.
Multiple nodes can be overridden in one file:

```json
{
  "relay0": [120.0, 45.0, 5.0],
  "ue3":    [75.0,  20.0, 1.6]
}
```

---

## buildings.json

A JSON array of building objects.  An empty array `[]` disables buildings
entirely.  When at least one building is present, `BuildingsChannelConditionModel`
is activated automatically, giving geometry-aware LOS/NLOS determination and
penetration loss.

### Building object fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `id` | string | no | `""` | Label for identification in logs. |
| `bounds.x_min` | float | yes | — | Bounding box in meters (axis-aligned rectangular prism). |
| `bounds.x_max` | float | yes | — | |
| `bounds.y_min` | float | yes | — | |
| `bounds.y_max` | float | yes | — | |
| `bounds.z_min` | float | yes | — | Typically `0.0` (ground level). |
| `bounds.z_max` | float | yes | — | Building height in meters.  ~3.5 m per floor is typical. |
| `type` | string | no | `"Residential"` | Building use type (see below). |
| `ext_walls` | string | no | `"ConcreteWithWindows"` | Exterior wall material (see below). |
| `n_floors` | int | no | `1` | Number of floors.  Used by ns-3 for inter-floor propagation models. |

### `type` options

| Value | Description |
|---|---|
| `"Residential"` | Apartment / housing block. |
| `"Office"` | Open-plan or cellular office.  Higher internal reflection. |
| `"Commercial"` | Retail / mixed-use.  Large open floor areas. |

### `ext_walls` options (Needs to be double checked)

Controls penetration loss through the exterior walls at 28 GHz.

| Value | Approx. penetration loss | Description |
|---|---|---|
| `"Wood"` | ~4 dB | Lightweight timber-frame construction. |
| `"ConcreteWithWindows"` | ~10–15 dB | Standard concrete with glazing (most common urban building). |
| `"ConcreteWithoutWindows"` | ~20–25 dB | Solid concrete / parking structure / basement wall. |
| `"StoneBlocks"` | ~12 dB | Masonry / stone construction (older buildings). |

### LOS/NLOS effect

When buildings are present, each link is classified as LOS or NLOS based on
whether the straight-line path between the two nodes intersects a building
bounding box.  NLOS links use a higher path-loss exponent and receive an
additional penetration loss term.  Classification updates dynamically as
nodes move.  See `links.csv` (`condition` column) to observe LOS/NLOS
transitions in real-time.

---

## run.ini — [network] section

The `[network]` section configures the **backhaul link** between the EPC
(Evolved Packet Core) PGW (Packet Data Gateway) and the remote traffic host.
The backhaul is the wired link that carries all traffic to and from the
simulated internet, every downlink (DL) packet flows from the remote host
through the EPC PGW and then over the mmWave (millimetre-wave) radio to the
UE; uplink (UL) traffic travels the reverse path.

In most mmWave experiments the backhaul is intentionally made a
**non-bottleneck** (100 Gb/s, 10 ms one-way delay) so that the mmWave radio
link is the only constrained resource.  Adjust these parameters if you want
to study how core network latency or capacity affects application-level metrics.

| Key | Default | Description |
|---|---|---|
| `backhaul_data_rate` | `100Gb/s` | Backhaul link capacity.  Use ns-3 DataRate strings: `"100Gb/s"`, `"10Gb/s"`, `"1Gb/s"`, `"100Mb/s"`. |
| `backhaul_delay_ms` | `10.0` | One-way backhaul latency in milliseconds.  Added on top of the radio air-interface delay. |

**When to change these:**
- **Studying end-to-end latency:** Increase `backhaul_delay_ms` to 50–100 ms
  to simulate a distant data center.
- **Studying congestion effects:** Reduce `backhaul_data_rate` to `100Mb/s`
  or `1Gb/s` to make the backhaul a potential bottleneck alongside the radio.
- **Default (mmWave radio studies):** Leave both at default, the backhaul
  carries no meaningful delay or capacity constraint.

---

## run.ini — [traffic] section

| Key | Default | Description |
|---|---|---|
| `direction` | `dl` | `dl` (downlink), `ul` (uplink), or `both`. |
| `packet_size_bytes` | `1400` | UDP (User Datagram Protocol) payload size. 1400 B avoids IP fragmentation on standard MTU. |
| `inter_packet_interval_us` | `100.0` | Time between packets (µs).  Sets offered load. |
| `app_start_offset_s` | `0.1` | Delay before first packet — allows RRC (Radio Resource Control) attach to complete. |
| `max_packets` | `0` | 0 = unlimited (traffic runs until `duration_s`).  Set > 0 to cap total packets per source. |

---

## run.ini — [output] section

The output directory is always auto-generated. Direct binary invocation produces
`outputs/<scenario_name>/seed-<N>/run-<run_id>/`; `run_sim.py` uses a
timestamped path.

| Key | Default | Description |
|---|---|---|
| `viz_tick_ms` | `100` | Snapshot interval for `positions.csv` / `links.csv` (ms). |
| `pcap_enabled` | `false` | When `true`, PCAP (packet capture) files written to `<output_dir>/pcap/`. |
