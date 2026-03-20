# mmwave-sim

A configurable 5G mmWave (millimeter-wave) simulation harness for ns3-mmwave.
Scenarios are defined entirely through JSON and INI files; no C++ changes are
needed to add nodes, adjust the channel, or vary traffic.

---

## Prerequisites

- ns3-mmwave built with examples enabled:
  ```bash
  cd /path/to/ns3-mmwave
  ./ns3 configure --enable-examples
  ./ns3 build
  ```
- Python 3.8+ (stdlib only — no pip packages required)
- C++17 compiler (for `std::filesystem`)

---

## Build

```bash
./ns3 build scratch/mmwave-sim/sim
```

---

## Quick start

### Run a single simulation

```bash
cd /path/to/ns3-mmwave
python3 scratch/mmwave-sim/scripts/run_sim.py \
    --config scratch/mmwave-sim/inputs/scenarios/single-enb-3ue/run.ini
```

Output is written to a timestamped directory under
`scratch/mmwave-sim/outputs/YYYY-MM/DD/HH-MM-SS/seed-<N>/`.

Override seed or run_id from the command line:

```bash
python3 scratch/mmwave-sim/scripts/run_sim.py \
    --config scratch/mmwave-sim/inputs/scenarios/single-enb-3ue/run.ini \
    --seed 7
```

### Run directly (no Python wrapper)

```bash
./ns3 run scratch/mmwave-sim/sim \
    -- --run-config=scratch/mmwave-sim/inputs/scenarios/single-enb-3ue/run.ini
```

### Run multiple seeds (statistical sweep)

Each iteration uses a **different seed value** to produce independent
realisations.  Results are averaged across seeds in `aggregated_summary.json`.

```bash
# 10 seeds (seeds 1..10)
python3 scratch/mmwave-sim/scripts/run_sweep.py \
    --config scratch/mmwave-sim/inputs/scenarios/single-enb-3ue/run.ini \
    --num-seeds 10

# Specific seeds
python3 scratch/mmwave-sim/scripts/run_sweep.py \
    --config scratch/mmwave-sim/inputs/scenarios/single-enb-3ue/run.ini \
    --seeds 1,7,42,100,999
```

Each seed runs as an independent process writing to its own directory.
`aggregated_summary.json` reports mean ± std across all seeds.

See `docs/tutorial.md` for step-by-step walkthroughs.

---

## Scenarios

### Validation scenarios

These scenarios test specific simulation behaviours.  Run them to verify the
simulator is working correctly before running novel experiments.

| Scenario | Tests                                                                    | Expected result                                                                   |
|---|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `ue-moving-away` | SINR (Signal-to-Interference-plus-Noise Ratio) degrades with distance    | SINR and throughput smoothly decrease as UE moves 5 m to 200 m from eNB over 10 s |
| `two-ues-one-approaching` | Symmetric verification, one UE improving, one static                     | ue1 SINR/throughput improve over time; ue0 stays roughly constant                 |
| `building-bypass` | LOS (Line of Sight)/NLOS (Non-Line of Sight) transition on building exit | SINR jumps sharply when UE moves out from behind building at ~t=2 s               |
| `building-two-ues` | Differential blockage (one stays blocked, one exits)                     | ue0 (fixed, NLOS) has lower mean SINR than ue1 (moves to LOS)                     |

---

## Output structure

```
outputs/
└── YYYY-MM/DD/HH-MM-SS/
    ├── inputs/
    │   ├── run.ini              original input snapshot
    │   ├── run_patched.ini      the exact ini passed to the binary
    │   ├── nodes.json
    │   └── buildings.json
    └── seed-42/
        ├── RxPacketTrace.txt    per-transport-block PHY trace
        ├── DlRlcStats.txt       per-UE DL RLC throughput / delay epochs
        ├── UlRlcStats.txt       per-UE UL RLC throughput / delay epochs
        ├── DlPdcpStats.txt      per-UE DL PDCP stats
        ├── UlPdcpStats.txt      per-UE UL PDCP stats
        ├── MmWaveSinrTime.txt   SINR time series
        ├── DlPhyTransmissionTrace.txt
        ├── UlPhyTransmissionTrace.txt
        ├── EnbSchedAllocTraces.txt
        ├── positions.csv        node position snapshots (viz)
        ├── links.csv            per-link channel state snapshots (viz)
        ├── summary.json         computed metrics (written by MetricsWriter)
        ├── stderr.txt           ns-3 log output
        └── pcap/                PCAP captures (when pcap_enabled = true)
```

See `docs/metrics.md` for full column definitions of every output file.

---

## RL extension point

> **TODO (RL):** The current mechanism only injects positions **before the
> simulation starts**, an RL agent cannot react to
> channel conditions mid-run.  For true step-based control (e.g., "move relay
> to destination X at Y m/s"), additions are needed.

To control node positions from an external agent (e.g., a reinforcement
learning policy), pass starting positions at launch:

1. Write a `positions_override.json`:
   ```json
   {"relay0": [120.0, 45.0, 5.0]}
   ```

2. Pass it at runtime:
   ```bash
   python3 scripts/run_sim.py \
       --config inputs/scenarios/two-enb-relay/run.ini \
       --positions-override /path/to/positions_override.json
   ```

The binary loads `nodes.json` first, then overwrites only the listed node
positions.  Roles, mobility types, and velocity fields are unchanged.

---

## Reference docs

| Document | Contents |
|---|---|
| `docs/input-schema.md` | Full schema for `nodes.json`, `buildings.json`, and `run.ini`; channel scenario vs buildings explained; simulation area discussion |
| `docs/metrics.md` | Complete column definitions for every output file; `summary.json` and `aggregated_summary.json` field reference |
| `docs/tutorial.md` | Step-by-step walkthroughs for each scenario |
| `docs/rl-extension.md` | Current RL integration state, limitations, and planned approaches (waypoint mobility, co-simulation, socket interface) |
| `run.ini` in each scenario | Line-by-line commentary on every configuration key |
