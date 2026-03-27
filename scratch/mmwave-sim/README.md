# mmwave-sim

mmWave 5G network simulator built on ns3-mmwave. Runs configurable scenarios with multiple seeds, produces per-UE metrics and visualization data.

## Project structure

```
mmwave-sim/
├── sim.cc                  Main entry point (seed loop, orchestration)
├── src/
│   ├── cli/                Command-line parsing, seed resolution
│   ├── config/             Configuration loading (run.ini + JSON)
│   ├── domain/             POD data types (no ns-3 dependency)
│   ├── io/                 Output writers (metrics, viz, progress)
│   ├── setup/              NS-3 topology, traffic, protocol defaults
│   └── util/               Generic helpers (INI parsing, strings)
├── inputs/scenarios/       Pre-built scenario configurations
├── outputs/                Timestamped simulation results
├── scripts/plotting/       Python post-sim plotting tools
└── third_party/            External deps (nlohmann/json)
```

## Running a scenario

```bash
./ns3 run "mmwave-sim --run-config=scratch/mmwave-sim/inputs/scenarios/uav-hover-baseline/run.ini"
```

### CLI options

| Flag | Description |
|---|---|
| `--run-config=<path>` | **(required)** Path to scenario `run.ini` |
| `--seeds=1,2,3` | Comma-separated seed list |
| `--seed=42` | Single seed override |
| `--run-id=1` | Override `run_id` from `run.ini` |
| `--positions-override=<path>` | Override node positions (JSON) |

Seed priority: `--seeds` > `--seed` > config file > default (42).

## Scenarios

Located in `inputs/scenarios/`. Each contains `run.ini`, `nodes.json`, and optionally `buildings.json`.

| Scenario | Description |
|---|---|
| `uav-hover-baseline` | Two hovering UAVs at fixed altitude and separation |
| `uav-flyaway-range` | UAV flying away from reference -- maps throughput vs distance |
| `building-bypass` | UE transitions from NLOS to LOS past a building |
| `building-two-ues` | Two UEs blocked by a building |
| `two-ues-one-approaching` | One mobile UE approaching a fixed eNB |
| `ue-moving-away` | Single UE moving away from eNB |

## Input format

### run.ini

INI file with sections:

- `[scenario]` -- name, duration, warmup, seed, node/building file paths
- `[channel]` -- frequency, tx power, scenario (UMi/UMa/RMa), channel model (3gpp/nyu), blockage
- `[nyu_channel]` -- NYU-specific params (bandwidth, atmosphere, foliage); ignored for 3gpp
- `[traffic]` -- direction (dl/ul/both), packet size, inter-packet interval
- `[network]` -- backhaul data rate and delay
- `[output]` -- viz tick interval, pcap toggle, trace level

### nodes.json

Array of node specs:
```json
[
  {"id": "enb0", "role": "enb", "mobility": "fixed", "position": {"x": 0, "y": 0, "z": 10}},
  {"id": "ue0", "role": "ue", "mobility": "constant_velocity",
   "position": {"x": 80, "y": 0, "z": 1.6}, "velocity": {"vx": 0, "vy": 1.0, "vz": 0}}
]
```

Mobility types: `fixed`, `constant_velocity`, `random_walk`.

### buildings.json (optional)

Array of building specs with axis-aligned bounding boxes:
```json
[{"id": "bldg0", "x_min": 40, "x_max": 70, "y_min": -20, "y_max": 20, "z_min": 0, "z_max": 20}]
```

## Output structure

Each run produces a timestamped directory:

```
outputs/YYYY-MM/DD/HH-MM-SS/
├── inputs/              Archived copy of scenario files
├── seed-1/
│   ├── summary.json     Per-UE + network metrics (throughput, SINR, delay, LOS fraction)
│   ├── positions.csv    Node positions over time
│   ├── links.csv        Per-link SINR, distance, LOS/NLOS over time
│   ├── RxPacketTrace.txt
│   ├── DlRlcStats.txt / UlRlcStats.txt
│   └── ...              Additional ns-3 trace files
├── seed-2/
└── figures/             Aggregated plots (if plotting scripts are run)
```

## Plotting

See [scripts/plotting/README.md](scripts/plotting/README.md) for post-sim plot generation.
