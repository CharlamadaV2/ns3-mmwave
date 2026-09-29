@page src_traffic src/traffic

@brief Flow-level traffic demand generation for the mesh simulation.

This module keeps the list of flows ("node A wants X Mbps to node B") and
updates it once per tick. It supports constant, Poisson-arrival and on-off
(bursty) models. `sim.cc` calls `Initialize()` once, then `Tick()` each step,
and passes `GetActiveFlows()` to the router. There are no packets.

## Module Layout

| File | Role |
|------|------|
| @ref traffic-matrix.h "traffic-matrix.h" | `Flow` struct and `TrafficMatrix` class, with full Doxygen docs. |
| @ref traffic-matrix.cc "traffic-matrix.cc" | Initialisation, per-tick expiry, Poisson arrivals, on-off state machine. |
| `CLAUDE.md` | Scope and dependency notes for AI assistants. |

## Setup

Nothing to install. The class is compiled into the simulator. The user builds
the simulator binary; do not build from here.

## Run

There is no CLI. Configure it in the scenario `run.ini`:

```ini
[traffic]
model             = constant     ; constant | poisson | on_off
flow_topology     = all_pairs    ; all_pairs | random_pairs | gateway
demand_mbps       = 10.0         ; per-flow demand, Mbps
arrival_rate_hz   = 1.0          ; poisson only, flows per second
on_time_s         = 1.0          ; on_off only, mean ON seconds
off_time_s        = 1.0          ; on_off only, mean OFF seconds
holding_time_s    = 0.0          ; mean flow lifetime, s; 0 = permanent
random_pair_count = 3            ; random_pairs only
gateway_node_id   =              ; gateway only, node id (required)
```

Run the unit tests (plain compiler, no ns-3) from `scratch/mesh-sim/`:

```bash
make -C tests/unit/traffic test
```

## Output

| Step | Where |
|------|-------|
| `GetActiveFlows()` returns the flow list (`const std::vector<Flow>&`) | In memory, valid until the next `Tick()` or `Initialize()`. It includes OFF-phase flows; the router filters them. No files are written. |
| `GetDemand(src, dst)` returns the active demand for a directed pair (Mbps) | In memory. |

## Conventions

- `end_time_s == 0` means a permanent flow. Only flows with `end_time_s > 0` that have expired are removed.
- Random draws use ns-3 random streams. The seed and run number are set once per seed in `sim.cc`, so the same seed gives the same traffic.
- Poisson mode also creates the initial flows at `Initialize()`. Without `holding_time_s`, arrived flows never expire.
- `gateway_node_id` is tried as a numeric index first, then as a node ID.
- Comments use Doxygen `/** */` blocks.

## Dependencies

- C++17.
- `domain/sim-config.h`.
- ns-3 `random-variable-stream.h`. Unit tests replace it with stubs in `tests/common/` and `tests/unit/traffic/`.
