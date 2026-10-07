@page src_routing src/routing

@brief Flow-level routing over the mesh link graph, with congestion scaling and latency.

This module takes the current link table (capacity and distance per node pair)
and the list of traffic flows for the tick. It finds a path for each flow,
scales down flows that overload a link, and returns one `FlowResult` per flow.
`sim.cc` calls it once per tick after the traffic update. There are no packets
or queues.

## Module Layout

| File | Role |
|------|------|
| @ref mesh-router.h "mesh-router.h" | `FlowResult` struct and `MeshRouter` class, with full Doxygen docs. |
| @ref mesh-router.cc "mesh-router.cc" | Dijkstra, widest-path and BFS path finding, congestion scaling, latency model. |
| `CLAUDE.md` | Scope and dependency notes for AI assistants. |

## Setup

Nothing to install. The router is compiled into the simulator. The user builds
the simulator binary; do not build from here.

## Run

The router has no CLI. Choose its behavior in the scenario `run.ini`:

```ini
[routing]
algorithm = shortest_path   ; shortest_path | max_throughput | min_hop
max_hops  = 5               ; 0 = unlimited
```

Run the unit tests (plain compiler, no ns-3) from `scratch/mesh-sim/`:

```bash
make -C tests/unit/routing test
```

| Algorithm | Strategy |
|-----------|----------|
| `shortest_path` (default) | Dijkstra, edge weight `1/capacity_mbps`. |
| `max_throughput` | Widest path: maximise the bottleneck link capacity. |
| `min_hop` | Breadth-first search; fewest hops, ignores link quality. |

## Output

| Step | Where |
|------|-------|
| `MeshRouter::Route()` returns `std::vector<FlowResult>`, same order as the input flows | In memory. `sim.cc` passes it to `io/` (metrics and viz writers) and `rl/`. The router writes no files. |

Each `FlowResult` holds `path`, `hop_count`, `demand_mbps`, `delivered_mbps`
(Mbps), `latency_ms` (ms), and `routable`.

## Conventions

- An edge exists only where the link capacity is above 0 Mbps.
- `max_hops = 0` means unlimited. A path longer than `max_hops` makes the flow unroutable; no longer path is used instead.
- Congestion scaling is per undirected edge: if total demand exceeds capacity, each flow on it is scaled by `capacity / total_demand`. A flow keeps the smallest factor over its hops.
- Latency per hop is `distance_m / 3e8 * 1000` ms plus a fixed 0.5 ms processing delay. Scaling does not change latency.
- Routing is deterministic for a given link table, so the RL agent is judged on positioning only.
- Comments use Doxygen `/** */` blocks.

## Dependencies

- C++17.
- `domain/` (`mesh-config.h`, `link-result.h`), `eval/link-table.h`, `traffic/traffic-matrix.h`.
- No ns-3 types are used directly. Unit tests use stubs from `tests/common/`.
