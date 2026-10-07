# routing/

## Scope
Flow-level routing over the per-tick `LinkTable`: picks a path per flow, applies
congestion scaling, and computes latency. No packets, queues, or ns-3 types.

## Files
- **mesh-router.h/cc** -- `FlowResult` struct; `MeshRouter::Route()` (public), private `FindPath` (dispatch + hop limit), `FindPathShortestPath` (Dijkstra, weight `1/capacity_mbps`), `FindPathMaxThroughput` (widest path), `FindPathMinHop` (BFS), `ReconstructPath`, `ApplyCongestionScaling`, `ComputeLatency`.

## Behavior to preserve
- Routing must stay deterministic and optimal for the current link state, so the
  RL agent is rewarded only for positioning, not routing.
- Config comes from `run.ini` `[routing]` (`algorithm`, `max_hops`, default
  `shortest_path` / 5) via `RoutingConfig`. `max_hops = 0` means unlimited; an
  over-long path makes the flow unroutable rather than falling back.
- An unknown `algorithm` silently uses shortest path here; `config-validator.cc`
  is what rejects it.
- An edge exists iff `LinkTable::Get(u,v).capacity_mbps > 0`.
- Results are one per input flow, same order. Zero-demand flows are unroutable
  with an empty path; self-flows (`src == dst`) are routable with empty path and
  0 delivered.
- Congestion scaling is per undirected edge using demanded (not delivered) Mbps,
  takes the minimum over hops, and never changes `latency_ms`.
- `NO_PREV` (`UINT32_MAX`) is the "no predecessor" sentinel in `prev[]`.

## Changing FlowResult or the algorithm list
Update together: `RoutingConfig` in `domain/mesh-config.h`, the `checkOneOf`
list in `config/config-validator.cc`, the dispatch in `FindPath`. `FlowResult`
is read by `io/metrics-writer`, `io/viz-writer`, and `rl/` (`rl-bridge`,
`rl-agent`), so field changes affect output columns and RL reward/observations.

## Tests
`make -C tests/unit/routing test` (from `scratch/mesh-sim/`; no ns-3).

## Dependencies
- Depends on: `domain/` (`mesh-config.h`, `link-result.h`), `eval/link-table.h`, `traffic/traffic-matrix.h` (`Flow`)
- Depended on by: `io/` (`metrics-writer.h`, `viz-writer.h`), `rl/` (`rl-bridge.h`, `rl-agent.h`), `sim.cc`
