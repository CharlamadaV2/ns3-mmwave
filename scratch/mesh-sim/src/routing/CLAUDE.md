# routing/

## Scope
Mesh routing -- given a link table and traffic demands, routes flows over the mesh and computes delivered throughput.

## Files
- **mesh-router.h/cc** -- `FlowResult` struct, `MeshRouter` class. Implements Dijkstra shortest-path, max-throughput (widest path), and min-hop routing. Applies proportional-fairness congestion scaling when links are overloaded.

## Dependencies
- Depends on: `domain/`, `eval/link-table`, `traffic/traffic-matrix`
- Depended on by: `io/`, `sim.cc`

## Rules
- Routing must be deterministic and optimal given current link state, so the RL agent is only rewarded/punished for positioning decisions, not routing decisions.
