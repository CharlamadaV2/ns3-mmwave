@page src_routing src/routing

@brief Files for network routing protocols based on simulation config  

Flow-level network routing for the mesh simulation. Given the current link
quality matrix and a set of active traffic demands, finds a path for each
flow and returns the delivered throughput after congestion scaling.

## Output
The routing module produces result of route flows over the mesh.

## Module Layout

| File Name | Description |
| -- | -- |
@ref mesh-router.h "mesh-router.h" | FlowResult struct and MeshRouter class declaration.
@ref mesh-router.cc "mesh-router.cc" | Path-finding algorithms, congestion scaling, and latency model.

## Conventions 

- Flow-level only. There are no packets, queues, or transmission delays
- max_hops = 0 means unlimited. Any positive value discards paths that exceed it, leaving the flow unroutable rather than using a longer path.
- Comment convention will enforce JavaDoc like styling (/** */). 

## Dependencies

No dependencies