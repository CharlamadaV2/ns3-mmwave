@page src_traffic src/traffic

@brief Component for generating network traffic flow per tick for simulation  

The traffic matrix advances the flow population each tick according to the configured model
and hands the current list to the routing layer.


## Output

Returns vector of active flows per tick in simulation.

## Module Layout

| File Name | Description |
| -- | -- |
@ref traffic-matrix.h "traffic-matrix.h" | Flow struct and TrafficMatrix class declaration.
@ref traffic-matrix.cc "traffic-matrix.cc" | Traffic model logic: initialisation, per-tick update, Poisson generation, on-off state machine.

## Conventions 

- Traffic patterns are deterministic for a given seed — the same seed always produces the same flow arrivals, random pairs, and on-off phase durations.
- Updates the 
- Purely tracking the traffic flow between two nodes.

## Dependencies

No dependencies