# traffic/

## Scope
Traffic demand generation. Models flow-level demands between node pairs (no packets).
Supports constant-rate, Poisson arrival, and on-off (bursty) traffic models.

## Files
- **traffic-matrix.h/cc** -- `Flow` struct, `TrafficMatrix` class. Generates and manages traffic demands. `Initialize()` creates initial flows, `Tick()` advances state (on/off transitions, flow arrivals/departures, holding times).

## Dependencies
- Depends on: `domain/`, ns-3 `RandomVariableStream` (for Poisson/on-off RNG)
- Depended on by: `routing/`, `sim.cc`
