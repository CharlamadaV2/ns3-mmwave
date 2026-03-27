# io/

## Scope
Output writers -- post-simulation metrics and in-simulation visualization snapshots.

## Files
- **metrics-writer.h/cc** -- Parses ns-3 trace files after `Simulator::Run()`, computes per-UE aggregates, writes `summary.json`. No ns-3 simulation deps.
- **viz-writer.h/cc** -- Periodic CSV snapshots of positions and link state during simulation. Uses ns-3 headers (`NodeContainer`, `ChannelConditionModel`).
- **progress-logger.h** -- Stderr progress output during simulation. Header-only.

## Dependencies
- Depends on: `domain/`, `util/`
- viz-writer and progress-logger additionally depend on ns-3 headers
- Depended on by: `sim.cc`

## Rules
- metrics-writer must stay ns-3-free (stdlib + nlohmann/json only).
