# inputs/

## Scope
Scenario definitions only. Each scenario is a directory under `scenarios/` containing configuration files that drive a simulation run.

## Per-scenario files
- **run.ini** -- Simulation parameters (channel, traffic, routing, mesh, output settings)
- **nodes.json** -- Node positions, roles, and mobility models
- **buildings.json** -- (optional) Building geometry for urban/obstruction scenarios

## Rules
- No output data here -- outputs go to `outputs/`.
- Node IDs must be unique within a scenario.
- All nodes have `role: "peer"` (no eNB/UE distinction in mesh-sim).
