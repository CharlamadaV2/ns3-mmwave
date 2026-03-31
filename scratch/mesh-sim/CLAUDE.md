# mesh-sim

Lightweight time-stepped mmWave mesh simulator built on ns3-mmwave.
All nodes are identical peers (no eNB/UE distinction). Uses ns-3 propagation
models as library functions. Uses `Simulator::Stop()`+`Run()` as a controlled
time-stepper each tick but does not use the ns-3 event loop for control flow.
C++ for simulation; Python only for post-sim plotting.
Entry point: `sim.cc`.

## Commit guidelines

- Use conventional commit format: `type(scope): short summary` (e.g. `feat(mesh-sim):`, `fix(mesh-sim):`).
- Keep the summary under 72 characters; use the body for the *why*, not the *what*.
- Group changes into logical commits -- one concern per commit. Prefer many focused commits over one monolithic commit.
- Types: `feat` (new functionality), `fix` (bug fix), `refactor` (restructure, no behavior change), `docs` (docs only), `chore`/`build` (tooling/CI/build).
- Scope should reflect the module being changed (e.g. `mesh-sim`, `link-evaluator`, `mesh-router`).
- Never add `Co-Authored-By` or `Signed-off-by` lines.

## Code guidelines

- **File scope awareness**: Understand what each directory owns before editing.
  Don't pile unrelated logic into one file. Each subdirectory has a CLAUDE.md
  describing its scope -- read it first.
- **Comment discipline**: Only comment when something isn't obvious. Keep
  comments short and descriptive. Don't restate what the code already says.
- Never run `./ns3 build` or `./ns3 run` -- the user handles builds.
- When adding new source files, update `CMakeLists.txt`'s source list.

## Architecture

- `src/` is split by concern: cli, config, domain, eval, io, routing, rl, setup, traffic, util
- `domain/` types are pure POD with no ns-3 dependency -- everything else depends on them
- `eval/` wraps ns-3 propagation models to compute per-link SINR and capacity
- `traffic/` generates demand matrices (no packets -- flow-level abstraction)
- `routing/` routes flows over the mesh using the link table
- `setup/` is the only layer that creates ns-3 objects (nodes, mobility, buildings, propagation models)
- `io/` writes CSV and JSON output compatible with the mmwave-sim GUI
- `config/` has no ns-3 headers -- can be compiled/tested independently
- All C++ code lives in namespace `mesh_sim`

## Dependency layers

```
domain  <--  config, cli, eval, io, routing, setup, traffic, util
util    <--  config, io
config  <--  sim.cc
cli     <--  sim.cc
setup   <--  sim.cc
eval    <--  sim.cc
traffic <--  sim.cc
routing <--  sim.cc
io      <--  sim.cc
rl      <--  sim.cc
```

## Input/output conventions

- Scenarios live in `inputs/scenarios/<name>/` with `run.ini` + `nodes.json` + optional `buildings.json`
- Outputs go to `outputs/YYYY-MM/DD/HH-MM-SS/seed-N/`
- Scenario inputs are archived into each output run for reproducibility

## Key differences from mmwave-sim

- No EPC, RRC, MAC, HARQ, RLC, PDCP -- direct propagation model calls only
- `Simulator::Stop()`+`Run()` advances the clock each tick; for-loop remains the master
- All nodes are peers (no base station / UE roles)
- Evaluates all N*(N-1)/2 links per tick (not just eNB-UE pairs)
- Traffic is flow-level demands, not packet-level
- Routing layer computes multi-hop paths over the mesh
