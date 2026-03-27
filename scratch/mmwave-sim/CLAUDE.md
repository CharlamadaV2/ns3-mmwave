# mmwave-sim

mmWave 5G simulation harness built on ns3-mmwave.
C++ for simulation orchestration; Python only for post-sim plotting.
Entry point: `sim.cc`.

## Commit guidelines

- Use conventional commit format: `type(scope): short summary` (e.g. `feat(mmwave-sim):`, `refactor(mmwave-helper):`).
- Keep the summary under 72 characters; use the body for the *why*, not the *what*.
- Group changes into logical commits — one concern per commit. Prefer many focused commits over one monolithic commit.
- Types: `feat` (new functionality), `fix` (bug fix), `refactor` (restructure, no behavior change), `docs` (docs only), `chore`/`build` (tooling/CI/build).
- Scope should reflect the module being changed (e.g. `mmwave-sim`, `mmwave-helper`, `ns3`).
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

- `src/` is split by concern: cli, config, domain, io, setup, util
- `domain/` types are pure POD with no ns-3 dependency -- everything else depends on them
- `setup/` is the only layer that creates ns-3 simulation objects
- `io/` runs post-simulation (metrics-writer) or via scheduled callbacks (viz-writer)
- `config/` has no ns-3 headers -- can be compiled/tested independently
- All C++ code lives in namespace `mmwave_sim`

## Dependency layers

```
domain  <--  config, cli, io, setup, util
util    <--  config, io
config  <--  sim.cc
cli     <--  sim.cc
setup   <--  sim.cc
io      <--  sim.cc
```

## Input/output conventions

- Scenarios live in `inputs/scenarios/<name>/` with `run.ini` + `nodes.json` + optional `buildings.json`
- Outputs go to `outputs/YYYY-MM/DD/HH-MM-SS/seed-N/`
- Scenario inputs are archived into each output run for reproducibility
