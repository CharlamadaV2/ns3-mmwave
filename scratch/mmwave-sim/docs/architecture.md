# Architecture

## Overview

`sim.cc` is the entry point. It orchestrates a linear pipeline:

```
CLI parsing --> Config loading --> NS-3 defaults --> Per-seed loop
```

Each seed iteration runs the full lifecycle: topology build, traffic install, viz setup, simulation run, metrics collection, teardown.

## Data flow

```
run.ini + nodes.json ----> ConfigLoader ----> SimConfig (POD)
                                                  |
CLI args ----> ParseCommandLine ----> CliArgs     |
                                                  v
                            +----------- Per Seed -----------+
                            | ApplyTraceFileDefaults         |
                            | ConfigureChannelDefaults       |
                            | CreateHelpers                  |
                            | TopologyBuilder::Build()       |
                            | TrafficSetup::Install()        |
                            | VizWriter::Start()             |
                            | Simulator::Run()               |
                            | VizWriter::Flush()             |
                            | MetricsWriter::Write()         |
                            | Simulator::Destroy()           |
                            +--------------------------------+
                                          |
                                          v
                            outputs/YYYY-MM/DD/HH-MM-SS/seed-N/
```

## Module ownership

### domain/ -- Data types
Pure POD structs. No logic, no ns-3 headers. Everything depends on this.
`SimConfig` is the central configuration struct that flows through the entire pipeline.

### cli/ -- Command-line interface
Parses args via `ns3::CommandLine`, resolves seeds, archives scenario inputs into the output directory for reproducibility.

### config/ -- Configuration loading
Reads `run.ini` via `ini-parser`, loads JSON specs via `spec-parser`. Produces a fully populated `SimConfig`. No ns-3 dependency -- can be tested independently.

### setup/ -- NS-3 wiring
The only layer that creates ns-3 simulation objects:
- **ns3-defaults** -- `Config::SetDefault` calls (protocol, tuning, trace paths)
- **topology-builder** -- nodes, mobility, buildings, EPC, devices, UE attachment
- **traffic-setup** -- UDP applications (DL/UL/both)

### io/ -- Output
- **viz-writer** -- periodic CSV snapshots during simulation (positions, links)
- **metrics-writer** -- post-run trace parsing, writes `summary.json`
- **progress-logger** -- stderr progress during simulation

### util/ -- Generic helpers
INI parsing, string manipulation, path resolution. No ns-3 or simulation knowledge.

### scripts/ -- Post-sim tooling
Python plotting CLI. Reads `summary.json` + `links.csv`, produces figures.

## Dependency layers

```
domain  <--  config, cli, io, setup, util
util    <--  config, io
config  <--  sim.cc
cli     <--  sim.cc
setup   <--  sim.cc
io      <--  sim.cc
```

`domain/` has no ns-3 headers, which allows config loading and metrics writing to compile and test without the ns-3 build system.

## Key design decisions

- **One seed loop in sim.cc**: each seed gets a fresh `Simulator` with full create/run/destroy lifecycle. This prevents state leaking between seeds.
- **Channel model selected at config time**: `channel_model` in `run.ini` chooses 3gpp or NYU. The selection is wired into ns-3 defaults before helper construction.
- **Scenario archiving**: input files are copied into the output directory so every result is self-contained and reproducible.
- **Positions override**: `--positions-override` allows external tools (e.g. RL agents) to inject node positions without modifying the base scenario.

## Calling order constraints

1. `ApplyProtocolDefaults()` -- once, before seed loop
2. `ApplyTuningDefaults(cfg)` -- once, before seed loop
3. Per seed:
   1. `ApplyTraceFileDefaults(output_dir)` -- redirects trace files
   2. `TopologyBuilder::ConfigureChannelDefaults(cfg)` -- sets channel model defaults
   3. `TopologyBuilder::CreateHelpers(cfg)` -- creates MmWaveHelper + EPC helper
   4. `TopologyBuilder::Build()` -- nodes, mobility, buildings, EPC, devices
   5. `TrafficSetup::Install()` -- UDP apps
   6. `VizWriter::Start()` -- opens files, connects traces, schedules ticks
   7. `Simulator::Run()` / `Simulator::Destroy()`
