@page src_cli src/cli

@brief Command-line parsing and pre-simulation setup for mesh-sim.  

This module contains `sim.cc` argument handling, seed resolution, 
and input archiving into standalone functions with no ns-3 types in 
their public signatures (except `ns3::CommandLine`inside the `.cc`).

## Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--run-config=<path>` | yes | — | Path to `run.ini` for the scenario. |
| `--seeds=<list>` | no | `run.ini` value | Comma-separated seed list. Each seed runs as an independent simulation with its own output subdirectory. |
| `--seed=<int>` | no | `run.ini` value | Single-seed override. Ignored when `--seeds` is also set. |
| `--run-id=<int>` | no | `run.ini` value | Override `run_id` metadata written to output files. |
| `--output-dir=<path>` | no | auto-timestamped | Override the batch output root. |
| `--positions-override=<path>` | no | — | JSON file that patches node positions before the simulation starts. Used by the RL controller. |
| `--debug-links` | no | `false` | Enable verbose per-link evaluation logging. |
| `--rl-mode` | no | `false` | Enable RL mode — the sim exchanges observations and actions via stdin/stdout JSON. |


## Output

Each seed produces its own subdirectory under the batch root:

```
<output-dir>/
  inputs/          # snapshot of all scenario input files (run.ini, nodes.json)
  seed-1/
    links.csv
    mcs.csv
    rx-power.csv
    positions.csv
    run.log
  seed-2/
```

The `inputs/` directory is written by `ArchiveScenarioInputs` before any seed
runs, so re-running with the archived files reproduces the original results
exactly.


## Module Layout

| File Name | Description |
| -- | -- |
 @ref cli-parser.h "cli-parser.h"   | CliArgs struct + three public function declarations
 @ref cli-parser.cc "cli-parser.cc"  | implementations; the only file in src/ that includes ns3/command-line.h

## Conventions

- **Hard exits on bad input.** `ParseCommandLine` and `ResolveSeeds` call
  `std::exit(1)` with a message to `stderr` rather than throwing.  This
  matches ns-3 convention for fatal startup errors and means `sim.cc` never
  needs to handle missing-flag cases.
- **Overwrite-safe archiving.** `ArchiveScenarioInputs` uses
  `copy_options::overwrite_existing` so re-running into the same output
  directory is safe.


## Dependencies

Depends on ns3 framework