# cli/

## Scope
Command-line parsing and pre-simulation setup (seed resolution, input archiving).
Does not touch ns-3 simulation APIs beyond `ns3::CommandLine`.

## Files
- **cli-parser.h/cc** -- `CliArgs` struct, `ParseCommandLine()`, `ResolveSeeds()`, `ArchiveScenarioInputs()`

## Dependencies
- Depends on: `domain/`
- Depended on by: `sim.cc`
