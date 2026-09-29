# cli/

## Scope
Command-line parsing and pre-simulation setup (seed resolution, input archiving).
Does not touch ns-3 simulation APIs beyond `ns3::CommandLine`; `cli-parser.cc` is
the only file in `src/` that includes `ns3/command-line.h`.

## Files
- **cli-parser.h/cc** -- `CliArgs` struct, `ParseCommandLine()`, `ResolveSeeds()`, `ArchiveScenarioInputs()`

## Behavior to preserve
- This module only parses and validates flags. `sim.cc` applies the overrides
  (`--run-id`, `--output-dir`, `--rl-mode`, `--band`) to `SimConfig` after
  `ConfigLoader::Load`.
- Bad user input prints `Error: ...` to stderr and calls `std::exit(1)`; it
  does not throw. The exception is `ArchiveScenarioInputs`, which can throw
  `std::filesystem::filesystem_error`; `sim.cc` catches it.
- "Not set" sentinels in `CliArgs`: an empty string for string flags, `-1` for
  `--seed` / `--run-id`.
- Seed precedence: `--seeds` > `--seed` > `run.ini`. `parseSeedList`
  (`util/`) handles invalid tokens.

## Adding or changing a flag
Update together: `CliArgs` + `ParseCommandLine` (validation), the override
logic in `sim.cc`, the flag table in `cli-parser.h` and `README.md`, and the
top-level `README.md`. Also check the "update together" table in
`CONTRIBUTING.md`. There is no unit-test suite for `cli/`, because it needs
ns-3. Integration tests (`make integration` in `tests/`) exercise the real
binary.

## Dependencies
- Depends on: `domain/`, `util/` (`string-utils.h`), ns-3 core (`CommandLine`)
- Depended on by: `sim.cc`
