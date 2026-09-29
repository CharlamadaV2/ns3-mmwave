@page src_cli src/cli

@brief Command-line parsing and pre-simulation setup for mesh-sim.

## Quick guide

This module handles the simulator's command-line flags (`ParseCommandLine`),
decides which random seeds to run (`ResolveSeeds`), and copies the scenario
input files into the output directory (`ArchiveScenarioInputs`). `sim.cc`
calls all three at startup. You use it indirectly every time you run the
simulator; you only edit it to add or change a flag.

Shortest run (the user builds the binary; do not build from here):

```bash
./ns3 run scratch/mesh-sim/sim -- --run-config=scratch/mesh-sim/inputs/calfex/06-25/1227-1413/run.ini
```

## Run / how to access the files

Run from the ns-3 root. Flags use the form `--name=value`. Only
`--run-config` is required.

```bash
./ns3 run scratch/mesh-sim/sim -- \
  --run-config=<path/to/run.ini> \
  --band=sub-6 \
  --seeds=1,2,3 \
  --output-dir=<dir>
```

If a flag is missing or invalid, the parser prints `Error: ...` to `stderr`
and exits with code 1 before anything else runs.

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--run-config=<path>` | yes | none | Path to the scenario `run.ini` (the file, not the directory). Must exist. |
| `--seeds=<list>` | no | seed from `run.ini` | Comma-separated seeds, e.g. `1,2,3`. Each seed runs independently in its own `seed-<N>/` subdirectory. |
| `--seed=<int>` | no | seed from `run.ini` | Single-seed override. Ignored when `--seeds` is also given. |
| `--run-id=<int>` | no | `run_id` from `run.ini` | Overrides the `run_id` value. Negative means not set. |
| `--output-dir=<path>` | no | `[output] dir` from `run.ini` / auto-timestamped | Overrides the batch output root. |
| `--band=<mmwave\|sub-6>` | no | `[channel] band` in `run.ini`, else `mmwave` | Radio band. Any other value is an error. Jammer interference applies only on `sub-6`. |
| `--positions-override=<path>` | no | none | JSON file that patches node positions after `nodes.json` is loaded. Must exist if given. Used by the RL controller. |
| `--debug-links` | no | `false` | Verbose per-link evaluation logging. |
| `--rl-mode` | no | `false` | Enables the RL bridge (observations and actions as JSON on stdin/stdout). |

`--debug-links` and `--rl-mode` can be given with no value.

Seed order of precedence: `--seeds`, then `--seed`, then `run.ini`. Empty
items in `--seeds` are skipped; a non-numeric item is an error.

## Files in this directory

| File | Role |
|------|------|
| @ref cli-parser.h "cli-parser.h" | `CliArgs` struct and declarations of the three public functions, with full Doxygen docs. |
| @ref cli-parser.cc "cli-parser.cc" | Implementations. The only file in `src/` that includes `ns3/command-line.h`. |
| `CLAUDE.md` | Scope and dependency notes for AI assistants. |
| `README.md` | This page. |

## Output

`ParseCommandLine` and `ResolveSeeds` write nothing. `ArchiveScenarioInputs`
creates `<output-dir>/inputs/` and copies every regular file from the
`run.ini` directory into it. The rest of the layout is written by other
modules (`sim.cc`, `src/io/`):

```
<output-dir>/
  run.log          # resolved config summary
  inputs/          # copy of the scenario directory (run.ini, nodes.json, ...)
  seed-1/
    positions.csv  links.csv  rx-power.csv  mcs.csv  flows.csv  routes.csv
  seed-2/
```

Re-running with the archived `inputs/` reproduces the original scenario.

## Conventions and gotchas

- **Hard exits.** `ParseCommandLine` and `ResolveSeeds` print to `stderr` and
  call `std::exit(1)`; they do not throw. `sim.cc` therefore does not handle
  these cases.
- **Files are only checked for existence.** `--run-config` and
  `--positions-override` are not validated here; `ConfigLoader` does that.
- **Archiving copies the whole directory.** Every regular file next to
  `run.ini` is copied (subdirectories are not), and existing files in
  `inputs/` are overwritten. Large files in the scenario directory get copied
  too.
- **Pass a directory in `--run-config`.** A bare filename such as
  `run.ini` has an empty parent path, and `ArchiveScenarioInputs` would throw
  `std::filesystem::filesystem_error` (caught in `sim.cc`, exit 1).
- **`--output-dir` is applied in `sim.cc`,** not in this module. It replaces
  `cfg.output_dir`, and `inputs/` is created under the replaced path.
- **Band resolution** is `--band` > `[channel] band` > `mmwave`. The
  resolution itself happens in `sim.cc`; this module only validates the value.

## Dependencies

- `ns3/command-line.h` (ns-3 core), used only in `cli-parser.cc`.
- `src/domain/sim-config.h` (`SimConfig`, read for `cfg.seed`).
- `src/util/string-utils.h` (`parseSeedList`).
- C++17 `<filesystem>`.
- Depended on by: `sim.cc`.
