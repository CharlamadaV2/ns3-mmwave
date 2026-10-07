@page src_cli src/cli

@brief Command-line parsing and pre-simulation setup for mesh-sim.

This module handles the simulator's command-line flags (`ParseCommandLine`),
decides which random seeds to run (`ResolveSeeds`, `ResolveQuerySeed`), and
copies the scenario input files into the output directory
(`ArchiveScenarioInputs`). `sim.cc` calls them at startup. You use it
indirectly every time you run the simulator; you only edit it to add or
change a flag.

## Module Layout

| File | Role |
|------|------|
| @ref cli-parser.h "cli-parser.h" | `CliArgs` struct and declarations of the public functions (including `ResolveQuerySeed` for `--channel-query`), with full Doxygen docs. |
| @ref cli-parser.cc "cli-parser.cc" | Implementations. The only file in `src/` that includes `ns3/command-line.h`. |
| `CLAUDE.md` | Scope and dependency notes for AI assistants. |
| `README.md` | This page. |

## Run

Shortest run (the user builds the binary; do not build from here):

```bash
./ns3 run scratch/mesh-sim/sim -- --run-config=scratch/mesh-sim/inputs/baselines/p0-smoke/run.ini
```

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
| `--channel-query` | no | `false` | Serves candidate-layout channel queries on stdin/stdout instead of running ([contract](@ref src_query)). Seeds must resolve to exactly one; `--output-dir` is ignored with a note; no output directory, archive, or `run.log` is written. |

`--debug-links`, `--rl-mode` and `--channel-query` can be given with no value.

Seed order of precedence: `--seeds`, then `--seed`, then `run.ini`. Empty
items in `--seeds` are skipped; a non-numeric item is an error.

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

## Conventions

- **Hard exits.** `ParseCommandLine` and `ResolveSeeds` print to `stderr` and
  call `std::exit(1)`; they do not throw. `sim.cc` therefore does not handle
  these cases.
- **Files are only checked for existence.** `--run-config` and
  `--positions-override` are not validated here; `ConfigLoader` does that.
- **Archiving copies the whole directory.** Every regular file next to
  `run.ini` is copied (subdirectories are not), and existing files in
  `inputs/` are overwritten. Large files in the scenario directory get copied
  too.
- **Include a directory in `--run-config`.** A bare filename such as
  `run.ini` has an empty parent path, and `ArchiveScenarioInputs` would throw
  `std::filesystem::filesystem_error` (caught in `sim.cc`, exit 1).
- **`--output-dir` is applied in `sim.cc`,** not in this module. It replaces
  `cfg.output_dir`, and `inputs/` is created under the replaced path.
- **Band resolution** is `--band` > `[channel] band` > `mmwave`. The
  resolution itself happens in `sim.cc`; this module only validates the value.
- **`--channel-query` seed check needs the config.** More than one resolved
  seed is rejected by `ResolveQuerySeed` (`Error: --channel-query needs exactly
  one seed`, exit 1), which `src/query/` calls after the config is loaded.

## Dependencies

- `ns3/command-line.h` (ns-3 core), used only in `cli-parser.cc`.
- `src/domain/sim-config.h` (`SimConfig`, read for `cfg.seed`).
- `src/util/string-utils.h` (`parseSeedList`).
- C++17 `<filesystem>`.
- Depended on by: `sim.cc`, `src/query/` (`ResolveQuerySeed`).
