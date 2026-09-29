@page src_config src/config

@brief Loads `run.ini` and its JSON files into a `SimConfig` and checks it for errors.

Three small pieces, none of which need ns-3. The loader reads the scenario
files, the validator reports every problem it finds, and the RL resolver turns
the `[rl]` settings into controlled-node slots and tick counts. `sim.cc` calls
all three at startup; you only edit this directory to add or change a
`run.ini` key or a validation rule.

## Module Layout

| File | Role |
|------|------|
| @ref config-loader.h "config-loader.h" | `ConfigLoader::Load()`; header lists every `run.ini` section, key, default, and JSON field. |
| @ref config-loader.cc "config-loader.cc" | INI and JSON parsing, output-directory defaulting, positions override. |
| @ref config-validator.h "config-validator.h" | `ValidationResult` and `ValidateConfig()`; header lists all rules. |
| @ref config-validator.cc "config-validator.cc" | The checks; every rule runs even after an earlier failure. |
| @ref rl-control.h "rl-control.h" | `ResolveRlControl()`, `ApplyRlControl()`, `ComputeTickCount()`, `ControlledStartPosition()`. |
| @ref rl-control.cc "rl-control.cc" | Selector, slot-count, cadence, and start-position resolution. |
| `CLAUDE.md` | Scope and change notes for AI assistants. |

## Setup

Nothing to install for the unit tests beyond a C++17 compiler (`g++`).

## Run

There is no executable here. Run the unit suite from `scratch/mesh-sim/`:

```bash
make -C tests/unit/config test
```

The suite compiles these three files plus `src/util/` with plain `g++`. Do not
use the top-level `make test` on Linux (its lint step needs macOS `xcrun`).

## Input Files

Paths inside `run.ini` are resolved against the `run.ini` directory.

| Input | Read by | Notes |
|-------|---------|-------|
| `run.ini` | `ConfigLoader::Load` | Sections `[scenario]`, `[output]`, `[channel]`, `[nyu_channel]`, `[traffic]`, `[routing]`, `[rl]`. |
| nodes file (`[scenario] nodes_file`, default `nodes.json`) | `ConfigLoader::Load` | Required. |
| `[scenario] jammers_file` | `ConfigLoader::Load` | Optional. |
| `[scenario] buildings_file` | `ConfigLoader::Load` | Optional. |
| positions override JSON | `ConfigLoader::Load` | Optional second argument; node id to `[x, y, z]`. |

## Output

| Step | Where |
|------|-------|
| `ConfigLoader::Load` | Returns a `SimConfig`; throws `std::runtime_error` for a bad `channel_model` or an unopenable file. Creates no files. |
| `ValidateConfig` | Returns a `ValidationResult`; `errors` holds one string per problem. |
| `ResolveRlControl` | Returns an `RlControlResolution`; `ApplyRlControl` copies it into `cfg.rl`. |

## Conventions

- Validation rules only append to `errors`; they never throw or stop early.
- Loaders fill `SimConfig` fields directly and do not range-check.
- Comments use Doxygen `/** */` blocks; full docs live in the headers.

## Dependencies

- C++17, `third_party/json.hpp` (nlohmann/json).
- `src/domain/` (config and spec types) and `src/util/` (`ini-parser.h`, `string-utils.h`).
