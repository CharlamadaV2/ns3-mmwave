# config/

## Scope
Turns `run.ini` + `nodes.json` / `jammers.json` / `buildings.json` into a `SimConfig`,
validates it, and resolves the `[rl]` control selectors. No ns-3 headers; compiled
and tested standalone.

## Files
- **config-loader.h/cc** -- `ConfigLoader::Load(run_ini, positions_override="")`. Header holds the full run.ini key/default table and JSON field table.
- **config-validator.h/cc** -- `ValidationResult`, `ValidateConfig()`.
- **rl-control.h/cc** -- `ResolveRlControl()`, `ApplyRlControl()`, `ComputeTickCount()`, `ControlledStartPosition()`.

## Behavior to preserve
- The validator collects all errors instead of failing fast; new rules must only `push_back`.
- The loader does not range-check. The one exception is `channel_model`, which throws
  `std::runtime_error` in `Load`. Bad numbers in the INI throw `std::stoul`/`std::stod` exceptions.
- `[rl] controlled_nodes` selects centralized mode by key presence (`controlled_nodes_set`),
  even when the value is empty (then it is an error).
- `ResolveRlControl` is the single source of RL slot, cadence, and tick rules. The validator
  reports its errors; `sim.cc` calls `ApplyRlControl`, which throws if errors exist.
- Legacy mode truncates the tick count; centralized mode rounds with a 1e-6 tolerance.
- `[channel] band` absent keeps `SimConfig`'s `"mmwave"` / `"default"`; `sim.cc` applies `--band`.
- Empty `[output] dir` derives the timestamped path from the run.ini location (three levels
  below the mesh-sim root).
- `[rl] reward_type = mean_sinr` is rewritten to `all_links_los`, old name kept in `reward_type_alias`.

## Adding a run.ini key
Update together: the `SimConfig` field in `domain/`, the read in `Load`, a rule in
`ValidateConfig` if it has a constrained range, the tables in `config-loader.h` /
`config-validator.h` and `README.md`, a case in `tests/unit/config/`, and the top-level
`README.md` and `CONTRIBUTING.md` "update together" row.

## Tests
`make -C tests/unit/config test` (from `scratch/mesh-sim/`).

## Dependencies
- Depends on: `domain/`, `util/` (`ini-parser.h`, `string-utils.h`), `third_party/json.hpp`
- Depended on by: `sim.cc`, `setup/` (`rl-control.h`), `tests/unit/config`
