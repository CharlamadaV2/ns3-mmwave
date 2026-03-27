# config/

## Scope
Loads `run.ini` + referenced `nodes.json` / `buildings.json` into a `SimConfig`.
No ns-3 headers -- can be compiled and tested independently of the ns-3 build.

## Files
- **config-loader.h/cc** -- `ConfigLoader::Load()` parses INI sections and JSON files into `SimConfig`
- **spec-parser.h/cc** -- `parseNodeSpec()`, `parseBuildingSpec()` deserialize JSON objects into domain PODs

## Dependencies
- Depends on: `domain/`, `util/`, `third_party/json.hpp`
- Depended on by: `sim.cc`
