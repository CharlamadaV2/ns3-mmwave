# config/

Loads scenario configuration files (`run.ini` + JSON) and produces a fully populated `SimConfig` struct. No ns-3 headers -- can be compiled and tested independently.

## Files

### config-loader.h / config-loader.cc

Parses `run.ini` and its referenced JSON files into a `SimConfig`.

**Key functions:**
- `ConfigLoader::Load(run_config_path, positions_override_path)` -- reads INI sections (`[scenario]`, `[channel]`, `[nyu_channel]`, `[traffic]`, `[network]`, `[output]`), loads `nodes.json` and optional `buildings.json`, applies positions override if provided.

The `positions_override_path` parameter is the RL extension point -- it overwrites node positions by matching node IDs while leaving all other fields unchanged.

### spec-parser.h / spec-parser.cc

Deserializes JSON objects into domain POD types.

**Key functions:**
- `parseNodeSpec(json)` -- JSON object to `NodeSpec`
- `parseBuildingSpec(json)` -- JSON object to `BuildingSpec`

## Usage

Called once at startup in `sim.cc`:
```cpp
auto cfg = mmwave_sim::ConfigLoader::Load(args.run_config_path, args.positions_override_path);
```
