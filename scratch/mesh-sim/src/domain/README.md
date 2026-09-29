@page src_domain src/domain

@brief Plain-struct configuration and result types shared by every simulator layer.

The headers hold no logic and no ns-3 code. `SimConfig` is the root object: `ConfigLoader`
fills it from `run.ini` and the JSON files, `ValidateConfig` checks it, and every
other layer reads it. `LinkResult` carries one link's radio result for one tick.
Units are in each field's `///<` comment (dB, dBm, metres, seconds, MHz, Mbps).

## Module Layout

| File | Role |
|------|------|
| @ref sim-config.h "sim-config.h" | Root `SimConfig`, `RlConfig` (RL controller), `TimingInfo` (wall-clock run times). |
| @ref channel-config.h "channel-config.h" | `ChannelConfig` (frequency, power, gains, models) and `NyuChannelConfig`. |
| @ref mesh-config.h "mesh-config.h" | `TrafficConfig` (flow demand model and topology) and `RoutingConfig`. |
| @ref node-spec.h "node-spec.h" | `NodeSpec`, mobility parameters, `BuildingSpec`, `MaxSpeedForType()`. |
| @ref link-result.h "link-result.h" | `LinkResult`: distance, LOS, path loss, power, SINR, capacity, MCS. |

## Run

Nothing runs on its own; these are header-only. To confirm they compile with plain
`g++` (no ns-3), run from `scratch/mesh-sim/`:

```bash
make -C tests/unit/config test
```

## Output

Nothing is written to disk. `TimingInfo` and `LinkResult` values are consumed by
`io/` writers, which produce the metrics JSON and viz CSV files.

## Conventions

- **Sub-structs are declared in their own header** and held as members of
  `SimConfig`.
- **Every field has a default** matching the fallback in `config/config-loader.cc`.
- **Documentation.** Doxygen `/** */` blocks on types; fields use trailing `///<`
  with units and defaults.
- **Coordinates** are metres in an arbitrary local frame.
- **Strings select models** (for example `"3gpp"`, `"shannon"`); allowed values
  are enforced by `config/config-validator.cc`, not here.
- **No ns-3 includes.** `sim-config.h` includes `src/jammer/jammer-spec.h` for `JammerSpec`.

## Dependencies

- C++17 standard library only.
- `sim-config.h` needs `src/jammer/jammer-spec.h`, which needs `node-spec.h`.
