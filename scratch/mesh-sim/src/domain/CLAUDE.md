# domain/

## Scope
Header-only POD config and result types. No ns-3 includes and no logic beyond
`MaxSpeedForType()`. Other layers depend on these headers; keep them light.

## Files
- **sim-config.h** -- `SimConfig` (root), `RlConfig`, `TimingInfo`
- **node-spec.h** -- `Position`, `Velocity`, `RandomWalkParams`, `Waypoint`, `NodeSpec`, `BuildingSpec`, `MaxSpeedForType()`
- **channel-config.h** -- `ChannelConfig`, `NyuChannelConfig`
- **mesh-config.h** -- `MeshConfig`, `TrafficConfig`, `RoutingConfig`
- **link-result.h** -- `LinkResult` (one per node pair per tick)

## Behavior to preserve
- Defaults in the structs must match the fallbacks in `ConfigLoader::Load` and
  `parseNodeSpec`/`parseBuildingSpec` (`config/config-loader.cc`); the loader
  passes its own default strings, so the two can drift silently.
- `RlConfig` "resolved fields" (`control_mode`, `controlled_indices`,
  `num_slots`, `decision_interval_ticks`, `num_ticks`) are never read from INI;
  `ApplyRlControl` fills them.
- `TimingInfo` is not loaded from any file; `sim.cc` fills it for `MetricsWriter`.
- `LinkResult` is one entry per unordered pair (`tx_id < rx_id`), not per direction.
- `blockage_enabled` and `beamforming_model` are parsed but read nowhere outside
  `config-loader.cc`.
- `sim-config.h` includes `src/jammer/jammer-spec.h`, which includes `node-spec.h`;
  do not make `node-spec.h` include `sim-config.h`.

## Adding or changing a SimConfig / sub-struct field
Update together: the struct here (with default and `///<` doc), `config/config-loader.cc`
(parse), `config/config-validator.cc` (checks), `io/run-logger.h` (the "Resolved
config" block in `run.log`), the key table in `config/config-loader.h`, and the
CLI override in `sim.cc` if one exists. Run `make -C tests/unit/config test`,
and check the "update together" table in `CONTRIBUTING.md`.

## Dependencies
- Depends on: standard library; `jammer/jammer-spec.h` (from `sim-config.h` only)
- Depended on by: `config/`, `cli/`, `eval/`, `io/`, `jammer/`, `routing/`, `rl/`, `setup/`, `traffic/`, `sim.cc`, `tests/unit/{config,routing,traffic}`
