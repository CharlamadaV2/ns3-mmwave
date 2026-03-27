# domain/

## Scope
Pure POD data types shared across the entire project. No logic, no ns-3 dependencies.
This is the dependency firewall -- everything else depends on domain, but domain depends on nothing.

## Files
- **sim-config.h** -- `SimConfig`, `TrafficConfig`, `NetworkConfig`, `TimingInfo`
- **node-spec.h** -- `Position`, `Velocity`, `RandomWalkParams`, `NodeSpec`, `BuildingSpec`
- **channel-config.h** -- `ChannelConfig`, `NyuChannelConfig`

## Rules
- Never add ns-3 `#include` directives here.
- Keep types as plain structs with default-initialized fields.
