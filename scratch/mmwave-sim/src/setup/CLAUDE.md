# setup/

## Scope
The only layer that creates ns-3 simulation objects -- topology, traffic, and protocol defaults.

## Files
- **topology-builder.h/cc** -- Creates nodes, mobility models, buildings, EPC, mmWave devices. Static methods `ConfigureChannelDefaults()` and `CreateHelpers()` must be called in order.
- **traffic-setup.h/cc** -- Installs UDP client/sink pairs based on traffic direction (dl/ul/both).
- **ns3-defaults.h/cc** -- Centralized `Config::SetDefault` calls for protocols, tuning, and trace paths.

## Dependencies
- Depends on: `domain/`, ns-3 modules (mmwave, internet, network, buildings)
- Depended on by: `sim.cc`, `io/viz-writer`

## Rules
- `ConfigureChannelDefaults()` must be called before `CreateHelpers()`.
- `ApplyTraceFileDefaults()` must be called per-seed before helper construction.
