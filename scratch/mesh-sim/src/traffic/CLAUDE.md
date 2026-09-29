# traffic/

## Scope
Flow-level demand generation (no packets). Owns the list of `Flow`s and updates
it once per tick for the constant, Poisson, and on-off models.

## Files
- **traffic-matrix.h/cc** -- `Flow` struct; `TrafficMatrix` with `Initialize()` (all_pairs / random_pairs / gateway), `Tick()` (expiry, model update, removal), `GetActiveFlows()`, `GetDemand()`; private `MakeFlow`, `TickOnOff`, `TickPoisson`, `Init*`.

## Behavior to preserve
- Config comes from `run.ini` `[traffic]` via `TrafficConfig` (`model`,
  `flow_topology`, `demand_mbps`, `arrival_rate_hz`, `on_time_s`, `off_time_s`,
  `holding_time_s`, `random_pair_count`, `gateway_node_id`); `tick_s` from `SimConfig`.
- `end_time_s == 0` means permanent. Only flows with `!active && end_time_s > 0`
  are erased; on-off OFF flows stay in the list with `in_on_phase = false`.
- `GetActiveFlows()` returns everything, including inactive/OFF flows.
  `MeshRouter` does the filtering; do not pre-filter here.
- Reproducibility comes from the ns-3 global seed/run set in `sim.cc` before
  construction; the RNG streams are not seeded in this class. Reordering RNG
  draws changes traffic for a given seed.
- Poisson flows also get initial flows from `Initialize()`. With
  `holding_time_s = 0` arrivals never expire.
- `gateway_node_id` is tried as a numeric index first, then as a node ID; the
  validator only accepts an ID match.
- `Initialize()` throws `std::runtime_error` on an unknown topology.

## Changing a traffic option
Update together: `TrafficConfig` in `domain/mesh-config.h`, the loader in
`config/config-loader.cc`, the checks in `config/config-validator.cc`, and
`README.md`. `Flow` fields are consumed by `routing/mesh-router`.

## Tests
`make -C tests/unit/traffic test` (from `scratch/mesh-sim/`; uses
`tests/unit/traffic/ns3-traffic-stub.h`, no ns-3).

## Dependencies
- Depends on: `domain/` (`sim-config.h`), ns-3 `random-variable-stream.h` (stubbed in unit tests)
- Depended on by: `routing/` (`Flow`), `sim.cc`
