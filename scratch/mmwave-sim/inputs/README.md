# inputs/

Scenario definitions for the mmwave-sim harness. Each scenario is a directory under `scenarios/` containing the configuration files needed for a simulation run.

## File format

### run.ini

INI file with the following sections:

| Section | Key fields |
|---|---|
| `[scenario]` | `name`, `seed`, `run_id`, `duration_s`, `warmup_s`, `nodes_file`, `buildings_file` |
| `[channel]` | `frequency_ghz`, `tx_power_dbm`, `scenario`, `channel_model`, `blockage_enabled` |
| `[nyu_channel]` | `rf_bandwidth_mhz`, `shadowing_enabled`, atmosphere/foliage params (ignored for 3gpp) |
| `[traffic]` | `direction` (dl/ul/both), `packet_size_bytes`, `inter_packet_interval_us` |
| `[network]` | `backhaul_data_rate`, `backhaul_delay_ms` |
| `[output]` | `viz_tick_ms`, `pcap_enabled`, `trace_level` |

### nodes.json

Array of node specs. Each node has:
- `id` -- unique identifier (e.g. `enb0`, `ue0`)
- `role` -- `"enb"` or `"ue"`
- `mobility` -- `"fixed"`, `"constant_velocity"`, or `"random_walk"`
- `position` -- `{x, y, z}` in metres
- `velocity` -- `{vx, vy, vz}` in m/s (for `constant_velocity`)
- `random_walk` -- `{x_min, x_max, y_min, y_max, speed_mps}` (for `random_walk`)

### buildings.json (optional)

Array of building specs with axis-aligned bounding boxes (`x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max`), plus `type`, `ext_walls`, and `n_floors`.

## Available scenarios

| Scenario | Description |
|---|---|
| `uav-hover-baseline` | Two hovering UAVs at 60m altitude, 250m separation. Baseline link performance. |
| `uav-flyaway-range` | UAV flying away at constant velocity. Maps throughput vs distance. |
| `building-bypass` | UE moving past a building. Tests NLOS-to-LOS transition. |
| `building-two-ues` | Two UEs blocked by a building. Multi-UE blockage scenario. |
| `two-ues-one-approaching` | One mobile UE approaching eNB. Dynamic range test. |
| `ue-moving-away` | Single UE moving away from eNB. Range degradation test. |
