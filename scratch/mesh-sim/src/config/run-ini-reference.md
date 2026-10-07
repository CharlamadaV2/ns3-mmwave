@page src_config_run_ini_reference src/config/run-ini-reference

@brief Every run.ini key the simulator reads, with defaults and units.

These are the keys read by the simulator, plus the five `[rl]` selection keys
read by Python. Omitted keys use the defaults below; unknown keys are silently
ignored. Input-file paths are relative to `run.ini`. For action and reward
behavior, see the [RL bridge](@ref src_rl); for `jammers.json`, see the
[jammer model](@ref src_jammer). Command-line overrides are listed in the
[CLI guide](@ref src_cli). `--channel-query` mode reads the same file and
keys as a run (no query-specific keys); see the [query contract](@ref src_query).

## [scenario] and [output]

| Key | Default | Controls |
| --- | --- | --- |
| `name` | `unnamed` | Scenario label. |
| `seed` | `42` | Simulation random seed; CLI `--seed`/`--seeds` can override it. |
| `run_id` | `1` | Run identifier; CLI `--run-id` can override it. |
| `duration_s` | `10` | Simulated duration in seconds. |
| `warmup_s` | `0` | Initial seconds excluded from output metrics, not from RL rewards. |
| `tick_s` | `0.1` | Simulation time advanced per tick, in seconds. |
| `nodes_file` | `nodes.json` | Mesh-node input file. |
| `buildings_file` | blank | Optional building input file. |
| `jammers_file` | blank | Optional jammer input file. Jamming requires `band = sub-6`. |
| `[output] dir` | timestamped `outputs/` path | Output root; a relative value resolves from `run.ini`. CLI `--output-dir` overrides it. |
| `[output] viz_tick_ms` | `100` | Milliseconds between visualization CSV snapshots. |

## [channel]

`band` chooses a simulator mode; it is not inferred from `frequency_ghz`.
Jammers contribute interference only in `sub-6` mode. The jammer's
`target_freq` then filters the carrier frequency.

The band resolves as `--band` on the command line, then `[channel] band`, then
the default `mmwave`. `run.log` records the result as `band` and `band_source`
(`cli`, `run.ini`, or `default`).

| Key | Default | Controls |
| --- | --- | --- |
| `band` | `mmwave` | `mmwave` or `sub-6`; CLI `--band` overrides it. |
| `frequency_ghz` | `28` | Link carrier frequency in GHz. |
| `tx_power_dbm` | `30` | Mesh-node transmitter power in dBm. |
| `scenario` | `UMi` | Propagation setting: `UMi`, `UMa`, `RMa`, `InH`, or `InF`. |
| `channel_model` | `3gpp` | Propagation model: `3gpp` or `nyu`. |
| `condition_model` | `auto` | `auto` or `static_los` channel-condition selection. |
| `blockage_enabled` | `true` | Enable channel-model blockage. |
| `beamforming_model` | `svd` | Parsed and stored, but not used by the current link evaluator. |
| `amc_model` | `shannon` | SINR-to-capacity method: `shannon`, `table`, or `silvus`. |
| `bandwidth_mhz` | `400` | Link bandwidth used for noise floor and capacity, in MHz. |
| `noise_figure_db` | `5` | Receiver noise figure in dB. |
| `tx_array_gain_dbi` | `12` | Default mesh-node transmit antenna gain in dBi. |
| `rx_array_gain_dbi` | `12` | Default mesh-node receive antenna gain in dBi. |

## [nyu_channel]

These settings matter only when `[channel] channel_model = nyu`, except
`rf_bandwidth_mhz`, which is currently parsed but not applied.

| Key | Default | Controls |
| --- | --- | --- |
| `rf_bandwidth_mhz` | `800` | Stored NYU RF bandwidth; currently not applied. |
| `shadowing_enabled` | `true` | NYU shadow fading. |
| `pressure_mbar` | `1013.25` | Atmospheric pressure. |
| `humidity_pct` | `50` | Relative humidity. |
| `temperature_c` | `20` | Air temperature. |
| `rain_rate_mm_hr` | `0` | Rain rate. |
| `atmospheric_loss_enabled` | `false` | Molecular absorption loss. |
| `foliage_loss_enabled` | `false` | Foliage excess loss. |
| `foliage_loss_db_m` | `0.4` | Foliage loss per metre. |
| `o2i_loss_type` | `Low Loss` | Outdoor-to-indoor loss class (`Low Loss` or `High Loss`). |

## [traffic] and [routing]

Traffic is flow-level demand, not packets. `constant` keeps flows active,
`poisson` starts flows randomly, and `on_off` alternates sending and silence.
`all_pairs` connects every pair, `random_pairs` samples pairs, and `gateway`
connects each other node to one gateway. See the [traffic](@ref src_traffic)
and [routing](@ref src_routing) modules for the code that owns them.

| Key | Default | Controls |
| --- | --- | --- |
| `[traffic] model` | `constant` | `constant`, `poisson`, or `on_off` demand. |
| `demand_mbps` | `10` | Requested rate per active flow. |
| `arrival_rate_hz` | `1` | New flows per second in `poisson` mode. |
| `on_time_s`, `off_time_s` | `1`, `1` | Mean on/off phase lengths in `on_off` mode. |
| `holding_time_s` | `0` | Flow lifetime in seconds; `0` means no expiry. |
| `flow_topology` | `all_pairs` | `all_pairs`, `random_pairs`, or `gateway`. |
| `random_pair_count` | `3` | Number of flows with `random_pairs` topology. |
| `gateway_node_id` | blank | Gateway node ID, required with `gateway` topology. |
| `[routing] algorithm` | `shortest_path` | Inverse-capacity shortest path, widest-capacity path (`max_throughput`), or fewest hops (`min_hop`). |
| `max_hops` | `5` | Maximum hops per route; `0` means unlimited. |

## [rl]

With `enabled = false`, the remaining C++ RL keys do not control the run. The
jammer-only smoke scenario uses this setting: its `[rl]` block is inert in the
regression run. CLI `--rl-mode` turns RL on even if the file says `false`.

| Key | Default | Controls |
| --- | --- | --- |
| `enabled` | `false` | Enable the simulator's action/observation bridge. |
| `controlled_node_id` | blank | Legacy single-node selector; blank selects the last mesh node. |
| `controlled_nodes` | absent | Centralized selector: `all` or comma-separated node IDs. Its presence selects centralized mode; cannot coexist with `controlled_node_id`. |
| `max_controlled_nodes` | `0` | Centralized action positions; `0` sizes to the selected nodes. Maximum `64`. |
| `action_type` | `discrete` | Legacy: `discrete` or `continuous`; centralized requires `discrete`. |
| `action_profile` | `move_2d` | Centralized movement profile; only `move_2d` is implemented. |
| `decision_interval_s` | `0` | Seconds between centralized decisions; `0` means `tick_s`, otherwise an integer multiple of it. |
| `reward_type` | `throughput` | C++ reward: `throughput` or `all_links_los`; `mean_sinr` is a deprecated alias for the latter: it still runs, prints a warning on stderr, and `run.log` records it as `rl.reward_alias`. |
| `step_size_m` | `50` | Nominal discrete move per simulation tick, capped by node speed in centralized mode. |
| `arrival_threshold_m` | `1` | Legacy continuous-target arrival distance, in metres. |
| `x_min`, `x_max` | `-1000`, `2000` | Allowed east/west movement bounds, in metres. Each min must be below its max (same for y and z). |
| `y_min`, `y_max` | `-1000`, `1000` | Allowed north/south movement bounds, in metres. |
| `z_min`, `z_max` | `0`, `100` | Allowed height bounds, in metres; centralized `move_2d` does not change height. |

In centralized mode `all_links_los` is true only when every controlled node
has at least one peer link and all of them are line-of-sight. `action_set` and
`dimensions` are not keys; the loader ignores unknown keys silently.

The next five keys are read by Python's centralized RL environment, **not** by
the C++ simulator. Each also has a training CLI override; see
[Selecting observations, rewards, and telemetry](@ref src_rl_policy_inputs_selection).

| Key | Default | Controls |
| --- | --- | --- |
| `observation_preset` | `raw_links_v1` | Observation representation given to the policy. |
| `reward_components` | absent | Comma-separated Python reward components; absent uses C++ `reward_type`. |
| `reward_weights` | `1` per component | Comma-separated weight for each selected component. |
| `telemetry` | `none` | `none` or `steps` decision-level records. |
| `telemetry_every` | `1` | Record every Nth decision when `telemetry = steps`. |

## [baseline]

Selects a placement baseline. The C++ simulator reads only `algorithm`, and
never plans placements: a direct run with `geometric` or `optimization` exits
with an error unless the CLI passes `--rl-mode`, in which case it prints a
notice and uses the scenario layout. `[rl] enabled = true` alone does not
bypass this guard. Python (`scripts/baselines/config.py`) owns every key below
and rejects unknown keys and values; the binary ignores keys other than
`algorithm`.

| Key | Default | Controls |
| --- | --- | --- |
| `algorithm` | `none` | `none`, `geometric`, or `optimization`; an absent section or key means `none`, and the binary also treats a blank value as `none`. |
| `objective` | none | `coverage`, `balanced`, or `resilience`; required when active. |
| `application` | `initial_positions` | Only `initial_positions` is accepted. |
| `movable_nodes` | none | Distinct, non-empty comma-separated node IDs the planner may move, or `all`. Required when active. |
| `seed` | none | Optimizer seed (integer ≥ 0), required for `optimization`; independent of the simulation and planning seeds. |
| `max_iterations` | none | Optimizer iterations (integer > 0), required for `optimization`. |
| `waypoint_policy` | `reject` | `reject` or `translate` for selected waypoint nodes. |
| `mapping_file` | none | Optional baseline mapping JSON (version 2); absent means the `[rl]` bounds geofence and platforms by `node_type`. Relative paths resolve from `run.ini`. |
| `aerial_fixed_cost_m2`, `ground_fixed_cost_m2` | `150000` | Coverage-equivalent m² charged for moving a node of that platform at all (≥ 0). |
| `aerial_cost_m2_per_m`, `ground_cost_m2_per_m` | `500` / `100` | Additional m² per metre moved (≥ 0). |
| `aerial_max_displacement_m`, `ground_max_displacement_m` | none | Hard displacement cap in metres (> 0); absent means unlimited. |
| `candidate_grid_cells`, `coverage_grid_cells` | `400` | Target cell counts (integer ≥ 1) for candidate positions and coverage probes. |
| `grid_min_resolution_m` | `5.0` | Smallest grid cell edge in metres (> 0). |
| `coverage_probe_height_m` | `1.5` | Coverage probe receiver height in metres (≥ 0). |
| `coverage_probe_rx_gain_dbi` | `[channel] rx_array_gain_dbi` | Coverage probe receive gain (finite, ≥ 0). |
| `coverage_sinr_db` | `-6.7` | SINR at which a probe counts as covered. |
| `balanced_core_fraction` | `0.5` | Share of selected nodes that need two anchor links under `balanced` ([0, 1]). |

`gateway_node_id` and `rf_config` are no longer accepted: baselines are
gateway-free and candidates are scored by the simulator channel (see
[`scripts/baselines/README.md`](@ref scripts_baselines)).
