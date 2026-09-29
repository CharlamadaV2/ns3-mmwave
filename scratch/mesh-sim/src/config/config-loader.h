/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file config-loader.h
 * @brief Parses @c run.ini and the JSON files it references into a fully
 *        populated @ref SimConfig.
 *
 * No ns-3 headers. Missing keys take the defaults listed below; the loader
 * does not range-check values (see @ref ValidateConfig).
 *
 * **run.ini keys (section: key = default)**
 *
 * | Section          | Keys and defaults |
 * |------------------|-------------------|
 * | @c [scenario]    | @c name=unnamed, @c seed=42, @c run_id=1, @c duration_s=10.0 (s), @c warmup_s=0.0 (s), @c tick_s=0.1 (s), @c nodes_file=nodes.json, @c jammers_file (none), @c buildings_file (none) |
 * | @c [output]      | @c dir (empty = auto @c outputs/YYYY-MM/DD/HH-MM-SS under the mesh-sim root; relative paths are resolved against the run.ini directory), @c viz_tick_ms=100 (ms) |
 * | @c [channel]     | @c frequency_ghz=28.0, @c tx_power_dbm=30.0, @c scenario=UMi, @c channel_model=3gpp, @c condition_model=auto, @c blockage_enabled=true, @c beamforming_model=svd, @c amc_model=shannon, @c noise_figure_db=5.0, @c bandwidth_mhz=400.0, @c tx_array_gain_dbi=12.0, @c rx_array_gain_dbi=12.0, @c band (absent keeps @c SimConfig defaults @c "mmwave" / source @c "default"; present sets source @c "run.ini") |
 * | @c [nyu_channel] | @c rf_bandwidth_mhz=800.0, @c shadowing_enabled=true, @c pressure_mbar=1013.25, @c humidity_pct=50.0, @c temperature_c=20.0, @c rain_rate_mm_hr=0.0, @c atmospheric_loss_enabled=false, @c foliage_loss_enabled=false, @c foliage_loss_db_m=0.4, @c o2i_loss_type="Low Loss". Always read; only used when @c channel_model is @c "nyu" |
 * | @c [traffic]     | @c model=constant, @c demand_mbps=10.0, @c arrival_rate_hz=1.0, @c on_time_s=1.0, @c off_time_s=1.0, @c holding_time_s=0.0, @c flow_topology=all_pairs, @c random_pair_count=3, @c gateway_node_id (empty) |
 * | @c [routing]     | @c algorithm=shortest_path, @c max_hops=5 |
 * | @c [rl]          | @c enabled=false, @c controlled_node_id (empty), @c action_type=discrete, @c reward_type=throughput (@c "mean_sinr" is rewritten to @c "all_links_los" and the old name kept in @c reward_type_alias), @c step_size_m=50.0, @c arrival_threshold_m=1.0, @c x_min=-1000, @c x_max=2000, @c y_min=-1000, @c y_max=1000, @c z_min=0, @c z_max=100 (m) |
 * | @c [rl] (centralized) | @c controlled_nodes (empty), @c max_controlled_nodes=0, @c action_profile=move_2d, @c decision_interval_s=0.0. Mere presence of the @c controlled_nodes key sets @c controlled_nodes_set and selects centralized mode |
 *
 * **JSON files** (paths resolved against the run.ini directory; each file is a
 * top-level array)
 *
 * | File | Required | Per-entry fields and defaults |
 * |------|----------|-------------------------------|
 * | nodes file | yes | @c id (required, string), @c role=peer, @c mobility=fixed, @c node_type=drone, @c position{x,y,z}=0, @c velocity{vx,vy,vz}=0, @c random_walk{bounds{x_min=-100,x_max=100,y_min=-100,y_max=100}, speed_mps=1.5}, @c waypoints[{t,x,y,z}]=0, optional @c tx_array_gain_dbi / @c rx_array_gain_dbi (unset = use @c [channel] value) |
 * | jammers file | no | @c enabled=false, @c id="", @c type=constant, @c target_freq=[], @c tx_power_dbm=25.0, @c tx_array_gain_dbi=12.0, @c duty_cycle=1.0, @c max_range_m=0.0, @c beamwidth_deg=360.0, @c azimuth_deg=0.0, @c zenith_deg=0.0, @c position, @c velocity, @c random_walk, @c waypoints (same shape as nodes), @c intervals[{start=0,end=0}] |
 * | buildings file | no | @c id="", @c bounds{x_min=0,x_max=1,y_min=0,y_max=1,z_min=0,z_max=1}, @c type=Residential, @c ext_walls=ConcreteWithWindows, @c n_floors=1, @c n_rooms_x=1, @c n_rooms_y=1 |
 * | positions override | no | Object mapping node id to @c [x, y, z]; applied after the nodes file. Ids not in the nodes file are ignored |
 */
#pragma once

#include "src/domain/sim-config.h"
#include <string>

namespace mesh_sim
{

/**
 * @brief Static factory that loads a complete @ref SimConfig from disk.
 *
 */
class ConfigLoader
{
  public:
    /**
     * @brief Load a @ref SimConfig from a @c run.ini file and its referenced assets.
     *
     * @param run_config_path          Path to the @c run.ini file (required).
     * @param positions_override_path  Optional path to a positions-override
     *                                 JSON file. Positions are applied after
     *                                 @c nodes.json is loaded. Pass an empty
     *                                 string (the default) to skip.
     * @return Fully populated @ref SimConfig ready for validation and use.
     * @throws std::runtime_error  If @c channel_model is neither @c "3gpp"
     *                             nor @c "nyu"; if the nodes file cannot be
     *                             opened; if a @c jammers_file or
     *                             @c buildings_file is specified but cannot
     *                             be opened; or if
     *                             @c positions_override_path is non-empty but
     *                             the file cannot be opened.
     * @throws std::invalid_argument / std::out_of_range  From @c std::stoul /
     *                             @c std::stod on a non-numeric or overflowing
     *                             INI value.
     * @throws nlohmann::json::exception  On malformed JSON, a node without
     *                             @c id, or a positions-override entry that
     *                             is not a 3-element numeric array.
     *
     * Steps: parse @c run.ini; fill the sections above; load the nodes,
     * jammers, and buildings files in that order; apply the positions
     * override. Does not validate ranges or enum values (except @c channel_model). When @c [output]
     * @c dir is empty the timestamped directory is derived from the local
     * clock and assumes the run.ini lives three levels below the mesh-sim
     * root (for example @c inputs/baselines/foo/run.ini). Reads files only;
     * creates no directories.
     */
    static SimConfig Load(const std::string& run_config_path,
                          const std::string& positions_override_path = "");
};

}  // namespace mesh_sim