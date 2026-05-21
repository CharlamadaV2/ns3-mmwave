/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file config-loader.h
 * @brief Parses @c run.ini and all referenced JSON files into a fully
 *        populated @ref SimConfig.
 *
 *
 * | Section        | Key fields loaded                                                    |
 * |----------------|----------------------------------------------------------------------|
 * | @c [scenario]  | @c name, @c seed, @c run_id, @c duration_s, @c warmup_s, @c tick_s, @c nodes_file, @c buildings_file |
 * | @c [output]    | @c dir, @c viz_tick_ms                                               |
 * | @c [channel]   | @c frequency_ghz, @c tx_power_dbm, @c scenario, @c channel_model, @c blockage_enabled, @c beamforming_model, @c amc_model, @c noise_figure_db, @c bandwidth_mhz, @c tx_array_gain_dbi, @c rx_array_gain_dbi |
 * | @c [nyu_channel] | All @ref NyuChannelConfig fields; only applied when @c channel_model is @c "nyu" |
 * | @c [traffic]   | @c model, @c demand_mbps, @c arrival_rate_hz, @c on_time_s, @c off_time_s, @c holding_time_s, @c flow_topology, @c random_pair_count, @c gateway_node_id |
 * | @c [routing]   | @c algorithm, @c max_hops                                            |
 * | @c [rl]        | @c enabled, @c controlled_node_id, @c action_type, @c reward_type, @c step_size_m, @c arrival_threshold_m, @c x_min/x_max/y_min/y_max |
 */
#pragma once

#include "src/domain/sim-config.h"
#include <string>

namespace mesh_sim
{

/**
 * @brief Static factory that loads a complete @ref SimConfig from disk.
 *
 * The class has no state — all functionality is exposed through the single
 * static method @ref Load.
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
     *                             nor @c "nyu"; if @c nodes.json cannot be
     *                             opened; if a @c buildings_file is specified
     *                             but cannot be opened; or if
     *                             @c positions_override_path is non-empty but
     *                             the file cannot be opened.
     */
    static SimConfig Load(const std::string& run_config_path,
                          const std::string& positions_override_path = "");
};

}  // namespace mesh_sim