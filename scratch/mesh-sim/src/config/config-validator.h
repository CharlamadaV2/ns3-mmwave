/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file config-validator.h
 * @brief Post-load validation of a fully populated @ref SimConfig.
 *
 * | Domain     | Rules enforced                                                             |
 * |------------|----------------------------------------------------------------------------|
 * | Timing     | @c duration_s > 0; @c tick_s > 0; @c tick_s <= @c duration_s; @c warmup_s >= 0; @c warmup_s < @c duration_s |
 * | Nodes      | At least 2 nodes; each node's @c mobility and @c node_type must be a recognised value; waypoint nodes require >= 2 waypoints with strictly monotonically increasing times and a non-negative first timestamp |
 * | Channel    | @c frequency_ghz > 0; @c bandwidth_mhz > 0; @c channel_model ∈ {3gpp, nyu}; @c scenario ∈ {UMi, UMa, RMa, InH, InF}; both array gains >= 0 |
 * | Traffic    | @c model ∈ {constant, poisson, on_off}; @c flow_topology ∈ {all_pairs, random_pairs, gateway}; @c demand_mbps > 0; @c gateway topology requires a non-empty @c gateway_node_id that matches a node; @c random_pairs requires @c random_pair_count > 0 |
 * | Routing    | @c algorithm ∈ {shortest_path, max_throughput, min_hop}                    |
 * | RL         | (when enabled) @c action_type ∈ {discrete, continuous}; @c reward_type ∈ {throughput, mean_sinr}; @c step_size_m > 0 (discrete); @c arrival_threshold_m > 0 (continuous); @c x_min < @c x_max; @c y_min < @c y_max; @c controlled_node_id must match a node if set |
 * | Buildings  | Each building's @c x_min < @c x_max; @c y_min < @c y_max; @c z_min < @c z_max |
 */
#pragma once

#include "src/domain/sim-config.h"

#include <string>
#include <vector>

namespace mesh_sim
{

/**
 * @brief Result of a @ref ValidateConfig call.
 *
 * On success @c errors is empty and @ref ok returns @c true.
 * On failure @c errors contains one human-readable string per problem found;
 * all problems are reported together so the user can fix them in one pass.
 */
struct ValidationResult
{
    std::vector<std::string> errors;  ///< One entry per validation failure;
                                      ///<   empty on success.

    /**
     * @brief Return @c true if no validation errors were found.
     * @return @c true when @c errors is empty, @c false otherwise.
     */
    bool ok() const { return errors.empty(); }
};

/**
 * @brief Validate a @ref SimConfig for invalid or inconsistent values.
 *
 * Intended to be called immediately after @ref ConfigLoader::Load, before
 * any ns-3 objects are constructed.  All checks are independent — every
 * rule is evaluated even if earlier ones have already failed — so the
 * returned @ref ValidationResult accumulates the full set of problems in
 * one call.
 *
 * @param cfg  Fully loaded @ref SimConfig to validate.
 * @return @ref ValidationResult containing all errors found, or an empty
 *         error list if the config is valid.
 */
ValidationResult ValidateConfig(const SimConfig& cfg);

}  // namespace mesh_sim
