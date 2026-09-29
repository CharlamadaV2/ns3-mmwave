/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file rl-control.h
 * @brief Pure resolution of the @c [rl] control selectors into slots, cadence,
 *        and tick counts.
 *
 * Single source of truth shared by @ref ValidateConfig (which reports the
 * errors) and @c sim.cc (which applies the resolved fields). No ns-3 headers.
 */
#pragma once

#include "src/domain/node-spec.h"
#include "src/domain/sim-config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mesh_sim
{

/**
 * @brief Result of a @ref ResolveRlControl call.
 *
 * On success @c errors is empty and the resolved fields are meaningful.
 * On failure @c errors holds one human-readable string per problem found and
 * the resolved fields must not be applied.
 */
struct RlControlResolution
{
    std::vector<std::string> errors;           ///< One entry per resolution failure;
                                               ///<   empty on success.
    std::string control_mode;                  ///< @c "legacy" or @c "centralized".
    std::vector<uint32_t> controlled_indices;  ///< Slot-order node indices
                                               ///<   (legacy: exactly one entry).
    uint32_t num_slots = 0;                    ///< Slot count @c M (legacy: 1).
    uint32_t decision_interval_ticks = 1;      ///< Decision cadence @c k in ticks.
    uint32_t num_ticks = 0;                    ///< Loop tick count for this mode.

    /**
     * @fn RlControlResolution::ok
     * @brief Return @c true if no resolution errors were found.
     * @return @c true when @c errors is empty, @c false otherwise.
     */
    bool ok() const { return errors.empty(); }
};

/**
 * @fn ResolveRlControl
 * @brief Resolve the RL control mode, controlled slots, cadence, and tick count.
 *
 * @param cfg  Loaded configuration; only the @c rl, @c nodes, @c jammers,
 *             @c duration_s, and @c tick_s fields are read.
 * @return Resolution with @c errors empty on success. When @c cfg.rl.enabled
 *         is false it returns @c control_mode "legacy" with no errors and all
 *         other fields at their defaults (@c num_ticks = 0).
 * @throws std::invalid_argument Not in practice: waypoint nodes without
 *         waypoints are reported as errors before start positions are read.
 *
 * Mode is centralized when @c cfg.rl.controlled_nodes_set is true, else legacy.
 * Legacy: one slot, the node named by @c controlled_node_id (falling back to
 * the last node if empty or unmatched), ticks = floor(duration/tick).
 * Centralized: parses @c controlled_nodes (@c "all" or comma-separated ids),
 * rejects mixing with @c controlled_node_id, non-discrete @c action_type, and
 * any @c action_profile other than @c move_2d; sizes slots from
 * @c max_controlled_nodes (0 = number of controlled nodes, at most 64); turns
 * @c decision_interval_s (0 = @c tick_s) into an integer tick multiple; and
 * checks each start position lies inside the [rl] bounds. Ticks use a
 * 1e-6-tolerant rounding in this mode. Pure; no I/O.
 */
RlControlResolution ResolveRlControl(const SimConfig& cfg);

/**
 * @fn ApplyRlControl
 * @brief Copy a successful resolution into @c cfg.rl's resolved fields.
 *
 * @param cfg  Configuration to update (@c control_mode, @c controlled_indices,
 *             @c num_slots, @c decision_interval_ticks, @c num_ticks).
 * @param r    Resolution from @ref ResolveRlControl.
 * @throws std::invalid_argument if @p r.errors is not empty.
 */
void ApplyRlControl(SimConfig& cfg, const RlControlResolution& r);

/**
 * @fn ComputeTickCount
 * @brief Return the loop tick count for a duration/tick pair.
 *
 * @param duration_s  Run length in seconds.
 * @param tick_s      Tick length in seconds.
 * @param robust      If true, round the ratio to the nearest integer when
 *                    within 1e-6 of it, else floor; if false, plain truncation.
 * @return Tick count, or 0 if either input is non-finite or <= 0, the ratio is
 *         below 1, or it exceeds the @c uint32_t range.
 */
uint32_t ComputeTickCount(double duration_s, double tick_s, bool robust);

/**
 * @fn ControlledStartPosition
 * @brief Return a node's control start position.
 *
 * @param spec  Node specification.
 * @return First waypoint for @c "waypoint" mobility, else @c spec.position.
 * @throws std::invalid_argument if the node uses waypoint mobility but has no
 *         waypoints.
 */
Position ControlledStartPosition(const NodeSpec& spec);

}  // namespace mesh_sim
