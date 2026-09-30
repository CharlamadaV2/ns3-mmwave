/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file link-result.h
 * @brief Per-link radio evaluation result POD type.
 *
 * One @ref mesh_sim::LinkResult is produced per unordered node pair (i < j) per
 * simulation tick by @c LinkEvaluator::EvaluateAll, with @c tx_id = i and
 * @c rx_id = j. The evaluator treats the link as symmetric, so there is no
 * separate (j, i) result.
 *
 * **Default values**
 * @c rx_power_dbm and @c sinr_db default to @c -999.0 as a "not evaluated"
 * marker. @c LinkEvaluator::Evaluate overwrites every field, so results it
 * returns never carry the marker.
 */
#pragma once

#include <cstdint>

namespace mesh_sim
{

/**
 * @brief Radio evaluation result for one link at one time step.
 *
 * Produced by the link evaluator and consumed by the link table, metrics
 * writer, routing, and the RL bridge.
 */
struct LinkResult
{
    uint32_t tx_id = 0;  ///< Index of the first node of the pair in @ref SimConfig::nodes.
    uint32_t rx_id = 0;  ///< Index of the second node of the pair in @ref SimConfig::nodes.

    double distance_m = 0.0;   ///< 3-D Euclidean distance between the two nodes (m).
    bool   is_los     = false;  ///< @c true if the link is Line-of-Sight, @c false if
                                ///<   Non-Line-of-Sight.

    double path_loss_db  = 0.0;     ///< Path loss in dB: the propagation-model loss, floored at
                                     ///<   free-space loss for a distance of at least 1 m.
    double rx_power_dbm  = -999.0;  ///< Received power in dBm: TX power - path loss + TX and RX
                                     ///<   array gains.
    double sinr_db       = -999.0;  ///< Signal-to-Interference-plus-Noise Ratio in dB. Equals SNR
                                     ///<   when no jammer is active; with an active jammer it is
                                     ///<   clamped to be >= 0 dB.
    double capacity_mbps = 0.0;     ///< Link capacity in Mbps from the configured AMC model
                                     ///<   (@c "shannon", @c "table", or @c "silvus").

    uint32_t mcs_index = 0;  ///< Modulation and coding scheme index in [0, 14] chosen from
                              ///<   @c sinr_db (Silvus table when @c amc_model is @c "silvus").

    bool condition_from_buildings = false;  ///< @c true when LOS/NLOS came from the buildings model
                                            ///<   (or @c static_los) rather than the statistical
                                            ///<   model. Set from the run-wide setting, not per link.
};

}  // namespace mesh_sim
