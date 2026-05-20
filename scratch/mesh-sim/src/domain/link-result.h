/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

/** @brief
 * Per-link evaluation result POD type.
 * One instance per ordered pair (tx, rx) per time step.
 * No ns-3 headers included.
 */
#pragma once

#include <cstdint>
/** @brief
*/
namespace mesh_sim
{
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
struct LinkResult
{
    uint32_t tx_id = 0;           // index into node list
    uint32_t rx_id = 0;
    double   distance_m = 0.0;    // 3D Euclidean distance
    bool     is_los = false;      // true = Line of Sight
    double   path_loss_db = 0.0;  // total path loss including shadow fading
    double   rx_power_dbm = -999.0;
    double   sinr_db = -999.0;
    double   capacity_mbps = 0.0; // Shannon or AMC-based capacity
    uint32_t mcs_index = 0;       // CQI/MCS table index (0-14)
    bool     condition_from_buildings = false; // true = deterministic (buildings)
};

}  // namespace mesh_sim
