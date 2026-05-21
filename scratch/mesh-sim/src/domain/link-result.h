/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file link-result.h
 * @brief Per-link radio evaluation result POD type.
 *
 *
 * One @ref LinkResult is produced per ordered (tx, rx) node pair per
 * simulation tick.  The pair is *directed*: a @c LinkResult for
 * (tx=0, rx=1) is independent of (tx=1, rx=0) because path loss,
 * shadow fading, and blockage are evaluated separately for each direction.
 *
 * **Sentinel values**
 * @c rx_power_dbm and @c sinr_db are initialised to @c -999.0 dBm/dB to
 * indicate "no valid measurement" (e.g. the link was not evaluated this
 * tick, or the path loss exceeds the noise floor). Consumers should
 * treat any value below a reasonable threshold (e.g. -200 dB) as invalid.
 */
#pragma once

#include <cstdint>

namespace mesh_sim
{

/**
 * @brief Radio evaluation result for one directed link at one time step.
 *
 * Produced by the link evaluator and consumed by the metrics writer,
 * routing engine, and RL reward calculator.
 */
struct LinkResult
{
    uint32_t tx_id = 0;  ///< Transmitter index into the @ref SimConfig::nodes array.
    uint32_t rx_id = 0;  ///< Receiver index into the @ref SimConfig::nodes array.

    double distance_m = 0.0;   ///< 3-D Euclidean distance between tx and rx (m).
    bool   is_los     = false;  ///< @c true if the link is Line-of-Sight; @c false
                                ///<   for Non-Line-of-Sight.

    double path_loss_db  = 0.0;     ///< Total path loss in dB, including shadow fading
                                     ///<   (and foliage / atmospheric components when
                                     ///<   those NYU options are enabled).
    double rx_power_dbm  = -999.0;  ///< Received signal power in dBm after applying
                                     ///<   TX power, TX/RX array gains, and path loss.
                                     ///<   Initialised to -999.0 as a sentinel for
                                     ///<   "not evaluated / below noise floor".
    double sinr_db       = -999.0;  ///< Signal-to-Interference-plus-Noise Ratio in dB.
                                     ///<   Initialised to -999.0 as a sentinel.
    double capacity_mbps = 0.0;     ///< Link capacity in Mbps, computed by the configured
                                     ///<   AMC model (@c "shannon" or @c "table").

    uint32_t mcs_index = 0;  ///< MCS / CQI table index in the range [0, 14] selected
                              ///<   by the AMC model based on @c sinr_db.

    bool condition_from_buildings = false;  ///< @c true when the LOS/NLOS condition was
                                            ///<   determined deterministically by the
                                            ///<   buildings model rather than statistically.
};

}  // namespace mesh_sim