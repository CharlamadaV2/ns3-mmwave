/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file sim-config.h
 * @brief Top-level simulation configuration and runtime metadata POD types.
 *
 * @ref SimConfig is the single root object that flows through the entire
 * simulation.  It is produced by @ref ConfigLoader::Load, validated by
 * @ref ValidateConfig, and then passed by const-reference to every
 * subsystem constructor.
 * @ref TimingInfo is intentionally separate from @ref SimConfig because it
 * is not loaded from any file — it is populated by @c sim.cc after the step
 * loop completes and written to the output JSON by @c MetricsWriter.
 */
#pragma once

#include "channel-config.h"
#include "mesh-config.h"
#include "node-spec.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace mesh_sim
{

/**
 * @brief Wall-clock timing metadata for one simulation run.
 *
 * Not loaded from any configuration file. Populated by @c sim.cc
 * immediately before and after the main step loop, then passed to
 * @c MetricsWriter which converts the @c time_point values to ISO-8601
 * strings in the output JSON.
 */
struct TimingInfo
{
    std::chrono::system_clock::time_point start;  ///< Wall-clock time at sim start.
    std::chrono::system_clock::time_point end;    ///< Wall-clock time at sim end.
    double elapsed_s = 0.0;                        ///< Total wall-clock duration in seconds
                                                   ///<   (@c end - @c start as a double).
};

/**
 * @brief Reinforcement-learning controller configuration.
 *
 * When @c enabled is @c false every other field is ignored. When @c true the
 * sim exchanges JSON observations and actions with an external RL agent via
 * stdin/stdout at each tick (activated by the @c --rl-mode CLI flag).
 *
 * **Action types**
 * | @c action_type   | Description                                                     |
 * |------------------|-----------------------------------------------------------------|
 * | @c "discrete"    | Left/right/up/down/stop offsets of @c step_size_m each tick.   |
 * | @c "continuous"  | Absolute (x, y) velocity vector; capped by @ref MaxSpeedForType. |
 *
 * **Reward types**
 * | @c reward_type    | Description                                          |
 * |-------------------|------------------------------------------------------|
 * | @c "throughput"   | Sum of routed flow demand across all active paths.   |
 * | @c "mean_sinr"    | Mean SINR in dB across all evaluated links.          |
 */
struct RlConfig
{
    bool        enabled             = false;         ///< Enable RL mode when @c true.
    std::string controlled_node_id;                  ///< ID of the node the RL agent controls.
                                                     ///<   Empty string means the last node
                                                     ///<   in @ref SimConfig::nodes.
    std::string action_type         = "discrete";    ///< Action space type:
                                                     ///<   @c "discrete" or @c "continuous".
    std::string reward_type         = "throughput";  ///< Reward signal:
                                                     ///<   @c "throughput" or @c "mean_sinr".
    double step_size_m          = 50.0;   ///< Per-tick displacement in metres for the
                                           ///<   @c "discrete" action type.
    double arrival_threshold_m  =  1.0;   ///< Distance in metres below which the controlled
                                           ///<   node is considered to have "arrived" at a
                                           ///<   target. Used only by @c "continuous" mode.

    // ---- Movement bounding box ---------------------------------------------
    double x_min = -1000.0;  ///< Western boundary of the controlled node's allowed area (m).
    double x_max =  2000.0;  ///< Eastern boundary of the controlled node's allowed area (m).
    double y_min = -1000.0;  ///< Southern boundary of the controlled node's allowed area (m).
    double y_max =  1000.0;  ///< Northern boundary of the controlled node's allowed area (m).
};

/**
 * @brief Root simulation configuration aggregating all subsystem parameters.
 *
 * Produced by @ref ConfigLoader::Load from @c run.ini and the JSON files it
 * references. Validated by @ref ValidateConfig before any ns-3 objects are
 * constructed. Passed by const-reference to every subsystem constructor.
 *
 * **Timing parameters**
 * - @c tick_s must be ≤ @c duration_s.
 * - @c warmup_s samples are excluded from all output metrics but the
 *   simulation still runs for the full @c duration_s.
 * - @c viz_tick_ms controls how frequently CSV position/link snapshots are
 *   written; it is independent of @c tick_s.
 */
struct SimConfig
{
    std::string scenario_name;        ///< Human-readable scenario label (from @c [scenario] name).
    uint32_t    seed       = 42;      ///< RNG seed passed to ns-3 @c RngSeedManager.
    uint32_t    run_id     = 1;       ///< Run identifier written to output metadata.
    double      duration_s = 10.0;    ///< Total simulated time in seconds.
    double      warmup_s   = 0.0;     ///< Seconds to exclude from output metrics.
                                       ///<   Must be < @c duration_s.
    double      tick_s     = 0.1;     ///< Simulation time step in seconds (100 ms default).
    std::string output_dir;           ///< Root output directory. Auto-generated as a
                                       ///<   timestamped path under @c outputs/ when empty
                                       ///<   in @c run.ini (see @ref ConfigLoader::Load).

    ChannelConfig channel;  ///< Radio and channel model parameters.
    MeshConfig    mesh;     ///< Traffic generation and routing parameters.

    uint32_t viz_tick_ms = 100;  ///< Interval in milliseconds at which CSV snapshots
                                  ///<   (positions, link results) are written to disk.

    RlConfig rl;  ///< Reinforcement-learning controller parameters.
                   ///<   Ignored when @c rl.enabled is @c false.

    std::vector<NodeSpec>     nodes;      ///< All simulation nodes, in index order.
                                           ///<   Node IDs assigned by @ref TrafficMatrix
                                           ///<   and the link evaluator are indices into
                                           ///<   this vector.
    std::vector<BuildingSpec> buildings;  ///< Optional building obstacles used for
                                           ///<   deterministic LOS/NLOS classification.
                                           ///<   Empty when no @c buildings_file is set.
};

}  // namespace mesh_sim