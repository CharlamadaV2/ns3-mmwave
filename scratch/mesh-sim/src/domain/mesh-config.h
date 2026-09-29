/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file mesh-config.h
 * @brief Mesh-specific configuration POD types: traffic generation and routing.
 *
 * These types model *flow-level* demand only; there are no packets.
 * The traffic model determines how many Mbps node A "wants" to send to
 * node B at each tick; the routing algorithm determines which path carries
 * that demand.
 */
#pragma once

#include <cstdint>
#include <string>

namespace mesh_sim
{

/**
 * @brief Flow-level traffic demand configuration.
 *
 * Three traffic models are supported, selected by @c model:
 *
 * | @c model     | Behaviour                                                         |
 * |--------------|-------------------------------------------------------------------|
 * | @c "constant"| All flows are active from the start at @c demand_mbps until @c holding_time_s expires (never, if 0). |
 * | @c "poisson" | New flows arrive at @c arrival_rate_hz (network-wide Poisson process). When @c holding_time_s > 0 each flow lasts an exponentially distributed time with that mean; otherwise it is permanent. |
 * | @c "on_off"  | Each flow alternates between ON (sending at @c demand_mbps) and OFF phases. Phase durations are drawn from exponential distributions with means @c on_time_s and @c off_time_s respectively. |
 *
 * Three flow topologies are supported, selected by @c flow_topology:
 *
 * | @c flow_topology   | Behaviour                                                  |
 * |--------------------|------------------------------------------------------------|
 * | @c "all_pairs"     | One flow per unordered node pair (N*(N-1)/2 flows total). |
 * | @c "random_pairs"  | @c random_pair_count flows with uniformly random src/dst.  |
 * | @c "gateway"       | Every non-gateway node sends to the gateway node.          |
 */
struct TrafficConfig
{
    std::string model = "constant";  ///< Traffic model: @c "constant", @c "poisson",
                                     ///<   or @c "on_off".

    // ---- constant model ----------------------------------------------------
    double demand_mbps = 10.0;  ///< Per-flow demand in Mbps (must be > 0). Used by all
                                 ///<   models; for @c "on_off" it is the rate while ON.

    // ---- poisson model -----------------------------------------------------
    double arrival_rate_hz = 1.0;  ///< Network-wide flow arrival rate (flows/s).
                                    ///<   Used only when @c model is @c "poisson".

    // ---- on_off model ------------------------------------------------------
    double on_time_s  = 1.0;  ///< Mean ON-phase duration in seconds (exponential).
                               ///<   Used only when @c model is @c "on_off".
    double off_time_s = 1.0;  ///< Mean OFF-phase duration in seconds (exponential).
                               ///<   Used only when @c model is @c "on_off".

    // ---- holding time (all models) -----------------------------------------
    double holding_time_s = 0.0;  ///< Flow lifetime in seconds. @c 0 means the flow lasts the
                                   ///<   entire simulation. For @c "poisson" it is the mean of an
                                   ///<   exponential draw per flow; for the other models every
                                   ///<   flow lasts exactly this long.

    // ---- flow topology -----------------------------------------------------
    std::string flow_topology     = "all_pairs";  ///< How node pairs are assigned flows:
                                                   ///<   @c "all_pairs", @c "random_pairs",
                                                   ///<   or @c "gateway".
    uint32_t    random_pair_count = 3;             ///< Number of randomly chosen src/dst
                                                   ///<   pairs. Used only when
                                                   ///<   @c flow_topology is @c "random_pairs".
    std::string gateway_node_id   = "";            ///< Node ID of the gateway. Required (and must match a
                                                   ///<   node ID) when @c flow_topology is @c "gateway";
                                                   ///<   ignored otherwise.
};

/**
 * @brief Routing algorithm configuration.
 *
 * The routing algorithm is applied once per tick to the current link-capacity
 * graph produced by the link evaluator.
 *
 * | @c algorithm         | Strategy                                                |
 * |----------------------|---------------------------------------------------------|
 * | @c "shortest_path"   | Dijkstra on the inverse-capacity-weighted graph (default). |
 * | @c "max_throughput"  | Widest-path: maximise the bottleneck link capacity.      |
 * | @c "min_hop"         | Minimum hop count, ignoring link quality entirely.       |
 */
struct RoutingConfig
{
    std::string algorithm = "shortest_path";  ///< Routing algorithm: @c "shortest_path",
                                               ///<   @c "max_throughput", or @c "min_hop".
    uint32_t max_hops = 5;  ///< Maximum number of hops allowed per path; longer
                             ///<   paths are treated as unroutable. @c 0 means unlimited.
};

/**
 * @brief Aggregates all mesh-layer configuration.
 *
 * Held as @ref SimConfig::mesh and populated from the @c [traffic] and
 * @c [routing] sections of @c run.ini by @ref ConfigLoader::Load.
 */
struct MeshConfig
{
    TrafficConfig traffic;  ///< Flow-level traffic demand parameters.
    RoutingConfig routing;  ///< Path-selection algorithm parameters.
};

}  // namespace mesh_sim