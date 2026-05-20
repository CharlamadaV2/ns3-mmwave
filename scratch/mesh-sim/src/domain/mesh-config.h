/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

/** @brief
 * Mesh-specific configuration POD types: traffic generation and routing.
 * No ns-3 headers included.
 */
#pragma once

#include <cstdint>
#include <string>

namespace mesh_sim
{

// How traffic demands are generated between node pairs.
struct TrafficConfig
{
    std::string model = "constant";  // "constant", "poisson", "on_off"

    // -- constant model --
    double demand_mbps = 10.0;

    // -- poisson model --
    double arrival_rate_hz = 1.0;  // flow arrivals per second (network-wide)

    // -- on_off model (bursty traffic) --
    double on_time_s  = 1.0;  // mean ON duration
    double off_time_s = 1.0;  // mean OFF duration

    // -- holding time (all models) --
    // 0 = flow lasts the entire simulation.
    double holding_time_s = 0.0;

    // -- flow topology --
    std::string flow_topology     = "all_pairs";  // "all_pairs", "random_pairs", "gateway"
    uint32_t    random_pair_count = 3;             // for "random_pairs"
    std::string gateway_node_id   = "";            // for "gateway"
};

struct RoutingConfig
{
    // "shortest_path":  Dijkstra on link capacity graph.
    // "max_throughput": widest-path (maximize bottleneck capacity).
    // "min_hop":        fewest hops (ignore link quality).
    std::string algorithm = "shortest_path";

    uint32_t max_hops = 5;  // 0 = unlimited
};

struct MeshConfig
{
    TrafficConfig traffic;
    RoutingConfig routing;
};

}  // namespace mesh_sim
