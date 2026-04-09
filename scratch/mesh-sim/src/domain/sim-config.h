/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Top-level simulation configuration and runtime metadata POD types.
 * No ns-3 headers included — keeps compilation fast and allows use in
 * post-run IO code without ns-3 linkage.
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

// Runtime-only timing metadata (not loaded from config).
// Populated by sim.cc after the step loop completes.
// MetricsWriter converts the time_points to ISO-8601 when writing JSON.
struct TimingInfo
{
    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;
    double elapsed_s = 0.0;
};

struct RlConfig
{
    bool        enabled               = false;
    std::string controlled_node_id;                 // empty → last node
    std::string action_type           = "discrete"; // "discrete" or "continuous"
    std::string reward_type           = "throughput"; // "throughput" or "mean_sinr"
    double      step_size_m           = 50.0;       // discrete only: offset for left/right
    double      arrival_threshold_m   = 1.0;        // continuous only: "arrived" distance
    double      x_min                 = -1000.0;
    double      x_max                 =  2000.0;
    double      y_min                 = -1000.0;
    double      y_max                 =  1000.0;
};

struct SimConfig
{
    std::string scenario_name;
    uint32_t    seed       = 42;
    uint32_t    run_id     = 1;
    double      duration_s = 10.0;  // total simulated time
    double      warmup_s   = 0.0;   // skip this many seconds for metrics
    double      tick_s     = 0.1;   // time step interval (100 ms)
    std::string output_dir;

    ChannelConfig channel;
    MeshConfig    mesh;

    uint32_t viz_tick_ms = 100;  // how often to write CSV snapshots

    RlConfig rl;

    std::vector<NodeSpec>     nodes;
    std::vector<BuildingSpec> buildings;
};

}  // namespace mesh_sim
