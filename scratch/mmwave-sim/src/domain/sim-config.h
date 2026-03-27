/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Top-level simulation configuration and runtime metadata POD types.
 * No ns-3 headers included — keeps compilation fast and allows use in
 * post-run IO code without ns-3 linkage.
 */
#pragma once

#include "channel-config.h"
#include "node-spec.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace mmwave_sim
{

struct TrafficConfig
{
    std::string direction                 = "dl";   // "dl", "ul", or "both"
    uint32_t    packet_size_bytes         = 1400;
    double      inter_packet_interval_us  = 100.0;
    double      app_start_offset_s        = 0.1;
    // 0 = unlimited (simulation is time-bounded by duration_s).
    // Set > 0 to cap the total packets sent per source, e.g. for delivery tests.
    uint32_t    max_packets               = 0;
};

struct NetworkConfig
{
    // Backhaul link between EPC PGW and the remote traffic host.
    // These model the core network connection — not the radio link.
    // Default values make the backhaul a non-bottleneck; reduce them to
    // study the effect of core network congestion or latency.
    std::string backhaul_data_rate = "100Gb/s";  // ns-3 DataRate string
    double      backhaul_delay_ms  = 10.0;        // one-way backhaul latency
};

// Runtime-only timing metadata (not loaded from config).
// Populated by sim.cc after Simulator::Run() completes.
// MetricsWriter converts the time_points to ISO-8601 when writing JSON.
struct TimingInfo
{
    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;
    double elapsed_s = 0.0;
};

struct SimConfig
{
    std::string scenario_name;
    uint32_t    seed       = 42;
    uint32_t    run_id     = 1;
    double      duration_s = 2.0;
    double      warmup_s   = 0.1;
    std::string output_dir;           // auto-generated if empty

    ChannelConfig channel;
    TrafficConfig traffic;
    NetworkConfig network;

    // Visualisation output
    uint32_t viz_tick_ms = 100;  // snapshot interval for positions/links CSVs

    // When true, PCAP captures are written to <output_dir>/pcap/
    bool pcap_enabled = false;

    // Trace output level: "full" (all ns3 traces), "minimal" (PHY + RLC only),
    // "none" (no file traces). "minimal" keeps what MetricsWriter needs.
    std::string trace_level = "full";

    std::vector<NodeSpec>     nodes;
    std::vector<BuildingSpec> buildings;
};

}  // namespace mmwave_sim
