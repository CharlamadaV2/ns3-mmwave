/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Plain POD types shared across all simulation components.
 * No ns-3 headers included here — keeps compilation fast
 * and allows use in post-run IO code without ns-3 linkage.
 */
#pragma once

#include <string>
#include <vector>

namespace mmwave_sim
{

// Coordinates are always stored as (x, y, z) triples in metres.
// For 2D scenarios: set z to a constant height.
// For 1D scenarios: additionally set y = 0 for all nodes.
// The coordinate system is arbitrary; relative geometry is what matters.
struct Position
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Velocity
{
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
};

struct RandomWalkParams
{
    double x_min = -100.0;
    double x_max = 100.0;
    double y_min = -100.0;
    double y_max = 100.0;
    double speed_mps = 1.5;
};

struct NodeSpec
{
    std::string id;
    std::string role;      // "enb" or "ue"
    std::string mobility;  // "fixed", "constant_velocity", "random_walk"
    Position    position;
    Velocity    velocity;
    RandomWalkParams random_walk;
};

struct BuildingSpec
{
    std::string id;
    double x_min = 0.0;
    double x_max = 1.0;
    double y_min = 0.0;
    double y_max = 1.0;
    double z_min = 0.0;
    double z_max = 1.0;
    std::string type      = "Residential";         // "Residential", "Office", "Commercial"
    std::string ext_walls = "ConcreteWithWindows";  // see ns3::Building::ExtWallsType
    int n_floors = 1;
};

struct ChannelConfig
{
    double      frequency_ghz    = 28.0;
    double      tx_power_dbm     = 30.0;  // eNB transmit power (dBm)
    std::string scenario         = "UMi";  // "UMi" or "UMa"
    bool        blockage_enabled = true;
};

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

    std::vector<NodeSpec>     nodes;
    std::vector<BuildingSpec> buildings;
};

}  // namespace mmwave_sim
