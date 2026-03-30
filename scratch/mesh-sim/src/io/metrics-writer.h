/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * MetricsWriter: accumulates per-tick metrics in-memory during the step loop,
 * then writes summary.json after the simulation completes.
 *
 * No ns-3 headers -- stdlib and nlohmann/json only.
 */
#pragma once

#include "src/domain/sim-config.h"
#include "src/eval/link-table.h"
#include "src/routing/mesh-router.h"

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace mesh_sim
{

class MetricsWriter
{
  public:
    explicit MetricsWriter(const SimConfig& cfg);

    void SetTiming(const TimingInfo& t);

    void AccumulateTick(double time_s,
                        const LinkTable& links,
                        const std::vector<FlowResult>& flows,
                        uint32_t numNodes);

    void Write() const;

  private:
    struct NodeStats
    {
        double   sinr_sum       = 0.0;
        double   sinr_min       = 1e9;
        double   sinr_max       = -1e9;
        uint64_t sinr_count     = 0;
        uint64_t los_link_sum   = 0;   // cumulative LOS links across ticks
        uint64_t conn_link_sum  = 0;   // cumulative connected links across ticks
        double   tx_delivered   = 0.0;  // sum of delivered_mbps where node is src
        double   rx_delivered   = 0.0;  // sum of delivered_mbps where node is dst
        uint64_t tick_count     = 0;
    };

    struct FlowStats
    {
        double   demand_sum     = 0.0;
        double   delivered_sum  = 0.0;
        double   latency_sum   = 0.0;
        double   hop_count_sum  = 0.0;
        uint64_t tick_count     = 0;
    };

    SimConfig  m_cfg;
    TimingInfo m_timing;

    std::map<uint32_t, NodeStats> m_nodeStats;
    std::map<std::pair<uint32_t, uint32_t>, FlowStats> m_flowStats;

    // Network-level accumulators
    uint64_t m_tickCount            = 0;
    double   m_netSinrSum           = 0.0;
    uint64_t m_netSinrCount         = 0;
    double   m_netDeliveredSum      = 0.0;
    uint64_t m_connectedPairsSum    = 0;
    uint64_t m_totalPairsSum        = 0;
    double   m_hopCountSum          = 0.0;
    uint64_t m_hopCountCount        = 0;
    uint64_t m_flowsRoutedSum       = 0;
    uint64_t m_flowsUnroutableSum   = 0;
};

}  // namespace mesh_sim
