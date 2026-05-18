/**
 * @brief 
 * 
 */

/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/io/metrics-writer.h"
#include "src/util/string-utils.h"
#include "third_party/json.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

using json = nlohmann::json;

namespace mesh_sim
{

MetricsWriter::MetricsWriter(const SimConfig& cfg)
    : m_cfg(cfg)
{
}

void
MetricsWriter::SetTiming(const TimingInfo& t)
{
    m_timing = t;
}

// ---------------------------------------------------------------------------
// AccumulateTick
// ---------------------------------------------------------------------------

void
MetricsWriter::AccumulateTick(double time_s,
                              const LinkTable& links,
                              const std::vector<FlowResult>& flows,
                              uint32_t numNodes)
{
    if (time_s < m_cfg.warmup_s)
    {
        return;
    }

    // Per-node link stats
    for (uint32_t i = 0; i < numNodes; ++i)
    {
        auto& ns = m_nodeStats[i];
        ns.tick_count++;

        uint64_t connLinks = 0;
        uint64_t losLinks  = 0;

        for (uint32_t j = 0; j < numNodes; ++j)
        {
            if (i == j)
            {
                continue;
            }
            const LinkResult& lr = links.Get(i, j);
            if (lr.sinr_db > -900.0)
            {
                ns.sinr_sum += lr.sinr_db;
                ns.sinr_min = std::min(ns.sinr_min, lr.sinr_db);
                ns.sinr_max = std::max(ns.sinr_max, lr.sinr_db);
                ns.sinr_count++;
                m_netSinrSum += lr.sinr_db;
                m_netSinrCount++;
            }
            if (links.IsConnected(i, j))
            {
                connLinks++;
            }
            if (lr.is_los)
            {
                losLinks++;
            }
        }

        ns.conn_link_sum += connLinks;
        ns.los_link_sum  += losLinks;
    }

    // Connectivity: count connected unordered pairs
    uint64_t connPairs = 0;
    uint64_t totalPairs = static_cast<uint64_t>(numNodes) * (numNodes - 1) / 2;
    for (uint32_t i = 0; i < numNodes; ++i)
    {
        for (uint32_t j = i + 1; j < numNodes; ++j)
        {
            if (links.IsConnected(i, j))
            {
                connPairs++;
            }
        }
    }
    m_connectedPairsSum += connPairs;
    m_totalPairsSum     += totalPairs;

    // Per-flow stats
    for (const auto& fr : flows)
    {
        auto& fs = m_flowStats[{fr.src, fr.dst}];
        fs.demand_sum    += fr.demand_mbps;
        fs.delivered_sum += fr.delivered_mbps;
        fs.latency_sum   += fr.latency_ms;
        fs.hop_count_sum += fr.hop_count;
        fs.tick_count++;

        m_nodeStats[fr.src].tx_delivered += fr.delivered_mbps;
        m_nodeStats[fr.dst].rx_delivered += fr.delivered_mbps;

        m_netDeliveredSum += fr.delivered_mbps;

        if (fr.routable)
        {
            m_flowsRoutedSum++;
            m_hopCountSum += fr.hop_count;
            m_hopCountCount++;
        }
        else
        {
            m_flowsUnroutableSum++;
        }
    }

    m_tickCount++;
}

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

void
MetricsWriter::Write() const
{
    json out;
    out["scenario"]   = m_cfg.scenario_name;
    out["seed"]       = m_cfg.seed;
    out["duration_s"] = m_cfg.duration_s;
    out["warmup_s"]   = m_cfg.warmup_s;

    if (m_timing.elapsed_s > 0.0)
    {
        out["wall_clock_start"] = toIso8601(m_timing.start);
        out["wall_clock_end"]   = toIso8601(m_timing.end);
        out["wall_elapsed_s"]   = m_timing.elapsed_s;
    }

    // Per-node metrics
    json per_node = json::object();
    for (const auto& kv : m_nodeStats)
    {
        uint32_t idx = kv.first;
        const NodeStats& ns = kv.second;
        json n;

        if (ns.sinr_count > 0)
        {
            n["mean_sinr_db"] = ns.sinr_sum / static_cast<double>(ns.sinr_count);
            n["min_sinr_db"]  = ns.sinr_min;
            n["max_sinr_db"]  = ns.sinr_max;
        }
        else
        {
            n["mean_sinr_db"] = nullptr;
            n["min_sinr_db"]  = nullptr;
            n["max_sinr_db"]  = nullptr;
        }

        double ticks = static_cast<double>(ns.tick_count);
        n["num_links"]     = (ticks > 0) ? ns.conn_link_sum / ticks : 0.0;
        n["num_los_links"] = (ticks > 0) ? ns.los_link_sum / ticks  : 0.0;
        n["tx_throughput_mbps"] = (ticks > 0) ? ns.tx_delivered / ticks : 0.0;
        n["rx_throughput_mbps"] = (ticks > 0) ? ns.rx_delivered / ticks : 0.0;

        // Use node id from config if available, otherwise "node<idx>"
        std::string nodeId = (idx < m_cfg.nodes.size())
                                 ? m_cfg.nodes[idx].id
                                 : "node" + std::to_string(idx);
        per_node[nodeId] = n;
    }
    out["per_node"] = per_node;

    // Per-flow metrics
    json per_flow = json::object();
    for (const auto& kv : m_flowStats)
    {
        const auto& key = kv.first;
        const FlowStats& fs = kv.second;

        std::string srcId = (key.first < m_cfg.nodes.size())
                                ? m_cfg.nodes[key.first].id
                                : "node" + std::to_string(key.first);
        std::string dstId = (key.second < m_cfg.nodes.size())
                                ? m_cfg.nodes[key.second].id
                                : "node" + std::to_string(key.second);

        json f;
        double ticks = static_cast<double>(fs.tick_count);
        f["demand_mbps"]    = (ticks > 0) ? fs.demand_sum / ticks    : 0.0;
        f["delivered_mbps"] = (ticks > 0) ? fs.delivered_sum / ticks : 0.0;
        f["latency_ms"]     = (ticks > 0) ? fs.latency_sum / ticks  : 0.0;
        f["hop_count"]      = (ticks > 0) ? fs.hop_count_sum / ticks : 0.0;

        per_flow[srcId + "->" + dstId] = f;
    }
    out["per_flow"] = per_flow;

    // Network summary
    json net;
    double ticks = static_cast<double>(m_tickCount);

    net["sum_throughput_mbps"] = (ticks > 0) ? m_netDeliveredSum / ticks : 0.0;
    if (m_netSinrCount > 0)
        net["mean_sinr_db"] = m_netSinrSum / static_cast<double>(m_netSinrCount);
    else
        net["mean_sinr_db"] = nullptr;
    net["connectivity"] = (m_totalPairsSum > 0)
                              ? static_cast<double>(m_connectedPairsSum) /
                                static_cast<double>(m_totalPairsSum)
                              : 0.0;
    net["mean_hop_count"] = (m_hopCountCount > 0)
                                ? m_hopCountSum / static_cast<double>(m_hopCountCount)
                                : 0.0;
    net["flows_routed"]    = (ticks > 0) ? m_flowsRoutedSum / ticks    : 0.0;
    net["flows_unroutable"] = (ticks > 0) ? m_flowsUnroutableSum / ticks : 0.0;

    out["network"] = net;

    // Write to file
    std::string path = m_cfg.output_dir + "/summary.json";
    std::ofstream f(path);
    if (!f.is_open())
    {
        std::cerr << "[MetricsWriter] ERROR: cannot write " << path << "\n";
        return;
    }
    f << out.dump(2) << "\n";
    std::cerr << "[MetricsWriter] Summary written to " << path << "\n";
}

}  // namespace mesh_sim
