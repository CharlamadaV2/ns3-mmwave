/**
 * @brief 
 * 
 */

/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

#include "src/routing/mesh-router.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <queue>
#include <utility>
#include <vector>

namespace mesh_sim
{

static const uint32_t NO_PREV = std::numeric_limits<uint32_t>::max();

MeshRouter::MeshRouter(const RoutingConfig& cfg)
    : m_cfg(cfg)
{
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::vector<FlowResult>
MeshRouter::Route(const LinkTable& links,
                  const std::vector<Flow>& flows,
                  uint32_t numNodes) const
{
    std::vector<FlowResult> results;
    results.reserve(flows.size());

    for (const auto& f : flows)
    {
        FlowResult fr;
        fr.src = f.src;
        fr.dst = f.dst;

        double effectiveDemand = (f.active && f.in_on_phase) ? f.demand_mbps : 0.0;
        fr.demand_mbps = effectiveDemand;

        if (effectiveDemand <= 0.0 || f.src == f.dst)
        {
            fr.routable = (f.src == f.dst);
            results.push_back(std::move(fr));
            continue;
        }

        auto path = FindPath(links, f.src, f.dst, numNodes);
        if (path.empty())
        {
            results.push_back(std::move(fr));
            continue;
        }

        fr.routable       = true;
        fr.path           = std::move(path);
        fr.hop_count      = static_cast<uint32_t>(fr.path.size() - 1);
        fr.delivered_mbps = fr.demand_mbps;
        fr.latency_ms     = ComputeLatency(fr.path, links);
        results.push_back(std::move(fr));
    }

    ApplyCongestionScaling(results, links);
    return results;
}

// ---------------------------------------------------------------------------
// Path finding -- dispatcher
// ---------------------------------------------------------------------------

std::vector<uint32_t>
MeshRouter::FindPath(const LinkTable& links,
                     uint32_t src,
                     uint32_t dst,
                     uint32_t numNodes) const
{
    std::vector<uint32_t> path;

    if (m_cfg.algorithm == "max_throughput")
        path = FindPathMaxThroughput(links, src, dst, numNodes);
    else if (m_cfg.algorithm == "min_hop")
        path = FindPathMinHop(links, src, dst, numNodes);
    else
        path = FindPathShortestPath(links, src, dst, numNodes);

    if (m_cfg.max_hops > 0 && path.size() > m_cfg.max_hops + 1)
        return {};

    return path;
}

// ---------------------------------------------------------------------------
// Dijkstra -- shortest path (weight = 1 / capacity_mbps)
// ---------------------------------------------------------------------------

std::vector<uint32_t>
MeshRouter::FindPathShortestPath(const LinkTable& links,
                                 uint32_t src,
                                 uint32_t dst,
                                 uint32_t numNodes) const
{
    std::vector<double>   dist(numNodes, std::numeric_limits<double>::infinity());
    std::vector<uint32_t> prev(numNodes, NO_PREV);
    std::vector<bool>     visited(numNodes, false);

    dist[src] = 0.0;

    // min-heap: (distance, node)
    using Entry = std::pair<double, uint32_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;
    pq.push({0.0, src});

    while (!pq.empty())
    {
        auto [d, u] = pq.top();
        pq.pop();

        if (visited[u])
            continue;
        visited[u] = true;

        if (u == dst)
            break;

        for (uint32_t v = 0; v < numNodes; ++v)
        {
            if (visited[v] || v == u)
                continue;

            double cap = links.Get(u, v).capacity_mbps;
            if (cap <= 0.0)
                continue;

            double w = 1.0 / cap;
            double newDist = dist[u] + w;
            if (newDist < dist[v])
            {
                dist[v] = newDist;
                prev[v] = u;
                pq.push({newDist, v});
            }
        }
    }

    return ReconstructPath(prev, src, dst);
}

// ---------------------------------------------------------------------------
// Widest path (maximize bottleneck bandwidth)
// ---------------------------------------------------------------------------

std::vector<uint32_t>
MeshRouter::FindPathMaxThroughput(const LinkTable& links,
                                  uint32_t src,
                                  uint32_t dst,
                                  uint32_t numNodes) const
{
    std::vector<double>   bw(numNodes, 0.0);
    std::vector<uint32_t> prev(numNodes, NO_PREV);
    std::vector<bool>     visited(numNodes, false);

    bw[src] = std::numeric_limits<double>::infinity();

    // max-heap: (bandwidth, node)
    using Entry = std::pair<double, uint32_t>;
    std::priority_queue<Entry> pq;
    pq.push({bw[src], src});

    while (!pq.empty())
    {
        auto [b, u] = pq.top();
        pq.pop();

        if (visited[u])
            continue;
        visited[u] = true;

        if (u == dst)
            break;

        for (uint32_t v = 0; v < numNodes; ++v)
        {
            if (visited[v] || v == u)
                continue;

            double cap = links.Get(u, v).capacity_mbps;
            if (cap <= 0.0)
                continue;

            double newBw = std::min(bw[u], cap);
            if (newBw > bw[v])
            {
                bw[v]   = newBw;
                prev[v] = u;
                pq.push({newBw, v});
            }
        }
    }

    return ReconstructPath(prev, src, dst);
}

// ---------------------------------------------------------------------------
// Min-hop (BFS)
// ---------------------------------------------------------------------------

std::vector<uint32_t>
MeshRouter::FindPathMinHop(const LinkTable& links,
                           uint32_t src,
                           uint32_t dst,
                           uint32_t numNodes) const
{
    std::vector<uint32_t> prev(numNodes, NO_PREV);
    std::vector<bool>     visited(numNodes, false);

    std::queue<uint32_t> q;
    q.push(src);
    visited[src] = true;

    while (!q.empty())
    {
        uint32_t u = q.front();
        q.pop();

        if (u == dst)
            break;

        for (uint32_t v = 0; v < numNodes; ++v)
        {
            if (visited[v] || v == u)
                continue;

            if (links.Get(u, v).capacity_mbps <= 0.0)
                continue;

            visited[v] = true;
            prev[v]    = u;
            q.push(v);
        }
    }

    return ReconstructPath(prev, src, dst);
}

// ---------------------------------------------------------------------------
// Path reconstruction from prev[] array
// ---------------------------------------------------------------------------

std::vector<uint32_t>
MeshRouter::ReconstructPath(const std::vector<uint32_t>& prev,
                            uint32_t src,
                            uint32_t dst) const
{
    if (prev[dst] == NO_PREV && src != dst)
        return {};

    std::vector<uint32_t> path;
    for (uint32_t cur = dst; cur != NO_PREV; cur = prev[cur])
    {
        path.push_back(cur);
        if (cur == src)
            break;
    }

    std::reverse(path.begin(), path.end());

    if (path.empty() || path.front() != src)
        return {};

    return path;
}

// ---------------------------------------------------------------------------
// Congestion scaling -- proportional fairness
// ---------------------------------------------------------------------------

void
MeshRouter::ApplyCongestionScaling(std::vector<FlowResult>& results,
                                   const LinkTable& links) const
{
    // linkKey (min,max) -> total demand on that link
    std::map<std::pair<uint32_t, uint32_t>, double> linkDemand;
    // linkKey -> flow indices using that link
    std::map<std::pair<uint32_t, uint32_t>, std::vector<size_t>> linkFlows;

    for (size_t fi = 0; fi < results.size(); ++fi)
    {
        const auto& fr = results[fi];
        if (!fr.routable || fr.demand_mbps <= 0.0)
            continue;

        for (size_t hi = 0; hi + 1 < fr.path.size(); ++hi)
        {
            auto key = std::make_pair(std::min(fr.path[hi], fr.path[hi + 1]),
                                      std::max(fr.path[hi], fr.path[hi + 1]));
            linkDemand[key] += fr.demand_mbps;
            linkFlows[key].push_back(fi);
        }
    }

    for (const auto& [key, totalDemand] : linkDemand)
    {
        double capacity = links.Get(key.first, key.second).capacity_mbps;
        if (totalDemand <= capacity || capacity <= 0.0)
            continue;

        double scale = capacity / totalDemand;
        for (size_t fi : linkFlows[key])
        {
            results[fi].delivered_mbps =
                std::min(results[fi].delivered_mbps,
                         results[fi].demand_mbps * scale);
        }
    }
}

// ---------------------------------------------------------------------------
// Latency proxy
//
// Standard store-and-forward model summing two components per hop:
//   1. Propagation delay  = distance / c  (free-space speed of light).
//   2. Processing delay   = 0.5 ms fixed per hop -- approximates L1/L2
//      processing at a mmWave small cell. Consistent with the 0.5 ms
//      one-way user-plane latency target in 3GPP TR 38.913 §7.
//
// No transmission delay term because traffic is flow-level, not packet-level.
// Reference: Rappaport, "Wireless Communications", Ch. 3 (propagation
// fundamentals); 3GPP TR 38.913 V17.0.0 (2022-03), Table 7.1-1 (latency).
// ---------------------------------------------------------------------------

double
MeshRouter::ComputeLatency(const std::vector<uint32_t>& path,
                           const LinkTable& links) const
{
    double total = 0.0;
    for (size_t i = 0; i + 1 < path.size(); ++i)
    {
        const auto& lr = links.Get(path[i], path[i + 1]);
        double propagation_ms = lr.distance_m / 3e8 * 1000.0;
        double processing_ms  = 0.5;
        total += propagation_ms + processing_ms;
    }
    return total;
}

}  // namespace mesh_sim
