/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/** @brief
 * MeshRouter: routes flows over the mesh link graph.
 * Supports Dijkstra shortest-path, widest-path (max throughput),
 * and min-hop (BFS) algorithms. Applies proportional-fairness
 * congestion scaling when links are overloaded.
 */
#pragma once

#include "src/domain/link-result.h"
#include "src/domain/mesh-config.h"
#include "src/eval/link-table.h"
#include "src/traffic/traffic-matrix.h"

#include <cstdint>
#include <vector>

namespace mesh_sim
{
/** @brief
*/
struct FlowResult
{
    uint32_t src = 0;
    uint32_t dst = 0;
    double   demand_mbps    = 0.0;
    double   delivered_mbps = 0.0;
    double   latency_ms     = 0.0;
    uint32_t hop_count      = 0;
    std::vector<uint32_t> path;
    bool     routable = false;
};

class MeshRouter
{
  public:
  /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    explicit MeshRouter(const RoutingConfig& cfg);
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    std::vector<FlowResult> Route(const LinkTable& links,
                                  const std::vector<Flow>& flows,
                                  uint32_t numNodes) const;

  private:
  /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    RoutingConfig m_cfg;
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    std::vector<uint32_t> FindPath(const LinkTable& links,
                                   uint32_t src,
                                   uint32_t dst,
                                   uint32_t numNodes) const;
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    std::vector<uint32_t> FindPathShortestPath(const LinkTable& links,
                                               uint32_t src,
                                               uint32_t dst,
                                               uint32_t numNodes) const;
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    std::vector<uint32_t> FindPathMaxThroughput(const LinkTable& links,
                                                uint32_t src,
                                                uint32_t dst,
                                                uint32_t numNodes) const;
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    std::vector<uint32_t> FindPathMinHop(const LinkTable& links,
                                         uint32_t src,
                                         uint32_t dst,
                                         uint32_t numNodes) const;
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void ApplyCongestionScaling(std::vector<FlowResult>& results,
                                const LinkTable& links) const;
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    double ComputeLatency(const std::vector<uint32_t>& path,
                          const LinkTable& links) const;
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    std::vector<uint32_t> ReconstructPath(const std::vector<uint32_t>& prev,
                                          uint32_t src,
                                          uint32_t dst) const;
};

}  // namespace mesh_sim
