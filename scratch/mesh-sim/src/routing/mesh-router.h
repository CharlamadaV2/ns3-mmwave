/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file mesh-router.h
 * @brief Flow-level routing over the mesh link graph.
 *
 * @ref MeshRouter takes the current @ref LinkTable and a list of active
 * @ref Flow objects, finds a path for each flow using the configured
 * algorithm, then applies proportional-fairness congestion scaling to
 * produce a @ref FlowResult for every flow.
 *
 * **Routing algorithms** (selected by @ref RoutingConfig::algorithm)
 *
 * | Algorithm          | Strategy                                                         | Data structure |
 * |--------------------|------------------------------------------------------------------|----------------|
 * | @c "shortest_path" | Dijkstra with edge weight @c 1/capacity_mbps.                   | min-heap       |
 * |                    | Prefers high-capacity links; default.                            |                |
 * | @c "max_throughput"| Widest-path: maximises the bottleneck link capacity.             | max-heap       |
 * | @c "min_hop"       | BFS; minimises hop count, ignoring link quality entirely.        | FIFO queue     |
 *
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

/**
 * @brief Routing result for one flow at one simulation tick.
 *
 * Produced by @ref MeshRouter::Route and consumed by @ref MetricsWriter,
 * @ref VizWriter, and the RL reward calculator.
 *
 * **Effective demand**
 * @c demand_mbps reflects the flow's *effective* demand this tick — it is
 * zero when the flow is inactive or in the OFF phase of an on-off model,
 * regardless of the configured per-flow demand. Only the on_off model sets
 * @c in_on_phase to @c false; constant and Poisson flows always have
 * @c demand_mbps equal to @ref Flow::demand_mbps when active.
 *
 * **Unroutable flows**
 * When @c routable is @c false the path is empty, @c delivered_mbps is 0,
 * @c latency_ms is 0, and @c hop_count is 0. The @c src and @c dst fields
 * are always set.
 *
 * **Self-flows** (@c src == @c dst) are marked @c routable = @c true with an
 * empty path and zero delivered demand, since they require no forwarding.
 */
struct FlowResult
{
    uint32_t src = 0;  ///< Source node index into @ref SimConfig::nodes.
    uint32_t dst = 0;  ///< Destination node index into @ref SimConfig::nodes.

    double demand_mbps    = 0.0;  ///< Effective demand this tick (Mbps); 0 when
                                   ///<   inactive or in the off-phase.
    double delivered_mbps = 0.0;  ///< Throughput after congestion scaling (Mbps);
                                   ///<   ≤ @c demand_mbps.
    double latency_ms     = 0.0;  ///< End-to-end path latency in ms (propagation +
                                   ///<   processing); 0 when unroutable.
    uint32_t hop_count    = 0;    ///< Number of hops: @c path.size() − 1; 0 when
                                   ///<   unroutable or self-flow.

    std::vector<uint32_t> path;   ///< Ordered node indices from @c src to @c dst.
                                   ///<   Empty when unroutable or self-flow.

    bool routable = false;         ///< @c true if a path was found (or @c src == @c dst).
};


/**
 * @brief Routes flows over the mesh link graph and applies congestion scaling.
 *
 * Stateless between @ref Route calls (all per-call state is local).
 * The @ref RoutingConfig is copied at construction and does not change.
 */
class MeshRouter
{
  public:
    /**
     * @brief Construct a router with the given routing configuration.
     *
     * @param cfg  Routing algorithm and max-hops settings from @ref SimConfig::mesh.
     */
    explicit MeshRouter(const RoutingConfig& cfg);

    /**
     * @brief Route all flows and return one @ref FlowResult per flow.
     *
     * **Per-flow pipeline:**
     * -# Compute effective demand: @c 0.0 for inactive or off-phase flows.
     * -# Skip path finding for zero-demand flows (marked unroutable unless
     *    @c src == @c dst).
     * -# Call @ref FindPath to select the best route under the configured
     *    algorithm and hop limit.
     * -# Set @c delivered_mbps = @c demand_mbps (full demand, before scaling).
     * -# Compute latency via @ref ComputeLatency.
     * -# After all flows are processed, call @ref ApplyCongestionScaling to
     *    reduce @c delivered_mbps on overloaded edges.
     *
     * @param links     Current @ref LinkTable from @ref LinkEvaluator::EvaluateAll.
     * @param flows     Active flows from @ref TrafficMatrix::GetActiveFlows.
     * @param numNodes  Total node count (@c cfg.nodes.size()); must match
     *                  the dimension of @c links.
     * @return Vector of @ref FlowResult, one per entry in @c flows, in the
     *         same order.
     */
    std::vector<FlowResult> Route(const LinkTable& links,
                                  const std::vector<Flow>& flows,
                                  uint32_t numNodes) const;

  private:
    RoutingConfig m_cfg;  ///< Routing algorithm and hop-limit settings.

    /**
     * @brief Dispatch to the appropriate path-finding algorithm and enforce
     *        the hop limit.
     *
     * Calls @ref FindPathShortestPath, @ref FindPathMaxThroughput, or
     * @ref FindPathMinHop based on @c m_cfg.algorithm.  If the found path
     * exceeds @c m_cfg.max_hops hops (and @c max_hops > 0), an empty vector
     * is returned and the flow is left unroutable.  @c max_hops == 0 means
     * unlimited.
     *
     * @param links     Current link quality matrix.
     * @param src       Source node index.
     * @param dst       Destination node index.
     * @param numNodes  Number of nodes in the graph.
     * @return Ordered path from @c src to @c dst, or an empty vector if
     *         no path exists or the hop limit is exceeded.
     */
    std::vector<uint32_t> FindPath(const LinkTable& links,
                                   uint32_t src,
                                   uint32_t dst,
                                   uint32_t numNodes) const;

    /**
     * @brief Dijkstra shortest path with edge weight @c 1/capacity_mbps.
     *
     * High-capacity links are preferred (lower weight). Edges with
     * @c capacity_mbps <= 0 are treated as absent. Uses a min-heap for
     * O((V+E)·log V) complexity.
     *
     * @param links     Current link quality matrix.
     * @param src       Source node index.
     * @param dst       Destination node index.
     * @param numNodes  Number of nodes in the graph.
     * @return Ordered path, or empty if @c dst is unreachable.
     */
    std::vector<uint32_t> FindPathShortestPath(const LinkTable& links,
                                               uint32_t src,
                                               uint32_t dst,
                                               uint32_t numNodes) const;

    /**
     * @brief Widest-path algorithm: maximises bottleneck link capacity.
     *
     * Modified Dijkstra using a max-heap. For each candidate next hop,
     * the path bandwidth is @c min(current_bottleneck, link_capacity);
     * the node with the highest such value is expanded first. Edges with
     * @c capacity_mbps <= 0 are skipped.
     *
     * @param links     Current link quality matrix.
     * @param src       Source node index.
     * @param dst       Destination node index.
     * @param numNodes  Number of nodes in the graph.
     * @return Ordered path that maximises bottleneck capacity, or empty if
     *         @c dst is unreachable.
     */
    std::vector<uint32_t> FindPathMaxThroughput(const LinkTable& links,
                                                uint32_t src,
                                                uint32_t dst,
                                                uint32_t numNodes) const;

    /**
     * @brief Minimum-hop path via BFS.
     *
     * Ignores link capacity entirely (beyond the > 0 connectivity check).
     * Returns the path with fewest hops; ties are broken by BFS expansion
     * order (lower node index expanded first within a level).
     *
     * @param links     Current link quality matrix.
     * @param src       Source node index.
     * @param dst       Destination node index.
     * @param numNodes  Number of nodes in the graph.
     * @return Ordered minimum-hop path, or empty if @c dst is unreachable.
     */
    std::vector<uint32_t> FindPathMinHop(const LinkTable& links,
                                         uint32_t src,
                                         uint32_t dst,
                                         uint32_t numNodes) const;

    /**
     * @brief Reconstruct a node path by walking the predecessor array.
     *
     * Walks @c prev[] backward from @c dst to @c src, then reverses the
     * result.  The sentinel value @c UINT32_MAX in @c prev indicates an
     * unvisited node (no predecessor).
     *
     * @param prev  Predecessor array filled by a path-finding algorithm;
     *              @c prev[v] is the node before @c v on the best path.
     * @param src   Source node index.
     * @param dst   Destination node index.
     * @return Ordered path @c [src, …, dst], or an empty vector if
     *         @c dst was not reached (@c prev[dst] == UINT32_MAX and
     *         @c src != @c dst).
     */
    std::vector<uint32_t> ReconstructPath(const std::vector<uint32_t>& prev,
                                          uint32_t src,
                                          uint32_t dst) const;

    /**
     * @brief Apply proportional-fairness congestion scaling to all results.
     *
     * For each physical edge (identified by the sorted node-index pair)
     * whose total demanded throughput exceeds its link capacity, scales
     * every flow using that edge by @c capacity/total_demand.
     * A flow traversing multiple bottleneck hops takes the minimum scaling
     * factor across all of them via:
     * @code
     *   fr.delivered_mbps = min(fr.delivered_mbps, fr.demand_mbps * scale)
     * @endcode
     * Edges where @c total_demand <= capacity are left unchanged.
     *
     * @param results  Flow results to update in-place.
     * @param links    Current @ref LinkTable (used to read link capacities).
     */
    void ApplyCongestionScaling(std::vector<FlowResult>& results,
                                const LinkTable& links) const;

    /**
     * @brief Compute end-to-end path latency in milliseconds.
     *
     * Uses a store-and-forward model summing two components per hop:
     * - **Propagation delay** = @c distance_m / 3×10⁸ × 1000 ms.
     * - **Processing delay**  = 0.5 ms (fixed per hop; approximates L1/L2
     *   processing at a mmWave small cell, consistent with the 3GPP
     *   TR 38.913 V17.0.0 §7 one-way user-plane latency target).
     *
     * No transmission-delay term is included because traffic is modelled
     * at flow granularity, not packet granularity.
     *
     * @param path   Ordered node indices from src to dst.
     * @param links  Current @ref LinkTable (used to read per-hop distances).
     * @return Total one-way latency in milliseconds.
     */
    double ComputeLatency(const std::vector<uint32_t>& path,
                          const LinkTable& links) const;
};

}  // namespace mesh_sim