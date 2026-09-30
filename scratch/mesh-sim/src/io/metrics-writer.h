/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file metrics-writer.h
 * @brief Accumulates per-tick simulation metrics and writes summary.json.
 *
 * One @ref mesh_sim::MetricsWriter is created per seed run. @ref mesh_sim::MetricsWriter::AccumulateTick
 * is called every tick; @ref mesh_sim::MetricsWriter::Write is called once after the step loop.
 * The class has no ns-3 dependency.
 *
 * **Output: summary.json**
 * Written to @c cfg.output_dir/summary.json (the per-seed directory
 * @c seed-N/) by @ref mesh_sim::MetricsWriter::Write. Indented with 2 spaces.
 *
 * | JSON key              | Value                                                     |
 * |-----------------------|-----------------------------------------------------------|
 * | @c scenario           | @c cfg.scenario_name. |
 * | @c seed               | @c cfg.seed. |
 * | @c duration_s         | Simulated duration in seconds. |
 * | @c warmup_s           | Warm-up period in seconds; earlier ticks are excluded from all averages. |
 * | @c wall_clock_start, @c wall_clock_end, @c wall_elapsed_s | ISO-8601 timestamps and elapsed wall seconds. Present only when @ref mesh_sim::MetricsWriter::SetTiming was called with @c elapsed_s > 0. |
 * | @c per_node[id]       | Keyed by node id. @c mean/min/max_sinr_db (over all valid link samples; @c null if none), @c num_links (mean connected links per tick), @c num_los_links (mean LOS links per tick), @c tx/rx_throughput_mbps (mean delivered Mbit/s of flows the node sources / sinks). |
 * | @c per_flow["A->B"]   | Keyed by source and destination node id. @c demand_mbps, @c delivered_mbps, @c latency_ms, @c hop_count, each averaged over the ticks the flow was reported. |
 * | @c network            | @c sum_throughput_mbps (mean total delivered per tick), @c mean_sinr_db (@c null if no valid sample), @c connectivity (connected node pairs / all pairs, using the default @c LinkTable::IsConnected threshold of -6.7 dB), @c mean_hop_count (over routable flow observations), @c flows_routed and @c flows_unroutable (mean per tick). |
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

/**
 * @class MetricsWriter
 * @brief Accumulates per-tick radio and flow metrics, then writes summary.json.
 *
 * All accumulation is done in-memory; the file is written exactly once by
 * @ref Write after the step loop completes.
 */
class MetricsWriter
{
  public:
    /**
     * @fn MetricsWriter::MetricsWriter
     * @brief Construct a writer bound to the given simulation configuration.
     *
     * Stores a copy of @c cfg so the writer remains valid after the
     * @c SimConfig is modified (e.g. by an RL position override between seeds).
     *
     * @param cfg  Fully loaded and validated simulation configuration.
     *             @c output_dir must be the per-seed directory by the time
     *             @ref Write runs, since the copy is taken at construction.
     */
    explicit MetricsWriter(const SimConfig& cfg);

    /**
     * @fn MetricsWriter::SetTiming
     * @brief Store wall-clock timing for inclusion in summary.json.
     *
     * Must be called after the step loop completes and before @ref Write.
     * If not called, the @c wall_clock_* fields are omitted from the output.
     *
     * @param t  @ref TimingInfo populated by @c sim.cc after the step loop.
     */
    void SetTiming(const TimingInfo& t);

    /**
     * @fn MetricsWriter::AccumulateTick
     * @brief Accumulate metrics for one simulation tick.
     *
     * Ticks with @c time_s < @c cfg.warmup_s are silently skipped so that
     * transient start-up behaviour is excluded from all averages.
     *
     * For each node pair the SINR sentinel value (-999.0 dB) is filtered:
     * only links with @c sinr_db > −900.0 contribute to SINR accumulators,
     * preventing unevaluated links from corrupting the statistics.
     *
     * @param time_s    Simulation time at this tick (seconds).
     * @param links     Current @ref LinkTable from @ref LinkEvaluator::EvaluateAll.
     * @param flows     Flow results from the routing engine for this tick.
     * @param numNodes  Number of active nodes (@c cfg.nodes.size()).
     */
    void AccumulateTick(double time_s,
                        const LinkTable& links,
                        const std::vector<FlowResult>& flows,
                        uint32_t numNodes);

    /**
     * @fn MetricsWriter::Write
     * @brief Write all accumulated metrics to @c output_dir/summary.json.
     *
     * All per-node and per-flow values are time-averaged over the number of
     * post-warmup ticks in which they were observed. Node IDs from
     * @c cfg.nodes are used as JSON keys; unrecognised indices fall back to
     * @c "node<idx>".
     *
     * Prints an error to @c stderr if the file cannot be opened and returns
     * without writing; does not throw. Prints a confirmation line to @c stderr
     * on success. Overwrites an existing @c summary.json. Does not create
     * the output directory.
     *
     * @return void.
     */
    void Write() const;

  private:
    /**
     * @brief Per-node running totals accumulated by @ref AccumulateTick.
     *
     * All values are summed over post-warmup ticks; time-averages are
     * computed in @ref Write by dividing by @c tick_count.
     */
    struct NodeStats
    {
        double   sinr_sum      = 0.0;   ///< Sum of SINR values across all peers and ticks (dB).
        double   sinr_min      = 1e9;   ///< Minimum SINR seen across all peers and ticks (dB).
        double   sinr_max      = -1e9;  ///< Maximum SINR seen across all peers and ticks (dB).
        uint64_t sinr_count    = 0;     ///< Number of valid (non-sentinel) SINR samples accumulated.
        uint64_t los_link_sum  = 0;     ///< Cumulative count of LOS links across ticks.
        uint64_t conn_link_sum = 0;     ///< Cumulative count of connected links across ticks.
        double   tx_delivered  = 0.0;   ///< Sum of delivered_mbps for flows where this node is src.
        double   rx_delivered  = 0.0;   ///< Sum of delivered_mbps for flows where this node is dst.
        uint64_t tick_count    = 0;     ///< Number of post-warmup ticks this node was observed.
    };

    /**
     * @brief Per-flow-pair running totals accumulated by @ref AccumulateTick.
     *
     * Keyed by @c (src_idx, dst_idx); time-averages computed in @ref Write.
     */
    struct FlowStats
    {
        double   demand_sum    = 0.0;  ///< Sum of demand_mbps across ticks.
        double   delivered_sum = 0.0;  ///< Sum of delivered_mbps across ticks.
        double   latency_sum   = 0.0;  ///< Sum of latency_ms across ticks.
        double   hop_count_sum = 0.0;  ///< Sum of hop_count across ticks.
        uint64_t tick_count    = 0;    ///< Number of ticks this flow pair was observed.
    };

    SimConfig  m_cfg;     ///< Simulation configuration (copied at construction).
    TimingInfo m_timing;  ///< Wall-clock timing set by @ref SetTiming.

    std::map<uint32_t, NodeStats>                       m_nodeStats;  ///< Per-node accumulators.
    std::map<std::pair<uint32_t, uint32_t>, FlowStats>  m_flowStats;  ///< Per-flow-pair accumulators.

    // ---- Network-level accumulators ----------------------------------------
    uint64_t m_tickCount          = 0;    ///< Post-warmup tick count.
    double   m_netSinrSum         = 0.0;  ///< Network-wide SINR sum (dB) across all links and ticks.
    uint64_t m_netSinrCount       = 0;    ///< Network-wide valid SINR sample count.
    double   m_netDeliveredSum    = 0.0;  ///< Network-wide sum of delivered_mbps across ticks.
    uint64_t m_connectedPairsSum  = 0;    ///< Sum of connected pair counts across ticks.
    uint64_t m_totalPairsSum      = 0;    ///< Sum of total pair counts across ticks.
    double   m_hopCountSum        = 0.0;  ///< Sum of hop counts for routable flows.
    uint64_t m_hopCountCount      = 0;    ///< Number of routable flow observations.
    uint64_t m_flowsRoutedSum     = 0;    ///< Cumulative count of routable flow ticks.
    uint64_t m_flowsUnroutableSum = 0;    ///< Cumulative count of unroutable flow ticks.
};

}  // namespace mesh_sim