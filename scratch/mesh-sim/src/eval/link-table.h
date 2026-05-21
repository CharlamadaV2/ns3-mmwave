/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file link-table.h
 * @brief NxN symmetric link-quality matrix with O(1) per-link lookup.
 *
 * @ref LinkTable wraps the flat vector produced by
 * @ref LinkEvaluator::EvaluateAll into a symmetric N×N matrix.
 * It is updated once per tick and then queried by the routing engine,
 * metrics writer, and RL reward calculator.
 *
 */
#pragma once

#include "src/domain/link-result.h"

#include <cstdint>
#include <vector>

namespace mesh_sim
{

/**
 * @brief Symmetric N×N link-quality matrix, updated each simulation tick.
 *
 * Holds one @ref LinkResult per ordered node pair. Diagonal entries
 * (@c Get(i,i)) return a default-constructed @ref LinkResult (all zeros /
 * sentinels) and should not be used for routing or metrics.
 */
class LinkTable
{
  public:
    /**
     * @brief Rebuild the matrix from a fresh set of link evaluations.
     *
     * Replaces the entire table with the results from
     * @ref LinkEvaluator::EvaluateAll. Each result is stored at both
     * @c [tx_id][rx_id] and @c [rx_id][tx_id] to make all lookups O(1).
     *
     * @param numNodes  Number of nodes @c N; determines the expected result
     *                  count of N·(N−1)/2 and the matrix dimensions.
     * @param results   Flat vector of link results from @ref LinkEvaluator::EvaluateAll,
     *                  in the same (i < j) index order.
     * @throws std::runtime_error if @c results.size() ≠ @c numNodes·(numNodes−1)/2.
     */
    void Update(uint32_t numNodes, const std::vector<LinkResult>& results);

    /**
     * @brief Return the number of nodes @c N the table was last built for.
     * @return Node count, or 0 if @ref Update has not been called.
     */
    uint32_t NumNodes() const;

    /**
     * @brief Return the @ref LinkResult for node pair (i, j).
     *
     * The table is symmetric: @c Get(i,j) == @c Get(j,i).
     * Diagonal entries (@c i == @c j) return a default @ref LinkResult.
     *
     * @param i  First node index (row).
     * @param j  Second node index (column).
     * @return Const reference to the stored @ref LinkResult. Valid until
     *         the next call to @ref Update.
     */
    const LinkResult& Get(uint32_t i, uint32_t j) const;

    /**
     * @brief Return the maximum capacity across all node pairs (Mbps).
     *
     * Scans the upper triangle of the matrix. Returns 0.0 if no links
     * have been evaluated yet.
     *
     * @return Highest @ref LinkResult::capacity_mbps in the table.
     */
    double MaxCapacity() const;

    /**
     * @brief Count unordered node pairs whose SINR meets the threshold.
     *
     * A pair is "connected" when its SINR is at or above
     * @c sinrThresholdDb, meaning it can carry at least the minimum
     * MCS rate (@ref SINR_MIN_DB = −6.7 dB by default).
     *
     * @param sinrThresholdDb  Minimum acceptable SINR in dB.
     *                         Defaults to @ref SINR_MIN_DB (−6.7 dB).
     * @return Number of connected unordered pairs in [0, N·(N−1)/2].
     */
    uint32_t ConnectedLinkCount(double sinrThresholdDb = -6.7) const;

    /**
     * @brief Test whether a specific node pair meets the SINR threshold.
     *
     * Equivalent to @c Get(i,j).sinr_db >= sinrThresholdDb.
     *
     * @param i              First node index.
     * @param j              Second node index.
     * @param sinrThresholdDb Minimum acceptable SINR in dB (default −6.7 dB).
     * @return @c true if the link SINR is at or above the threshold.
     */
    bool IsConnected(uint32_t i, uint32_t j, double sinrThresholdDb = -6.7) const;

  private:
    uint32_t m_numNodes = 0;  ///< Node count set by the last @ref Update call.

    /// N×N symmetric matrix of link results.
    /// Indexed as @c m_table[tx_id][rx_id]; both orderings are populated.
    std::vector<std::vector<LinkResult>> m_table;
};

}  // namespace mesh_sim