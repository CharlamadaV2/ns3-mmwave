/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file link-table.h
 * @brief NxN symmetric link-quality matrix with O(1) per-link lookup.
 *
 * @ref mesh_sim::LinkTable wraps the flat vector produced by
 * @ref mesh_sim::LinkEvaluator::EvaluateAll into a symmetric N×N matrix.
 * It is updated once per tick and then queried by the routing engine,
 * metrics and viz writers, and the RL bridge. Pure C++ apart from
 * @c NS_LOG in the .cc, so it is unit-tested standalone.
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
 * Holds one @ref LinkResult per ordered node pair (both orders store the
 * same values). Diagonal entries (@c Get(i,i)) are default-constructed
 * @ref LinkResult objects (zeros, @c sinr_db = −999) and must not be used
 * for routing or metrics. Indices are not bounds-checked.
 */
class LinkTable
{
  public:
    /**
     * @fn LinkTable::Update
     * @brief Rebuild the matrix from a fresh set of link evaluations.
     *
     * @param numNodes  Number of nodes N; sets the matrix size and the expected
     *                  result count N·(N−1)/2.
     * @param results   Link results from @ref LinkEvaluator::EvaluateAll.
     *                  Each @c tx_id and @c rx_id must be below @p numNodes.
     * @return Nothing.
     * @throws std::runtime_error if @c results.size() ≠ @c numNodes·(numNodes−1)/2.
     *
     * Discards the old table and allocates a fresh N×N one. Each result is
     * stored at both @c [tx_id][rx_id] and @c [rx_id][tx_id]. Pairs missing
     * from @p results are not detected beyond the size check; they stay default.
     * Also emits @c NS_LOG output (per-link detail only at LOGIC level).
     */
    void Update(uint32_t numNodes, const std::vector<LinkResult>& results);

    /**
     * @fn LinkTable::NumNodes
     * @brief Return the number of nodes @c N the table was last built for.
     * @return Node count, or 0 if @ref Update has not been called.
     * @throws Nothing.
     */
    uint32_t NumNodes() const;

    /**
     * @fn LinkTable::Get
     * @brief Return the @ref LinkResult for node pair (i, j).
     *
     * @param i  First node index (row), in [0, N).
     * @param j  Second node index (column), in [0, N).
     * @return Const reference to the stored @ref LinkResult. Valid until
     *         the next call to @ref Update.
     * @throws Nothing; out-of-range indices are undefined behaviour.
     *
     * Symmetric: @c Get(i,j) and @c Get(j,i) hold the same values. The
     * diagonal returns a default @ref LinkResult.
     */
    const LinkResult& Get(uint32_t i, uint32_t j) const;

    /**
     * @fn LinkTable::MaxCapacity
     * @brief Return the maximum capacity across all node pairs (Mbps).
     *
     * @return Highest @ref LinkResult::capacity_mbps in the table; 0.0 if the
     *         table is empty or every link has zero capacity.
     * @throws Nothing.
     *
     * Scans the upper triangle only.
     */
    double MaxCapacity() const;

    /**
     * @fn LinkTable::ConnectedLinkCount
     * @brief Count unordered node pairs whose SINR meets the threshold.
     *
     * @param sinrThresholdDb  Minimum acceptable SINR in dB. Default −6.7,
     *                         a literal equal to @c SINR_MIN_DB (not linked to it).
     * @return Number of connected unordered pairs in [0, N·(N−1)/2].
     * @throws Nothing.
     *
     * A pair is connected when @c sinr_db >= @p sinrThresholdDb (inclusive).
     * Scans the upper triangle only.
     */
    uint32_t ConnectedLinkCount(double sinrThresholdDb = -6.7) const;

    /**
     * @fn LinkTable::IsConnected
     * @brief Test whether a specific node pair meets the SINR threshold.
     *
     * @param i               First node index, in [0, N).
     * @param j               Second node index, in [0, N).
     * @param sinrThresholdDb Minimum acceptable SINR in dB (default −6.7, equal to @c SINR_MIN_DB).
     * @return @c true if @c Get(i,j).sinr_db >= @p sinrThresholdDb.
     * @throws Nothing; out-of-range indices are undefined behaviour.

     */
    bool IsConnected(uint32_t i, uint32_t j, double sinrThresholdDb = -6.7) const;

  private:
    uint32_t m_numNodes = 0;  ///< Node count set by the last @ref Update call.

    /// N×N symmetric matrix of link results.
    /// Indexed as @c m_table[tx_id][rx_id]; both orderings are populated.
    std::vector<std::vector<LinkResult>> m_table;
};

}  // namespace mesh_sim