/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

/** @brief
 * LinkTable: NxN link quality matrix updated each tick.
 * Wraps the flat output of LinkEvaluator::EvaluateAll() into
 * a symmetric matrix with O(1) per-link lookup.
 */
#pragma once

#include "src/domain/link-result.h"

#include <cstdint>
#include <vector>
/** @brief
*/
namespace mesh_sim
{
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
class LinkTable
{
  public:
    void Update(uint32_t numNodes, const std::vector<LinkResult>& results);

    uint32_t NumNodes() const;

    // O(1) lookup. Symmetric: Get(i,j) == Get(j,i). Diagonal returns a default LinkResult.
    const LinkResult& Get(uint32_t i, uint32_t j) const;

    // Aggregate queries.
    double   MaxCapacity() const;
    uint32_t ConnectedLinkCount(double sinrThresholdDb = -6.7) const;
    bool     IsConnected(uint32_t i, uint32_t j, double sinrThresholdDb = -6.7) const;

  private:
    uint32_t m_numNodes = 0;
    std::vector<std::vector<LinkResult>> m_table; // NxN, symmetric
};

}  // namespace mesh_sim
