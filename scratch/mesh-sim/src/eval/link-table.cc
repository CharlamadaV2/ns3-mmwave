/**
 * @brief 
 * 
 */

/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/eval/link-table.h"

#include "ns3/log.h"

#include <algorithm>
#include <stdexcept>

NS_LOG_COMPONENT_DEFINE("LinkTable");

namespace mesh_sim
{

void
LinkTable::Update(uint32_t numNodes, const std::vector<LinkResult>& results)
{
    const uint32_t expected = numNodes * (numNodes - 1) / 2;
    if (results.size() != expected)
    {
        throw std::runtime_error("[LinkTable] Expected " + std::to_string(expected) +
                                 " link results for " + std::to_string(numNodes) +
                                 " nodes, got " + std::to_string(results.size()));
    }

    m_numNodes = numNodes;
    m_table.assign(numNodes, std::vector<LinkResult>(numNodes));

    for (const auto& r : results)
    {
        m_table[r.tx_id][r.rx_id] = r;
        m_table[r.rx_id][r.tx_id] = r;
    }

    NS_LOG_DEBUG("Update: " << numNodes << " nodes, "
                 << ConnectedLinkCount() << "/" << expected << " links connected"
                 << ", maxCap=" << MaxCapacity() << " Mbps");

    // Log the full matrix at LOGIC level (very verbose).
    for (uint32_t i = 0; i < numNodes; ++i)
    {
        for (uint32_t j = i + 1; j < numNodes; ++j)
        {
            const auto& l = m_table[i][j];
            NS_LOG_LOGIC("  [" << i << "," << j << "]"
                         << " d=" << l.distance_m << "m"
                         << " LOS=" << l.is_los
                         << " SINR=" << l.sinr_db << "dB"
                         << " cap=" << l.capacity_mbps << "Mbps");
        }
    }
}

uint32_t
LinkTable::NumNodes() const
{
    return m_numNodes;
}

const LinkResult&
LinkTable::Get(uint32_t i, uint32_t j) const
{
    return m_table[i][j];
}

double
LinkTable::MaxCapacity() const
{
    double max = 0.0;
    for (uint32_t i = 0; i < m_numNodes; ++i)
    {
        for (uint32_t j = i + 1; j < m_numNodes; ++j)
        {
            if (m_table[i][j].capacity_mbps > max)
            {
                max = m_table[i][j].capacity_mbps;
            }
        }
    }
    return max;
}

uint32_t
LinkTable::ConnectedLinkCount(double sinrThresholdDb) const
{
    uint32_t count = 0;
    for (uint32_t i = 0; i < m_numNodes; ++i)
    {
        for (uint32_t j = i + 1; j < m_numNodes; ++j)
        {
            if (m_table[i][j].sinr_db >= sinrThresholdDb)
            {
                ++count;
            }
        }
    }
    return count;
}

bool
LinkTable::IsConnected(uint32_t i, uint32_t j, double sinrThresholdDb) const
{
    return m_table[i][j].sinr_db >= sinrThresholdDb;
}

}  // namespace mesh_sim
