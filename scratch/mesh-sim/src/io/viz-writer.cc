/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/io/viz-writer.h"

#include "ns3/mobility-model.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <utility>

using namespace ns3;

namespace mesh_sim
{

VizWriter::VizWriter(const SimConfig& cfg)
    : m_cfg(cfg),
      m_vizTickS(cfg.viz_tick_ms / 1000.0),
      m_nextWriteS(0.0)
{
}

VizWriter::~VizWriter()
{
    Close();
}

void
VizWriter::Open()
{
    const std::string posPath  = m_cfg.output_dir + "/positions.csv";
    const std::string linkPath = m_cfg.output_dir + "/links.csv";

    m_posFile.open(posPath);
    if (!m_posFile.is_open())
    {
        throw std::runtime_error("VizWriter: cannot open " + posPath);
    }

    m_linkFile.open(linkPath);
    if (!m_linkFile.is_open())
    {
        throw std::runtime_error("VizWriter: cannot open " + linkPath);
    }

    // Metadata header (consumed by GUI's parseMeta)
    uint32_t numNodes  = static_cast<uint32_t>(m_cfg.nodes.size());
    uint32_t simDurMs  = static_cast<uint32_t>(m_cfg.duration_s * 1000.0);
    uint64_t freqHz    = static_cast<uint64_t>(m_cfg.channel.frequency_ghz * 1e9);

    m_posFile << "# scenario="    << m_cfg.scenario_name  << "\n"
              << "# frequency="   << freqHz               << "\n"
              << "# txPower="     << m_cfg.channel.tx_power_dbm << "\n"
              << "# numNodes="    << numNodes              << "\n"
              << "# simDuration=" << simDurMs              << "\n"
              << "# tickMs="      << m_cfg.viz_tick_ms     << "\n"
              << "# dimensions=3\n";

    m_posFile  << "time_s,node_id,x,y,z,node_type,active\n";
    m_linkFile << "time_s,node_a,node_b,dist_m,sinr_db,condition,"
                  "capacity_mbps,delivered_mbps,hop_count\n";
}

void
VizWriter::WriteTick(double time_s,
                     const std::vector<Ptr<MobilityModel>>& mobs,
                     const LinkTable& links,
                     const std::vector<FlowResult>& flows)
{
    if (time_s < m_nextWriteS - 1e-9)
    {
        return;
    }

    WritePositions(time_s, mobs);
    WriteLinks(time_s, links, flows);

    m_nextWriteS += m_vizTickS;
}

void
VizWriter::Close()
{
    if (m_posFile.is_open())
    {
        m_posFile.flush();
        m_posFile.close();
    }
    if (m_linkFile.is_open())
    {
        m_linkFile.flush();
        m_linkFile.close();
    }
}

// ---------------------------------------------------------------------------

void
VizWriter::WritePositions(double time_s,
                          const std::vector<Ptr<MobilityModel>>& mobs)
{
    m_posFile << std::fixed << std::setprecision(6);

    for (uint32_t i = 0; i < mobs.size(); ++i)
    {
        Vector pos = mobs[i]->GetPosition();
        m_posFile << time_s << "," << i << ","
                  << pos.x << "," << pos.y << "," << pos.z
                  << ",peer,1\n";
    }
}

void
VizWriter::WriteLinks(double time_s,
                      const LinkTable& links,
                      const std::vector<FlowResult>& flows)
{
    // Build per-edge map from routable flows
    using Edge = std::pair<uint32_t, uint32_t>;
    struct EdgeInfo
    {
        double   delivered_mbps = 0.0;
        uint32_t max_hop_count  = 0;
    };
    std::map<Edge, EdgeInfo> edgeMap;

    for (const auto& fr : flows)
    {
        if (!fr.routable || fr.path.size() < 2)
        {
            continue;
        }
        for (size_t k = 0; k + 1 < fr.path.size(); ++k)
        {
            uint32_t a = std::min(fr.path[k], fr.path[k + 1]);
            uint32_t b = std::max(fr.path[k], fr.path[k + 1]);
            auto& info = edgeMap[{a, b}];
            info.delivered_mbps += fr.delivered_mbps;
            info.max_hop_count = std::max(info.max_hop_count, fr.hop_count);
        }
    }

    uint32_t N = links.NumNodes();
    m_linkFile << std::fixed << std::setprecision(6);

    for (uint32_t i = 0; i < N; ++i)
    {
        for (uint32_t j = i + 1; j < N; ++j)
        {
            const LinkResult& lr = links.Get(i, j);
            const char* cond = lr.is_los ? "LOS" : "NLOS";

            double delivered = 0.0;
            uint32_t hops    = 0;
            auto it = edgeMap.find({i, j});
            if (it != edgeMap.end())
            {
                delivered = it->second.delivered_mbps;
                hops      = it->second.max_hop_count;
            }

            m_linkFile << time_s << ","
                       << i << "," << j << ","
                       << lr.distance_m << ","
                       << lr.sinr_db << ","
                       << cond << ","
                       << std::setprecision(1)
                       << lr.capacity_mbps << ","
                       << delivered << ","
                       << hops << "\n";
            m_linkFile << std::setprecision(6);
        }
    }
}

}  // namespace mesh_sim
