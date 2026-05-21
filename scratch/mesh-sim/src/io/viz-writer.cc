/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/** @file viz-writer.cc */

#include "src/io/viz-writer.h"
#include "src/eval/sinr-capacity.h"

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

// ---------------------------------------------------------------------------
// Open / Close
// ---------------------------------------------------------------------------

void
VizWriter::Open()
{
    const std::string& dir = m_cfg.output_dir;

    auto openFile = [&](std::ofstream& f, const std::string& name)
    {
        std::string path = dir + "/" + name;
        f.open(path);
        if (!f.is_open())
        {
            throw std::runtime_error("VizWriter: cannot open " + path);
        }
    };

    openFile(m_posFile,     "positions.csv");
    openFile(m_linkFile,    "links.csv");
    openFile(m_rxPowerFile, "rx-power.csv");
    openFile(m_mcsFile,     "mcs.csv");
    openFile(m_flowFile,    "flows.csv");
    openFile(m_routeFile,   "routes.csv");

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
              << "# dimensions=3\n"
              << "# rainRate="    << m_cfg.channel.nyu.rain_rate_mm_hr << "\n"
              << "# channelModel=" << m_cfg.channel.channel_model << "\n"
              << "# flowTopology=" << m_cfg.mesh.traffic.flow_topology << "\n"
              << "# trafficModel=" << m_cfg.mesh.traffic.model << "\n";

    m_posFile     << "time_s,node_id,x,y,z,node_type,active\n";
    m_linkFile    << "time_s,node_a,node_b,dist_m,sinr_db,condition,"
                     "condition_reason,capacity_mbps,delivered_mbps,hop_count\n";
    m_rxPowerFile << "time_s,node_a,node_b,rx_power_dbm\n";
    m_mcsFile     << "time_s,node_a,node_b,mcs_index,spectral_eff\n";
    m_flowFile    << "time_s,src,dst,demand_mbps,delivered_mbps,"
                     "latency_ms,hop_count,routable\n";
    m_routeFile   << "time_s,src,dst,path,bottleneck_mbps,hop_count,routable\n";
}

void
VizWriter::Close()
{
    auto closeFile = [](std::ofstream& f)
    {
        if (f.is_open())
        {
            f.flush();
            f.close();
        }
    };

    closeFile(m_posFile);
    closeFile(m_linkFile);
    closeFile(m_rxPowerFile);
    closeFile(m_mcsFile);
    closeFile(m_flowFile);
    closeFile(m_routeFile);
}

// ---------------------------------------------------------------------------
// WriteTick
// ---------------------------------------------------------------------------

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
    WriteRxPower(time_s, links);
    WriteMcs(time_s, links);
    WriteFlows(time_s, flows);
    WriteRoutes(time_s, flows, links);

    m_nextWriteS += m_vizTickS;
}

// ---------------------------------------------------------------------------
// WritePositions
// ---------------------------------------------------------------------------

void
VizWriter::WritePositions(double time_s,
                          const std::vector<Ptr<MobilityModel>>& mobs)
{
    m_posFile << std::fixed << std::setprecision(6);

    for (uint32_t i = 0; i < mobs.size(); ++i)
    {
        Vector pos = mobs[i]->GetPosition();
        const std::string& role = (i < m_cfg.nodes.size()) ? m_cfg.nodes[i].role : "peer";
        m_posFile << time_s << "," << i << ","
                  << pos.x << "," << pos.y << "," << pos.z
                  << "," << role << ",1\n";
    }
}

// ---------------------------------------------------------------------------
// WriteLinks
// ---------------------------------------------------------------------------

void
VizWriter::WriteLinks(double time_s,
                      const LinkTable& links,
                      const std::vector<FlowResult>& flows)
{
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
            const char* reason = lr.condition_from_buildings ? "building" : "probabilistic";

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
                       << reason << ","
                       << std::setprecision(1)
                       << lr.capacity_mbps << ","
                       << delivered << ","
                       << hops << "\n";
            m_linkFile << std::setprecision(6);
        }
    }
}

// ---------------------------------------------------------------------------
// WriteRxPower
// ---------------------------------------------------------------------------

void
VizWriter::WriteRxPower(double time_s, const LinkTable& links)
{
    uint32_t N = links.NumNodes();
    m_rxPowerFile << std::fixed << std::setprecision(6);

    for (uint32_t i = 0; i < N; ++i)
    {
        for (uint32_t j = i + 1; j < N; ++j)
        {
            const LinkResult& lr = links.Get(i, j);
            m_rxPowerFile << time_s << ","
                          << i << "," << j << ","
                          << lr.rx_power_dbm << "\n";
        }
    }
}

// ---------------------------------------------------------------------------
// WriteMcs
// ---------------------------------------------------------------------------

void
VizWriter::WriteMcs(double time_s, const LinkTable& links)
{
    uint32_t N = links.NumNodes();
    m_mcsFile << std::fixed << std::setprecision(6);

    for (uint32_t i = 0; i < N; ++i)
    {
        for (uint32_t j = i + 1; j < N; ++j)
        {
            const LinkResult& lr = links.Get(i, j);
            double spectral_eff = MCS_TABLE[lr.mcs_index].spectral_eff;
            m_mcsFile << time_s << ","
                      << i << "," << j << ","
                      << lr.mcs_index << ","
                      << std::setprecision(2)
                      << spectral_eff << "\n";
            m_mcsFile << std::setprecision(6);
        }
    }
}

// ---------------------------------------------------------------------------
// WriteFlows
// ---------------------------------------------------------------------------

void
VizWriter::WriteFlows(double time_s, const std::vector<FlowResult>& flows)
{
    m_flowFile << std::fixed << std::setprecision(6);

    for (const auto& fr : flows)
    {
        m_flowFile << time_s << ","
                   << fr.src << "," << fr.dst << ","
                   << std::setprecision(1)
                   << fr.demand_mbps << ","
                   << fr.delivered_mbps << ","
                   << std::setprecision(3)
                   << fr.latency_ms << ","
                   << fr.hop_count << ","
                   << (fr.routable ? 1 : 0) << "\n";
        m_flowFile << std::setprecision(6);
    }
}

// ---------------------------------------------------------------------------
// WriteRoutes
// ---------------------------------------------------------------------------

void
VizWriter::WriteRoutes(double time_s,
                       const std::vector<FlowResult>& flows,
                       const LinkTable& links)
{
    m_routeFile << std::fixed << std::setprecision(6);

    for (const auto& fr : flows)
    {
        // Build path string (semicolon-separated node indices)
        std::string pathStr;
        for (size_t k = 0; k < fr.path.size(); ++k)
        {
            if (k > 0)
            {
                pathStr += ';';
            }
            pathStr += std::to_string(fr.path[k]);
        }

        // Bottleneck = min capacity along the path
        double bottleneck = 0.0;
        if (fr.routable && fr.path.size() >= 2)
        {
            bottleneck = 1e9;
            for (size_t k = 0; k + 1 < fr.path.size(); ++k)
            {
                double cap = links.Get(fr.path[k], fr.path[k + 1]).capacity_mbps;
                bottleneck = std::min(bottleneck, cap);
            }
        }

        m_routeFile << time_s << ","
                    << fr.src << "," << fr.dst << ","
                    << pathStr << ","
                    << std::setprecision(1)
                    << bottleneck << ","
                    << fr.hop_count << ","
                    << (fr.routable ? 1 : 0) << "\n";
        m_routeFile << std::setprecision(6);
    }
}

}  // namespace mesh_sim
