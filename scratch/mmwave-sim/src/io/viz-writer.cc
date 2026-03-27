/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/io/viz-writer.h"

#include "ns3/config.h"
#include "ns3/mobility-model.h"
#include "ns3/simulator.h"
#include "ns3/spectrum-value.h"

#include <cmath>
#include <iomanip>
#include <stdexcept>

using namespace ns3;

namespace mmwave_sim
{

VizWriter::VizWriter(const SimConfig&                     cfg,
                     const NodeContainer&                 enbNodes,
                     const NodeContainer&                 ueNodes,
                     Ptr<ChannelConditionModel>           condModel)
    : m_cfg(cfg),
      m_enbNodes(enbNodes),
      m_ueNodes(ueNodes),
      m_condModel(condModel),
      m_tickS(cfg.viz_tick_ms / 1000.0)
{
}

VizWriter::~VizWriter()
{
    if (m_posFile.is_open())
    {
        m_posFile.close();
    }
    if (m_linkFile.is_open())
    {
        m_linkFile.close();
    }
}

void
VizWriter::Flush()
{
    if (m_posFile.is_open())
    {
        m_posFile.flush();
    }
    if (m_linkFile.is_open())
    {
        m_linkFile.flush();
    }
}

void
VizWriter::Start()
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

    // Metadata header (consumed by useSimData.ts parseMeta)
    const uint32_t totalNodes = m_enbNodes.GetN() + m_ueNodes.GetN();
    const uint32_t simDurMs   = static_cast<uint32_t>(m_cfg.duration_s * 1000.0);
    m_posFile << "# scenario="   << m_cfg.scenario_name                      << "\n"
              << "# frequency="  << static_cast<uint64_t>(
                                        m_cfg.channel.frequency_ghz * 1e9)   << "\n"
              << "# txPower="    << m_cfg.channel.tx_power_dbm               << "\n"
              << "# numNodes="   << totalNodes                                << "\n"
              << "# simDuration=" << simDurMs                                 << "\n"
              << "# tickMs="     << m_cfg.viz_tick_ms                        << "\n"
              // positions are always written as (x,y,z); set z=0 for 2D, y=z=0 for 1D
              << "# dimensions=3\n";

    // Column headers
    m_posFile  << "time_s,node_id,x,y,z,node_type,active\n";
    // Value is -999 for ticks that fire before the first measurement arrives.
    m_linkFile << "time_s,node_a,node_b,dist_m,sinr_db,condition\n";

    // Connect to the UE PHY CQI report trace.
    // Path: NodeList -> DeviceList (MmWaveUeNetDevice) -> ComponentCarrierMap
    //       (MmWaveNetDevice attribute) -> MmWaveComponentCarrierUe -> MmWaveUePhy.
    // Fired every CQI period with per-subcarrier SINR as a SpectrumValue.
    // The callback computes the wideband average and caches it per IMSI.
    Config::ConnectWithoutContext(
        "/NodeList/*/DeviceList/*/ComponentCarrierMap/*/MmWaveUePhy/ReportCurrentCellRsrpSinr",
        MakeCallback(&VizWriter::OnSinrReport, this));

    Simulator::Schedule(Seconds(m_cfg.warmup_s), &VizWriter::Tick, this);
}

// ---------------------------------------------------------------------------
// OnSinrReport: cache latest per-UE SINR from the simulation
// ---------------------------------------------------------------------------

void
VizWriter::OnSinrReport(uint64_t imsi, SpectrumValue& sinr, SpectrumValue& /* rsrp */)
{
    // Compute wideband average SINR in linear domain across all subcarriers,
    // then convert to dB.  The SpectrumValue elements are linear SINR ratios.
    double sum = 0.0;
    uint32_t n = 0;
    for (auto it = sinr.ConstValuesBegin(); it != sinr.ConstValuesEnd(); ++it)
    {
        sum += *it;
        ++n;
    }
    const double sinr_linear_avg = (n > 0) ? sum / n : 0.0;
    const double sinr_db = (sinr_linear_avg > 0.0)
                               ? 10.0 * std::log10(sinr_linear_avg)
                               : -999.0;
    m_sinrCacheDb[imsi] = sinr_db;
}

// ---------------------------------------------------------------------------
// Tick: snapshot positions and links, then reschedule
// ---------------------------------------------------------------------------

void
VizWriter::Tick()
{
    const double t = Simulator::Now().GetSeconds();

    m_posFile  << std::fixed << std::setprecision(6);
    m_linkFile << std::fixed << std::setprecision(6);

    // --- Positions ---
    for (uint32_t i = 0; i < m_enbNodes.GetN(); i++)
    {
        Ptr<Node>          node = m_enbNodes.Get(i);
        Ptr<MobilityModel> mob  = node->GetObject<MobilityModel>();
        Vector pos = mob->GetPosition();
        m_posFile << t << "," << node->GetId() << ","
                  << pos.x << "," << pos.y << "," << pos.z
                  << ",bs,1\n";
    }
    for (uint32_t i = 0; i < m_ueNodes.GetN(); i++)
    {
        Ptr<Node>          node = m_ueNodes.Get(i);
        Ptr<MobilityModel> mob  = node->GetObject<MobilityModel>();
        Vector pos = mob->GetPosition();
        const std::string nodeType = (pos.z > 5.0) ? "air" : "ground";
        m_posFile << t << "," << node->GetId() << ","
                  << pos.x << "," << pos.y << "," << pos.z
                  << "," << nodeType << ",1\n";
    }

    // --- Links (all eNB-UE pairs) ---
    for (uint32_t e = 0; e < m_enbNodes.GetN(); e++)
    {
        Ptr<Node>          enb    = m_enbNodes.Get(e);
        Ptr<MobilityModel> mobEnb = enb->GetObject<MobilityModel>();
        Vector posEnb = mobEnb->GetPosition();

        for (uint32_t u = 0; u < m_ueNodes.GetN(); u++)
        {
            Ptr<Node>          ue    = m_ueNodes.Get(u);
            Ptr<MobilityModel> mobUe = ue->GetObject<MobilityModel>();
            Vector posUe = mobUe->GetPosition();

            const double dx   = posEnb.x - posUe.x;
            const double dy   = posEnb.y - posUe.y;
            const double dz   = posEnb.z - posUe.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

            // LOS/NLOS via buildings geometry
            Ptr<ChannelCondition> cond = m_condModel->GetChannelCondition(mobEnb, mobUe);
            const bool isLos =
                (cond->GetLosCondition() == ChannelCondition::LosConditionValue::LOS);
            const std::string condStr = isLos ? "LOS" : "NLOS";

            // UE IMSI is 1-indexed: ue at index u has IMSI u+1
            const uint64_t imsi = static_cast<uint64_t>(u + 1);
            auto it = m_sinrCacheDb.find(imsi);
            const double sinr_db = (it != m_sinrCacheDb.end()) ? it->second : -999.0;

            m_linkFile << t << ","
                       << enb->GetId() << "," << ue->GetId() << ","
                       << dist << "," << sinr_db << ","
                       << condStr << "\n";
        }
    }

    // Reschedule while simulation still has time remaining
    const double nextT = t + m_tickS;
    if (nextT <= m_cfg.duration_s)
    {
        Simulator::Schedule(Seconds(m_tickS), &VizWriter::Tick, this);
    }
}

}  // namespace mmwave_sim
