/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * VizWriter: periodically snapshots node positions and per-link channel state
 * into positions.csv and links.csv for consumption by the ns3-mmwave-viz tool.
 *
 * Fires at cfg.viz_tick_ms intervals starting from t=0.
 *
 * Link SINR values are sourced from the MmWaveUePhy/ReportCurrentCellRsrpSinr
 * trace, which fires every CQI period directly from the UE PHY.  The wideband
 * average SINR is computed from the SpectrumValue and cached per IMSI.
 * For ticks before any SINR report arrives the sinr_db column outputs -999.
 *
 * positions.csv columns:  time_s, node_id, x, y, z, node_type, active
 * links.csv columns:      time_s, node_a, node_b, dist_m, sinr_db, condition
 */
#pragma once

#include "src/domain/types.h"

#include "ns3/channel-condition-model.h"
#include "ns3/network-module.h"
#include "ns3/spectrum-value.h"

#include <fstream>
#include <map>

namespace mmwave_sim
{

class VizWriter
{
  public:
    VizWriter(const SimConfig&                     cfg,
              const ns3::NodeContainer&            enbNodes,
              const ns3::NodeContainer&            ueNodes,
              ns3::Ptr<ns3::ChannelConditionModel> condModel);

    ~VizWriter();

    /**
     * Open output files, connect SINR trace callbacks, and schedule first tick.
     * Must be called after topology is built (devices and nodes exist) but
     * before Simulator::Run().
     */
    void Start();

  private:
    /** Called at each tick; writes one row per node / per link, then reschedules. */
    void Tick();

    /**
     * Trace sink connected to MmWaveUePhy/ReportCurrentCellRsrpSinr.
     * Computes the wideband average SINR from the SpectrumValue and caches it
     * per IMSI (dB).  Fired by the UE PHY every CQI reporting period.
     */
    void OnSinrReport(uint64_t imsi, ns3::SpectrumValue& sinr, ns3::SpectrumValue& rsrp);

    const SimConfig&          m_cfg;
    ns3::NodeContainer        m_enbNodes;
    ns3::NodeContainer        m_ueNodes;
    ns3::Ptr<ns3::ChannelConditionModel> m_condModel;

    std::ofstream m_posFile;
    std::ofstream m_linkFile;
    double        m_tickS;

    // Latest SINR (dB) per UE, keyed by IMSI (1-indexed, matching nodes.json order).
    // Updated by OnSinrReport; read by Tick().
    std::map<uint64_t, double> m_sinrCacheDb;
};

}  // namespace mmwave_sim
