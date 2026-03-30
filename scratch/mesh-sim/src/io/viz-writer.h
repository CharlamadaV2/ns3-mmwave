/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * VizWriter: writes positions.csv and links.csv for GUI consumption.
 * Called from the step loop at viz_tick_ms intervals.
 *
 * positions.csv columns:  time_s, node_id, x, y, z, node_type, active
 * links.csv columns:      time_s, node_a, node_b, dist_m, sinr_db, condition,
 *                          capacity_mbps, delivered_mbps, hop_count
 */
#pragma once

#include "src/domain/sim-config.h"
#include "src/eval/link-table.h"
#include "src/routing/mesh-router.h"

#include "ns3/mobility-model.h"

#include <cstdint>
#include <fstream>
#include <vector>

namespace mesh_sim
{

class VizWriter
{
  public:
    explicit VizWriter(const SimConfig& cfg);
    ~VizWriter();

    void Open();

    void WriteTick(double time_s,
                   const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs,
                   const LinkTable& links,
                   const std::vector<FlowResult>& flows);

    void Close();

  private:
    void WritePositions(double time_s,
                        const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs);

    void WriteLinks(double time_s,
                    const LinkTable& links,
                    const std::vector<FlowResult>& flows);

    const SimConfig& m_cfg;
    std::ofstream    m_posFile;
    std::ofstream    m_linkFile;
    double           m_vizTickS;
    double           m_nextWriteS;
};

}  // namespace mesh_sim
