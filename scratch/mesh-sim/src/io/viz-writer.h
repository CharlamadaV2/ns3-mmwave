/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/** @brief
 * VizWriter: writes per-tick CSV snapshots for GUI and plotting consumption.
 * Called from the step loop at viz_tick_ms intervals.
 *
 * Output files:
 *   positions.csv  -- time_s, node_id, x, y, z, node_type, active
 *   links.csv      -- time_s, node_a, node_b, dist_m, sinr_db, condition,
 *                     condition_reason, capacity_mbps, delivered_mbps, hop_count
 *   rx-power.csv   -- time_s, node_a, node_b, rx_power_dbm
 *   mcs.csv        -- time_s, node_a, node_b, mcs_index, spectral_eff
 *   flows.csv      -- time_s, src, dst, demand_mbps, delivered_mbps, latency_ms,
 *                     hop_count, routable
 *   routes.csv     -- time_s, src, dst, path, bottleneck_mbps, hop_count, routable
 */
#pragma once

#include "src/domain/sim-config.h"
#include "src/eval/link-table.h"
#include "src/routing/mesh-router.h"

#include "ns3/mobility-model.h"

#include <cstdint>
#include <fstream>
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
class VizWriter
{
  public:
  /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    explicit VizWriter(const SimConfig& cfg);
    ~VizWriter();
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void Open();
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void WriteTick(double time_s,
                   const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs,
                   const LinkTable& links,
                   const std::vector<FlowResult>& flows);
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void Close();

  private:
  /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void WritePositions(double time_s,
                        const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs);
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void WriteLinks(double time_s,
                    const LinkTable& links,
                    const std::vector<FlowResult>& flows);
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void WriteRxPower(double time_s, const LinkTable& links);
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void WriteMcs(double time_s, const LinkTable& links);
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void WriteFlows(double time_s, const std::vector<FlowResult>& flows);
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void WriteRoutes(double time_s,
                     const std::vector<FlowResult>& flows,
                     const LinkTable& links);

    const SimConfig& m_cfg;
    std::ofstream    m_posFile;
    std::ofstream    m_linkFile;
    std::ofstream    m_rxPowerFile;
    std::ofstream    m_mcsFile;
    std::ofstream    m_flowFile;
    std::ofstream    m_routeFile;
    double           m_vizTickS;
    double           m_nextWriteS;
};

}  // namespace mesh_sim
