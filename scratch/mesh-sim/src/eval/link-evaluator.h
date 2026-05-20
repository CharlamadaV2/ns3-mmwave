/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

/** @brief
 * LinkEvaluator: wraps ns-3 propagation models to compute per-link
 * path loss, SINR, and capacity.  Core physics engine of mesh-sim.
 */
#pragma once

#include "src/domain/link-result.h"
#include "src/domain/sim-config.h"

#include "ns3/channel-condition-model.h"
#include "ns3/mobility-model.h"
#include "ns3/propagation-loss-model.h"

#include <string>
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
class LinkEvaluator
{
  public:
    void Configure(const SimConfig& cfg,
                   ns3::Ptr<ns3::PropagationLossModel> plModel,
                   ns3::Ptr<ns3::ChannelConditionModel> condModel);

    LinkResult Evaluate(ns3::Ptr<ns3::MobilityModel> txMob,
                        ns3::Ptr<ns3::MobilityModel> rxMob,
                        uint32_t txIdx,
                        uint32_t rxIdx) const;

    std::vector<LinkResult> EvaluateAll(
        const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs) const;

  private:
    double      m_txPowerDbm    = 30.0;
    double      m_noiseFloorDbm = -174.0;
    double      m_bandwidthHz   = 400e6;
    double      m_bfGainDb      = 0.0;
    std::string m_amcModel      = "shannon";
    bool        m_buildingsEnabled = false;

    ns3::Ptr<ns3::PropagationLossModel>  m_plModel;
    ns3::Ptr<ns3::ChannelConditionModel> m_condModel;
};

}  // namespace mesh_sim
