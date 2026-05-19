/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
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

namespace mesh_sim
{

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
    std::string m_amcModel      = "shannon";
    bool        m_buildingsEnabled = false;

    // Per-node array gains, indexed parallel to cfg.nodes (and to mobs in
    // EvaluateAll). Resolved at Configure() from each NodeSpec's override
    // or the channel default.
    std::vector<double> m_txGainDbi;
    std::vector<double> m_rxGainDbi;

    ns3::Ptr<ns3::PropagationLossModel>  m_plModel;
    ns3::Ptr<ns3::ChannelConditionModel> m_condModel;
};

}  // namespace mesh_sim
