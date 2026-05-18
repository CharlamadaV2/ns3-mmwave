/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * TopologyBuilder: creates ns-3 mobility models, buildings, and propagation
 * models for the mesh topology.  This is the only layer that creates ns-3
 * objects.  No EPC, RRC, MAC, or protocol stack.
 */
#pragma once

#include "src/domain/sim-config.h"

#include "ns3/channel-condition-model.h"
#include "ns3/mobility-model.h"
#include "ns3/node-container.h"
#include "ns3/propagation-loss-model.h"

#include <vector>

namespace mesh_sim
{

class TopologyBuilder
{
  public:
    explicit TopologyBuilder(const SimConfig& cfg);

    void Build();

    std::vector<ns3::Ptr<ns3::MobilityModel>> GetMobilityModels() const;
    ns3::Ptr<ns3::PropagationLossModel>       GetPropagationModel() const;
    ns3::Ptr<ns3::ChannelConditionModel>      GetConditionModel() const;

  private:
    const SimConfig& m_cfg;

    ns3::NodeContainer                         m_nodes;
    std::vector<ns3::Ptr<ns3::MobilityModel>>  m_mobilityModels;
    ns3::Ptr<ns3::PropagationLossModel>        m_propagationModel;
    ns3::Ptr<ns3::ChannelConditionModel>       m_conditionModel;

    void CreateNodesAndMobility();
    void CreateBuildings();
    void ConfigurePropagationModel();

    void InstallMobilityFixed(const ns3::Ptr<ns3::Node>& node, const NodeSpec& spec);
    void InstallMobilityConstantVelocity(const ns3::Ptr<ns3::Node>& node, const NodeSpec& spec);
    void InstallMobilityRandomWalk(const ns3::Ptr<ns3::Node>& node, const NodeSpec& spec);
    void InstallMobilityWaypoint(const ns3::Ptr<ns3::Node>& node, const NodeSpec& spec);
};

}  // namespace mesh_sim
