/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/** @brief
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
/** @brief
*/
class TopologyBuilder
{
  public:
  /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    explicit TopologyBuilder(const SimConfig& cfg);
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void Build();
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    std::vector<ns3::Ptr<ns3::MobilityModel>> GetMobilityModels() const;
    /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    ns3::Ptr<ns3::PropagationLossModel>       GetPropagationModel() const;
    /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    ns3::Ptr<ns3::ChannelConditionModel>      GetConditionModel() const;

  private:
    const SimConfig& m_cfg;

    ns3::NodeContainer                         m_nodes;
    std::vector<ns3::Ptr<ns3::MobilityModel>>  m_mobilityModels;
    ns3::Ptr<ns3::PropagationLossModel>        m_propagationModel;
    ns3::Ptr<ns3::ChannelConditionModel>       m_conditionModel;
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void CreateNodesAndMobility();
    /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void CreateBuildings();
    /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void ConfigurePropagationModel();
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void InstallMobilityFixed(const ns3::Ptr<ns3::Node>& node, const NodeSpec& spec);
    /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void InstallMobilityConstantVelocity(const ns3::Ptr<ns3::Node>& node, const NodeSpec& spec);
    /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void InstallMobilityRandomWalk(const ns3::Ptr<ns3::Node>& node, const NodeSpec& spec);
    /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void InstallMobilityWaypoint(const ns3::Ptr<ns3::Node>& node, const NodeSpec& spec);
};

}  // namespace mesh_sim
