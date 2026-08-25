/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * RL bridge: stdin/stdout JSON IPC between the C++ sim and a Python RL agent.
 *
 * Each tick the bridge:
 *   1. Computes reward + observation from current sim state
 *   2. Writes a JSON line to stdout
 *   3. Reads an action JSON line from stdin (unless done)
 *   4. Applies the action by setting the controlled node's velocity
 *      toward the desired position via ConstantVelocityMobilityModel
 */
#pragma once

#include "src/domain/sim-config.h"
#include "src/eval/link-table.h"
#include "src/routing/mesh-router.h"

#include "ns3/constant-velocity-mobility-model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mesh_sim
{

class RlBridge
{
  public:
    RlBridge(const SimConfig& cfg, uint32_t controlledIdx);

    // Write obs+reward to stdout, read action from stdin.
    // Returns raw action value (int for discrete, parsed internally for continuous).
    // If done==true, writes final obs but does not read action.
    void Step(uint32_t tick,
              double time_s,
              const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs,
              const LinkTable& linkTable,
              const std::vector<FlowResult>& flowResults,
              bool done);

    // Apply the last received action to the controlled node.
    // Computes desired position, derives velocity, calls SetVelocity().
    void ApplyAction(ns3::Ptr<ns3::MobilityModel> mob);

  private:
    RlConfig    m_rl;
    double      m_tickS;
    uint32_t    m_controlledIdx;
    uint32_t    m_numNodes;
    double      m_maxSpeed;
    std::string m_nodeType;

    // Last action received from Python
    int    m_lastDiscreteAction = 0;
    double m_lastTargetX = 0.0;
    double m_lastTargetY = 0.0;
    double m_lastTargetZ = 0.0;
    
    double ComputeReward(const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs,
                         const LinkTable& linkTable,
                         const std::vector<FlowResult>& flows) const;
    void WriteObs(uint32_t tick, double time_s,
                  const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs,
                  const LinkTable& linkTable,
                  double reward, bool done) const;
    void ReadAction();
};

}  // namespace mesh_sim
