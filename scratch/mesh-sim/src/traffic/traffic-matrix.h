/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * TrafficMatrix: flow-level traffic demand generation for the mesh.
 * Supports constant-rate, Poisson arrival, and on-off (bursty) models.
 * No packets -- just "node A wants X Mbps to node B."
 */
#pragma once

#include "src/domain/sim-config.h"

#include "ns3/random-variable-stream.h"

#include <cstdint>
#include <vector>

namespace mesh_sim
{

struct Flow
{
    uint32_t src = 0;
    uint32_t dst = 0;
    double   demand_mbps  = 0.0;
    double   start_time_s = 0.0;
    double   end_time_s   = 0.0;  // 0 = permanent
    bool     active       = true;

    // on_off model state
    bool   in_on_phase = true;
    double phase_end_s = 0.0;
};

class TrafficMatrix
{
  public:
    explicit TrafficMatrix(const SimConfig& cfg);

    void Initialize(uint32_t numNodes, double currentTime);

    void Tick(double currentTime);

    const std::vector<Flow>& GetActiveFlows() const;

    double GetDemand(uint32_t src, uint32_t dst) const;

  private:
    void InitAllPairs(uint32_t numNodes, double currentTime);
    void InitRandomPairs(uint32_t numNodes, double currentTime);
    void InitGateway(uint32_t numNodes, double currentTime);

    Flow MakeFlow(uint32_t src, uint32_t dst, double currentTime) const;

    void TickOnOff(double currentTime);
    void TickPoisson(double currentTime);

    TrafficConfig              m_trafficCfg;
    std::vector<NodeSpec>      m_nodeSpecs;
    uint32_t                   m_numNodes = 0;
    double                     m_tickS    = 0.1;

    std::vector<Flow> m_flows;

    ns3::Ptr<ns3::UniformRandomVariable>      m_uniformRng;
    ns3::Ptr<ns3::ExponentialRandomVariable>   m_expRng;
};

}  // namespace mesh_sim
