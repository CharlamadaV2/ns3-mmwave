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

/** @brief
*/

class TrafficMatrix
{
  public:
  /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    explicit TrafficMatrix(const SimConfig& cfg);
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void Initialize(uint32_t numNodes, double currentTime);
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void Tick(double currentTime);
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    const std::vector<Flow>& GetActiveFlows() const;
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    double GetDemand(uint32_t src, uint32_t dst) const;

  private:
  /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void InitAllPairs(uint32_t numNodes, double currentTime);
    /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void InitRandomPairs(uint32_t numNodes, double currentTime);
    /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void InitGateway(uint32_t numNodes, double currentTime);
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    Flow MakeFlow(uint32_t src, uint32_t dst, double currentTime) const;
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void TickOnOff(double currentTime);
    /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void TickPoisson(double currentTime);

    TrafficConfig              m_trafficCfg;
    std::vector<NodeSpec>      m_nodeSpecs;
    uint32_t                   m_numNodes = 0;
    double                     m_tickS    = 0.1;

    std::vector<Flow> m_flows;
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    ns3::Ptr<ns3::UniformRandomVariable>      m_uniformRng;
    /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    ns3::Ptr<ns3::ExponentialRandomVariable>   m_expRng;
};

}  // namespace mesh_sim
