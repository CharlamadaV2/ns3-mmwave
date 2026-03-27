/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * TrafficSetup: installs UDP application sources and sinks based on SimConfig.
 * Depends on TopologyBuilder for node containers and IP addresses.
 */
#pragma once

#include "src/domain/sim-config.h"
#include "src/setup/topology-builder.h"

namespace mmwave_sim
{

class TrafficSetup
{
  public:
    TrafficSetup(const SimConfig& cfg, const TopologyBuilder& topology);

    /**
     * Install applications according to cfg.traffic.direction:
     *   "dl"   - UdpClient on remote host → PacketSink on each UE
     *   "ul"   - UdpClient on each UE → PacketSink on remote host
     *   "both" - both DL and UL flows
     */
    void Install();

  private:
    const SimConfig&       m_cfg;
    const TopologyBuilder& m_topology;

    void InstallDl();
    void InstallUl();
};

}  // namespace mmwave_sim
