/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/setup/traffic-setup.h"

#include "ns3/applications-module.h"
#include "ns3/inet-socket-address.h"
#include "ns3/ipv4-address.h"

using namespace ns3;

namespace mmwave_sim
{

TrafficSetup::TrafficSetup(const SimConfig& cfg, const TopologyBuilder& topology)
    : m_cfg(cfg), m_topology(topology)
{
}

void
TrafficSetup::Install()
{
    const std::string& dir = m_cfg.traffic.direction;
    if (dir == "dl" || dir == "both")
    {
        InstallDl();
    }
    if (dir == "ul" || dir == "both")
    {
        InstallUl();
    }
}

void
TrafficSetup::InstallDl()
{
    const auto& tc        = m_cfg.traffic;
    NodeContainer ueNodes = m_topology.GetUeNodes();
    Ipv4InterfaceContainer ueIpIfaces = m_topology.GetUeIpInterfaces();
    Ptr<Node> remoteHost  = m_topology.GetRemoteHost();

    uint16_t dlPort = 1234;

    ApplicationContainer serverApps;
    ApplicationContainer clientApps;

    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        // Sink on UE
        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), dlPort));
        serverApps.Add(sinkHelper.Install(ueNodes.Get(u)));

        // Source on remote host → UE
        UdpClientHelper client(ueIpIfaces.GetAddress(u), dlPort);
        client.SetAttribute("Interval",    TimeValue(MicroSeconds(static_cast<uint64_t>(tc.inter_packet_interval_us))));
        client.SetAttribute("MaxPackets",  UintegerValue(tc.max_packets == 0 ? 0xFFFFFFFF : tc.max_packets));
        client.SetAttribute("PacketSize",  UintegerValue(tc.packet_size_bytes));
        clientApps.Add(client.Install(remoteHost));
    }

    serverApps.Start(Seconds(tc.app_start_offset_s));
    clientApps.Start(Seconds(tc.app_start_offset_s));
    serverApps.Stop(Seconds(m_cfg.duration_s));
    clientApps.Stop(Seconds(m_cfg.duration_s));
}

void
TrafficSetup::InstallUl()
{
    const auto& tc        = m_cfg.traffic;
    NodeContainer ueNodes = m_topology.GetUeNodes();
    Ipv4Address remoteHostAddr = m_topology.GetRemoteHostAddr();
    Ptr<Node> remoteHost  = m_topology.GetRemoteHost();

    uint16_t ulBasePort = 2000;

    ApplicationContainer serverApps;
    ApplicationContainer clientApps;

    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        uint16_t ulPort = ulBasePort + u + 1;

        // Sink on remote host
        PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), ulPort));
        serverApps.Add(sinkHelper.Install(remoteHost));

        // Source on UE → remote host
        UdpClientHelper client(remoteHostAddr, ulPort);
        client.SetAttribute("Interval",   TimeValue(MicroSeconds(static_cast<uint64_t>(tc.inter_packet_interval_us))));
        client.SetAttribute("MaxPackets", UintegerValue(tc.max_packets == 0 ? 0xFFFFFFFF : tc.max_packets));
        client.SetAttribute("PacketSize", UintegerValue(tc.packet_size_bytes));
        clientApps.Add(client.Install(ueNodes.Get(u)));
    }

    serverApps.Start(Seconds(tc.app_start_offset_s));
    clientApps.Start(Seconds(tc.app_start_offset_s));
    serverApps.Stop(Seconds(m_cfg.duration_s));
    clientApps.Stop(Seconds(m_cfg.duration_s));
}

}  // namespace mmwave_sim
