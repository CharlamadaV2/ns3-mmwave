/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * TopologyBuilder: creates all ns-3 nodes, installs mobility models,
 * creates buildings, wires EPC, and attaches UEs to eNBs.
 *
 * Exposes the node containers and IP interfaces needed by TrafficSetup.
 */
#pragma once

#include "src/domain/sim-config.h"

#include "ns3/channel-condition-model.h"
#include "ns3/internet-module.h"
#include "ns3/mmwave-helper.h"
#include "ns3/mmwave-point-to-point-epc-helper.h"
#include "ns3/network-module.h"

namespace mmwave_sim
{

struct MmWaveHelpers
{
    ns3::Ptr<ns3::mmwave::MmWaveHelper>                mmwaveHelper;
    ns3::Ptr<ns3::mmwave::MmWavePointToPointEpcHelper> epcHelper;
};

class TopologyBuilder
{
  public:
    TopologyBuilder(const SimConfig&                                         cfg,
                    const ns3::Ptr<ns3::mmwave::MmWaveHelper>&               mmwH,
                    const ns3::Ptr<ns3::mmwave::MmWavePointToPointEpcHelper>& epc);

    /**
     * Create MmWaveHelper and EPC helper, configured per SimConfig.
     * Must be called AFTER ConfigureChannelDefaults().
     */
    static MmWaveHelpers CreateHelpers(const SimConfig& cfg);

    /**
     * Apply channel Config::SetDefault values.
     * MUST be called BEFORE CreateObject<MmWaveHelper>() so the helper
     * picks up the correct ChannelModel and PathlossModel at construction.
     */
    static void ConfigureChannelDefaults(const SimConfig& cfg);

    /**
     * Execute the full topology build sequence:
     *   1. Set frequency via CC params
     *   2. Create eNB and UE nodes with the right mobility models
     *   3. Create building obstacles (if any) and install BuildingsHelper
     *   4. Install mmWave devices
     *   5. Wire EPC: remote host, P2P backhaul, static routing
     *   6. Assign UE IP addresses
     *   7. Attach UEs to closest eNB
     */
    void Build();

    // Accessors for TrafficSetup and VizWriter
    ns3::NodeContainer               GetEnbNodes()              const { return m_enbNodes; }
    ns3::NodeContainer               GetUeNodes()               const { return m_ueNodes; }
    ns3::Ipv4InterfaceContainer      GetUeIpInterfaces()        const { return m_ueIpIfaces; }
    ns3::Ptr<ns3::Node>              GetRemoteHost()            const { return m_remoteHost; }
    ns3::Ipv4Address                 GetRemoteHostAddr()        const { return m_remoteHostAddr; }
    ns3::Ptr<ns3::ChannelConditionModel> GetChannelConditionModel() const { return m_condModel; }

  private:
    const SimConfig&                                   m_cfg;
    ns3::Ptr<ns3::mmwave::MmWaveHelper>                m_mmwH;
    ns3::Ptr<ns3::mmwave::MmWavePointToPointEpcHelper> m_epc;

    ns3::NodeContainer                   m_enbNodes;
    ns3::NodeContainer                   m_ueNodes;
    ns3::Ipv4InterfaceContainer          m_ueIpIfaces;
    ns3::Ptr<ns3::Node>                  m_remoteHost;
    ns3::Ipv4Address                     m_remoteHostAddr;
    ns3::Ptr<ns3::ChannelConditionModel> m_condModel;

    void ConfigureChannel();
    void ConfigureChannelNyu(const std::string& sc, double freqHz);
    void ConfigureChannel3gpp(const std::string& sc, double freqHz);
    void CreateNodes();
    void InstallMobility();
    void CreateBuildings();
    void WireEpc();
    void InstallDevicesAndAttach();

    // Per-node mobility helpers
    void InstallMobilityFixed(const ns3::Ptr<ns3::Node>& node, const NodeSpec& spec);
    void InstallMobilityConstantVelocity(const ns3::Ptr<ns3::Node>& node, const NodeSpec& spec);
    void InstallMobilityRandomWalk(const ns3::Ptr<ns3::Node>& node, const NodeSpec& spec);
};

}  // namespace mmwave_sim
