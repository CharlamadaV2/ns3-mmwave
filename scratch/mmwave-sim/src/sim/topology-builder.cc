/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/sim/topology-builder.h"

#include "ns3/applications-module.h"
#include "ns3/buildings-helper.h"
#include "ns3/buildings-module.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/mobility-module.h"
#include "ns3/mmwave-component-carrier.h"
#include "ns3/mmwave-phy-mac-common.h"
#include "ns3/point-to-point-helper.h"

#include <iostream>
#include <map>
#include <stdexcept>

using namespace ns3;
using namespace ns3::mmwave;

namespace mmwave_sim
{

TopologyBuilder::TopologyBuilder(const SimConfig&                                          cfg,
                                 const Ptr<MmWaveHelper>&                                  mmwH,
                                 const Ptr<MmWavePointToPointEpcHelper>&                   epc)
    : m_cfg(cfg), m_mmwH(mmwH), m_epc(epc)
{
}

void
TopologyBuilder::Build()
{
    ConfigureChannel();
    CreateNodes();
    InstallMobility();
    CreateBuildings();
    WireEpc();
    InstallDevicesAndAttach();
    // Retrieve the exact CCM the simulation is using (set during device installation)
    // so VizWriter can share the same instance.
    m_condModel = m_mmwH->GetChannelConditionModel();
}

// ---------------------------------------------------------------------------
// Channel and blockage configuration
// ---------------------------------------------------------------------------

void
TopologyBuilder::ConfigureChannel()
{
    // 3GPP channel model
    Config::SetDefault("ns3::MmWaveHelper::ChannelModel",
                       StringValue("ns3::ThreeGppSpectrumPropagationLossModel"));

    // Normalise shorthand aliases to the full 3GPP scenario strings that
    // ThreeGppChannelModel accepts.
    const std::string& sc = m_cfg.channel.scenario;
    std::string sc3gpp = sc;
    if (sc == "UMi")
    {
        sc3gpp = "UMi-StreetCanyon";
    }
    else if (sc == "UMa")
    {
        sc3gpp = "UMa";  // already canonical
    }
    Config::SetDefault("ns3::ThreeGppChannelModel::Scenario", StringValue(sc3gpp));

    if (sc3gpp == "UMi-StreetCanyon")
    {
        Config::SetDefault("ns3::MmWaveHelper::PathlossModel",
                           StringValue("ns3::ThreeGppUmiStreetCanyonPropagationLossModel"));
    }
    else if (sc3gpp == "UMa")
    {
        Config::SetDefault("ns3::MmWaveHelper::PathlossModel",
                           StringValue("ns3::ThreeGppUmaPropagationLossModel"));
    }
    else
    {
        std::cerr << "[TopologyBuilder] Unknown channel scenario '" << sc3gpp
                  << "', using UMi pathloss model.\n";
        Config::SetDefault("ns3::MmWaveHelper::PathlossModel",
                           StringValue("ns3::ThreeGppUmiStreetCanyonPropagationLossModel"));
    }

    // Blockage
    Config::SetDefault("ns3::ThreeGppChannelModel::Blockage",
                       BooleanValue(m_cfg.channel.blockage_enabled));

    // Use BuildingsChannelConditionModel when buildings are present
    if (!m_cfg.buildings.empty())
    {
        m_mmwH->SetChannelConditionModelType("ns3::BuildingsChannelConditionModel");
    }

    // Frequency via ComponentCarrier object
    Ptr<MmWavePhyMacCommon> phyMacConfig = CreateObject<MmWavePhyMacCommon>();
    phyMacConfig->SetCentreFrequency(m_cfg.channel.frequency_ghz * 1e9);

    Ptr<MmWaveComponentCarrier> cc = CreateObject<MmWaveComponentCarrier>();
    cc->SetConfigurationParameters(phyMacConfig);
    cc->SetAsPrimary(true);

    std::map<uint8_t, MmWaveComponentCarrier> ccMap;
    ccMap[0] = *cc;
    m_mmwH->SetCcPhyParams(ccMap);
}

// ---------------------------------------------------------------------------
// Node creation
// ---------------------------------------------------------------------------

void
TopologyBuilder::CreateNodes()
{
    for (const auto& spec : m_cfg.nodes)
    {
        NodeContainer nc;
        nc.Create(1);
        if (spec.role == "enb")
        {
            m_enbNodes.Add(nc);
        }
        else if (spec.role == "ue")
        {
            m_ueNodes.Add(nc);
        }
        else
        {
            throw std::runtime_error("Unknown node role '" + spec.role + "' for node '" + spec.id + "'");
        }
    }
}

// ---------------------------------------------------------------------------
// Mobility installation
// ---------------------------------------------------------------------------

void
TopologyBuilder::InstallMobility()
{
    // Collect eNB and UE specs in order
    std::vector<const NodeSpec*> enbSpecs;
    std::vector<const NodeSpec*> ueSpecs;
    for (const auto& spec : m_cfg.nodes)
    {
        if (spec.role == "enb")
        {
            enbSpecs.push_back(&spec);
        }
        else
        {
            ueSpecs.push_back(&spec);
        }
    }

    // Install per-node
    for (uint32_t i = 0; i < m_enbNodes.GetN(); i++)
    {
        InstallMobilityFixed(m_enbNodes.Get(i), *enbSpecs[i]);  // eNBs always fixed
    }

    for (uint32_t i = 0; i < m_ueNodes.GetN(); i++)
    {
        const NodeSpec& spec = *ueSpecs[i];
        if (spec.mobility == "fixed")
        {
            InstallMobilityFixed(m_ueNodes.Get(i), spec);
        }
        else if (spec.mobility == "constant_velocity")
        {
            InstallMobilityConstantVelocity(m_ueNodes.Get(i), spec);
        }
        else if (spec.mobility == "random_walk")
        {
            InstallMobilityRandomWalk(m_ueNodes.Get(i), spec);
        }
        else
        {
            throw std::runtime_error("Unknown mobility '" + spec.mobility + "' for node '" + spec.id + "'");
        }
    }
}

void
TopologyBuilder::InstallMobilityFixed(const Ptr<Node>& node, const NodeSpec& spec)
{
    MobilityHelper mob;
    Ptr<ListPositionAllocator> posAlloc = CreateObject<ListPositionAllocator>();
    posAlloc->Add(Vector(spec.position.x, spec.position.y, spec.position.z));
    mob.SetPositionAllocator(posAlloc);
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    NodeContainer nc;
    nc.Add(node);
    mob.Install(nc);
}

void
TopologyBuilder::InstallMobilityConstantVelocity(const Ptr<Node>& node, const NodeSpec& spec)
{
    MobilityHelper mob;
    mob.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    NodeContainer nc;
    nc.Add(node);
    mob.Install(nc);
    node->GetObject<MobilityModel>()->SetPosition(
        Vector(spec.position.x, spec.position.y, spec.position.z));
    node->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(
        Vector(spec.velocity.vx, spec.velocity.vy, spec.velocity.vz));
}

void
TopologyBuilder::InstallMobilityRandomWalk(const Ptr<Node>& node, const NodeSpec& spec)
{
    const auto& rw = spec.random_walk;
    MobilityHelper mob;

    // Set the initial position via the allocator so that Install places the node
    // inside the bounds before RandomWalk2dMobilityModel validates the position.
    Ptr<ListPositionAllocator> posAlloc = CreateObject<ListPositionAllocator>();
    posAlloc->Add(Vector(spec.position.x, spec.position.y, spec.position.z));
    mob.SetPositionAllocator(posAlloc);

    mob.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
        "Bounds",
        RectangleValue(Rectangle(rw.x_min, rw.x_max, rw.y_min, rw.y_max)),
        "Speed",
        StringValue("ns3::ConstantRandomVariable[Constant=" +
                    std::to_string(rw.speed_mps) + "]"),
        "Mode",
        StringValue("Time"),
        "Time",
        TimeValue(Seconds(1.0)));
    NodeContainer nc;
    nc.Add(node);
    mob.Install(nc);
}

// ---------------------------------------------------------------------------
// Building creation
// ---------------------------------------------------------------------------

static Building::BuildingType_t
parseBuildingType(const std::string& s)
{
    if (s == "Residential")  { return Building::Residential; }
    if (s == "Office")       { return Building::Office; }
    if (s == "Commercial")   { return Building::Commercial; }
    return Building::Residential;
}

static Building::ExtWallsType_t
parseExtWalls(const std::string& s)
{
    if (s == "Wood")                   { return Building::Wood; }
    if (s == "ConcreteWithWindows")    { return Building::ConcreteWithWindows; }
    if (s == "ConcreteWithoutWindows") { return Building::ConcreteWithoutWindows; }
    if (s == "StoneBlocks")            { return Building::StoneBlocks; }
    return Building::ConcreteWithWindows;
}

void
TopologyBuilder::CreateBuildings()
{
    if (m_cfg.buildings.empty())
    {
        return;
    }

    for (const auto& bspec : m_cfg.buildings)
    {
        Ptr<Building> b = Create<Building>();
        b->SetBoundaries(Box(bspec.x_min, bspec.x_max,
                             bspec.y_min, bspec.y_max,
                             bspec.z_min, bspec.z_max));
        b->SetBuildingType(parseBuildingType(bspec.type));
        b->SetExtWallsType(parseExtWalls(bspec.ext_walls));
        b->SetNFloors(bspec.n_floors);
    }

    // Install BuildingsHelper on all nodes (required for LOS/NLOS tracking)
    BuildingsHelper::Install(m_enbNodes);
    BuildingsHelper::Install(m_ueNodes);
}

// ---------------------------------------------------------------------------
// EPC wiring: remote host, backhaul, routing
// ---------------------------------------------------------------------------

void
TopologyBuilder::WireEpc()
{
    Ptr<Node> pgw = m_epc->GetPgwNode();

    // Remote host for traffic generation
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    m_remoteHost = remoteHostContainer.Get(0);

    InternetStackHelper internet;
    internet.Install(remoteHostContainer);
    internet.Install(m_ueNodes);  // must be called before AssignUeIpv4Address

    // Backhaul P2P link between EPC PGW and remote host.
    // Data rate and delay are configurable via [network] in run.ini.
    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate(m_cfg.network.backhaul_data_rate)));
    p2ph.SetDeviceAttribute("Mtu",      UintegerValue(1500));
    p2ph.SetChannelAttribute("Delay",   TimeValue(Seconds(m_cfg.network.backhaul_delay_ms / 1000.0)));

    NetDeviceContainer internetDevices = p2ph.Install(pgw, m_remoteHost);

    // PCAP capture on the backhaul link (one file per direction per device)
    if (m_cfg.pcap_enabled)
    {
        p2ph.EnablePcapAll(m_cfg.output_dir + "/pcap/backhaul");
    }
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIfaces = ipv4h.Assign(internetDevices);
    m_remoteHostAddr = internetIfaces.GetAddress(1);

    // Route from remote host → UE subnet (7.0.0.0/8 is the EPC UE address pool)
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(m_remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"),
                                               Ipv4Mask("255.0.0.0"), 1);
}

// ---------------------------------------------------------------------------
// Device installation and UE attachment
// ---------------------------------------------------------------------------

void
TopologyBuilder::InstallDevicesAndAttach()
{
    NetDeviceContainer enbDevs = m_mmwH->InstallEnbDevice(m_enbNodes);
    NetDeviceContainer ueDevs  = m_mmwH->InstallUeDevice(m_ueNodes);

    // Assign UE IP addresses now that devices exist
    m_ueIpIfaces = m_epc->AssignUeIpv4Address(NetDeviceContainer(ueDevs));

    // Set default gateway for each UE
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    for (uint32_t u = 0; u < m_ueNodes.GetN(); ++u)
    {
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(m_ueNodes.Get(u)->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(m_epc->GetUeDefaultGatewayAddress(), 1);
    }

    m_mmwH->AttachToClosestEnb(ueDevs, enbDevs);
}

}  // namespace mmwave_sim
