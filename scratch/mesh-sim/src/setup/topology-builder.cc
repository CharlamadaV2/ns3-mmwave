/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/** @file topology-builder.cc*/

#include "src/setup/topology-builder.h"

#include "ns3/boolean.h"
#include "ns3/buildings-helper.h"
#include "ns3/buildings-module.h"
#include "ns3/constant-position-mobility-model.h"
#include "ns3/constant-velocity-mobility-model.h"
#include "ns3/double.h"
#include "ns3/mobility-helper.h"
#include "ns3/nyu-propagation-loss-model.h"
#include "ns3/string.h"
#include "ns3/three-gpp-propagation-loss-model.h"
#include "ns3/waypoint-mobility-model.h"
#include "ns3/waypoint.h"

#include <stdexcept>

using namespace ns3;

namespace mesh_sim
{

// ---------------------------------------------------------------------------
// Building-spec string → ns-3 enum helpers
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

// ---------------------------------------------------------------------------
// Construction / orchestration
// ---------------------------------------------------------------------------

TopologyBuilder::TopologyBuilder(const SimConfig& cfg)
    : m_cfg(cfg)
{
}

void
TopologyBuilder::Build()
{
    CreateNodesAndMobility();
    CreateJammersAndMobility();
    CreateBuildings();
    ConfigurePropagationModel();
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

std::vector<Ptr<MobilityModel>>
TopologyBuilder::GetMobilityModels() const
{
    return m_mobilityModels;
}

std::vector<Ptr<MobilityModel>>
TopologyBuilder::GetJammerMobilityModels() const
{
    return m_jammerMobilityModels;
}

Ptr<PropagationLossModel>
TopologyBuilder::GetPropagationModel() const
{
    return m_propagationModel;
}

Ptr<ChannelConditionModel>
TopologyBuilder::GetConditionModel() const
{
    return m_conditionModel;
}

// ---------------------------------------------------------------------------
// Node creation and mobility installation
// ---------------------------------------------------------------------------

void
TopologyBuilder::CreateNodesAndMobility()
{
    for (const auto& spec : m_cfg.nodes)
    {
        NodeContainer nc;
        nc.Create(1);
        Ptr<Node> node = nc.Get(0);
        m_nodes.Add(nc);

        if (spec.mobility == "fixed")
        {
            InstallMobilityFixed(node, spec);
        }
        else if (spec.mobility == "constant_velocity")
        {
            InstallMobilityConstantVelocity(node, spec);
        }
        else if (spec.mobility == "random_walk")
        {
            InstallMobilityRandomWalk(node, spec);
        }
        else if (spec.mobility == "waypoint")
        {
            InstallMobilityWaypoint(node, spec);
        }
        else
        {
            throw std::runtime_error("Unknown mobility '" + spec.mobility +
                                     "' for node '" + spec.id + "'");
        }

        m_mobilityModels.push_back(node->GetObject<MobilityModel>());
    }
}

// ---------------------------------------------------------------------------
// Jammer creation and mobility installation
// ---------------------------------------------------------------------------

void
TopologyBuilder::CreateJammersAndMobility()
{
    // One ns-3 mobility model per jammer, in the SAME order as cfg.jammers
    // (JammerModel::Configure asserts the two vectors are equal length).
    for (const auto& spec : m_cfg.jammers)
    {
        NodeContainer nc;
        nc.Create(1);
        Ptr<Node> node = nc.Get(0);
        m_nodes.Add(nc);

        MobilityHelper mob;

        if (!spec.waypoints.empty())
        {
            mob.SetMobilityModel("ns3::WaypointMobilityModel");
            NodeContainer one; one.Add(node);
            mob.Install(one);
            Ptr<WaypointMobilityModel> wm = node->GetObject<WaypointMobilityModel>();
            for (const auto& wp : spec.waypoints)
            {
                wm->AddWaypoint(ns3::Waypoint(Seconds(wp.t), Vector(wp.x, wp.y, wp.z)));
            }
        }
        else if (spec.velocity.vx != 0.0 || spec.velocity.vy != 0.0 ||
                 spec.velocity.vz != 0.0)
        {
            mob.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
            NodeContainer one; one.Add(node);
            mob.Install(one);
            node->GetObject<MobilityModel>()->SetPosition(
                Vector(spec.position.x, spec.position.y, spec.position.z));
            node->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(
                Vector(spec.velocity.vx, spec.velocity.vy, spec.velocity.vz));
        }
        else
        {
            // Stationary jammer (the common case for the logged EW positions).
            Ptr<ListPositionAllocator> posAlloc = CreateObject<ListPositionAllocator>();
            posAlloc->Add(Vector(spec.position.x, spec.position.y, spec.position.z));
            mob.SetPositionAllocator(posAlloc);
            mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
            NodeContainer one; one.Add(node);
            mob.Install(one);
        }

        m_jammerMobilityModels.push_back(node->GetObject<MobilityModel>());
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
TopologyBuilder::InstallMobilityWaypoint(const Ptr<Node>& node, const NodeSpec& spec)
{
    MobilityHelper mob;
    mob.SetMobilityModel("ns3::WaypointMobilityModel");
    NodeContainer nc;
    nc.Add(node);
    mob.Install(nc);

    Ptr<WaypointMobilityModel> wm = node->GetObject<WaypointMobilityModel>();
    for (const auto& wp : spec.waypoints)
    {
        wm->AddWaypoint(ns3::Waypoint(Seconds(wp.t), Vector(wp.x, wp.y, wp.z)));
    }
}

void
TopologyBuilder::InstallMobilityRandomWalk(const Ptr<Node>& node, const NodeSpec& spec)
{
    const auto& rw = spec.random_walk;
    MobilityHelper mob;

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

void
TopologyBuilder::CreateBuildings()
{
    // BuildingsChannelConditionModel needs MobilityBuildingInfo on every
    // node, even with zero buildings, so install the helper whenever the
    // user asked for static_los.
    const bool needs_helper =
        !m_cfg.buildings.empty() || m_cfg.channel.condition_model == "static_los";
    if (!needs_helper)
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
        b->SetNRoomsX(bspec.n_rooms_x);
        b->SetNRoomsY(bspec.n_rooms_y);
    }

    BuildingsHelper::Install(m_nodes);
}

// ---------------------------------------------------------------------------
// Propagation model configuration
// ---------------------------------------------------------------------------

void
TopologyBuilder::ConfigurePropagationModel()
{
    const auto& ch = m_cfg.channel;
    const std::string& sc = ch.scenario;
    const double freqHz = ch.frequency_ghz * 1e9;

    if (ch.channel_model == "nyu")
    {
        // --- NYU propagation model ---
        Ptr<NYUPropagationLossModel> plModel;
        if      (sc == "UMi") { plModel = CreateObject<NYUUmiPropagationLossModel>(); }
        else if (sc == "UMa") { plModel = CreateObject<NYUUmaPropagationLossModel>(); }
        else if (sc == "RMa") { plModel = CreateObject<NYURmaPropagationLossModel>(); }
        else if (sc == "InH") { plModel = CreateObject<NYUInHPropagationLossModel>(); }
        else if (sc == "InF") { plModel = CreateObject<NYUInFPropagationLossModel>(); }
        else
        {
            throw std::runtime_error("Unknown NYU scenario '" + sc +
                                     "'. Valid: UMi, UMa, RMa, InH, InF.");
        }

        plModel->SetFrequency(freqHz);

        const auto& nyu = ch.nyu;
        plModel->SetAttribute("ShadowingEnabled",       BooleanValue(nyu.shadowing_enabled));
        plModel->SetAttribute("Pressure",               DoubleValue(nyu.pressure_mbar));
        plModel->SetAttribute("Humidity",               DoubleValue(nyu.humidity_pct));
        plModel->SetAttribute("Temperature",            DoubleValue(nyu.temperature_c));
        plModel->SetAttribute("RainRate",               DoubleValue(nyu.rain_rate_mm_hr));
        plModel->SetAttribute("AtmosphericLossEnabled", BooleanValue(nyu.atmospheric_loss_enabled));
        plModel->SetAttribute("FoliageLossEnabled",     BooleanValue(nyu.foliage_loss_enabled));
        plModel->SetAttribute("FoliageLoss",            DoubleValue(nyu.foliage_loss_db_m));
        plModel->SetAttribute("O2ILosstype",            StringValue(nyu.o2i_loss_type));

        if (ch.condition_model == "static_los" || !m_cfg.buildings.empty())
        {
            auto ccm = CreateObject<BuildingsChannelConditionModel>();
            plModel->SetChannelConditionModel(ccm);
            m_conditionModel = ccm;
        }
        else
        {
            m_conditionModel = plModel->GetChannelConditionModel();
        }

        m_propagationModel = plModel;
    }
    else
    {
        // --- 3GPP propagation model ---
        Ptr<ThreeGppPropagationLossModel> plModel;
        if      (sc == "UMi")
        {
            plModel = CreateObject<ThreeGppUmiStreetCanyonPropagationLossModel>();
        }
        else if (sc == "UMa")
        {
            plModel = CreateObject<ThreeGppUmaPropagationLossModel>();
        }
        else if (sc == "RMa")
        {
            plModel = CreateObject<ThreeGppRmaPropagationLossModel>();
        }
        else if (sc == "InH" || sc == "InH-Mixed")
        {
            plModel = CreateObject<ThreeGppIndoorOfficePropagationLossModel>();
        }
        else if (sc == "InH-Open")
        {
            plModel = CreateObject<ThreeGppIndoorOfficePropagationLossModel>();
        }
        else
        {
            throw std::runtime_error("Unknown 3GPP scenario '" + sc +
                                     "'. Valid: UMi, UMa, RMa, InH, InH-Mixed, InH-Open.");
        }

        plModel->SetFrequency(freqHz);

        if (ch.condition_model == "static_los" || !m_cfg.buildings.empty())
        {
            auto ccm = CreateObject<BuildingsChannelConditionModel>();
            plModel->SetChannelConditionModel(ccm);
            m_conditionModel = ccm;
        }
        else
        {
            m_conditionModel = plModel->GetChannelConditionModel();
        }

        m_propagationModel = plModel;
    }
}

}  // namespace mesh_sim
