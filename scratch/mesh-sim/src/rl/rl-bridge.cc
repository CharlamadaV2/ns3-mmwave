/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

#include "src/rl/rl-bridge.h"
#include "third_party/json.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;

namespace mesh_sim
{

namespace
{
constexpr double kWallEps = 1e-6;
constexpr int    kHold = 4;
}  // namespace

RlBridge::RlBridge(const SimConfig& cfg)
    : m_rl(cfg.rl),
      m_tickS(cfg.tick_s),
      m_controlledIdx(cfg.rl.controlled_indices.empty()
                          ? static_cast<uint32_t>(cfg.nodes.size()) - 1
                          : cfg.rl.controlled_indices[0]),
      m_numNodes(static_cast<uint32_t>(cfg.nodes.size())),
      m_maxSpeed(MaxSpeedForType(cfg.nodes[m_controlledIdx].node_type)),
      m_nodeType(cfg.nodes[m_controlledIdx].node_type),
      m_centralized(cfg.rl.control_mode == "centralized")
{
    m_numSlots = m_centralized ? std::max(1u, cfg.rl.num_slots) : 1u;
    m_k        = std::max(1u, cfg.rl.decision_interval_ticks);
    m_numTicks = cfg.rl.num_ticks;

    m_slots.resize(m_numSlots);
    const uint32_t resolved = static_cast<uint32_t>(cfg.rl.controlled_indices.size());
    for (uint32_t i = 0; i < m_numSlots; ++i)
    {
        if (!m_centralized || i >= resolved)
        {
            if (i == 0)
            {
                // Legacy: the single controlled node, speed capped as today.
                m_slots[0] = ControlSlot{m_controlledIdx, cfg.nodes[m_controlledIdx].id,
                                         m_maxSpeed, true};
            }
            continue;
        }
        const uint32_t idx = cfg.rl.controlled_indices[i];
        const auto& spec = cfg.nodes[idx];
        m_slots[i] = ControlSlot{idx, spec.id,
                                 std::min(m_rl.step_size_m / m_tickS,
                                          MaxSpeedForType(spec.node_type)),
                                 true};
    }

    m_lastJoint.assign(m_numSlots, kHold);
    m_lastMask.assign(5 * m_numSlots, 0);

    m_nodeIds.reserve(cfg.nodes.size());
    for (const auto& spec : cfg.nodes)
    {
        m_nodeIds.push_back(spec.id);
    }
    m_band    = cfg.band;
    m_warmupS = cfg.warmup_s;

    size_t jammersEnabled = 0;
    for (const auto& j : cfg.jammers)
    {
        if (j.enabled)
        {
            ++jammersEnabled;
        }
    }
    m_jammerPathEnabled = (cfg.band == "sub-6" && jammersEnabled > 0);
}

// @brief "throughput" sums delivered_mbps; "all_links_los" returns +1 only when
//        every controlled node has at least one peer link and all are LOS.
double
RlBridge::ComputeRewardTick(const LinkTable& linkTable,
                            const std::vector<FlowResult>& flows) const
{
    if (m_rl.reward_type == "all_links_los")
    {
        for (const auto& slot : m_slots)
        {
            if (!slot.active)
            {
                continue;
            }
            uint32_t losCount = 0;
            uint32_t count = 0;
            for (uint32_t j = 0; j < m_numNodes; ++j)
            {
                if (j == slot.node_index)
                {
                    continue;
                }
                if (linkTable.Get(slot.node_index, j).is_los)
                {
                    ++losCount;
                }
                ++count;
            }
            if (count == 0 || losCount != count)
            {
                return -1.0;
            }
        }
        return 1.0;
    }

    // Default: "throughput" — sum of delivered_mbps across all flows
    double total = 0.0;
    for (const auto& fr : flows)
    {
        total += fr.delivered_mbps;
    }
    return total;
}

// Adds one tick to the open decision window that feeds `facts.window` and the mean reward.
void
RlBridge::AccumulateTick(const LinkTable& linkTable, const std::vector<FlowResult>& flows)
{
    ++m_window.ticks;
    for (const auto& fr : flows)
    {
        m_window.demand_mbps_sum += fr.demand_mbps;
        m_window.delivered_mbps_sum += fr.delivered_mbps;
        if (fr.demand_mbps > 0.0)
        {
            ++m_window.flow_ticks_with_demand;
            if (!fr.routable)
            {
                ++m_window.unroutable_flow_ticks;
            }
        }
    }
    m_window.connected_pairs_sum += linkTable.ConnectedLinkCount();
    for (uint32_t i = 0; i < m_numNodes; ++i)
    {
        for (uint32_t j = i + 1; j < m_numNodes; ++j)
        {
            if (linkTable.Get(i, j).is_los)
            {
                ++m_window.los_pairs_sum;
            }
        }
    }
    m_window.legacy_reward_sum += ComputeRewardTick(linkTable, flows);
}

bool
RlBridge::IsDecisionTick(uint32_t ti) const
{
    return m_centralized && ti < m_numTicks && (ti % m_k) == 0;
}

// ---------------------------------------------------------------------------
// JSON output (C++ → Python)
// ---------------------------------------------------------------------------

void
RlBridge::WriteObs(uint32_t tick, double time_s,
                   const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs,
                   const LinkTable& linkTable,
                   double reward, bool done) const
{
    auto ctrlPos = mobs[m_controlledIdx]->GetPosition();

    // Per-link SINR and capacity from controlled node to all others
    json linkSinrs = json::array();
    json linkCaps  = json::array();
    for (uint32_t j = 0; j < m_numNodes; ++j)
    {
        if (j == m_controlledIdx)
            continue;
        const auto& lr = linkTable.Get(m_controlledIdx, j);
        linkSinrs.push_back(lr.sinr_db);
        linkCaps.push_back(lr.capacity_mbps);
    }

    json obs;
    // Send full 3-D position so the Python action mask can range-check z.
    obs["controlled_pos"]    = {ctrlPos.x, ctrlPos.y, ctrlPos.z};
    obs["link_sinrs"]        = linkSinrs;
    obs["link_capacities"]   = linkCaps;

    json msg;
    msg["type"]       = "step";
    msg["tick"]       = tick;
    msg["time_s"]     = time_s;
    msg["obs"]        = obs;
    msg["reward"]     = reward;
    msg["done"]       = done;
    msg["action_type"] = m_rl.action_type;

    std::cout << msg.dump() << "\n" << std::flush;
}

void
RlBridge::WriteInit() const
{
    if (!m_centralized)
    {
        return;
    }

    ojson slotIds = ojson::array();
    ojson slotSpeeds = ojson::array();
    uint32_t numControlled = 0;
    for (const auto& slot : m_slots)
    {
        if (slot.active)
        {
            ++numControlled;
            slotIds.push_back(slot.node_id);
            slotSpeeds.push_back(slot.speed_mps);
        }
        else
        {
            slotIds.push_back(nullptr);
            slotSpeeds.push_back(nullptr);
        }
    }

    const uint32_t obsDim  = m_numSlots * (4 + 2 * (m_numNodes - 1));
    const uint32_t maskDim = 5 * m_numSlots;
    const uint32_t numDecisions = (m_numTicks + m_k - 1) / m_k;

    ojson msg;
    msg["type"]                   = "init";
    msg["contract"]               = "mesh_move_2d_v1";
    msg["dimensions"]             = 2;
    msg["action_meanings"]        = {"west", "east", "south", "north", "hold"};
    msg["max_controlled_nodes"]   = m_numSlots;
    msg["num_controlled"]         = numControlled;
    msg["slot_node_ids"]          = slotIds;
    msg["slot_speed_mps"]         = slotSpeeds;
    msg["num_mesh_nodes"]         = m_numNodes;
    msg["obs_dim"]                = obsDim;
    msg["mask_dim"]               = maskDim;
    msg["tick_s"]                 = m_tickS;
    msg["decision_interval_s"]    = m_k * m_tickS;
    msg["decision_interval_ticks"] = m_k;
    msg["num_ticks"]              = m_numTicks;
    msg["num_decisions"]          = numDecisions;
    msg["reward_type"]            = m_rl.reward_type;
    msg["reward_window"]          = "mean";
    msg["wall_policy"]            = "clip";
    msg["facts_schema"]           = "mesh_facts_v1";
    msg["facts_columns"]          = ojson{{"nodes", {"x", "y", "z", "vx", "vy", "vz", "slot"}},
                                          {"links", {"sinr_db", "capacity_mbps", "is_los"}}};
    msg["node_ids"]               = m_nodeIds;
    msg["num_links"]              = m_numNodes * (m_numNodes - 1) / 2;
    msg["bounds"]                 = ojson{{"x_min", m_rl.x_min}, {"x_max", m_rl.x_max},
                                          {"y_min", m_rl.y_min}, {"y_max", m_rl.y_max},
                                          {"z_min", m_rl.z_min}, {"z_max", m_rl.z_max}};
    msg["band"]                   = m_band;
    msg["jammer_path_enabled"]    = m_jammerPathEnabled;
    msg["warmup_s"]               = m_warmupS;

    std::cout << msg.dump() << "\n" << std::flush;
}

std::vector<int>
RlBridge::ComputeMask(const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs) const
{
    std::vector<int> mask(5 * m_numSlots, 0);
    for (uint32_t i = 0; i < m_numSlots; ++i)
    {
        const auto& slot = m_slots[i];
        if (!slot.active)
        {
            mask[5 * i + 4] = 1;  // padded slot: hold only
            continue;
        }
        auto p = mobs[slot.node_index]->GetPosition();
        mask[5 * i + 0] = (p.x - m_rl.x_min > kWallEps) ? 1 : 0;
        mask[5 * i + 1] = (m_rl.x_max - p.x > kWallEps) ? 1 : 0;
        mask[5 * i + 2] = (p.y - m_rl.y_min > kWallEps) ? 1 : 0;
        mask[5 * i + 3] = (m_rl.y_max - p.y > kWallEps) ? 1 : 0;
        mask[5 * i + 4] = 1;
    }
    return mask;
}

void
RlBridge::WriteStep(uint32_t tick, double time_s,
                    const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs,
                    const LinkTable& linkTable,
                    const std::vector<int>& mask,
                    double reward, WindowFacts window, bool done) const
{
    ojson obs = ojson::array();
    for (const auto& slot : m_slots)
    {
        if (!slot.active)
        {
            for (uint32_t k = 0; k < 4 + 2 * (m_numNodes - 1); ++k)
            {
                obs.push_back(0.0);
            }
            continue;
        }
        auto p = mobs[slot.node_index]->GetPosition();
        obs.push_back(1.0);
        obs.push_back(p.x);
        obs.push_back(p.y);
        obs.push_back(p.z);
        for (uint32_t j = 0; j < m_numNodes; ++j)
        {
            if (j == slot.node_index)
            {
                continue;
            }
            const auto& lr = linkTable.Get(slot.node_index, j);
            obs.push_back(lr.sinr_db);
            obs.push_back(lr.capacity_mbps);
        }
    }

    ojson factNodes = ojson::array();
    for (uint32_t n = 0; n < m_numNodes; ++n)
    {
        auto p = mobs[n]->GetPosition();
        auto v = mobs[n]->GetVelocity();
        int slot = -1;
        for (uint32_t i = 0; i < m_numSlots; ++i)
        {
            if (m_slots[i].active && m_slots[i].node_index == n)
            {
                slot = static_cast<int>(i);
                break;
            }
        }
        factNodes.push_back(ojson::array({p.x, p.y, p.z, v.x, v.y, v.z, slot}));
    }

    ojson factLinks = ojson::array();
    for (uint32_t i = 0; i < m_numNodes; ++i)
    {
        for (uint32_t j = i + 1; j < m_numNodes; ++j)
        {
            const auto& lr = linkTable.Get(i, j);
            factLinks.push_back(
                ojson::array({lr.sinr_db, lr.capacity_mbps, lr.is_los ? 1 : 0}));
        }
    }

    ojson factWindow;
    factWindow["ticks"]                  = window.ticks;
    factWindow["demand_mbps_sum"]        = window.demand_mbps_sum;
    factWindow["delivered_mbps_sum"]     = window.delivered_mbps_sum;
    factWindow["flow_ticks_with_demand"] = window.flow_ticks_with_demand;
    factWindow["unroutable_flow_ticks"]  = window.unroutable_flow_ticks;
    factWindow["connected_pairs_sum"]    = window.connected_pairs_sum;
    factWindow["los_pairs_sum"]          = window.los_pairs_sum;
    factWindow["legacy_reward_sum"]      = window.legacy_reward_sum;

    ojson facts;
    facts["nodes"]  = factNodes;
    facts["links"]  = factLinks;
    facts["window"] = factWindow;

    ojson msg;
    msg["type"]              = "step";
    msg["tick"]              = tick;
    msg["time_s"]            = time_s;
    msg["decision"]          = m_decision;
    msg["ticks_in_step"]     = window.ticks;
    msg["obs"]               = obs;
    msg["mask"]              = mask;
    msg["reward"]            = reward;
    msg["done"]              = done;
    msg["revalidated_slots"] = m_revalidatedSlots;
    msg["facts"]             = facts;

    std::cout << msg.dump() << "\n" << std::flush;
}

// ---------------------------------------------------------------------------
// JSON input (Python → C++)
// ---------------------------------------------------------------------------

void
RlBridge::ReadAction()
{
    // One warning per process; rl-bridge.h carries no state for this.
    static bool warned = false;

    std::string line;
    if (!std::getline(std::cin, line))
    {
        // stdin closed — hold position
        m_lastDiscreteAction = 6;
        if (!warned)
        {
            warned = true;
            std::cerr << "Warning: RL action stream closed; holding position (Stay).\n";
        }
        return;
    }

    auto j = json::parse(line, nullptr, false);
    if (j.is_discarded())
    {
        m_lastDiscreteAction = 6;
        if (!warned)
        {
            warned = true;
            std::cerr << "Warning: malformed RL action JSON; holding position (Stay).\n";
        }
        return;
    }

    // A non-object line or a non-integer "action" (e.g. a joint-action list) is
    // malformed for the legacy encoding; it must not abort the run.
    if (!j.is_object() ||
        (m_rl.action_type != "continuous" && j.contains("action") &&
         !j["action"].is_number_integer()))
    {
        m_lastDiscreteAction = 6;
        if (!warned)
        {
            warned = true;
            std::cerr << "Warning: malformed RL action JSON; holding position (Stay).\n";
        }
        return;
    }

    if (m_rl.action_type == "continuous")
    {
        const auto& a = j["action"];
        m_lastTargetX = a[0].get<double>();
        m_lastTargetY = a[1].get<double>();
        // Optional third component for 3-D continuous control; keep current z if absent.
        m_lastTargetZ = (a.size() > 2) ? a[2].get<double>() : m_lastTargetZ;
    }
    else
    {
        m_lastDiscreteAction = j.value("action", 0);
    }
}

void
RlBridge::ReadJointAction()
{
    static bool warnedClosed = false;
    static bool warnedMalformed = false;
    static bool warnedRevalidated = false;

    m_revalidatedSlots.clear();

    if (m_streamClosed)
    {
        m_lastJoint.assign(m_numSlots, kHold);
        return;
    }

    std::string line;
    if (!std::getline(std::cin, line))
    {
        m_streamClosed = true;
        m_lastJoint.assign(m_numSlots, kHold);
        if (!warnedClosed)
        {
            warnedClosed = true;
            std::cerr << "Warning: RL action stream closed; all controlled nodes hold.\n";
        }
        return;
    }

    auto j = json::parse(line, nullptr, false);
    bool structural = !j.is_discarded() && j.is_object() && j.contains("action") &&
                      j["action"].is_array() && j["action"].size() == m_numSlots;
    std::vector<int> proposed;
    if (structural)
    {
        for (const auto& entry : j["action"])
        {
            if (!entry.is_number_integer())
            {
                structural = false;
                break;
            }
            const int a = entry.get<int>();
            if (a < 0 || a > kHold)
            {
                structural = false;
                break;
            }
            proposed.push_back(a);
        }
    }

    if (!structural)
    {
        m_lastJoint.assign(m_numSlots, kHold);
        if (!warnedMalformed)
        {
            warnedMalformed = true;
            std::cerr << "Warning: malformed RL joint action; all controlled nodes hold.\n";
        }
        return;
    }

    for (uint32_t i = 0; i < m_numSlots; ++i)
    {
        const bool padded = !m_slots[i].active;
        const bool masked = !padded && m_lastMask[5 * i + proposed[i]] == 0;
        if ((padded && proposed[i] != kHold) || masked)
        {
            proposed[i] = kHold;
            m_revalidatedSlots.push_back(i);
        }
    }

    if (!m_revalidatedSlots.empty() && !warnedRevalidated)
    {
        warnedRevalidated = true;
        std::cerr << "Warning: RL joint action revalidated; invalid slot actions replaced by hold.\n";
    }

    m_lastJoint = proposed;
}

// ---------------------------------------------------------------------------
// Step: write obs, read action
// ---------------------------------------------------------------------------

void
RlBridge::Step(uint32_t tick, double time_s,
               const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs,
               const LinkTable& linkTable,
               const std::vector<FlowResult>& flowResults,
               bool done)
{
    if (m_centralized)
    {
        const WindowFacts window = m_window;
        const double reward =
            (window.ticks > 0) ? window.legacy_reward_sum / window.ticks : 0.0;
        m_window = WindowFacts{};

        m_lastMask = ComputeMask(mobs);
        WriteStep(tick, time_s, mobs, linkTable, m_lastMask, reward, window, done);
        ++m_decision;

        if (!done)
        {
            ReadJointAction();
        }
        return;
    }

    double reward = ComputeRewardTick(linkTable, flowResults);
    WriteObs(tick, time_s, mobs, linkTable, reward, done);

    if (!done)
    {
        ReadAction();
    }
}

// ---------------------------------------------------------------------------
// Movement: per-tick clamp and action application
// ---------------------------------------------------------------------------

ns3::Vector
RlBridge::ClampVelocityForTick(const ns3::Vector& pos, const ns3::Vector& vel) const
{
    auto clampAxis = [this](double p, double v, double lo, double hi) {
        const double next = p + v * m_tickS;
        if (next > hi)
        {
            return (hi > p) ? (hi - p) / m_tickS : 0.0;
        }
        if (next < lo)
        {
            return (lo < p) ? (lo - p) / m_tickS : 0.0;
        }
        return v;
    };

    return ns3::Vector(clampAxis(pos.x, vel.x, m_rl.x_min, m_rl.x_max),
                       clampAxis(pos.y, vel.y, m_rl.y_min, m_rl.y_max),
                       vel.z);
}

void
RlBridge::BeforeAdvance(const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs)
{
    if (!m_centralized)
    {
        return;
    }

    for (const auto& slot : m_slots)
    {
        if (!slot.active)
        {
            continue;
        }
        auto cvmm = mobs[slot.node_index]->GetObject<ns3::ConstantVelocityMobilityModel>();
        if (!cvmm)
        {
            continue;
        }
        cvmm->SetVelocity(ClampVelocityForTick(cvmm->GetPosition(), cvmm->GetVelocity()));
    }
}

// ---------------------------------------------------------------------------
// ApplyAction: compute desired position, derive velocity, SetVelocity()
// ---------------------------------------------------------------------------

// @brief Environment action moves the controlled node in x, y, or z.
//
// Discrete action indices MUST match the Python env's action_masks():
//   0:-X  1:+X  2:-Y  3:+Y  4:-Z  5:+Z  6:Stay
void
RlBridge::ApplyAction(const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs)
{
    if (m_centralized)
    {
        for (uint32_t i = 0; i < m_numSlots; ++i)
        {
            const auto& slot = m_slots[i];
            if (!slot.active)
            {
                continue;
            }
            double vx = 0.0;
            double vy = 0.0;
            switch (m_lastJoint[i])
            {
            case 0: vx = -slot.speed_mps; break;  // west
            case 1: vx =  slot.speed_mps; break;  // east
            case 2: vy = -slot.speed_mps; break;  // south
            case 3: vy =  slot.speed_mps; break;  // north
            default: break;                       // hold
            }
            auto cvmm = mobs[slot.node_index]->GetObject<ns3::ConstantVelocityMobilityModel>();
            if (cvmm)
            {
                cvmm->SetVelocity(ns3::Vector(vx, vy, 0.0));
            }
        }
        return;
    }

    ns3::Ptr<ns3::MobilityModel> mob = mobs[m_slots[0].node_index];

    auto pos = mob->GetPosition();
    double desiredX = pos.x;
    double desiredY = pos.y;
    double desiredZ = pos.z;

    if (m_rl.action_type == "continuous")
    {
        desiredX = m_lastTargetX;
        desiredY = m_lastTargetY;
        desiredZ = m_lastTargetZ;
    }
    else
    {
        // Discrete 7-action set (must match Python action_masks ordering).
        switch (m_lastDiscreteAction)
        {
        case 0:
            desiredX = pos.x - m_rl.step_size_m;  // -X
            break;
        case 1:
            desiredX = pos.x + m_rl.step_size_m;  // +X
            break;
        case 2:
            desiredY = pos.y - m_rl.step_size_m;  // -Y
            break;
        case 3:
            desiredY = pos.y + m_rl.step_size_m;  // +Y
            break;
        case 4:
            desiredZ = pos.z - m_rl.step_size_m;  // -Z
            break;
        case 5:
            desiredZ = pos.z + m_rl.step_size_m;  // +Z
            break;
        case 6:
        default:
            break;                                // Stay
        }
    }

    // Clamp desired position to the arena bounds (x/y/z).
    desiredX = std::clamp(desiredX, m_rl.x_min, m_rl.x_max);
    desiredY = std::clamp(desiredY, m_rl.y_min, m_rl.y_max);
    desiredZ = std::clamp(desiredZ, m_rl.z_min, m_rl.z_max);

    double dx = desiredX - pos.x;
    double dy = desiredY - pos.y;
    double dz = desiredZ - pos.z;
    double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;

    if (m_rl.action_type == "continuous" && dist < m_rl.arrival_threshold_m)
    {
        // Arrived — stop
        vx = 0.0;
        vy = 0.0;
        vz = 0.0;
    }
    else if (dist > 1e-9)
    {
        // Velocity toward desired, capped at max speed (now in 3-D).
        double speed = std::min(dist / m_tickS, m_maxSpeed);
        vx = (dx / dist) * speed;
        vy = (dy / dist) * speed;
        vz = (dz / dist) * speed;
    }

    auto cvmm = mob->GetObject<ns3::ConstantVelocityMobilityModel>();
    if (cvmm)
    {
        cvmm->SetVelocity(ns3::Vector(vx, vy, vz));
    }
}

}  // namespace mesh_sim
