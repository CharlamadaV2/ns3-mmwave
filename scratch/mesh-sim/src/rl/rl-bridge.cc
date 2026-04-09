/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

#include "src/rl/rl-bridge.h"
#include "third_party/json.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

using json = nlohmann::json;

namespace mesh_sim
{

RlBridge::RlBridge(const SimConfig& cfg, uint32_t controlledIdx)
    : m_rl(cfg.rl),
      m_tickS(cfg.tick_s),
      m_controlledIdx(controlledIdx),
      m_numNodes(static_cast<uint32_t>(cfg.nodes.size())),
      m_maxSpeed(MaxSpeedForType(cfg.nodes[controlledIdx].node_type)),
      m_nodeType(cfg.nodes[controlledIdx].node_type)
{
}

// ---------------------------------------------------------------------------
// Reward computation
// ---------------------------------------------------------------------------

double
RlBridge::ComputeReward(const std::vector<ns3::Ptr<ns3::MobilityModel>>& /* mobs */,
                        const LinkTable& linkTable,
                        const std::vector<FlowResult>& flows) const
{
    if (m_rl.reward_type == "mean_sinr")
    {
        uint32_t losCount = 0;
        uint32_t count = 0;
        for (uint32_t j = 0; j < m_numNodes; ++j)
        {
            if (j == m_controlledIdx)
            {
                continue;
            }
            if (linkTable.Get(m_controlledIdx, j).is_los)
            {
                ++losCount;
            }
            ++count;
        }
        return (count > 0 && losCount == count) ? 1.0 : -1.0;
    }

    // Default: "throughput" — sum of delivered_mbps across all flows
    double total = 0.0;
    for (const auto& fr : flows)
    {
        total += fr.delivered_mbps;
    }
    return total;
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
    obs["controlled_pos"]    = {ctrlPos.x, ctrlPos.y};
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

// ---------------------------------------------------------------------------
// JSON input (Python → C++)
// ---------------------------------------------------------------------------

void
RlBridge::ReadAction()
{
    std::string line;
    if (!std::getline(std::cin, line))
    {
        // stdin closed — default to stay
        m_lastDiscreteAction = 0;
        return;
    }

    auto j = json::parse(line, nullptr, false);
    if (j.is_discarded())
    {
        m_lastDiscreteAction = 0;
        return;
    }

    if (m_rl.action_type == "continuous")
    {
        const auto& a = j["action"];
        m_lastTargetX = a[0].get<double>();
        m_lastTargetY = a[1].get<double>();
    }
    else
    {
        m_lastDiscreteAction = j.value("action", 0);
    }
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
    double reward = ComputeReward(mobs, linkTable, flowResults);
    WriteObs(tick, time_s, mobs, linkTable, reward, done);

    if (!done)
    {
        ReadAction();
    }
}

// ---------------------------------------------------------------------------
// ApplyAction: compute desired position, derive velocity, SetVelocity()
// ---------------------------------------------------------------------------

void
RlBridge::ApplyAction(ns3::Ptr<ns3::MobilityModel> mob)
{
    auto pos = mob->GetPosition();
    double desiredX = pos.x;
    double desiredY = pos.y;

    if (m_rl.action_type == "continuous")
    {
        desiredX = m_lastTargetX;
        desiredY = m_lastTargetY;
    }
    else
    {
        // Discrete: 0=stay, 1=-x, 2=+x, 3=-y, 4=+y
        switch (m_lastDiscreteAction)
        {
        case 1:
            desiredX = pos.x - m_rl.step_size_m;
            break;
        case 2:
            desiredX = pos.x + m_rl.step_size_m;
            break;
        case 3:
            desiredY = pos.y - m_rl.step_size_m;
            break;
        case 4:
            desiredY = pos.y + m_rl.step_size_m;
            break;
        default:
            break;
        }
    }

    // Clamp desired position to bounds
    desiredX = std::clamp(desiredX, m_rl.x_min, m_rl.x_max);
    desiredY = std::clamp(desiredY, m_rl.y_min, m_rl.y_max);

    double dx = desiredX - pos.x;
    double dy = desiredY - pos.y;
    double dist = std::sqrt(dx * dx + dy * dy);

    double vx = 0.0;
    double vy = 0.0;

    if (m_rl.action_type == "continuous" && dist < m_rl.arrival_threshold_m)
    {
        // Arrived — stop
        vx = 0.0;
        vy = 0.0;
    }
    else if (dist > 1e-9)
    {
        // Compute velocity toward desired, capped at max speed
        double speed = std::min(dist / m_tickS, m_maxSpeed);
        vx = (dx / dist) * speed;
        vy = (dy / dist) * speed;
    }

    auto cvmm = mob->GetObject<ns3::ConstantVelocityMobilityModel>();
    if (cvmm)
    {
        cvmm->SetVelocity(ns3::Vector(vx, vy, 0.0));
    }
}

}  // namespace mesh_sim
