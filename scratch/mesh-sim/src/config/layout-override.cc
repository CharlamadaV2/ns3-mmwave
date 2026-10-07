/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file layout-override.cc
 * @brief Implements @ref mesh_sim::ApplyLayout; see layout-override.h.
 */

#include "src/config/layout-override.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace mesh_sim
{

namespace
{

constexpr double kZToleranceM = 1e-6;

std::string
Num(double v)
{
    std::ostringstream os;
    os.precision(17);
    os << v;
    return os.str();
}

bool
CentrallyControlled(const SimConfig& cfg, uint32_t i)
{
    if (!cfg.rl.enabled || cfg.rl.control_mode != "centralized")
    {
        return false;
    }
    const auto& idx = cfg.rl.controlled_indices;
    return std::find(idx.begin(), idx.end(), i) != idx.end();
}

}  // namespace

std::vector<std::string>
ApplyLayout(SimConfig& cfg, const std::vector<Position>& layout)
{
    std::vector<std::string> errors;
    if (layout.size() != cfg.nodes.size())
    {
        errors.push_back("layout has " + std::to_string(layout.size()) +
                         " positions, expected " + std::to_string(cfg.nodes.size()));
        return errors;
    }

    for (uint32_t i = 0; i < cfg.nodes.size(); ++i)
    {
        const NodeSpec& spec = cfg.nodes[i];
        const Position& p = layout[i];
        const std::string label = "node '" + spec.id + "'";
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
        {
            errors.push_back(label + ": non-finite position");
            continue;
        }
        const bool waypoint = (spec.mobility == "waypoint");
        if (waypoint && spec.waypoints.empty())
        {
            errors.push_back(label + ": waypoint mobility without waypoints");
            continue;
        }
        const double startZ = waypoint ? spec.waypoints.front().z : spec.position.z;
        if (std::fabs(p.z - startZ) > kZToleranceM)
        {
            errors.push_back(label + ": z " + Num(p.z) + " differs from start z " + Num(startZ));
        }
        if (spec.mobility == "random_walk" && !CentrallyControlled(cfg, i))
        {
            // RandomWalk2dMobilityModel asserts the start lies inside its (inclusive) bounds.
            const auto& rw = spec.random_walk;
            if (p.x < rw.x_min || p.x > rw.x_max || p.y < rw.y_min || p.y > rw.y_max)
            {
                errors.push_back(label + ": (" + Num(p.x) + ", " + Num(p.y) +
                                 ") is outside its random_walk bounds x [" + Num(rw.x_min) +
                                 ", " + Num(rw.x_max) + "] y [" + Num(rw.y_min) + ", " +
                                 Num(rw.y_max) + "]");
            }
        }
    }
    if (!errors.empty())
    {
        return errors;
    }

    for (uint32_t i = 0; i < cfg.nodes.size(); ++i)
    {
        NodeSpec& spec = cfg.nodes[i];
        const Position& p = layout[i];
        if (spec.mobility == "waypoint")
        {
            // Same arithmetic as scripts/baselines/effective_inputs.py rewrite_nodes.
            const double dx = p.x - spec.waypoints.front().x;
            const double dy = p.y - spec.waypoints.front().y;
            for (auto& wp : spec.waypoints)
            {
                wp.x = wp.x + dx;
                wp.y = wp.y + dy;
            }
        }
        spec.position.x = p.x;
        spec.position.y = p.y;
    }
    return errors;
}

}  // namespace mesh_sim
