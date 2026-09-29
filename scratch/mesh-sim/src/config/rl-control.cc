/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file rl-control.cc
 * @brief Implements the RL control resolver; behavior is documented in rl-control.h.
 */

#include "src/config/rl-control.h"
#include "src/util/string-utils.h"

#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace mesh_sim
{

namespace
{

constexpr double kTickEps    = 1e-6;  // integer-ratio tolerance
constexpr uint32_t kMaxSlots = 64;

// Upper bound on a tick ratio we are willing to convert to uint32_t.
const double kMaxTickRatio = static_cast<double>(std::numeric_limits<uint32_t>::max() - 1);

bool
isFinitePositive(double v)
{
    return std::isfinite(v) && v > 0.0;
}

std::string
num(double v)
{
    std::ostringstream os;
    os << v;
    return os.str();
}

// Split on ',', trim each token, and keep only non-empty tokens.
std::vector<std::string>
tokenize(const std::string& s)
{
    std::vector<std::string> out;
    std::string cur;
    std::istringstream is(s);
    while (std::getline(is, cur, ','))
    {
        std::string t = trimStr(cur);
        if (!t.empty())
        {
            out.push_back(t);
        }
    }
    return out;
}

}  // namespace

uint32_t
ComputeTickCount(double duration_s, double tick_s, bool robust)
{
    if (!isFinitePositive(duration_s) || !isFinitePositive(tick_s))
    {
        return 0;
    }
    const double q = duration_s / tick_s;
    if (!std::isfinite(q) || q < 1.0 || q > kMaxTickRatio)
    {
        return 0;
    }
    if (!robust)
    {
        return static_cast<uint32_t>(duration_s / tick_s);
    }
    const double n = static_cast<double>(std::llround(q));
    return static_cast<uint32_t>(std::fabs(q - n) <= kTickEps ? n : std::floor(q));
}

Position
ControlledStartPosition(const NodeSpec& spec)
{
    if (spec.mobility == "waypoint")
    {
        if (spec.waypoints.empty())
        {
            throw std::invalid_argument("ControlledStartPosition: node '" + spec.id +
                                        "' has no waypoints");
        }
        const Waypoint& w = spec.waypoints.front();
        return Position{w.x, w.y, w.z};
    }
    return spec.position;
}

namespace
{

// Keep the single-node selector's existing fallback behavior.
void
resolveLegacy(const SimConfig& cfg, RlControlResolution& r)
{
    uint32_t idx = static_cast<uint32_t>(cfg.nodes.size()) - 1;
    if (!cfg.rl.controlled_node_id.empty())
    {
        for (uint32_t i = 0; i < cfg.nodes.size(); ++i)
        {
            if (cfg.nodes[i].id == cfg.rl.controlled_node_id)
            {
                idx = i;
                break;
            }
        }
    }
    r.controlled_indices = {idx};
    r.num_slots          = 1;
}

// Validate the selector and map node ids to their indices.
void
resolveSelection(const SimConfig& cfg, RlControlResolution& r)
{
    if (!cfg.rl.controlled_node_id.empty())
    {
        r.errors.push_back("rl.controlled_node_id and rl.controlled_nodes are mutually exclusive");
    }

    if (cfg.rl.action_type != "discrete")
    {
        r.errors.push_back("rl.action_type '" + cfg.rl.action_type +
                           "' is not supported with rl.controlled_nodes");
    }
    if (cfg.rl.action_profile == "move_3d")
    {
        r.errors.push_back("rl.action_profile 'move_3d' is reserved and not implemented");
    }
    else if (cfg.rl.action_profile != "move_2d")
    {
        r.errors.push_back("rl.action_profile: unknown value '" + cfg.rl.action_profile +
                           "' (must be one of: 'move_2d')");
    }

    // Node ids must be unique to select exactly one node.
    std::set<std::string> seen;
    std::set<std::string> reported;
    for (const auto& n : cfg.nodes)
    {
        if (!seen.insert(n.id).second && reported.insert(n.id).second)
        {
            r.errors.push_back("nodes.json has duplicate node id '" + n.id +
                               "'; ids must be unique when rl.controlled_nodes is used");
        }
    }

    const std::vector<std::string> tokens = tokenize(cfg.rl.controlled_nodes);
    if (tokens.empty())
    {
        r.errors.push_back("rl.controlled_nodes is set but empty; use 'all', a comma-separated "
                           "list of node ids, or remove the key for legacy single-node control");
        return;
    }

    bool hasAll = false;
    for (const auto& t : tokens)
    {
        if (t == "all")
        {
            hasAll = true;
            break;
        }
    }
    if (hasAll)
    {
        if (tokens.size() != 1)
        {
            r.errors.push_back("rl.controlled_nodes: 'all' cannot be combined with explicit ids");
            return;
        }
        for (uint32_t i = 0; i < cfg.nodes.size(); ++i)
        {
            r.controlled_indices.push_back(i);
        }
        return;
    }

    std::set<std::string> used;
    for (const auto& t : tokens)
    {
        if (!used.insert(t).second)
        {
            r.errors.push_back("rl.controlled_nodes: duplicate id '" + t + "'");
            continue;
        }

        bool found = false;
        for (uint32_t i = 0; i < cfg.nodes.size(); ++i)
        {
            if (cfg.nodes[i].id == t)
            {
                r.controlled_indices.push_back(i);
                found = true;
                break;
            }
        }
        if (found)
        {
            continue;
        }

        bool isJammer = false;
        for (const auto& j : cfg.jammers)
        {
            if (j.id == t)
            {
                isJammer = true;
                break;
            }
        }
        r.errors.push_back(isJammer
                               ? "rl.controlled_nodes: '" + t +
                                     "' is a jammer id; jammers cannot be RL-controlled"
                               : "rl.controlled_nodes: unknown node id '" + t + "'");
    }
}

// Resolve the policy vector's controlled-node capacity.
void
resolveSlotCount(const SimConfig& cfg, RlControlResolution& r)
{
    if (cfg.rl.max_controlled_nodes < 0)
    {
        r.errors.push_back("rl.max_controlled_nodes must be >= 0 (0 means auto-size)");
        return;
    }

    const uint32_t count = static_cast<uint32_t>(r.controlled_indices.size());
    if (count == 0)
    {
        if (r.errors.empty())
        {
            r.errors.push_back("rl.controlled_nodes resolved to no controlled nodes");
        }
        return;
    }

    const uint32_t m = cfg.rl.max_controlled_nodes == 0
                           ? count
                           : static_cast<uint32_t>(cfg.rl.max_controlled_nodes);
    if (m < count)
    {
        r.errors.push_back("rl.max_controlled_nodes (" + std::to_string(m) +
                           ") is smaller than the number of controlled nodes (" +
                           std::to_string(count) + ")");
        return;
    }
    if (m > kMaxSlots)
    {
        r.errors.push_back("rl.max_controlled_nodes (" + std::to_string(m) +
                           ") exceeds the maximum of " + std::to_string(kMaxSlots));
        return;
    }
    r.num_slots = m;
}

// Resolve the interval between policy decisions in simulator ticks.
void
resolveCadence(const SimConfig& cfg, RlControlResolution& r, bool timingOk)
{
    const double raw = cfg.rl.decision_interval_s;
    if (!std::isfinite(raw) || raw < 0.0)
    {
        r.errors.push_back("rl.decision_interval_s must be finite and >= 0 (0 means tick_s)");
        return;
    }
    if (!timingOk)
    {
        return;
    }

    const double interval = raw == 0.0 ? cfg.tick_s : raw;
    const double kf       = interval / cfg.tick_s;
    if (!std::isfinite(kf) || kf > kMaxTickRatio)
    {
        r.errors.push_back("rl.decision_interval_s exceeds duration_s");
        return;
    }

    const double n = static_cast<double>(std::llround(kf));
    if (n < 1.0 || std::fabs(kf - n) > kTickEps)
    {
        r.errors.push_back("rl.decision_interval_s must be an integer multiple of tick_s");
        return;
    }

    const uint32_t k = static_cast<uint32_t>(n);
    if (k > r.num_ticks)
    {
        r.errors.push_back("rl.decision_interval_s exceeds duration_s");
        return;
    }
    r.decision_interval_ticks = k;
}

// Keep every controlled start position within the movement bounds.
void
resolveStartPositions(const SimConfig& cfg, RlControlResolution& r)
{
    if (!isFinitePositive(cfg.rl.step_size_m))
    {
        r.errors.push_back("rl.step_size_m must be finite and > 0");
    }

    const bool boundsOk = std::isfinite(cfg.rl.x_min) && std::isfinite(cfg.rl.x_max) &&
                          std::isfinite(cfg.rl.y_min) && std::isfinite(cfg.rl.y_max) &&
                          std::isfinite(cfg.rl.z_min) && std::isfinite(cfg.rl.z_max) &&
                          cfg.rl.x_min < cfg.rl.x_max && cfg.rl.y_min < cfg.rl.y_max &&
                          cfg.rl.z_min < cfg.rl.z_max;
    if (!boundsOk)
    {
        r.errors.push_back("rl bounds must be finite and ordered");
    }

    for (uint32_t idx : r.controlled_indices)
    {
        const NodeSpec& spec = cfg.nodes[idx];
        if (spec.mobility == "waypoint" && spec.waypoints.empty())
        {
            r.errors.push_back("rl.controlled_nodes: waypoint node '" + spec.id +
                               "' has no waypoints");
            continue;
        }

        const Position p = ControlledStartPosition(spec);
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
        {
            r.errors.push_back("rl.controlled_nodes: node '" + spec.id +
                               "' has a non-finite start position");
            continue;
        }
        if (!boundsOk)
        {
            continue;
        }
        if (p.x < cfg.rl.x_min || p.x > cfg.rl.x_max || p.y < cfg.rl.y_min ||
            p.y > cfg.rl.y_max || p.z < cfg.rl.z_min || p.z > cfg.rl.z_max)
        {
            r.errors.push_back("rl.controlled_nodes: node '" + spec.id + "' starts at (" +
                               num(p.x) + "," + num(p.y) + "," + num(p.z) +
                               "), outside the [rl] bounds");
        }
    }
}

}  // namespace

RlControlResolution
ResolveRlControl(const SimConfig& cfg)
{
    RlControlResolution r;
    r.control_mode = "legacy";

    if (!cfg.rl.enabled)
    {
        return r;
    }

    const bool centralized = cfg.rl.controlled_nodes_set;
    r.control_mode         = centralized ? "centralized" : "legacy";

    r.num_ticks           = ComputeTickCount(cfg.duration_s, cfg.tick_s, centralized);
    const bool timingOk   = r.num_ticks > 0;
    if (!timingOk)
    {
        r.errors.push_back("rl timing is invalid or exceeds the supported tick range");
    }

    const bool haveNodes = !cfg.nodes.empty();
    if (!haveNodes)
    {
        r.errors.push_back("rl control requires at least one mesh node");
    }

    if (!centralized)
    {
        if (haveNodes)
        {
            resolveLegacy(cfg, r);
        }
        return r;
    }

    if (haveNodes)
    {
        resolveSelection(cfg, r);
    }
    resolveSlotCount(cfg, r);
    resolveCadence(cfg, r, timingOk);
    resolveStartPositions(cfg, r);
    return r;
}

void
ApplyRlControl(SimConfig& cfg, const RlControlResolution& r)
{
    if (!r.ok())
    {
        throw std::invalid_argument("ApplyRlControl: refusing to apply an RL control "
                                    "resolution that reported errors");
    }
    cfg.rl.control_mode            = r.control_mode;
    cfg.rl.controlled_indices      = r.controlled_indices;
    cfg.rl.num_slots               = r.num_slots;
    cfg.rl.decision_interval_ticks = r.decision_interval_ticks;
    cfg.rl.num_ticks               = r.num_ticks;
}

}  // namespace mesh_sim
