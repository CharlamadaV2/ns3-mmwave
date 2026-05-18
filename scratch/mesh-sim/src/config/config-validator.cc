/**
 * @brief 
 * 
 */

/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

#include "src/config/config-validator.h"

#include <algorithm>
#include <initializer_list>

namespace mesh_sim
{

static void
checkPositive(ValidationResult& r, const std::string& field, double val)
{
    if (val <= 0.0)
    {
        r.errors.push_back(field + " must be > 0 (got " + std::to_string(val) + ")");
    }
}

static void
checkOneOf(ValidationResult& r,
           const std::string& field,
           const std::string& val,
           std::initializer_list<std::string> allowed)
{
    if (std::find(allowed.begin(), allowed.end(), val) == allowed.end())
    {
        std::string opts;
        for (const auto& a : allowed)
        {
            if (!opts.empty())
                opts += ", ";
            opts += "'" + a + "'";
        }
        r.errors.push_back(field + ": unknown value '" + val +
                           "' (must be one of: " + opts + ")");
    }
}

ValidationResult
ValidateConfig(const SimConfig& cfg)
{
    ValidationResult r;

    // -- timing --
    checkPositive(r, "duration_s", cfg.duration_s);
    checkPositive(r, "tick_s", cfg.tick_s);

    if (cfg.tick_s > cfg.duration_s)
    {
        r.errors.push_back("tick_s (" + std::to_string(cfg.tick_s) +
                           ") must be <= duration_s (" +
                           std::to_string(cfg.duration_s) + ")");
    }
    if (cfg.warmup_s < 0.0)
    {
        r.errors.push_back("warmup_s must be >= 0 (got " +
                           std::to_string(cfg.warmup_s) + ")");
    }
    if (cfg.warmup_s >= cfg.duration_s && cfg.duration_s > 0.0)
    {
        r.errors.push_back("warmup_s (" + std::to_string(cfg.warmup_s) +
                           ") must be < duration_s (" +
                           std::to_string(cfg.duration_s) + ")");
    }

    // -- nodes --
    if (cfg.nodes.size() < 2)
    {
        r.errors.push_back("at least 2 nodes required (got " +
                           std::to_string(cfg.nodes.size()) + ")");
    }

    for (const auto& node : cfg.nodes)
    {
        checkOneOf(r, "node '" + node.id + "' mobility", node.mobility,
                   {"fixed", "constant_velocity", "random_walk", "waypoint"});

        if (node.mobility == "waypoint")
        {
            const std::string label = "node '" + node.id + "' waypoints";
            if (node.waypoints.size() < 2)
            {
                r.errors.push_back(label + ": need at least 2 waypoints (got " +
                                   std::to_string(node.waypoints.size()) + ")");
            }
            for (size_t i = 1; i < node.waypoints.size(); ++i)
            {
                if (node.waypoints[i].t <= node.waypoints[i - 1].t)
                {
                    r.errors.push_back(label + ": time must be strictly monotonic " +
                                       "(index " + std::to_string(i) + " t=" +
                                       std::to_string(node.waypoints[i].t) + " <= " +
                                       std::to_string(node.waypoints[i - 1].t) + ")");
                    break;
                }
            }
            if (!node.waypoints.empty() && node.waypoints.front().t < 0.0)
            {
                r.errors.push_back(label + ": first waypoint t must be >= 0 (got " +
                                   std::to_string(node.waypoints.front().t) + ")");
            }
        }
    }

    // -- channel --
    checkPositive(r, "channel.frequency_ghz", cfg.channel.frequency_ghz);
    checkPositive(r, "channel.bandwidth_mhz", cfg.channel.bandwidth_mhz);
    checkOneOf(r, "channel.channel_model", cfg.channel.channel_model,
               {"3gpp", "nyu"});
    checkOneOf(r, "channel.scenario", cfg.channel.scenario,
               {"UMi", "UMa", "RMa", "InH", "InF"});
    if (cfg.channel.tx_array_gain_dbi < 0.0)
    {
        r.errors.push_back("channel.tx_array_gain_dbi must be >= 0 (got " +
                           std::to_string(cfg.channel.tx_array_gain_dbi) + ")");
    }
    if (cfg.channel.rx_array_gain_dbi < 0.0)
    {
        r.errors.push_back("channel.rx_array_gain_dbi must be >= 0 (got " +
                           std::to_string(cfg.channel.rx_array_gain_dbi) + ")");
    }

    // -- traffic --
    const auto& tc = cfg.mesh.traffic;
    checkOneOf(r, "traffic.model", tc.model,
               {"constant", "poisson", "on_off"});
    checkOneOf(r, "traffic.flow_topology", tc.flow_topology,
               {"all_pairs", "random_pairs", "gateway"});
    checkPositive(r, "traffic.demand_mbps", tc.demand_mbps);

    if (tc.flow_topology == "gateway")
    {
        if (tc.gateway_node_id.empty())
        {
            r.errors.push_back(
                "traffic.gateway_node_id is required when flow_topology='gateway'");
        }
        else
        {
            bool found = false;
            for (const auto& n : cfg.nodes)
            {
                if (n.id == tc.gateway_node_id)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                r.errors.push_back(
                    "traffic.gateway_node_id '" + tc.gateway_node_id +
                    "' does not match any node ID");
            }
        }
    }

    if (tc.flow_topology == "random_pairs" && tc.random_pair_count == 0)
    {
        r.errors.push_back(
            "traffic.random_pair_count must be > 0 when flow_topology='random_pairs'");
    }

    // -- routing --
    checkOneOf(r, "routing.algorithm", cfg.mesh.routing.algorithm,
               {"shortest_path", "max_throughput", "min_hop"});

    // -- node types --
    for (const auto& node : cfg.nodes)
    {
        checkOneOf(r, "node '" + node.id + "' node_type", node.node_type,
                   {"drone", "vehicle", "pedestrian"});
    }

    // -- rl --
    if (cfg.rl.enabled)
    {
        checkOneOf(r, "rl.action_type", cfg.rl.action_type,
                   {"discrete", "continuous"});
        checkOneOf(r, "rl.reward_type", cfg.rl.reward_type,
                   {"throughput", "mean_sinr"});

        if (cfg.rl.action_type == "discrete")
        {
            checkPositive(r, "rl.step_size_m", cfg.rl.step_size_m);
        }
        if (cfg.rl.action_type == "continuous")
        {
            checkPositive(r, "rl.arrival_threshold_m", cfg.rl.arrival_threshold_m);
        }

        if (cfg.rl.x_min >= cfg.rl.x_max)
            r.errors.push_back("rl.x_min must be < rl.x_max");
        if (cfg.rl.y_min >= cfg.rl.y_max)
            r.errors.push_back("rl.y_min must be < rl.y_max");

        if (!cfg.rl.controlled_node_id.empty())
        {
            bool found = false;
            for (const auto& n : cfg.nodes)
            {
                if (n.id == cfg.rl.controlled_node_id)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                r.errors.push_back(
                    "rl.controlled_node_id '" + cfg.rl.controlled_node_id +
                    "' does not match any node ID");
            }
        }
    }

    // -- buildings --
    for (const auto& b : cfg.buildings)
    {
        std::string label = "building '" + b.id + "'";
        if (b.x_min >= b.x_max)
            r.errors.push_back(label + ": x_min must be < x_max");
        if (b.y_min >= b.y_max)
            r.errors.push_back(label + ": y_min must be < y_max");
        if (b.z_min >= b.z_max)
            r.errors.push_back(label + ": z_min must be < z_max");
    }

    return r;
}

}  // namespace mesh_sim
