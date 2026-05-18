/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/config/config-loader.h"
#include "src/util/ini-parser.h"
#include "src/util/string-utils.h"
#include "third_party/json.hpp"

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

namespace mesh_sim
{

// ---------------------------------------------------------------------------
// JSON -> POD helpers (inlined here; no separate spec-parser file needed)
// ---------------------------------------------------------------------------

static NodeSpec
parseNodeSpec(const json& j)
{
    NodeSpec n;
    n.id       = j.at("id").get<std::string>();
    n.role     = j.value("role", "peer");
    n.mobility  = j.value("mobility", "fixed");
    n.node_type = j.value("node_type", "drone");

    if (j.contains("position"))
    {
        const auto& p = j["position"];
        n.position.x  = p.value("x", 0.0);
        n.position.y  = p.value("y", 0.0);
        n.position.z  = p.value("z", 0.0);
    }

    if (j.contains("velocity"))
    {
        const auto& v = j["velocity"];
        n.velocity.vx = v.value("vx", 0.0);
        n.velocity.vy = v.value("vy", 0.0);
        n.velocity.vz = v.value("vz", 0.0);
    }

    if (j.contains("random_walk"))
    {
        const auto& rw = j["random_walk"];
        if (rw.contains("bounds"))
        {
            const auto& b   = rw["bounds"];
            n.random_walk.x_min = b.value("x_min", -100.0);
            n.random_walk.x_max = b.value("x_max",  100.0);
            n.random_walk.y_min = b.value("y_min", -100.0);
            n.random_walk.y_max = b.value("y_max",  100.0);
        }
        n.random_walk.speed_mps = rw.value("speed_mps", 1.5);
    }

    if (j.contains("waypoints"))
    {
        for (const auto& jw : j["waypoints"])
        {
            Waypoint w;
            w.t = jw.value("t", 0.0);
            w.x = jw.value("x", 0.0);
            w.y = jw.value("y", 0.0);
            w.z = jw.value("z", 0.0);
            n.waypoints.push_back(w);
        }
    }

    return n;
}

static BuildingSpec
parseBuildingSpec(const json& j)
{
    BuildingSpec b;
    b.id = j.value("id", "");

    if (j.contains("bounds"))
    {
        const auto& bnd = j["bounds"];
        b.x_min = bnd.value("x_min", 0.0);
        b.x_max = bnd.value("x_max", 1.0);
        b.y_min = bnd.value("y_min", 0.0);
        b.y_max = bnd.value("y_max", 1.0);
        b.z_min = bnd.value("z_min", 0.0);
        b.z_max = bnd.value("z_max", 1.0);
    }

    b.type      = j.value("type",      "Residential");
    b.ext_walls = j.value("ext_walls", "ConcreteWithWindows");
    b.n_floors  = j.value("n_floors",  1);
    b.n_rooms_x = j.value("n_rooms_x", 1);
    b.n_rooms_y = j.value("n_rooms_y", 1);
    return b;
}

// ---------------------------------------------------------------------------
// ConfigLoader::Load
// ---------------------------------------------------------------------------

SimConfig
ConfigLoader::Load(const std::string& run_config_path,
                   const std::string& positions_override_path)
{
    namespace fs = std::filesystem;

    const std::string base_dir = dirOf(run_config_path);

    // --- Parse run.ini ---
    auto ini = parseIni(run_config_path);

    SimConfig cfg;

    // [scenario]
    cfg.scenario_name = iniGet(ini, "scenario", "name", "unnamed");
    cfg.seed      = static_cast<uint32_t>(std::stoul(iniGet(ini, "scenario", "seed",       "42")));
    cfg.run_id    = static_cast<uint32_t>(std::stoul(iniGet(ini, "scenario", "run_id",     "1")));
    cfg.duration_s = std::stod(iniGet(ini, "scenario", "duration_s", "10.0"));
    cfg.warmup_s   = std::stod(iniGet(ini, "scenario", "warmup_s",   "0.0"));
    cfg.tick_s     = std::stod(iniGet(ini, "scenario", "tick_s",     "0.1"));

    // [output]
    cfg.output_dir = iniGet(ini, "output", "dir", "");
    if (cfg.output_dir.empty())
    {
        // Auto-generate a timestamped output directory anchored to
        // scratch/mesh-sim/outputs/ regardless of cwd.
        // base_dir points at the scenario dir (e.g. .../inputs/baselines/foo)
        // so we go up three levels to reach scratch/mesh-sim/.
        fs::path mesh_sim_dir = fs::path(base_dir).parent_path()  // baselines/
                                                    .parent_path()  // inputs/
                                                    .parent_path(); // mesh-sim/
        std::time_t now = std::time(nullptr);
        std::tm* lt = std::localtime(&now);

        std::ostringstream ts_month, ts_day, ts_time;
        ts_month << std::put_time(lt, "%Y-%m");
        ts_day   << std::put_time(lt, "%d");
        ts_time  << std::put_time(lt, "%H-%M-%S");

        fs::path out = fs::weakly_canonical(mesh_sim_dir)
                       / "outputs" / ts_month.str() / ts_day.str() / ts_time.str();
        cfg.output_dir = out.string();
    }
    else if (!fs::path(cfg.output_dir).is_absolute())
    {
        cfg.output_dir = (fs::path(base_dir) / cfg.output_dir).string();
    }

    cfg.viz_tick_ms = static_cast<uint32_t>(
        std::stoul(iniGet(ini, "output", "viz_tick_ms", "100")));

    // [channel] — shared params
    cfg.channel.frequency_ghz    = std::stod(iniGet(ini, "channel", "frequency_ghz",    "28.0"));
    cfg.channel.tx_power_dbm     = std::stod(iniGet(ini, "channel", "tx_power_dbm",     "30.0"));
    cfg.channel.scenario         = iniGet(ini, "channel", "scenario",         "UMi");
    cfg.channel.channel_model    = iniGet(ini, "channel", "channel_model",    "3gpp");
    cfg.channel.blockage_enabled = iniGetBool(ini, "channel", "blockage_enabled", true);
    cfg.channel.beamforming_model = iniGet(ini, "channel", "beamforming_model", "svd");
    cfg.channel.amc_model        = iniGet(ini, "channel", "amc_model",        "shannon");
    cfg.channel.noise_figure_db  = std::stod(iniGet(ini, "channel", "noise_figure_db",  "5.0"));
    cfg.channel.bandwidth_mhz   = std::stod(iniGet(ini, "channel", "bandwidth_mhz",   "400.0"));
    cfg.channel.tx_array_gain_dbi = std::stod(iniGet(ini, "channel", "tx_array_gain_dbi", "12.0"));
    cfg.channel.rx_array_gain_dbi = std::stod(iniGet(ini, "channel", "rx_array_gain_dbi", "12.0"));

    // Validate channel_model
    {
        const std::string& cm = cfg.channel.channel_model;
        if (cm != "3gpp" && cm != "nyu")
        {
            throw std::runtime_error(
                "[config] Unknown channel_model '" + cm +
                "'. Must be '3gpp' or 'nyu'.");
        }
    }

    // [nyu_channel] — NYU-specific params
    cfg.channel.nyu.rf_bandwidth_mhz         = std::stod(iniGet(ini, "nyu_channel", "rf_bandwidth_mhz",         "800.0"));
    cfg.channel.nyu.shadowing_enabled         = iniGetBool(ini, "nyu_channel", "shadowing_enabled",         true);
    cfg.channel.nyu.pressure_mbar            = std::stod(iniGet(ini, "nyu_channel", "pressure_mbar",            "1013.25"));
    cfg.channel.nyu.humidity_pct             = std::stod(iniGet(ini, "nyu_channel", "humidity_pct",             "50.0"));
    cfg.channel.nyu.temperature_c            = std::stod(iniGet(ini, "nyu_channel", "temperature_c",            "20.0"));
    cfg.channel.nyu.rain_rate_mm_hr          = std::stod(iniGet(ini, "nyu_channel", "rain_rate_mm_hr",          "0.0"));
    cfg.channel.nyu.atmospheric_loss_enabled = iniGetBool(ini, "nyu_channel", "atmospheric_loss_enabled", false);
    cfg.channel.nyu.foliage_loss_enabled     = iniGetBool(ini, "nyu_channel", "foliage_loss_enabled",     false);
    cfg.channel.nyu.foliage_loss_db_m        = std::stod(iniGet(ini, "nyu_channel", "foliage_loss_db_m",        "0.4"));
    cfg.channel.nyu.o2i_loss_type            = iniGet(ini, "nyu_channel", "o2i_loss_type", "Low Loss");

    // [traffic] — mesh-specific flow-level traffic config
    cfg.mesh.traffic.model            = iniGet(ini, "traffic", "model",            "constant");
    cfg.mesh.traffic.demand_mbps      = std::stod(iniGet(ini, "traffic", "demand_mbps",      "10.0"));
    cfg.mesh.traffic.arrival_rate_hz  = std::stod(iniGet(ini, "traffic", "arrival_rate_hz",  "1.0"));
    cfg.mesh.traffic.on_time_s        = std::stod(iniGet(ini, "traffic", "on_time_s",        "1.0"));
    cfg.mesh.traffic.off_time_s       = std::stod(iniGet(ini, "traffic", "off_time_s",       "1.0"));
    cfg.mesh.traffic.holding_time_s   = std::stod(iniGet(ini, "traffic", "holding_time_s",   "0.0"));
    cfg.mesh.traffic.flow_topology    = iniGet(ini, "traffic", "flow_topology",    "all_pairs");
    cfg.mesh.traffic.random_pair_count = static_cast<uint32_t>(
        std::stoul(iniGet(ini, "traffic", "random_pair_count", "3")));
    cfg.mesh.traffic.gateway_node_id  = iniGet(ini, "traffic", "gateway_node_id",  "");

    // [routing] — mesh-specific routing config
    cfg.mesh.routing.algorithm = iniGet(ini, "routing", "algorithm", "shortest_path");
    cfg.mesh.routing.max_hops  = static_cast<uint32_t>(
        std::stoul(iniGet(ini, "routing", "max_hops", "5")));

    // [rl] — reinforcement learning config
    cfg.rl.enabled              = iniGetBool(ini, "rl", "enabled", false);
    cfg.rl.controlled_node_id   = iniGet(ini, "rl", "controlled_node_id", "");
    cfg.rl.action_type          = iniGet(ini, "rl", "action_type", "discrete");
    cfg.rl.reward_type          = iniGet(ini, "rl", "reward_type", "throughput");
    cfg.rl.step_size_m          = std::stod(iniGet(ini, "rl", "step_size_m", "50.0"));
    cfg.rl.arrival_threshold_m  = std::stod(iniGet(ini, "rl", "arrival_threshold_m", "1.0"));
    cfg.rl.x_min                = std::stod(iniGet(ini, "rl", "x_min", "-1000.0"));
    cfg.rl.x_max                = std::stod(iniGet(ini, "rl", "x_max", "2000.0"));
    cfg.rl.y_min                = std::stod(iniGet(ini, "rl", "y_min", "-1000.0"));
    cfg.rl.y_max                = std::stod(iniGet(ini, "rl", "y_max", "1000.0"));

    // --- Load nodes.json ---
    std::string nodes_file = iniGet(ini, "scenario", "nodes_file", "nodes.json");
    std::string nodes_path = resolvePath(base_dir, nodes_file);
    {
        std::ifstream nf(nodes_path);
        if (!nf.is_open())
        {
            throw std::runtime_error("Cannot open nodes file: " + nodes_path);
        }
        json jnodes;
        nf >> jnodes;
        for (const auto& jn : jnodes)
        {
            cfg.nodes.push_back(parseNodeSpec(jn));
        }
    }

    // --- Load buildings.json (optional) ---
    std::string buildings_file = iniGet(ini, "scenario", "buildings_file", "");
    if (!buildings_file.empty())
    {
        std::string buildings_path = resolvePath(base_dir, buildings_file);
        std::ifstream bf(buildings_path);
        if (!bf.is_open())
        {
            throw std::runtime_error("Cannot open buildings file: " + buildings_path);
        }
        json jbuildings;
        bf >> jbuildings;
        for (const auto& jb : jbuildings)
        {
            cfg.buildings.push_back(parseBuildingSpec(jb));
        }
    }

    // --- Apply positions override (RL extension point) ---
    if (!positions_override_path.empty())
    {
        std::ifstream pf(positions_override_path);
        if (!pf.is_open())
        {
            throw std::runtime_error("Cannot open positions override: " + positions_override_path);
        }
        json jpos;
        pf >> jpos;
        for (auto& node : cfg.nodes)
        {
            if (jpos.contains(node.id))
            {
                const auto& arr = jpos[node.id];
                node.position.x = arr.at(0).get<double>();
                node.position.y = arr.at(1).get<double>();
                node.position.z = arr.at(2).get<double>();
            }
        }
    }

    return cfg;
}

}  // namespace mesh_sim
