/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/config/config-loader.h"
#include "src/config/spec-parser.h"
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

namespace mmwave_sim
{

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
    cfg.scenario_name = iniGet(ini, "scenario", "name", "unnamed");
    cfg.seed          = static_cast<uint32_t>(std::stoul(iniGet(ini, "scenario", "seed",    "42")));
    cfg.run_id        = static_cast<uint32_t>(std::stoul(iniGet(ini, "scenario", "run_id",  "1")));
    cfg.duration_s    = std::stod(iniGet(ini, "scenario", "duration_s", "2.0"));
    cfg.warmup_s      = std::stod(iniGet(ini, "scenario", "warmup_s",   "0.1"));

    cfg.output_dir = iniGet(ini, "output", "dir", "");
    if (cfg.output_dir.empty())
    {
        // Auto-generate a timestamped output directory anchored to
        // scratch/mmwave-sim/outputs/ regardless of cwd.
        // base_dir points at the scenario dir (e.g. .../inputs/scenarios/foo)
        // so we go up three levels to reach scratch/mmwave-sim/.
        fs::path mmwave_sim_dir = fs::path(base_dir).parent_path()  // scenarios/
                                                     .parent_path()  // inputs/
                                                     .parent_path(); // mmwave-sim/
        std::time_t now = std::time(nullptr);
        std::tm* lt = std::localtime(&now);

        std::ostringstream ts_month, ts_day, ts_time;
        ts_month << std::put_time(lt, "%Y-%m");
        ts_day   << std::put_time(lt, "%d");
        ts_time  << std::put_time(lt, "%H-%M-%S");

        fs::path out = fs::weakly_canonical(mmwave_sim_dir)
                       / "outputs" / ts_month.str() / ts_day.str() / ts_time.str();
        cfg.output_dir = out.string();
    }
    else if (!fs::path(cfg.output_dir).is_absolute())
    {
        // Relative output_dir in INI — resolve against the scenario directory
        cfg.output_dir = (fs::path(base_dir) / cfg.output_dir).string();
    }

    // Channel — shared params
    cfg.channel.frequency_ghz    = std::stod(iniGet(ini, "channel", "frequency_ghz",    "28.0"));
    cfg.channel.tx_power_dbm     = std::stod(iniGet(ini, "channel", "tx_power_dbm",     "30.0"));
    cfg.channel.scenario         = iniGet(ini, "channel", "scenario",         "UMi");
    cfg.channel.channel_model    = iniGet(ini, "channel", "channel_model",    "3gpp");
    cfg.channel.blockage_enabled = iniGetBool(ini, "channel", "blockage_enabled", true);

    // Validate channel_model before anything downstream relies on it
    {
        const std::string& cm = cfg.channel.channel_model;
        if (cm != "3gpp" && cm != "nyu")
        {
            throw std::runtime_error(
                "[config] Unknown channel_model '" + cm +
                "'. Must be '3gpp' or 'nyu'.");
        }
    }

    // Performance tuning
    cfg.channel.channel_update_period_ms = static_cast<uint32_t>(
        std::stoul(iniGet(ini, "channel", "channel_update_period_ms", "0")));
    cfg.channel.condition_update_period_ms = static_cast<uint32_t>(
        std::stoul(iniGet(ini, "channel", "condition_update_period_ms", "0")));
    cfg.channel.beamforming_model = iniGet(ini, "channel", "beamforming_model", "svd");
    cfg.channel.cqi_period_slots = static_cast<uint32_t>(
        std::stoul(iniGet(ini, "channel", "cqi_period_slots", "20")));
    cfg.channel.amc_model = iniGet(ini, "channel", "amc_model", "error");

    // NYU-specific channel params — only relevant when channel_model = "nyu"
    cfg.channel.nyu.rf_bandwidth_mhz          = std::stod(iniGet(ini, "nyu_channel", "rf_bandwidth_mhz",          "800.0"));
    cfg.channel.nyu.shadowing_enabled          = iniGetBool(ini, "nyu_channel", "shadowing_enabled",          true);
    cfg.channel.nyu.pressure_mbar             = std::stod(iniGet(ini, "nyu_channel", "pressure_mbar",             "1013.25"));
    cfg.channel.nyu.humidity_pct              = std::stod(iniGet(ini, "nyu_channel", "humidity_pct",              "50.0"));
    cfg.channel.nyu.temperature_c             = std::stod(iniGet(ini, "nyu_channel", "temperature_c",             "20.0"));
    cfg.channel.nyu.rain_rate_mm_hr           = std::stod(iniGet(ini, "nyu_channel", "rain_rate_mm_hr",           "0.0"));
    cfg.channel.nyu.atmospheric_loss_enabled  = iniGetBool(ini, "nyu_channel", "atmospheric_loss_enabled",  false);
    cfg.channel.nyu.foliage_loss_enabled      = iniGetBool(ini, "nyu_channel", "foliage_loss_enabled",      false);
    cfg.channel.nyu.foliage_loss_db_m         = std::stod(iniGet(ini, "nyu_channel", "foliage_loss_db_m",         "0.4"));
    cfg.channel.nyu.o2i_loss_type             = iniGet(ini, "nyu_channel", "o2i_loss_type", "Low Loss");

    // Traffic
    cfg.traffic.direction               = iniGet(ini, "traffic", "direction",               "dl");
    cfg.traffic.packet_size_bytes       = static_cast<uint32_t>(
        std::stoul(iniGet(ini, "traffic", "packet_size_bytes", "1400")));
    cfg.traffic.inter_packet_interval_us = std::stod(
        iniGet(ini, "traffic", "inter_packet_interval_us", "100.0"));
    cfg.traffic.app_start_offset_s = std::stod(
        iniGet(ini, "traffic", "app_start_offset_s", "0.1"));
    cfg.traffic.max_packets = static_cast<uint32_t>(
        std::stoul(iniGet(ini, "traffic", "max_packets", "0")));

    // Network (backhaul)
    cfg.network.backhaul_data_rate = iniGet(ini, "network", "backhaul_data_rate", "100Gb/s");
    cfg.network.backhaul_delay_ms  = std::stod(iniGet(ini, "network", "backhaul_delay_ms", "10.0"));

    // Visualisation
    cfg.viz_tick_ms = static_cast<uint32_t>(
        std::stoul(iniGet(ini, "output", "viz_tick_ms", "100")));

    // PCAP
    cfg.pcap_enabled = iniGetBool(ini, "output", "pcap_enabled", false);

    // Trace level
    cfg.trace_level = iniGet(ini, "output", "trace_level", "full");

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
    // Overrides node positions before simulation starts.
    // Runtime position changes require a separate control channel (future work).
    if (!positions_override_path.empty())
    {
        std::ifstream pf(positions_override_path);
        if (!pf.is_open())
        {
            throw std::runtime_error("Cannot open positions override: " + positions_override_path);
        }
        json jpos;
        pf >> jpos;
        // Format: {"node_id": [x, y, z], ...}
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

}  // namespace mmwave_sim
