/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/config/config-loader.h"
#include "third_party/json.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

namespace mmwave_sim
{

static std::string
trimStr(const std::string& s)
{
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos)
    {
        return "";
    }
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static std::map<std::string, std::map<std::string, std::string>>
parseIni(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    std::map<std::string, std::map<std::string, std::string>> sections;
    std::string section;
    std::string line;

    while (std::getline(f, line))
    {
        // Strip inline comments
        auto cpos = line.find('#');
        if (cpos != std::string::npos)
        {
            line = line.substr(0, cpos);
        }
        cpos = line.find(';');
        if (cpos != std::string::npos)
        {
            line = line.substr(0, cpos);
        }

        line = trimStr(line);
        if (line.empty())
        {
            continue;
        }

        if (line.front() == '[' && line.back() == ']')
        {
            section = trimStr(line.substr(1, line.size() - 2));
        }
        else
        {
            auto eq = line.find('=');
            if (eq == std::string::npos)
            {
                continue;
            }
            std::string key = trimStr(line.substr(0, eq));
            std::string val = trimStr(line.substr(eq + 1));
            sections[section][key] = val;
        }
    }
    return sections;
}

static std::string
iniGet(const std::map<std::string, std::map<std::string, std::string>>& ini,
       const std::string& section,
       const std::string& key,
       const std::string& def = "")
{
    auto sit = ini.find(section);
    if (sit == ini.end())
    {
        return def;
    }
    auto kit = sit->second.find(key);
    if (kit == sit->second.end())
    {
        return def;
    }
    return kit->second;
}

static bool
iniGetBool(const std::map<std::string, std::map<std::string, std::string>>& ini,
           const std::string& section,
           const std::string& key,
           bool def)
{
    std::string v = iniGet(ini, section, key, def ? "true" : "false");
    std::string lv = v;
    std::transform(lv.begin(), lv.end(), lv.begin(), ::tolower);
    return (lv == "true" || lv == "1" || lv == "yes");
}

static std::string
resolvePath(const std::string& base_dir, const std::string& path)
{
    if (path.empty())
    {
        return path;
    }
    if (path.front() == '/')
    {
        return path;  // already absolute
    }
    return base_dir + "/" + path;
}

static std::string
dirOf(const std::string& path)
{
    auto pos = path.rfind('/');
    if (pos == std::string::npos)
    {
        return ".";
    }
    return path.substr(0, pos);
}

static NodeSpec
parseNodeSpec(const json& j)
{
    NodeSpec n;
    n.id       = j.at("id").get<std::string>();
    n.role     = j.at("role").get<std::string>();
    n.mobility = j.value("mobility", "fixed");

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
    return b;
}

// ---------------------------------------------------------------------------
// ConfigLoader::Load
// ---------------------------------------------------------------------------

SimConfig
ConfigLoader::Load(const std::string& run_config_path,
                   const std::string& positions_override_path)
{
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
        cfg.output_dir = "outputs/" + cfg.scenario_name
                       + "/seed-" + std::to_string(cfg.seed)
                       + "/run-"  + std::to_string(cfg.run_id);
    }

    // Channel
    cfg.channel.frequency_ghz    = std::stod(iniGet(ini, "channel", "frequency_ghz",    "28.0"));
    cfg.channel.tx_power_dbm     = std::stod(iniGet(ini, "channel", "tx_power_dbm",     "30.0"));
    cfg.channel.scenario         = iniGet(ini, "channel", "scenario",         "UMi");
    cfg.channel.blockage_enabled = iniGetBool(ini, "channel", "blockage_enabled", true);

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
    // TODO (RL): This only sets node positions before the simulation starts.
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
