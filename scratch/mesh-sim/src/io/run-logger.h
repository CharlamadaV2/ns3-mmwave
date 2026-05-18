/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/** @brief
 * RunLogger: writes a run.log capturing seeds, CLI overrides, and resolved
 * config for reproducibility.  Header-only (same pattern as progress-logger.h).
 * No ns-3 dependency.
 */
#pragma once

#include "src/cli/cli-parser.h"
#include "src/domain/sim-config.h"
#include "src/util/string-utils.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

  /** @brief
  */
namespace mesh_sim
{

/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
inline void
WriteRunLog(const std::string& base_output_dir,
            const CliArgs& args,
            const SimConfig& cfg,
            const std::vector<uint32_t>& seeds)
{
    namespace fs = std::filesystem;
    fs::create_directories(base_output_dir);

    std::ofstream out(base_output_dir + "/run.log");
    if (!out.is_open())
    {
        return;
    }

    auto now = std::chrono::system_clock::now();

    out << "mesh-sim run log\n";
    out << "================\n";
    out << "timestamp:           " << toIso8601(now) << "\n";
    out << "scenario:            " << cfg.scenario_name << "\n";
    out << "run_config:          " << args.run_config_path << "\n";

    // Seeds
    out << "seeds:               [";
    for (size_t i = 0; i < seeds.size(); ++i)
    {
        if (i > 0) out << ", ";
        out << seeds[i];
    }
    out << "]\n";

    if (!args.seeds_arg.empty())
        out << "seed_source:         --seeds CLI argument\n";
    else if (args.seed_override >= 0)
        out << "seed_source:         --seed CLI override\n";
    else
        out << "seed_source:         config default\n";

    // CLI overrides
    out << "\nCLI overrides:\n";
    out << "  --run-config       = " << args.run_config_path << "\n";
    out << "  --seeds            = " << (args.seeds_arg.empty() ? "(not set)" : args.seeds_arg) << "\n";
    out << "  --seed             = " << (args.seed_override < 0 ? "(not set)" : std::to_string(args.seed_override)) << "\n";
    out << "  --run-id           = " << (args.run_id_override < 0 ? "(not set)" : std::to_string(args.run_id_override)) << "\n";
    out << "  --positions-override = " << (args.positions_override_path.empty() ? "(not set)" : args.positions_override_path) << "\n";

    // Resolved config
    out << "\nResolved config:\n";
    out << "  duration_s         = " << cfg.duration_s << "\n";
    out << "  tick_s             = " << cfg.tick_s << "\n";
    out << "  warmup_s           = " << cfg.warmup_s << "\n";
    out << "  run_id             = " << cfg.run_id << "\n";
    out << "  nodes              = " << cfg.nodes.size() << "\n";
    out << "  buildings          = " << cfg.buildings.size() << "\n";
    out << "  channel_model      = " << cfg.channel.channel_model << "\n";
    out << "  scenario           = " << cfg.channel.scenario << "\n";
    out << "  frequency_ghz      = " << cfg.channel.frequency_ghz << "\n";
    out << "  bandwidth_mhz      = " << cfg.channel.bandwidth_mhz << "\n";
    out << "  tx_power_dbm       = " << cfg.channel.tx_power_dbm << "\n";
    out << "  traffic.model      = " << cfg.mesh.traffic.model << "\n";
    out << "  traffic.topology   = " << cfg.mesh.traffic.flow_topology << "\n";
    out << "  traffic.demand_mbps = " << cfg.mesh.traffic.demand_mbps << "\n";
    out << "  routing.algorithm  = " << cfg.mesh.routing.algorithm << "\n";
    out << "  routing.max_hops   = " << cfg.mesh.routing.max_hops << "\n";
}

}  // namespace mesh_sim
