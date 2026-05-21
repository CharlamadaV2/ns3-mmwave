/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file run-logger.h
 * @brief Writes a human-readable run.log capturing seeds, CLI overrides,
 *        and the fully resolved configuration for reproducibility.
 *
 *
 * Call @ref WriteRunLog once per batch run (not per seed) immediately after
 * @ref ResolveSeeds, before the per-seed simulation loop begins.  The log
 * captures the state of @ref CliArgs and @ref SimConfig at that point so
 * that any subsequent seed run can be reproduced from the archived inputs
 * alone.
 *
 * **Output: <base_output_dir>/run.log**
 *
 * | Section            | Contents                                                       |
 * |--------------------|----------------------------------------------------------------|
 * | Header             | ISO-8601 timestamp, scenario name, run_config path.            |
 * | Seeds              | The resolved seed list and which source provided them          |
 * |                    | (@c --seeds / @c --seed / config default).                     |
 * | CLI overrides      | Every @ref CliArgs field, with @c "(not set)" for unused flags. |
 * | Resolved config    | Key @ref SimConfig scalar fields: timing, channel, traffic,    |
 * |                    | routing, node/building counts.                                 |
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

namespace mesh_sim
{

/**
 * @brief Write a human-readable reproducibility log for a batch run.
 *
 * Creates @c <base_output_dir>/run.log (creating intermediate directories
 * as needed). If the file cannot be opened the function returns silently;
 * this is a best-effort log and should not abort the simulation.
 *
 * **Seed source label**
 * The log annotates which source provided the seed list:
 * - @c "--seeds CLI argument" when @c args.seeds_arg is non-empty.
 * - @c "--seed CLI override" when @c args.seed_override >= 0.
 * - @c "config default" otherwise.
 *
 * @param base_output_dir  Batch output root directory; @c run.log is written
 *                         directly inside it (not per-seed).
 * @param args             Parsed CLI arguments from @ref ParseCommandLine.
 * @param cfg              Fully loaded @ref SimConfig after @ref ConfigLoader::Load.
 *                         Only scalar fields are logged; node and building
 *                         details are preserved in the archived @c nodes.json.
 * @param seeds            Resolved seed list from @ref ResolveSeeds.
 */
inline void
WriteRunLog(const std::string&       base_output_dir,
            const CliArgs&           args,
            const SimConfig&         cfg,
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
    out << "timestamp:           " << toIso8601(now)          << "\n";
    out << "scenario:            " << cfg.scenario_name        << "\n";
    out << "run_config:          " << args.run_config_path     << "\n";

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

    // CLI overrides — every CliArgs field, "(not set)" for absent flags
    out << "\nCLI overrides:\n";
    out << "  --run-config         = " << args.run_config_path << "\n";
    out << "  --seeds              = "
        << (args.seeds_arg.empty() ? "(not set)" : args.seeds_arg) << "\n";
    out << "  --seed               = "
        << (args.seed_override < 0 ? "(not set)" : std::to_string(args.seed_override)) << "\n";
    out << "  --run-id             = "
        << (args.run_id_override < 0 ? "(not set)" : std::to_string(args.run_id_override)) << "\n";
    out << "  --positions-override = "
        << (args.positions_override_path.empty() ? "(not set)" : args.positions_override_path) << "\n";

    // Resolved config — key scalar fields
    out << "\nResolved config:\n";
    out << "  duration_s          = " << cfg.duration_s                    << "\n";
    out << "  tick_s              = " << cfg.tick_s                        << "\n";
    out << "  warmup_s            = " << cfg.warmup_s                      << "\n";
    out << "  run_id              = " << cfg.run_id                        << "\n";
    out << "  nodes               = " << cfg.nodes.size()                  << "\n";
    out << "  buildings           = " << cfg.buildings.size()              << "\n";
    out << "  channel_model       = " << cfg.channel.channel_model         << "\n";
    out << "  scenario            = " << cfg.channel.scenario              << "\n";
    out << "  frequency_ghz       = " << cfg.channel.frequency_ghz        << "\n";
    out << "  bandwidth_mhz       = " << cfg.channel.bandwidth_mhz        << "\n";
    out << "  tx_power_dbm        = " << cfg.channel.tx_power_dbm         << "\n";
    out << "  traffic.model       = " << cfg.mesh.traffic.model            << "\n";
    out << "  traffic.topology    = " << cfg.mesh.traffic.flow_topology    << "\n";
    out << "  traffic.demand_mbps = " << cfg.mesh.traffic.demand_mbps      << "\n";
    out << "  routing.algorithm   = " << cfg.mesh.routing.algorithm        << "\n";
    out << "  routing.max_hops    = " << cfg.mesh.routing.max_hops         << "\n";
}

}  // namespace mesh_sim