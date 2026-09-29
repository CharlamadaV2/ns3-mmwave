/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file run-logger.h
 * @brief Header-only writer for run.log: seeds, CLI overrides, and resolved config.
 *
 * Call @ref WriteRunLog once per batch run (not per seed) immediately after
 * @ref ResolveSeeds, before the per-seed simulation loop begins.  The log
 * captures the state of @ref CliArgs and @ref SimConfig at that point. Its
 * "Resolved config" block is the source of truth for a run.
 *
 * **Output: @c \<base_output_dir\>/run.log** (batch root, not per seed)
 *
 * | Section            | Contents                                                       |
 * |--------------------|----------------------------------------------------------------|
 * | Header             | ISO-8601 timestamp, scenario name, run_config path.            |
 * | Seeds              | The resolved seed list and its source                          |
 * |                    | (@c --seeds / @c --seed / config default).                     |
 * | CLI overrides      | Every @ref CliArgs field, with @c "(not set)" for unused flags. |
 * | Resolved config    | Key @ref SimConfig scalar fields: timing, channel, traffic,    |
 * |                    | routing, node/building counts, plus band/RL-reward provenance,  |
 * |                    | the resolved RL control block, and jammer counts.              |
 * | Jammers            | One line per configured @ref JammerSpec (omitted when none).    |
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
 * @fn WriteRunLog
 * @brief Write a human-readable reproducibility log for a batch run.
 *
 * Creates @c \<base_output_dir\>/run.log (creating intermediate directories
 * as needed). If the file cannot be opened the function returns silently;
 * this is a best-effort log and should not abort the simulation. The
 * directory creation is not guarded, so it can throw.
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
 * @return void.
 * @throws std::filesystem::filesystem_error if @c create_directories fails.
 *
 * Also derives @c rl.contract from @c cfg.rl (@c mesh_move_2d_v1 for
 * centralized control, @c legacy_continuous or @c legacy_discrete7
 * otherwise), lists controlled node ids, and counts enabled jammers.
 * @c jammer_path_enabled is true only for band @c sub-6 with at least one
 * enabled jammer. Overwrites an existing @c run.log.
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
    out << "  --output-dir         = "
        << (args.output_dir.empty() ? "(not set)" : args.output_dir) << "\n";
    out << "  --debug-links        = " << (args.debug_links ? "true" : "false") << "\n";
    out << "  --rl-mode            = " << (args.rl_mode ? "true" : "false") << "\n";
    out << "  --band               = "
        << (args.band.empty() ? "(not set)" : args.band) << "\n";

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

    size_t jammersEnabled = 0;
    for (const auto& j : cfg.jammers)
    {
        if (j.enabled)
            ++jammersEnabled;
    }
    const bool jammerPathEnabled = (cfg.band == "sub-6" && jammersEnabled > 0);

    out << "  band                = " << cfg.band                          << "\n";
    out << "  band_source         = " << cfg.band_source                   << "\n";
    out << "  rl.reward_type      = " << cfg.rl.reward_type                << "\n";
    out << "  rl.reward_alias     = "
        << (cfg.rl.reward_type_alias.empty() ? "none" : cfg.rl.reward_type_alias) << "\n";

    // RL control provenance: resolved by ResolveRlControl before this is called.
    std::string rlContract = "legacy_discrete7";
    if (cfg.rl.control_mode == "centralized")
        rlContract = "mesh_move_2d_v1";
    else if (cfg.rl.action_type == "continuous")
        rlContract = "legacy_continuous";

    std::string rlControlled;
    for (uint32_t idx : cfg.rl.controlled_indices)
    {
        if (idx >= cfg.nodes.size())
            continue;
        if (!rlControlled.empty())
            rlControlled += ",";
        rlControlled += cfg.nodes[idx].id;
    }

    out << "  rl.control_mode     = " << cfg.rl.control_mode                 << "\n";
    out << "  rl.contract         = " << rlContract                          << "\n";
    out << "  rl.controlled_nodes = "
        << (rlControlled.empty() ? "(none)" : rlControlled)                  << "\n";
    out << "  rl.max_controlled_nodes = " << cfg.rl.num_slots                << "\n";
    out << "  rl.decision_interval_s  = "
        << (cfg.rl.decision_interval_ticks * cfg.tick_s)                     << "\n";
    out << "  rl.decision_interval_ticks = " << cfg.rl.decision_interval_ticks << "\n";
    out << "  jammers.configured  = " << cfg.jammers.size()                << "\n";
    out << "  jammers.enabled     = " << jammersEnabled                    << "\n";
    out << "  jammer_path_enabled = " << (jammerPathEnabled ? "true" : "false") << "\n";

    if (!cfg.jammers.empty())
    {
        out << "\nJammers:\n";
        for (const auto& j : cfg.jammers)
        {
            const char* motion = !j.waypoints.empty() ? "waypoints"
                               : (j.velocity.vx != 0.0 || j.velocity.vy != 0.0 ||
                                  j.velocity.vz != 0.0) ? "velocity"
                               : "static";

            out << "  " << (j.id.empty() ? "(unnamed)" : j.id)
                << ": enabled=" << (j.enabled ? "true" : "false")
                << " type=" << j.type
                << " target_freq_mhz=[";
            for (size_t i = 0; i < j.target_freq.size(); ++i)
            {
                if (i > 0) out << ", ";
                out << j.target_freq[i];
            }
            out << "]"
                << " tx_power_dbm=" << j.tx_power_dbm
                << " tx_array_gain_dbi=" << j.tx_array_gain_dbi
                << " duty_cycle=" << j.duty_cycle
                << " max_range_m=" << j.max_range_m
                << " beamwidth_deg=" << j.beamwidth_deg
                << " azimuth_deg=" << j.azimuth_deg
                << " zenith_deg=" << j.zenith_deg
                << " motion=" << motion
                << "\n";
        }
    }
}

}  // namespace mesh_sim