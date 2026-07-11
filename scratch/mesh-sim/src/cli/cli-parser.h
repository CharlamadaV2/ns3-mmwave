/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

/**
 * @file cli-parser.h
 * @brief Command-line parsing and pre-simulation setup helpers.
 *
 * Separates argument handling and reproducibility bookkeeping from the
 * top-level orchestration in @c sim.cc.
 *
 * **Typical call sequence in sim.cc**
 * @code
 * CliArgs    args  = ParseCommandLine(argc, argv);
 * SimConfig  cfg   = ConfigLoader::Load(args.run_config_path,
 *                                       args.positions_override_path);
 * auto       seeds = ResolveSeeds(args, cfg);
 * ArchiveScenarioInputs(args.output_dir, args.run_config_path);
 * @endcode
 */
#pragma once

#include "src/domain/sim-config.h"

#include <string>
#include <vector>

namespace mesh_sim
{

/**
 * @brief Parsed command-line arguments for one simulator invocation.
 *
 * Populated by @ref ParseCommandLine and consumed by @ref ResolveSeeds
 * and the rest of @c sim.cc.  String fields are empty when the
 * corresponding flag was not supplied on the command line.
 */
struct CliArgs
{
    std::string run_config_path;          ///< Path to @c run.ini (required).
    std::string positions_override_path;  ///< Optional path to a positions-override JSON.
                                          ///<   When non-empty, node positions are patched
                                          ///<   after @c nodes.json is loaded (RL extension point).
    std::string output_dir;               ///< Output root directory override.
                                          ///<   Empty means @c sim.cc auto-generates a
                                          ///<   timestamped path.
    std::string seeds_arg;                ///< Raw comma-separated seed list from @c --seeds
                                          ///<   (e.g. @c "1,2,3,4,5"). Empty when not supplied.
    int         seed_override   = -1;     ///< Single-seed override from @c --seed.
                                          ///<   Negative means not set.
    int         run_id_override = -1;     ///< Run-ID override from @c --run-id.
                                          ///<   Negative means not set.
    bool        debug_links     = false;  ///< Enable verbose per-link debug logging when @c true.
    bool        rl_mode         = false;  ///< Enable RL mode (stdin/stdout JSON exchange) when @c true.
    std::string band            = "mmwave"; ///< Radio band: "mmwave" (no interference, default)
                                            ///<   or "sub-6" (co-channel interference summed).
};


/**
 * @brief Parse @c argv via @c ns3::CommandLine and return a populated @ref CliArgs.
 *
 * Registered flags:
 * | Flag                    | Type   | Required | Description                                  |
 * |-------------------------|--------|----------|----------------------------------------------|
 * | @c --run-config         | string | yes      | Path to @c run.ini.                          |
 * | @c --positions-override | string | no       | Path to positions-override JSON (RL hook).   |
 * | @c --seeds              | string | no       | Comma-separated seed list (e.g. @c "1,2,3"). |
 * | @c --seed               | int    | no       | Single-seed override.                        |
 * | @c --run-id             | int    | no       | Run-ID override.                             |
 * | @c --output-dir         | string | no       | Output directory override.                   |
 * | @c --debug-links        | bool   | no       | Enable verbose link-evaluation logging.      |
 * | @c --rl-mode            | bool   | no       | Enable RL stdin/stdout JSON exchange.        |
 *
 * Exits with code 1 if:
 * - @c --run-config is not provided.
 * - The @c run-config file does not exist on disk.
 * - @c --positions-override is provided but the file does not exist.
 *
 * @param argc  Argument count from @c main.
 * @param argv  Argument vector from @c main.
 * @return Populated @ref CliArgs struct.
 */
CliArgs ParseCommandLine(int argc, char* argv[]);

/**
 * @brief Determine the list of random seeds to run from CLI args and config defaults.
 *
 * Resolution priority (highest to lowest):
 * -# @c --seeds (comma-separated list) — parsed via @c parseSeedList().
 * -# @c --seed  (single integer override).
 * -# @c cfg.seed (value from @c run.ini).
 *
 * Exits with code 1 if the resolved list is empty (e.g. @c --seeds was
 * supplied but contained no valid integers after parsing).
 *
 * @param args  Parsed CLI arguments from @ref ParseCommandLine.
 * @param cfg   Fully loaded simulation config from @c ConfigLoader::Load.
 * @return Non-empty vector of seed values in the order they should be run.
 */
std::vector<uint32_t> ResolveSeeds(const CliArgs& args, const SimConfig& cfg);

/**
 * @brief Copy all scenario input files into the batch output directory for reproducibility.
 *
 * Creates @c <base_output_dir>/inputs/ and copies every regular file from
 * the directory containing @c run_config_path (typically @c nodes.json,
 * @c buildings.json, and @c run.ini itself).  Subdirectories are not
 * traversed.  Existing files in the destination are silently overwritten.
 *
 * This ensures every batch output directory is self-contained: re-running
 * with the archived inputs produces identical results without needing the
 * original scenario directory.
 *
 * @param base_output_dir  Root output directory for this batch run.
 * @param run_config_path  Path to @c run.ini; its parent directory is the
 *                         source of files to archive.
 */
void ArchiveScenarioInputs(const std::string& base_output_dir,
                            const std::string& run_config_path);

}  // namespace mesh_sim