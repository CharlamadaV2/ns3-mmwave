/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

#include "src/cli/cli-parser.h"
#include "src/util/string-utils.h"

#include "ns3/command-line.h"

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace mesh_sim
{

CliArgs
ParseCommandLine(int argc, char* argv[])
{
    CliArgs args;

    ns3::CommandLine cmd;
    cmd.AddValue("run-config",
                 "Path to run.ini scenario configuration file",
                 args.run_config_path);
    cmd.AddValue("positions-override",
                 "Optional JSON file overriding node positions. "
                 "Positions are applied before the simulation starts only.",
                 args.positions_override_path);
    cmd.AddValue("seeds",
                 "Comma-separated list of seeds to run (e.g. 1,2,3,4,5). "
                 "Each seed runs as an independent simulation with its own output subdirectory.",
                 args.seeds_arg);
    cmd.AddValue("seed",
                 "Override the seed value from run.ini (single seed).",
                 args.seed_override);
    cmd.AddValue("run-id",
                 "Override the run_id value from run.ini.",
                 args.run_id_override);
    cmd.AddValue("debug-links",
                 "Enable verbose link-evaluation debug logging.",
                 args.debug_links);
    cmd.Parse(argc, argv);

    if (args.run_config_path.empty())
    {
        std::cerr << "Error: --run-config=<path> is required.\n";
        std::exit(1);
    }

    if (!fs::exists(args.run_config_path))
    {
        std::cerr << "Error: run-config file not found: "
                  << args.run_config_path << "\n";
        std::exit(1);
    }

    if (!args.positions_override_path.empty() &&
        !fs::exists(args.positions_override_path))
    {
        std::cerr << "Error: positions-override file not found: "
                  << args.positions_override_path << "\n";
        std::exit(1);
    }

    return args;
}

std::vector<uint32_t>
ResolveSeeds(const CliArgs& args, const SimConfig& cfg)
{
    std::vector<uint32_t> seeds;

    if (!args.seeds_arg.empty())
    {
        seeds = parseSeedList(args.seeds_arg);
    }
    else if (args.seed_override >= 0)
    {
        seeds.push_back(static_cast<uint32_t>(args.seed_override));
    }
    else
    {
        seeds.push_back(cfg.seed);
    }

    if (seeds.empty())
    {
        std::cerr << "Error: no seeds to run.\n";
        std::exit(1);
    }

    return seeds;
}

void
ArchiveScenarioInputs(const std::string& base_output_dir,
                       const std::string& run_config_path)
{
    fs::path inputsArchive = fs::path(base_output_dir) / "inputs";
    fs::create_directories(inputsArchive);
    fs::path scenarioDir = fs::path(run_config_path).parent_path();
    for (const auto& entry : fs::directory_iterator(scenarioDir))
    {
        if (fs::is_regular_file(entry))
        {
            fs::copy_file(entry.path(),
                          inputsArchive / entry.path().filename(),
                          fs::copy_options::overwrite_existing);
        }
    }
}

}  // namespace mesh_sim
