/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Command-line parsing and pre-simulation setup helpers.
 * Keeps sim.cc focused on orchestration.
 */
#pragma once

#include "src/domain/sim-config.h"

#include <string>
#include <vector>

namespace mesh_sim
{

struct CliArgs
{
    std::string run_config_path;
    std::string positions_override_path;
    std::string seeds_arg;       // raw comma-separated seed list
    int         seed_override  = -1;
    int         run_id_override = -1;
    bool        debug_links    = false;
};

/**
 * Parse command-line arguments via ns3::CommandLine.
 * Exits with error if --run-config is not provided.
 */
CliArgs ParseCommandLine(int argc, char* argv[]);

/**
 * Determine the seed list from CLI args and config defaults.
 * Priority: --seeds > --seed > cfg.seed.
 */
std::vector<uint32_t> ResolveSeeds(const CliArgs& args, const SimConfig& cfg);

/**
 * Copy all regular files from the scenario input directory
 * into <base_output_dir>/inputs/ for reproducibility.
 */
void ArchiveScenarioInputs(const std::string& base_output_dir,
                            const std::string& run_config_path);

}  // namespace mesh_sim
