/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

/** @brief
 * ConfigLoader: parses run.ini + referenced nodes.json / buildings.json
 * and returns a fully populated SimConfig struct.
 *
 * No ns-3 headers included — this code can be compiled and tested
 * independently of the ns-3 build.
 */
#pragma once

#include "src/domain/sim-config.h"
#include <string>

namespace mesh_sim
{

class ConfigLoader
{
  public:
    /**
     * Load a complete SimConfig from a run.ini file.
     *
     * @param run_config_path  Path to run.ini (required).
     * @param positions_override_path  Optional path to positions_override.json.
     *        If non-empty, the positions of matching node IDs are overwritten after
     *        loading nodes.json.  All other node fields are unchanged.
     *        This is the RL extension point.
     */
    static SimConfig Load(const std::string& run_config_path,
                          const std::string& positions_override_path = "");
};

}  // namespace mesh_sim
