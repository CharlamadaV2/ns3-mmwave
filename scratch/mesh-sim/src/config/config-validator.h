/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

/** @file config-validator.h
 * @brief
 * Config validation: checks a fully-loaded SimConfig for invalid
 * or inconsistent values before the simulation runs.
 * No ns-3 headers -- can be tested independently.
 */
#pragma once

#include "src/domain/sim-config.h"

#include <string>
#include <vector>

namespace mesh_sim
{
/** @brief
*
*/
struct ValidationResult
{
    std::vector<std::string> errors;
    bool ok() const { return errors.empty(); }
};

/** @brief 

 * Validate a SimConfig after loading.  Returns all errors found
 * (not just the first) so the user can fix them in one pass.
 * 
 * @param
 * @return
 */
ValidationResult ValidateConfig(const SimConfig& cfg);

}  // namespace mesh_sim
