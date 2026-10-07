/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file layout-override.h
 * @brief Applies a candidate start layout to a resolved @ref mesh_sim::SimConfig (no ns-3).
 */
#pragma once

#include "src/domain/node-spec.h"
#include "src/domain/sim-config.h"

#include <string>
#include <vector>

namespace mesh_sim
{

/**
 * @fn ApplyLayout
 * @brief Move every node's start x/y to @p layout, keeping z and scenario mobility.
 *
 * @param cfg     Validated config with RL control already applied; mutated only on success.
 * @param layout  One position per @c cfg.nodes entry, in the same order.
 * @return One message per problem; empty means the layout was applied.
 *
 * The whole layout is checked before anything changes: the size must match,
 * every coordinate must be finite, each requested z must equal the node's
 * start z (first waypoint for waypoint mobility) within 1e-6, and a
 * @c random_walk node that is not centrally RL-controlled must start inside
 * its inclusive walk bounds. Waypoint nodes shift every waypoint by the start
 * delta (z and t kept) and set @c position x/y to the target; other nodes set
 * @c position x/y. Jammers are untouched.
 */
std::vector<std::string> ApplyLayout(SimConfig& cfg, const std::vector<Position>& layout);

}  // namespace mesh_sim
