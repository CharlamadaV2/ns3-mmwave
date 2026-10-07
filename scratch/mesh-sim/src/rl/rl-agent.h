/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file rl-agent.h
 * @brief Unused no-op placeholder; nothing includes it. The live agent path is rl-bridge.h.
 */
#pragma once

#include "src/domain/node-spec.h"
#include "src/domain/link-result.h"
#include "src/routing/mesh-router.h"

#include <vector>

namespace mesh_sim
{

class RlAgent
{
  public:
    /**
     * @fn RlAgent::GetActions
     * @brief Return the current positions unchanged (no-op placeholder).
     *
     * @param currentPositions  Node positions, metres.
     * @return A copy of @p currentPositions; the link and flow arguments are ignored.
     */
    std::vector<Position> GetActions(
        const std::vector<Position>& currentPositions,
        const std::vector<LinkResult>& /* links */,
        const std::vector<FlowResult>& /* flows */)
    {
        return currentPositions;
    }
};

}  // namespace mesh_sim
