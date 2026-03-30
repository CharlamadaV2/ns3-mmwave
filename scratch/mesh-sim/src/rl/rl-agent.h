/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * RL agent placeholder. Returns current positions unchanged (no-op).
 * Replace with actual RL agent (Python IPC via ZMQ, stdin/stdout, or shared memory).
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
    std::vector<Position> GetActions(
        const std::vector<Position>& currentPositions,
        const std::vector<LinkResult>& /* links */,
        const std::vector<FlowResult>& /* flows */)
    {
        return currentPositions;
    }
};

}  // namespace mesh_sim
