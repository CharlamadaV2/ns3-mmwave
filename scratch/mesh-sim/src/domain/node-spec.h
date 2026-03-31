/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Topology / geometry POD types: nodes, buildings, and their spatial properties.
 * No ns-3 headers included — keeps compilation fast and allows use in
 * post-run IO code without ns-3 linkage.
 */
#pragma once

#include <string>

namespace mesh_sim
{

// Coordinates are always stored as (x, y, z) triples in metres.
// For 2D scenarios: set z to a constant height.
// For 1D scenarios: additionally set y = 0 for all nodes.
// The coordinate system is arbitrary; relative geometry is what matters.
struct Position
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Velocity
{
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
};

struct RandomWalkParams
{
    double x_min = -100.0;
    double x_max = 100.0;
    double y_min = -100.0;
    double y_max = 100.0;
    double speed_mps = 1.5;
};

struct NodeSpec
{
    std::string id;
    std::string role;      // "peer" (all nodes are peers in mesh-sim)
    std::string mobility;  // "fixed", "constant_velocity", "random_walk"
    Position    position;
    Velocity    velocity;
    RandomWalkParams random_walk;
};

struct BuildingSpec
{
    std::string id;
    double x_min = 0.0;
    double x_max = 1.0;
    double y_min = 0.0;
    double y_max = 1.0;
    double z_min = 0.0;
    double z_max = 1.0;
    std::string type      = "Residential";         // "Residential", "Office", "Commercial"
    std::string ext_walls = "ConcreteWithWindows";  // see ns3::Building::ExtWallsType
    int n_floors  = 1;
    int n_rooms_x = 1;
    int n_rooms_y = 1;
};

}  // namespace mesh_sim
