/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

/**
 * @file node-spec.h
 * @brief Topology and geometry POD types: nodes, buildings, and spatial properties.
 *
 * **Coordinate convention**
 * All positions are stored as (x, y, z) triples in metres.
 * - 2-D scenarios: set z to a constant height for all nodes.
 * - 1-D scenarios: additionally set y = 0 for all nodes.
 * The coordinate system is arbitrary; only relative geometry matters.
 */
#pragma once

#include <string>
#include <vector>

namespace mesh_sim
{

/**
 * @brief Cartesian position in metres.
 */
struct Position
{
    double x = 0.0;  ///< East offset from scenario origin (m).
    double y = 0.0;  ///< North offset from scenario origin (m).
    double z = 0.0;  ///< Altitude above scenario ground plane (m).
};

/**
 * @brief Constant velocity vector in metres per second.
 *
 * Used by the @c constant_velocity mobility model.
 * A stationary node should use the @c fixed model instead of a zero velocity here.
 */
struct Velocity
{
    double vx = 0.0;  ///< East component (m/s).
    double vy = 0.0;  ///< North component (m/s).
    double vz = 0.0;  ///< Vertical component (m/s).
};

/**
 * @brief Bounding box and speed parameters for the random-walk mobility model.
 *
 * The node moves at a constant @c speed_mps in a uniformly random direction,
 * reflecting at the box boundaries.
 */
struct RandomWalkParams
{
    double x_min    = -100.0;  ///< Western boundary of the walk area (m).
    double x_max    =  100.0;  ///< Eastern boundary of the walk area (m).
    double y_min    = -100.0;  ///< Southern boundary of the walk area (m).
    double y_max    =  100.0;  ///< Northern boundary of the walk area (m).
    double speed_mps =   1.5;  ///< Constant walk speed (m/s).
};

/**
 * @brief A single timed waypoint in a node's mobility trajectory.
 *
 * Waypoints are stored in time order in @ref NodeSpec::waypoints.
 * The simulator linearly interpolates position between consecutive waypoints.
 */
struct Waypoint
{
    double t = 0.0;  ///< Time since scenario start at which this position is reached (s).
    double x = 0.0;  ///< East position at time @c t (m).
    double y = 0.0;  ///< North position at time @c t (m).
    double z = 0.0;  ///< Altitude at time @c t (m).
};

/**
 * @brief Full specification for one simulation node.
 *
 * Loaded from the @c nodes array in @c nodes.json.
 * The active mobility model is selected by @c mobility; only the
 * corresponding parameter struct (@c velocity, @c random_walk, or
 * @c waypoints) is used at runtime.
 */
struct NodeSpec
{
    std::string id;         ///< Unique node identifier (e.g. @c "rab1").
    std::string role;       ///< Node role; always @c "peer" in mesh-sim.
    std::string mobility;   ///< Mobility model: @c "fixed", @c "constant_velocity",
                            ///<   @c "random_walk", or @c "waypoint".
    std::string node_type;  ///< Physical category: @c "drone", @c "vehicle", or
                            ///<   @c "pedestrian". Determines the RL speed cap.
    Position    position;   ///< Initial position (and fixed position for @c "fixed" nodes).
    Velocity    velocity;   ///< Constant velocity (used only when @c mobility == "constant_velocity").
    RandomWalkParams random_walk;           ///< Walk parameters (used only when @c mobility == "random_walk").
    std::vector<Waypoint> waypoints;        ///< Ordered waypoint list (used only when @c mobility == "waypoint").
};

/**
 * @brief Return the maximum speed (m/s) allowed for a given node type.
 *
 * Used by the RL controller to cap continuous velocity actions before they
 * are applied to the node.
 *
 * | node_type      | max speed |
 * |----------------|-----------|
 * | @c "vehicle"   | 15.0 m/s  |
 * | @c "pedestrian"| 1.5 m/s   |
 * | @c "drone" (default) | 20.0 m/s |
 *
 * @param node_type  Node type string from @ref NodeSpec::node_type.
 * @return Maximum speed in metres per second.
 */
inline double MaxSpeedForType(const std::string& node_type)
{
    if (node_type == "vehicle")    return 15.0;
    if (node_type == "pedestrian") return  1.5;
    return 20.0;  // drone (default)
}

/**
 * @brief Axis-aligned bounding-box specification for one building.
 *
 * Loaded from @c buildings.json. Buildings are used by the ns-3 building
 * propagation model to determine LOS/NLOS conditions deterministically.
 */
struct BuildingSpec
{
    std::string id;                                  ///< Unique building identifier.
    double x_min = 0.0;                              ///< Western face x-coordinate (m).
    double x_max = 1.0;                              ///< Eastern face x-coordinate (m).
    double y_min = 0.0;                              ///< Southern face y-coordinate (m).
    double y_max = 1.0;                              ///< Northern face y-coordinate (m).
    double z_min = 0.0;                              ///< Ground level z-coordinate (m).
    double z_max = 1.0;                              ///< Roof z-coordinate (m).
    std::string type      = "Residential";           ///< ns3::Building type: @c "Residential",
                                                     ///<   @c "Office", or @c "Commercial".
    std::string ext_walls = "ConcreteWithWindows";   ///< ns3::Building::ExtWallsType string.
    int n_floors  = 1;  ///< Number of floors; affects internal attenuation.
    int n_rooms_x = 1;  ///< Room columns along x-axis.
    int n_rooms_y = 1;  ///< Room rows along y-axis.
};

}  // namespace mesh_sim