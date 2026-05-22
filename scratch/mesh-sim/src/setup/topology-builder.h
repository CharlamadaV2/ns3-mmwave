/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file topology-builder.h
 * @brief Creates ns-3 mobility models, buildings, and propagation models
 *        for the mesh topology.
 *
 * @ref TopologyBuilder is the **only layer in mesh-sim that creates ns-3
 * objects**. All other subsystems receive the resulting pointers via the
 * accessor methods and operate entirely through the domain types defined
 * in @c src/domain/. No EPC, RRC, MAC, or protocol stack is created.
 *
 *
 * **Mobility models**
 *
 * | @ref NodeSpec::mobility  | ns-3 class                          | Notes                                  |
 * |--------------------------|-------------------------------------|----------------------------------------|
 * | @c "fixed"               | @c ConstantPositionMobilityModel    | Position set from @c NodeSpec::position. |
 * | @c "constant_velocity"   | @c ConstantVelocityMobilityModel    | Position and velocity set after install. |
 * | @c "random_walk"         | @c RandomWalk2dMobilityModel        | Direction changes every 1 simulated second; bounds and speed from @ref RandomWalkParams. |
 * | @c "waypoint"            | @c WaypointMobilityModel            | All @ref Waypoint entries added via @c ns3::Seconds(wp.t). |
 *
 * **Channel condition model selection**
 * When @c cfg.buildings is non-empty a @c BuildingsChannelConditionModel is
 * installed, giving deterministic LOS/NLOS based on building geometry.
 * When no buildings are configured the propagation model's default statistical
 * condition model is used instead.
 *
 * **Propagation model selection**
 *
 * | @c channel_model | @c scenario | ns-3 class                                        |
 * |------------------|-------------|---------------------------------------------------|
 * | @c "3gpp"        | @c "UMi"    | @c ThreeGppUmiStreetCanyonPropagationLossModel     |
 * | @c "3gpp"        | @c "UMa"    | @c ThreeGppUmaPropagationLossModel                 |
 * | @c "3gpp"        | @c "RMa"    | @c ThreeGppRmaPropagationLossModel                 |
 * | @c "3gpp"        | @c "InH" / @c "InH-Mixed" / @c "InH-Open" | @c ThreeGppIndoorOfficePropagationLossModel |
 * | @c "nyu"         | @c "UMi"    | @c NYUUmiPropagationLossModel                      |
 * | @c "nyu"         | @c "UMa"    | @c NYUUmaPropagationLossModel                      |
 * | @c "nyu"         | @c "RMa"    | @c NYURmaPropagationLossModel                      |
 * | @c "nyu"         | @c "InH"    | @c NYUInHPropagationLossModel                      |
 * | @c "nyu"         | @c "InF"    | @c NYUInFPropagationLossModel                      |
 */
#pragma once

#include "src/domain/sim-config.h"

#include "ns3/channel-condition-model.h"
#include "ns3/mobility-model.h"
#include "ns3/node-container.h"
#include "ns3/propagation-loss-model.h"

#include <vector>

namespace mesh_sim
{

/**
 * @brief Translates a @ref SimConfig into ns-3 mobility, building, and
 *        propagation model objects.
 *
 * Holds a const reference to the @ref SimConfig — the config must remain
 * valid for the lifetime of the builder.  All ns-3 smart pointers are
 * retained internally after @ref Build and returned by the accessor methods.
 */
class TopologyBuilder
{
  public:
    /**
     * @brief Construct a builder for the given simulation configuration.
     *
     * Stores a const reference to @c cfg; does not create any ns-3 objects.
     * Call @ref Build before any accessor.
     *
     * @param cfg  Fully loaded and validated simulation configuration.
     *             Must outlive this object.
     */
    explicit TopologyBuilder(const SimConfig& cfg);

    /**
     * @brief Create all ns-3 topology objects in the required order.
     *
     * Executes three steps in sequence:
     * -# @ref CreateNodesAndMobility — one ns-3 node per @ref NodeSpec,
     *    with the appropriate mobility model installed.
     * -# @ref CreateBuildings — ns-3 @c Building objects plus
     *    @c BuildingsHelper::Install; skipped when @c cfg.buildings is empty.
     * -# @ref ConfigurePropagationModel — selects and configures the 3GPP
     *    or NYU propagation loss model and the channel condition model.
     *
     * Must be called exactly once, before any accessor method.
     *
     * @throws std::runtime_error for an unrecognised @c NodeSpec::mobility
     *         string, an unknown propagation scenario, or an unknown channel
     *         model string.
     */
    void Build();

    /**
     * @brief Return the mobility model for each node in @c cfg.nodes order.
     *
     * The vector index matches the node index used by @ref LinkEvaluator
     * and @ref LinkTable: @c GetMobilityModels()[k] corresponds to
     * @c cfg.nodes[k].
     *
     * @return Vector of @c N mobility model smart pointers, where @c N is
     *         @c cfg.nodes.size(). Empty until @ref Build is called.
     */
    std::vector<ns3::Ptr<ns3::MobilityModel>> GetMobilityModels() const;

    /**
     * @brief Return the configured propagation loss model.
     *
     * The pointer is valid after @ref Build. Pass directly to
     * @ref LinkEvaluator::Configure.
     *
     * @return ns-3 propagation loss model (3GPP or NYU subclass).
     */
    ns3::Ptr<ns3::PropagationLossModel> GetPropagationModel() const;

    /**
     * @brief Return the channel condition model used for LOS/NLOS determination.
     *
     * When buildings are present this is a @c BuildingsChannelConditionModel
     * (deterministic). Otherwise it is the propagation model's default
     * statistical condition model. Pass directly to
     * @ref LinkEvaluator::Configure.
     *
     * @return ns-3 channel condition model.
     */
    ns3::Ptr<ns3::ChannelConditionModel> GetConditionModel() const;

  private:
    const SimConfig& m_cfg;  ///< Simulation configuration (const reference; not owned).

    ns3::NodeContainer                        m_nodes;            ///< All created ns-3 nodes.
    std::vector<ns3::Ptr<ns3::MobilityModel>> m_mobilityModels;   ///< One entry per cfg.nodes entry, in index order.
    ns3::Ptr<ns3::PropagationLossModel>       m_propagationModel; ///< Configured after Build().
    ns3::Ptr<ns3::ChannelConditionModel>      m_conditionModel;   ///< Configured after Build().

    /**
     * @brief Create one ns-3 node per @ref NodeSpec and install its mobility model.
     *
     * Iterates @c cfg.nodes in order, creates a single-node @c NodeContainer
     * for each, dispatches to the appropriate @c InstallMobility* helper, and
     * appends the resulting @c MobilityModel pointer to @c m_mobilityModels.
     *
     * @throws std::runtime_error for any unrecognised @c NodeSpec::mobility value.
     */
    void CreateNodesAndMobility();

    /**
     * @brief Create ns-3 @c Building objects from @c cfg.buildings.
     *
     * A no-op when @c cfg.buildings is empty. For non-empty configs,
     * creates one @c ns3::Building per @ref BuildingSpec, sets its bounding
     * box, type, exterior wall material, floor count, and room layout, then
     * calls @c BuildingsHelper::Install(m_nodes) to associate all nodes with
     * the building model.  @c BuildingsHelper::Install must be called after
     * all buildings are created and all nodes have mobility models installed.
     */
    void CreateBuildings();

    /**
     * @brief Select and configure the propagation and channel condition models.
     *
     * Reads @c cfg.channel.channel_model (@c "3gpp" or @c "nyu") and
     * @c cfg.channel.scenario to instantiate the correct ns-3 subclass.
     * Sets the carrier frequency on the model. For NYU, also applies every
     * field from @ref NyuChannelConfig via @c SetAttribute.
     *
     * **Channel condition model:**
     * - Buildings present: @c BuildingsChannelConditionModel is installed on
     *   the propagation model and stored as @c m_conditionModel.
     * - No buildings: the propagation model's default statistical condition
     *   model is retrieved and stored as @c m_conditionModel.
     *
     * @throws std::runtime_error for an unrecognised scenario or channel model
     *         string.
     */
    void ConfigurePropagationModel();

    /**
     * @brief Install @c ns3::ConstantPositionMobilityModel on a node.
     *
     * Uses a @c ListPositionAllocator to set the initial position from
     * @ref NodeSpec::position. The node does not move for the entire simulation.
     *
     * @param node  ns-3 node to install the model on.
     * @param spec  Node specification providing the fixed position.
     */
    void InstallMobilityFixed(const ns3::Ptr<ns3::Node>& node, const NodeSpec& spec);

    /**
     * @brief Install @c ns3::ConstantVelocityMobilityModel on a node.
     *
     * Sets the initial position from @ref NodeSpec::position and the constant
     * velocity vector from @ref NodeSpec::velocity after the model is installed,
     * since @c MobilityHelper does not accept a velocity at install time.
     *
     * @param node  ns-3 node to install the model on.
     * @param spec  Node specification providing the initial position and velocity.
     */
    void InstallMobilityConstantVelocity(const ns3::Ptr<ns3::Node>& node, const NodeSpec& spec);

    /**
     * @brief Install @c ns3::RandomWalk2dMobilityModel on a node.
     *
     * Configures the walk with:
     * - **Bounds**: rectangle from @ref RandomWalkParams.
     * - **Speed**: @c ConstantRandomVariable at @c speed_mps.
     * - **Mode**: @c Time — direction is resampled every 1 simulated second.
     *
     * @param node  ns-3 node to install the model on.
     * @param spec  Node specification providing the initial position and walk params.
     */
    void InstallMobilityRandomWalk(const ns3::Ptr<ns3::Node>& node, const NodeSpec& spec);

    /**
     * @brief Install @c ns3::WaypointMobilityModel on a node.
     *
     * Adds every @ref Waypoint from @c spec.waypoints via
     * @c WaypointMobilityModel::AddWaypoint, using @c ns3::Seconds(wp.t)
     * as the arrival time. The validator ensures at least two waypoints
     * with strictly increasing times exist before this is called.
     *
     * @param node  ns-3 node to install the model on.
     * @param spec  Node specification providing the ordered waypoint list.
     */
    void InstallMobilityWaypoint(const ns3::Ptr<ns3::Node>& node, const NodeSpec& spec);
};

}  // namespace mesh_sim