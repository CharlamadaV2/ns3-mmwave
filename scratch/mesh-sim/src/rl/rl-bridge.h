/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file rl-bridge.h
 * @brief RL bridge: stdin/stdout JSON IPC between the C++ sim and a Python RL agent.
 *
 * Legacy mode (one controlled node, Discrete(7)) writes one message per tick.
 * Centralized mode (`rl.controlled_nodes`) writes an `init` line once, then one
 * `step` message every `decision_interval_ticks` ticks carrying the joint
 * observation, the authoritative per-slot action mask, and the windowed reward;
 * the action is a MultiDiscrete([5]*M) list. C++ owns masks, speed caps, and the
 * per-tick bounds clamp -- Python never re-derives them. See @ref src_rl.
 */
#pragma once

#include "src/domain/sim-config.h"
#include "src/eval/link-table.h"
#include "src/routing/mesh-router.h"

#include "ns3/constant-velocity-mobility-model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mesh_sim
{

/// One action-space slot: a resolved controlled node, or padding when !active.
struct ControlSlot
{
    uint32_t    node_index = 0;    ///< Index into SimConfig::nodes (undefined when padding).
    std::string node_id;           ///< Node id (empty when padding).
    double      speed_mps = 0.0;   ///< min(step_size_m / tick_s, MaxSpeedForType(node_type)).
    bool        active = false;    ///< False for padded slots (hold only).
};

/// Sums over the ticks of one decision window (centralized facts export).
struct WindowFacts
{
    uint32_t ticks = 0;
    double   demand_mbps_sum = 0.0;
    double   delivered_mbps_sum = 0.0;
    uint64_t flow_ticks_with_demand = 0;
    uint64_t unroutable_flow_ticks = 0;
    uint64_t connected_pairs_sum = 0;
    uint64_t los_pairs_sum = 0;
    double   legacy_reward_sum = 0.0;
};

/// Owns RL IPC and physics for the controlled node(s); legacy and centralized modes.
class RlBridge
{
  public:
    /**
     * @fn RlBridge::RlBridge
     * @brief Resolve the controlled slots and copy the settings the bridge needs.
     *
     * @param cfg  Validated config with RL control already resolved
     *             (@c cfg.rl.controlled_indices, @c num_slots, @c decision_interval_ticks).
     *
     * Legacy mode uses one slot: the first controlled index, or the last node when none
     * is set. Centralized mode builds @c num_slots slots; slots beyond the resolved
     * indices are inactive padding. Nothing is printed. @c cfg must contain at least
     * one node.
     */
    explicit RlBridge(const SimConfig& cfg);

    /**
     * @fn RlBridge::WriteInit
     * @brief Write the centralized @c init contract line to stdout.
     *
     * Does nothing in legacy mode. Call once before the tick loop.
     */
    void WriteInit() const;

    /**
     * @fn RlBridge::IsDecisionTick
     * @brief Tell whether tick @p ti is a centralized decision boundary.
     *
     * @param ti  Zero-based tick index.
     * @return @c true only in centralized mode when @p ti is below the run's tick
     *         count and a multiple of @c decision_interval_ticks; always @c false in
     *         legacy mode.
     */
    bool IsDecisionTick(uint32_t ti) const;

    /**
     * @fn RlBridge::BeforeAdvance
     * @brief Clamp each controlled slot's velocity so this tick cannot leave the bounds.
     *
     * @param mobs  Mobility model per node, indexed by node index.
     *
     * Centralized mode only (no-op in legacy mode). Limits x and y against
     * @c rl.x_min..x_max and @c y_min..y_max; z is left alone. Slots whose model is
     * not a @c ConstantVelocityMobilityModel are skipped. Call before advancing the clock.
     */
    void BeforeAdvance(const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs);

    /**
     * @fn RlBridge::AccumulateTick
     * @brief Add this tick's reward and fact sums to the open decision window.
     *
     * @param linkTable  Link table for this tick.
     * @param flows      Routed flow results for this tick.
     *
     * Feeds the window mean reward and the @c facts block of the next step message.
     */
    void AccumulateTick(const LinkTable& linkTable, const std::vector<FlowResult>& flows);

    /**
     * @fn RlBridge::Step
     * @brief Write the observation and reward to stdout, then read the next action from stdin.
     *
     * @param tick         Zero-based tick index.
     * @param time_s       Simulation time in seconds.
     * @param mobs         Mobility model per node, indexed by node index.
     * @param linkTable    Link table for this tick.
     * @param flowResults  Routed flow results for this tick (legacy reward).
     * @param done         @c true on the final message; no action is read afterwards.
     *
     * Blocks on stdin until Python replies. Centralized mode writes a @c step message
     * (window mean reward, mask) and clears the window. Legacy mode writes one
     * observation per call. A closed stdin holds position (one warning on stderr).
     */
    void Step(uint32_t tick,
              double time_s,
              const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs,
              const LinkTable& linkTable,
              const std::vector<FlowResult>& flowResults,
              bool done);

    /**
     * @fn RlBridge::ApplyAction
     * @brief Turn the last received action into velocity on the controlled node(s).
     *
     * @param mobs  Mobility model per node, indexed by node index.
     *
     * Centralized mode sets a constant velocity of @c speed_mps along -x, +x, -y, +y,
     * or zero for hold (action 4). Legacy discrete mode uses indices 0..6
     * (-X, +X, -Y, +Y, -Z, +Z, stay); continuous mode steers toward the last target.
     */
    void ApplyAction(const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs);

  private:
    RlConfig    m_rl;
    double      m_tickS;
    uint32_t    m_controlledIdx;
    uint32_t    m_numNodes;
    double      m_maxSpeed;
    std::string m_nodeType;

    // Centralized state
    bool                     m_centralized = false;
    std::vector<ControlSlot> m_slots;
    uint32_t                 m_numSlots = 1;
    uint32_t                 m_k = 1;
    uint32_t                 m_numTicks = 0;
    std::vector<int>         m_lastJoint;        ///< Last joint action (4 == hold).
    std::vector<int>         m_lastMask;         ///< Mask sent in the last step message.
    std::vector<uint32_t>    m_revalidatedSlots; ///< Slots held by revalidation, for the next message.
    WindowFacts              m_window;           ///< Sums for the open decision window.
    uint32_t                 m_decision = 0;
    bool                     m_streamClosed = false;

    // Copied from SimConfig at construction: WriteInit runs after cfg is gone.
    std::vector<std::string> m_nodeIds;              ///< nodes.json ids in file order.
    std::string              m_band;
    double                   m_warmupS = 0.0;
    bool                     m_jammerPathEnabled = false;

    // Last action received from Python
    int    m_lastDiscreteAction = 0;
    double m_lastTargetX = 0.0;
    double m_lastTargetY = 0.0;
    double m_lastTargetZ = 0.0;

    double ComputeRewardTick(const LinkTable& linkTable,
                             const std::vector<FlowResult>& flows) const;
    void WriteObs(uint32_t tick, double time_s,
                  const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs,
                  const LinkTable& linkTable,
                  double reward, bool done) const;
    void ReadAction();

    void WriteStep(uint32_t tick, double time_s,
                   const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs,
                   const LinkTable& linkTable,
                   const std::vector<int>& mask,
                   double reward, WindowFacts window, bool done) const;
    std::vector<int> ComputeMask(const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs) const;
    void ReadJointAction();
    ns3::Vector ClampVelocityForTick(const ns3::Vector& pos, const ns3::Vector& vel) const;
};

}  // namespace mesh_sim
