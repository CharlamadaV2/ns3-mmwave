/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file traffic-matrix.h
 * @brief Flow-level traffic demand generation for the mesh.
 *
 * @ref TrafficMatrix maintains a list of @ref Flow objects and updates
 * them each tick according to the configured traffic model. There are no
 * packets — a flow simply represents "node A wants X Mbps to node B."
 *
 *
 * **Traffic models**
 *
 * | @c model      | Flow creation                           | Per-tick update                          |
 * |---------------|-----------------------------------------|------------------------------------------|
 * | @c "constant" | All flows created at @ref Initialize.   | None; flows run for the full simulation. |
 * | @c "poisson"  | New flows arrive each tick via Poisson. | New flows added; expired flows removed.  |
 * | @c "on_off"   | All flows created at @ref Initialize.   | ON/OFF state machine per flow.           |
 *
 * **Flow removal**
 * Expired flows (@c end_time_s > 0 and @c active == false) are erased from
 * @c m_flows each tick. Permanent flows (@c end_time_s == 0) are never removed,
 * even when they transition to inactive in the on-off model.
 *
 * **Gateway resolution**
 * For the @c "gateway" topology, @ref TrafficConfig::gateway_node_id is
 * resolved as a numeric index first (@c stoul); if that throws, it falls
 * back to a linear search by @ref NodeSpec::id string. Index 0 is used
 * when @c gateway_node_id is empty.
 */
#pragma once

#include "src/domain/sim-config.h"

#include "ns3/random-variable-stream.h"

#include <cstdint>
#include <vector>

namespace mesh_sim
{

/**
 * @brief A single directed traffic flow between two nodes.
 *
 * Flows are created by @ref TrafficMatrix and consumed by @ref MeshRouter.
 * The on-off model state (@c in_on_phase, @c phase_end_s) is only meaningful
 * when the traffic model is @c "on_off"; for other models these fields retain
 * their default values and are ignored by the router.
 */
struct Flow
{
    uint32_t src = 0;  ///< Source node index into @ref SimConfig::nodes.
    uint32_t dst = 0;  ///< Destination node index into @ref SimConfig::nodes.

    double demand_mbps  = 0.0;  ///< Per-flow demand in Mbps (from @ref TrafficConfig::demand_mbps).
    double start_time_s = 0.0;  ///< Simulation time at which the flow was created (s).
    double end_time_s   = 0.0;  ///< Expiry time in seconds; @c 0 = permanent (no expiry).

    bool active = true;  ///< @c false once @c currentTime >= @c end_time_s.
                          ///<   Inactive flows are removed from @c m_flows each
                          ///<   tick unless they are permanent (@c end_time_s == 0).

    // ---- on_off model state ------------------------------------------------
    bool   in_on_phase = true;  ///< @c true when the flow is in its ON (transmitting) phase.
                                 ///<   Always @c true for non-on-off models.
    double phase_end_s = 0.0;   ///< Simulation time at which the current ON or OFF
                                 ///<   phase ends. Irrelevant for non-on-off models.
};


/**
 * @brief Generates and maintains the set of active flows each simulation tick.
 *
 * Copies @ref TrafficConfig and the node spec list at construction; does not
 * hold a reference to @ref SimConfig after the constructor returns.
 */
class TrafficMatrix
{
  public:
    /**
     * @brief Construct a traffic matrix from the simulation configuration.
     *
     * Copies @c cfg.mesh.traffic, @c cfg.nodes, and @c cfg.tick_s.
     * Creates two ns-3 RNG objects (@c UniformRandomVariable for Poisson
     * generation / random pairs, @c ExponentialRandomVariable for on-off
     * phase durations and Poisson holding times) and seeds them via the
     * ns-3 global @c RngSeedManager.
     *
     * @param cfg  Fully loaded and validated simulation configuration.
     */
    explicit TrafficMatrix(const SimConfig& cfg);

    /**
     * @brief Create the initial set of flows for the configured topology.
     *
     * Clears any existing flows, sets the node count, then calls the
     * appropriate initialiser:
     * - @c "all_pairs"    → @ref InitAllPairs
     * - @c "random_pairs" → @ref InitRandomPairs
     * - @c "gateway"      → @ref InitGateway
     *
     * Must be called once before the step loop, at @c currentTime = 0 for
     * normal use (or at warmup end when pre-warming traffic state).
     *
     * @param numNodes    Total number of active simulation nodes.
     * @param currentTime Simulation time at which flows start (seconds).
     * @throws std::runtime_error for an unrecognised @c flow_topology string.
     */
    void Initialize(uint32_t numNodes, double currentTime);

    /**
     * @brief Advance the traffic state by one simulation tick.
     *
     * Executed in three phases each tick:
     * -# **Expiry**: any flow whose @c end_time_s > 0 and
     *    @c currentTime >= @c end_time_s is marked @c active = @c false.
     * -# **Model update**: @ref TickOnOff or @ref TickPoisson is called
     *    for the respective model. The @c "constant" model requires no
     *    per-tick work.
     * -# **Removal**: flows matching @c !active && end_time_s > 0 are
     *    erased from @c m_flows. Permanent flows (@c end_time_s == 0) are
     *    never removed.
     *
     * @param currentTime  Current simulation time in seconds.
     */
    void Tick(double currentTime);

    /**
     * @brief Return a const reference to the internal flow list.
     *
     * Returns *all* flows — including inactive flows (in the OFF phase or
     * expiring this tick) that have not yet been removed. @ref MeshRouter
     * filters by @c f.active and @c f.in_on_phase internally, so callers
     * do not need to pre-filter.
     *
     * The reference is valid until the next call to @ref Tick or
     * @ref Initialize.
     *
     * @return Const reference to @c m_flows.
     */
    const std::vector<Flow>& GetActiveFlows() const;

    /**
     * @brief Return the total active demand from @c src to @c dst (Mbps).
     *
     * Sums @c demand_mbps across all flows where @c f.src == src,
     * @c f.dst == dst, and @c f.active == true. For the @c "on_off"
     * model, flows in the OFF phase (@c in_on_phase == false) are excluded.
     *
     * @param src  Source node index.
     * @param dst  Destination node index.
     * @return Total demand in Mbps; 0.0 if no active flows exist for this pair.
     */
    double GetDemand(uint32_t src, uint32_t dst) const;

  private:
    /**
     * @brief Create one flow for every unordered node pair (i < j).
     *
     * Produces N*(N-1)/2 flows for @c N nodes.  All flows start active at
     * @c currentTime with permanent lifetime (@c end_time_s == 0) unless
     * @c holding_time_s > 0.
     *
     * @param numNodes    Number of nodes.
     * @param currentTime Flow creation time (seconds).
     */
    void InitAllPairs(uint32_t numNodes, double currentTime);

    /**
     * @brief Create @c random_pair_count flows with uniformly random src/dst.
     *
     * Each pair is drawn independently. Self-flows (@c src == @c dst) are
     * rejected via a rejection loop (uniform resample until @c dst != @c src).
     *
     * @param numNodes    Number of nodes.
     * @param currentTime Flow creation time (seconds).
     */
    void InitRandomPairs(uint32_t numNodes, double currentTime);

    /**
     * @brief Create one flow from every non-gateway node to the gateway.
     *
     * Resolves the gateway index from @ref TrafficConfig::gateway_node_id:
     * -# Attempt @c std::stoul (numeric index).
     * -# On @c std::invalid_argument, search @c m_nodeSpecs by @c id string.
     * -# Default to index 0 when @c gateway_node_id is empty.
     *
     * @param numNodes    Number of nodes.
     * @param currentTime Flow creation time (seconds).
     * @throws std::runtime_error if @c gateway_node_id is a non-numeric string
     *         that does not match any node ID.
     */
    void InitGateway(uint32_t numNodes, double currentTime);

    /**
     * @brief Construct a single @ref Flow for the given (src, dst) pair.
     *
     * Sets @c demand_mbps, @c start_time_s, and @c active = true.
     * When @c holding_time_s > 0 sets @c end_time_s = currentTime + holding_time_s.
     * For the @c "on_off" model, sets @c in_on_phase = true and draws the
     * first ON-phase duration from @c Exponential(on_time_s).
     *
     * @note For @c "poisson" flows, @ref TickPoisson overrides @c end_time_s
     *       with a fresh exponential draw after calling this factory.
     *
     * @param src         Source node index.
     * @param dst         Destination node index.
     * @param currentTime Flow creation time (seconds).
     * @return Fully initialised @ref Flow.
     */
    Flow MakeFlow(uint32_t src, uint32_t dst, double currentTime) const;

    /**
     * @brief Advance the ON/OFF state machine for all active flows.
     *
     * For each active flow, checks whether @c phase_end_s <= currentTime.
     * A @c while loop handles multiple phase transitions within a single
     * tick (possible when @c tick_s is large relative to mean phase duration).
     * Each transition: toggles @c in_on_phase and adds an exponential sample
     * of the new phase's mean duration to @c phase_end_s.
     *
     * @param currentTime  Current simulation time (seconds).
     */
    void TickOnOff(double currentTime);

    /**
     * @brief Generate Poisson-distributed new flow arrivals for this tick.
     *
     * Computes the expected arrivals: @c lambda = arrival_rate_hz × tick_s.
     * The actual count is drawn using Knuth's product-of-uniforms method:
     * multiply uniform(0,1) samples until their product falls below
     * @c exp(-lambda); the count is the number of multiplications minus one.
     *
     * Each new flow has a uniformly random src != dst (rejection sampling).
     * When @c holding_time_s > 0, @c end_time_s is overridden with an
     * independent @c Exponential(holding_time_s) draw so Poisson flows have
     * exponentially distributed lifetimes rather than fixed ones.
     *
     * @param currentTime  Current simulation time (seconds).
     */
    void TickPoisson(double currentTime);

    TrafficConfig         m_trafficCfg;  ///< Copy of traffic configuration.
    std::vector<NodeSpec> m_nodeSpecs;   ///< Copy of node specs (for gateway ID lookup).
    uint32_t              m_numNodes = 0; ///< Node count set by @ref Initialize.
    double                m_tickS    = 0.1; ///< Tick duration in seconds (from @c cfg.tick_s).

    std::vector<Flow> m_flows;  ///< All flows (active, inactive, and off-phase).

    /// Uniform RNG [0, 1] used for Poisson arrival generation and random-pair selection.
    ns3::Ptr<ns3::UniformRandomVariable>    m_uniformRng;
    /// Exponential RNG used for on-off phase durations and Poisson flow holding times.
    ns3::Ptr<ns3::ExponentialRandomVariable> m_expRng;
};

}  // namespace mesh_sim