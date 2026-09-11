/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file jammer-model.h
 * @brief Per-link jammer interference contribution.
 *
 * @ref JammerModel computes the total received jammer power (in watts)
 * at a receiver node, summing contributions from every enabled
 * @ref JammerSpec. This value is added to the noise-plus-interference
 * denominator inside @ref LinkEvaluator::EvaluateAll when
 * @c m_computeInterference is true (@c band == @c "sub-6").
 *
 * **Design**
 *
 * JammerModel is intentionally separate from @ref LinkEvaluator so that:
 * - It can be disabled with zero code changes in the evaluator (just don't
 *   call @ref Configure or pass an empty jammer list).
 * - It can be extended (e.g. directional antenna pattern, frequency sweep)
 *   without touching the core SINR calculation.
 * - Tests can verify jammer power independently of the full link pipeline.
 *
 * **Usage in EvaluateAll**
 * @code
 *   double jamWatt = m_jammerModel.InterfPowerAtReceiver(rxMob);
 *   r.sinr_db = 10.0 * std::log10(signalWatt / (noiseWatt + interfWatt + jamWatt));
 * @endcode
 */
#pragma once

#include "src/jammer/jammer-spec.h"

#include <cstdint>

#include "ns3/mobility-model.h"
#include "ns3/propagation-loss-model.h"

#include <vector>

namespace mesh_sim
{

/**
 * @brief Computes the aggregate received jammer power at a node.
 *
 * Stateless between calls — all per-call state is local to
 * @ref InterfPowerAtReceiver. The propagation loss model pointer and jammer
 * specs are set once by @ref Configure.
 */
class JammerModel
{
  public:
    /**
     * @brief Bind the propagation model and load jammer specs.
     *
     * Filters out disabled jammers immediately so they never appear
     * in the per-tick interference sum.
     *
     * @param jammers  List of @ref JammerSpec instances (from SimConfig).
     * @param plModel  The same @ref PropagationLossModel used by
     *                 @ref LinkEvaluator — jammer path loss is computed
     *                 with identical physics as mesh-node path loss.
     * @param mobModels Mobility models for jammer nodes, in the same index
     *                  order as @p jammers. Built by @ref TopologyBuilder.
     */
    void Configure(const std::vector<JammerSpec>& jammers,
                   ns3::Ptr<ns3::PropagationLossModel> plModel,
                   const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobModels,
                   double carrierHz = 0.0,
                   uint32_t seed = 1);

    /**
     * @brief Return true if at least one enabled jammer was loaded.
     *
     * LinkEvaluator uses this to skip the jammer computation entirely
     * when no jammers are present, keeping the no-jammer case as fast
     * as before.
     */
    bool HasJammers() const;

    /**
     * @brief Compute total jammer interference power received at @p rxMob.
     *
     * For each enabled jammer, computes the received power using the
     * shared propagation loss model (same RMa/UMa/NYU physics as mesh
     * links), then sums the linear watt values across all jammers.
     *
     * @param rxMob  Mobility model of the receiving mesh node.
     * @return       Total jammer interference power in watts. Returns 0.0
     *               if no jammers are configured.
     */
    double InterfPowerAtReceiver(ns3::Ptr<ns3::MobilityModel> rxMob,
                                 double nowS = 0.0) const;

  private:
    /// @brief True if @p spec is transmitting at time @p nowS (per its intervals).
    ///        A jammer with no intervals is treated as always on.
    static bool ActiveAt(const JammerSpec& spec, double nowS);

    /// @brief True if the link carrier falls within the jammer's target band.
    ///        An empty target_freq means "no frequency filtering" (all links).
    bool InBand(const JammerSpec& spec) const;

    /// @brief True if @p rxMob is inside the jammer's 3-D beam cone
    ///        (azimuth+zenith pointing, half-angle = beamwidth/2). Omni when
    ///        beamwidth >= 360.
    static bool InBeam(const JammerSpec& spec,
                       const ns3::Vector& jamPos, const ns3::Vector& rxPos);

    /// @brief On/off decision for a 'random'-type jammer at @p nowS
    ///        (deterministic, seed-dependent). 'constant' jammers are always on.
    bool BurstOn(const JammerSpec& spec, double nowS) const;

    /// @brief Internal per-jammer state after Configure().
    struct JammerEntry
    {
        JammerSpec                     spec;    ///< Jammer parameters.
        ns3::Ptr<ns3::MobilityModel>   mob;     ///< ns-3 mobility model.
    };

    std::vector<JammerEntry>              m_jammers;  ///< Enabled jammers only.
    ns3::Ptr<ns3::PropagationLossModel>   m_plModel;   ///< Shared propagation model.
    double                                m_carrierMhz = 0.0; ///< Link carrier freq (MHz) for InBand.
    uint32_t                              m_seed = 1;    ///< Sim seed for 'random' bursts.
};

}  // namespace mesh_sim
