/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file jammer-model.h
 * @brief Per-receiver jammer interference power for the sub-6 band.
 *
 * @ref mesh_sim::JammerModel sums the received power (watts) of every enabled
 * @ref JammerSpec that passes the time, random-burst, frequency, range and
 * beam gates. @ref LinkEvaluator::Evaluate adds the result to the SINR
 * denominator when @c band is @c "sub-6" and @ref JammerModel::HasJammers is true.
 *
 * Path loss uses the same ns-3 @c PropagationLossModel as mesh links.
 * The model is separate from @ref LinkEvaluator so it can be tested or
 * extended without touching the core SINR code.
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
 * Holds only the state set by @ref Configure; @ref InterfPowerAtReceiver is
 * const and has no side effects, so results depend only on the arguments
 * (including @c nowS) and the configured seed.
 */
class JammerModel
{
  public:
    /**
     * @fn JammerModel::Configure
     * @brief Bind the propagation model and load the enabled jammers.
     *
     * @param jammers   Jammer specs, normally @c SimConfig::jammers.
     * @param plModel   Propagation loss model shared with @ref LinkEvaluator.
     * @param mobModels One mobility model per entry of @p jammers, same order
     *                  (built by @c TopologyBuilder::GetJammerMobilityModels).
     * @param carrierHz Link carrier frequency in Hz; converted to MHz for the
     *                  frequency gate. Default 0.0, which disables that gate.
     * @param seed      Simulation seed mixed into the @c random burst draw. Default 1.
     * @return Nothing.
     * @throws Nothing directly; @c NS_ASSERT_MSG aborts (debug builds) if
     *         @p jammers and @p mobModels differ in size.
     *
     * Replaces any previously loaded jammers. Specs with @c enabled == false
     * are dropped here and never reach the per-tick sum. Copies each spec.
     * Not called by @ref LinkEvaluator::Configure when @c cfg.jammers is empty.
     */
    void Configure(const std::vector<JammerSpec>& jammers,
                   ns3::Ptr<ns3::PropagationLossModel> plModel,
                   const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobModels,
                   double carrierHz = 0.0,
                   uint32_t seed = 1);

    /**
     * @fn JammerModel::HasJammers
     * @brief Report whether at least one enabled jammer was loaded.
     *
     * @return @c true if any enabled jammer is loaded; @c false before
     *         @ref Configure or when all jammers are disabled.
     *
     * @ref LinkEvaluator uses this to skip jammer work entirely.
     */
    bool HasJammers() const;

    /**
     * @fn JammerModel::InterfPowerAtReceiver
     * @brief Sum the jammer power received at one node at a given time.
     *
     * @param rxMob Mobility model of the receiving node.
     * @param nowS  Current simulation time in seconds since scenario start. Default 0.0.
     * @return Total interference in watts (linear, not dBm); 0.0 if no jammer
     *         is loaded or none passes its gates.
     * @throws Nothing.
     *
     * Per jammer, in order: interval gate, random-burst gate, frequency gate,
     * range gate (@c max_range_m, 0 = off), beam gate. A passing jammer gives
     * EIRP = @c tx_power_dbm + @c tx_array_gain_dbi. Received power is
     * @c CalcRxPower(EIRP), or the EIRP itself when the distance is under 1 m.
     * @c constant jammers are scaled by @c duty_cycle; @c random jammers count
     * at full power when their burst is on. The receive-side antenna gain is
     * not applied. Does not read the receiver's own gain or the mesh TX power.
     */
    double InterfPowerAtReceiver(ns3::Ptr<ns3::MobilityModel> rxMob,
                                 double nowS = 0.0) const;

  private:
    /// @brief True if @p spec is inside one of its half-open intervals at @p nowS.
    ///        A jammer with no intervals is treated as always on.
    static bool ActiveAt(const JammerSpec& spec, double nowS);

    /// @brief True if the link carrier falls within the jammer's target band.
    ///        Empty target_freq or an unset carrier (0 MHz) means no filtering.
    ///        One value is a spot (+/-2.5 MHz); two or more give [min,max].
    bool InBand(const JammerSpec& spec) const;

    /// @brief True if @p rxMob is inside the jammer's 3-D beam cone
    ///        (azimuth+zenith pointing, half-angle = beamwidth/2). Omni when
    ///        beamwidth >= 360.
    static bool InBeam(const JammerSpec& spec,
                       const ns3::Vector& jamPos, const ns3::Vector& rxPos);

    /// @brief On/off decision for a 'random'-type jammer at @p nowS: one hash
    ///        draw per whole second from (id, seed, second) against duty_cycle.
    ///        Any other type is always on.
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
