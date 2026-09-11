/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/** @file jammer-model.cc */

#include "src/jammer/jammer-model.h"

#include <ns3/log.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>

NS_LOG_COMPONENT_DEFINE("JammerModel");

namespace {
    /// @brief Convert dBm to watts — same helper as link-evaluator.cc.
    inline double DbmToWatt(double dbm) { return std::pow(10.0, (dbm - 30.0) / 10.0); }
}
 
namespace mesh_sim
{
 
// ---------------------------------------------------------------------------
// Configure
// ---------------------------------------------------------------------------
 
void
JammerModel::Configure(const std::vector<JammerSpec>& jammers,
                       ns3::Ptr<ns3::PropagationLossModel> plModel,
                       const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobModels,
                       double carrierHz,
                       uint32_t seed)
{
    m_plModel = plModel;
    m_carrierMhz = carrierHz / 1e6;
    m_seed = seed;
    m_jammers.clear();
 
    NS_ASSERT_MSG(jammers.size() == mobModels.size(),
                  "[JammerModel] jammers and mobModels vectors must be the same size");
 
    for (std::size_t i = 0; i < jammers.size(); ++i)
    {
        const auto& spec = jammers[i];
        if (!spec.enabled)
        {
            NS_LOG_DEBUG("JammerModel: skipping disabled jammer '" << spec.id << "'");
            continue;
        }
        m_jammers.push_back({spec, mobModels[i]});
        NS_LOG_DEBUG("JammerModel: loaded jammer '" << spec.id
                     << "' type=" << spec.type
                     << " power=" << spec.tx_power_dbm << " dBm");
    }
 
    NS_LOG_DEBUG("JammerModel::Configure: " << m_jammers.size()
                 << "/" << jammers.size() << " jammers active");
}
 
// ---------------------------------------------------------------------------
// HasJammers
// ---------------------------------------------------------------------------
 
bool
JammerModel::HasJammers() const
{
    return !m_jammers.empty();
}
 
// ---------------------------------------------------------------------------
// InterfPowerAtReceiver
// ---------------------------------------------------------------------------
 
// @brief True if the jammer is transmitting at nowS. No intervals => always on.
bool
JammerModel::ActiveAt(const JammerSpec& spec, double nowS)
{
    if (spec.intervals.empty())
    {
        return true;
    }
    for (const auto& iv : spec.intervals)
    {
        if (nowS >= iv.start && nowS < iv.end)
        {
            return true;
        }
    }
    return false;
}


// ---------------------------------------------------------------------------
// InBand — frequency gate (uses target_freq)
// ---------------------------------------------------------------------------
//
// Empty target_freq => no filtering (jammer affects every link). With entries,
// the link carrier must fall within [min,max] of the listed MHz values (a band),
// or within +/-2.5 MHz of a single listed frequency (a spot).
bool
JammerModel::InBand(const JammerSpec& spec) const
{
    if (spec.target_freq.empty() || m_carrierMhz <= 0.0)
    {
        return true;
    }
    if (spec.target_freq.size() == 1)
    {
        return std::abs(m_carrierMhz - spec.target_freq.front()) <= 2.5;
    }
    double lo = *std::min_element(spec.target_freq.begin(), spec.target_freq.end());
    double hi = *std::max_element(spec.target_freq.begin(), spec.target_freq.end());
    return m_carrierMhz >= lo && m_carrierMhz <= hi;
}

// ---------------------------------------------------------------------------
// InBeam — 3-D directional gate (uses azimuth_deg, zenith_deg, beamwidth_deg)
// ---------------------------------------------------------------------------
//
// Builds the pointing unit vector from azimuth (compass, deg from North) and
// zenith (deg from straight up: 0=up, 90=horizontal, 180=down), then tests the
// angle to the receiver against the cone half-angle. Omni when beamwidth>=360.
bool
JammerModel::InBeam(const JammerSpec& spec,
                    const ns3::Vector& jamPos, const ns3::Vector& rxPos)
{
    if (spec.beamwidth_deg >= 360.0)
    {
        return true;  // omnidirectional
    }
    const double az = spec.azimuth_deg * M_PI / 180.0;
    const double ze = spec.zenith_deg  * M_PI / 180.0;
    // Pointing vector: x=East, y=North, z=Up.
    const double px = std::sin(ze) * std::sin(az);
    const double py = std::sin(ze) * std::cos(az);
    const double pz = std::cos(ze);

    double dx = rxPos.x - jamPos.x;
    double dy = rxPos.y - jamPos.y;
    double dz = rxPos.z - jamPos.z;
    const double dn = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (dn < 1e-9)
    {
        return true;  // co-located: inside any beam
    }
    dx /= dn; dy /= dn; dz /= dn;

    double dot = px*dx + py*dy + pz*dz;
    dot = std::clamp(dot, -1.0, 1.0);
    const double angleDeg = std::acos(dot) * 180.0 / M_PI;
    return angleDeg <= spec.beamwidth_deg / 2.0;
}

// ---------------------------------------------------------------------------
// BurstOn — temporal on/off for 'random' jammers (uses type, duty_cycle)
// ---------------------------------------------------------------------------
//
// 'constant' jammers are always on (their power is scaled by duty_cycle in the
// caller). 'random' jammers turn fully on/off per integer second with
// probability duty_cycle, decided deterministically from (id, second, seed) so
// runs are reproducible and seed-dependent.
bool
JammerModel::BurstOn(const JammerSpec& spec, double nowS) const
{
    if (spec.type != "random")
    {
        return true;
    }
    std::uint64_t h = std::hash<std::string>{}(spec.id);
    h ^= static_cast<std::uint64_t>(m_seed) * 0x9E3779B97F4A7C15ULL;
    h ^= static_cast<std::uint64_t>(std::floor(nowS)) * 0xD1B54A32D192ED03ULL;
    // splitmix64 finalizer -> uniform in [0,1)
    h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 27; h *= 0x94D049BB133111EBULL;
    h ^= h >> 31;
    double u = (h >> 11) * (1.0 / 9007199254740992.0);  // 53-bit mantissa
    return u < spec.duty_cycle;
}

double
JammerModel::InterfPowerAtReceiver(ns3::Ptr<ns3::MobilityModel> rxMob,
                                   double nowS) const
{
    if (m_jammers.empty())
    {
        return 0.0;
    }
 
    double totalWatt = 0.0;
 
    for (const auto& entry : m_jammers)
    {
        const JammerSpec& spec = entry.spec;

        // ---- Time gate: scheduled intervals -----------------------------------
        if (!ActiveAt(spec, nowS))
        {
            continue;
        }

        // ---- Temporal gate: 'random'-type burst on/off (uses type,duty_cycle) --
        if (!BurstOn(spec, nowS))
        {
            continue;
        }

        // ---- Frequency gate: link carrier in the jammer band (uses target_freq) -
        if (!InBand(spec))
        {
            continue;
        }

        const double dist = entry.mob->GetDistanceFrom(rxMob);

        // ---- Range gate -------------------------------------------------------
        if (spec.max_range_m > 0.0 && dist > spec.max_range_m)
        {
            continue;
        }

        // ---- Directional gate: 3-D cone (uses azimuth,zenith,beamwidth) --------
        if (!InBeam(spec, entry.mob->GetPosition(), rxMob->GetPosition()))
        {
            continue;
        }

        // ---- Interference power -----------------------------------------------
        // Effective EIRP includes the jammer's antenna gain (uses tx_array_gain_dbi).
        const double eirpDbm = spec.tx_power_dbm + spec.tx_array_gain_dbi;
        double rxPowerDbm;
        if (dist < 1.0)
        {
            rxPowerDbm = eirpDbm;  // co-located guard
        }
        else
        {
            rxPowerDbm = m_plModel->CalcRxPower(eirpDbm, entry.mob, rxMob);
        }

        // 'constant' jammers are scaled by duty_cycle (average power); 'random'
        // jammers already gated on/off above, so they contribute full power when on.
        const double duty = (spec.type == "random") ? 1.0 : spec.duty_cycle;
        const double effectiveWatt = DbmToWatt(rxPowerDbm) * duty;
        totalWatt += effectiveWatt;

        NS_LOG_DEBUG("JammerModel: '" << spec.id << "' -> d=" << dist
                     << "m eirp=" << eirpDbm << "dBm rxPow=" << rxPowerDbm
                     << "dBm eff=" << effectiveWatt << "W");
    }
 
    return totalWatt;
}


} //mesh-sim
