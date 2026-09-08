/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/** @file jammer-model.cc */

#include "src/jammer/jammer-model.h"

#include <ns3/log.h>

#include <cmath>

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
                       const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobModels)
{
    m_plModel = plModel;
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
        // ---- Time gate ------------------------------------------------------
        // Skip jammers not scheduled to transmit at the current sim time.
        if (!ActiveAt(entry.spec, nowS))
        {
            continue;
        }

        const double dist = entry.mob->GetDistanceFrom(rxMob);
 
        // ---- Range gate -------------------------------------------------------
        // If max_range_m is set and receiver is beyond it, skip this jammer.
        if (entry.spec.max_range_m > 0.0 && dist > entry.spec.max_range_m)
        {
            NS_LOG_DEBUG("JammerModel: '" << entry.spec.id
                         << "' -> receiver out of range (" << dist
                         << "m > " << entry.spec.max_range_m << "m), skipping");
            continue;
        }
 
        
	//TODO: Update to match with spec
	// ---- Directional gate -------------------------------------------------
        // If beamwidth < 360, check whether the receiver falls within the cone.
        if (entry.spec.beamwidth_deg < 360.0)
        {
            ns3::Vector jamPos = entry.mob->GetPosition();
            ns3::Vector rxPos  = rxMob->GetPosition();
 
            // Vector from jammer to receiver in the XY plane
            double dx = rxPos.x - jamPos.x;
            double dy = rxPos.y - jamPos.y;
 
            // Bearing from jammer to receiver: degrees clockwise from North
            // atan2 returns angle from +X axis (East), so convert to compass bearing.
            double bearing_deg = std::fmod(
                90.0 - std::atan2(dy, dx) * 180.0 / M_PI + 360.0, 360.0);
 
            // Angular difference between jammer pointing and receiver bearing.
            // Wrap into [-180, 180] so we get the shortest angular distance.
            double angle_diff = std::fmod(
                bearing_deg - entry.spec.azimuth_deg + 540.0, 360.0) - 180.0;
            double abs_diff = std::abs(angle_diff);
 
            const double half_beam = entry.spec.beamwidth_deg / 2.0;
            if (abs_diff > half_beam)
            {
                NS_LOG_DEBUG("JammerModel: '" << entry.spec.id
                             << "' -> receiver at bearing " << bearing_deg
                             << " deg, off-beam by " << abs_diff
                             << " deg (half-beam=" << half_beam << "), skipping");
                continue;
            }
 
            NS_LOG_DEBUG("JammerModel: '" << entry.spec.id
                         << "' -> receiver at bearing " << bearing_deg
                         << " deg, within beam (diff=" << abs_diff << " deg)");
        }
 
	//TODO: Update to match with spec
        // ---- Interference power -----------------------------------------------
        double rxPowerDbm;
        if (dist < 1.0)
        {
            rxPowerDbm = entry.spec.tx_power_dbm;  // co-located guard
        }
        else
        {
            rxPowerDbm = m_plModel->CalcRxPower(entry.spec.tx_power_dbm,
                                                 entry.mob, rxMob);
        }
 
        const double effectiveWatt = DbmToWatt(rxPowerDbm) * entry.spec.duty_cycle;
        totalWatt += effectiveWatt;
 
        NS_LOG_DEBUG("JammerModel: '" << entry.spec.id
                     << "' -> d=" << dist << "m"
                     << " rxPow=" << rxPowerDbm << "dBm"
                     << " duty=" << entry.spec.duty_cycle
                     << " effective=" << effectiveWatt << "W");
    }
 
    return totalWatt;
}


} //mesh-sim
