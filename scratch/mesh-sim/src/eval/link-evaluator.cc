/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/eval/link-evaluator.h"
#include "src/eval/sinr-capacity.h"

#include "ns3/channel-condition-model.h"
#include "ns3/log.h"

#include <cmath>

NS_LOG_COMPONENT_DEFINE("LinkEvaluator");

namespace mesh_sim
{

// Ideal beamforming: 4x4 = 16 element array per node (v1 hardcode).
static constexpr uint32_t BF_ELEMENTS_PER_NODE = 16;

// ---------------------------------------------------------------------------
// Configure
// ---------------------------------------------------------------------------

void
LinkEvaluator::Configure(const SimConfig& cfg,
                         ns3::Ptr<ns3::PropagationLossModel> plModel,
                         ns3::Ptr<ns3::ChannelConditionModel> condModel)
{
    if (!plModel || !condModel)
    {
        throw std::runtime_error(
            "[LinkEvaluator] PropagationLossModel and ChannelConditionModel must not be null");
    }

    m_plModel   = plModel;
    m_condModel = condModel;

    m_txPowerDbm       = cfg.channel.tx_power_dbm;
    m_bandwidthHz      = cfg.channel.bandwidth_mhz * 1e6;
    m_amcModel         = cfg.channel.amc_model;
    m_buildingsEnabled = !cfg.buildings.empty();

    // Thermal noise floor: kTB in dBm = -174 dBm/Hz + 10*log10(B_hz) + noise figure.
    // -174 dBm/Hz is thermal noise power spectral density at T=290K (room temp).
    // Reference: Johnson-Nyquist noise, N = kTB where k = Boltzmann's constant.
    m_noiseFloorDbm = -174.0 + 10.0 * std::log10(m_bandwidthHz) + cfg.channel.noise_figure_db;

    // Ideal beamforming gain: 10*log10(N_elements) per array, both TX and RX.
    // For a uniform linear/planar array, max directivity gain = N_elements.
    // With arrays on both ends, total BF gain = 2 * 10*log10(N).
    // Reference: Balanis, "Antenna Theory", Ch. 6 (array factor).
    double gain_per_node = 10.0 * std::log10(static_cast<double>(BF_ELEMENTS_PER_NODE));
    m_bfGainDb = 2.0 * gain_per_node;

    NS_LOG_DEBUG("Configure: txPower=" << m_txPowerDbm << " dBm, BW="
                 << m_bandwidthHz / 1e6 << " MHz, noiseFloor="
                 << m_noiseFloorDbm << " dBm, bfGain=" << m_bfGainDb
                 << " dB, amc=" << m_amcModel);
}

// ---------------------------------------------------------------------------
// Evaluate (single link)
// ---------------------------------------------------------------------------

LinkResult
LinkEvaluator::Evaluate(ns3::Ptr<ns3::MobilityModel> txMob,
                        ns3::Ptr<ns3::MobilityModel> rxMob,
                        uint32_t txIdx,
                        uint32_t rxIdx) const
{
    LinkResult r;
    r.tx_id = txIdx;
    r.rx_id = rxIdx;

    r.distance_m = txMob->GetDistanceFrom(rxMob);

    // Guard against co-located nodes (log10(0) in path-loss equations).
    // Treat as perfect LOS link: no path loss, just TX power + BF gain.
    if (r.distance_m < 1.0)
    {
        r.is_los                   = true;
        r.path_loss_db             = 0.0;
        r.rx_power_dbm             = m_txPowerDbm + m_bfGainDb;
        r.sinr_db                  = r.rx_power_dbm - m_noiseFloorDbm;
        r.capacity_mbps            = SinrToCapacity(r.sinr_db, m_bandwidthHz, m_amcModel);
        r.mcs_index                = SinrToMcsIndex(r.sinr_db);
        r.condition_from_buildings = m_buildingsEnabled;
        NS_LOG_DEBUG("Link " << txIdx << "->" << rxIdx
                     << ": co-located (d<1m), SINR=" << r.sinr_db << " dB");
        return r;
    }

    // LOS / NLOS determination
    auto cond = m_condModel->GetChannelCondition(txMob, rxMob);
    r.is_los = (cond->GetLosCondition() == ns3::ChannelCondition::LosConditionValue::LOS);

    // Propagation: CalcRxPower includes path loss + shadow fading, but NOT BF gain.
    double rxPowerDbm = m_plModel->CalcRxPower(m_txPowerDbm, txMob, rxMob);
    r.path_loss_db = m_txPowerDbm - rxPowerDbm;

    // Effective received power with ideal beamforming on both ends.
    r.rx_power_dbm = rxPowerDbm + m_bfGainDb;

    // SINR assuming orthogonal channels (no inter-node interference).
    // mmWave beams are narrow enough that spatial isolation is high between
    // non-aligned pairs. Full interference modeling would require a joint
    // scheduling+routing layer — deferred to a future version. A simple
    // interference margin (e.g., subtract 3 dB) can be added to ChannelConfig
    // if needed for conservative estimates.
    r.sinr_db = r.rx_power_dbm - m_noiseFloorDbm;

    r.capacity_mbps            = SinrToCapacity(r.sinr_db, m_bandwidthHz, m_amcModel);
    r.mcs_index                = SinrToMcsIndex(r.sinr_db);
    r.condition_from_buildings = m_buildingsEnabled;

    NS_LOG_DEBUG("Link " << txIdx << "->" << rxIdx
                 << ": d=" << r.distance_m << "m"
                 << " LOS=" << r.is_los
                 << " PL=" << r.path_loss_db << "dB"
                 << " rxPow=" << r.rx_power_dbm << "dBm"
                 << " SINR=" << r.sinr_db << "dB"
                 << " cap=" << r.capacity_mbps << "Mbps");

    return r;
}

// ---------------------------------------------------------------------------
// EvaluateAll (N*(N-1)/2 undirected pairs)
// ---------------------------------------------------------------------------

std::vector<LinkResult>
LinkEvaluator::EvaluateAll(
    const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs) const
{
    const auto n = static_cast<uint32_t>(mobs.size());
    std::vector<LinkResult> results;
    results.reserve(n * (n - 1) / 2);

    for (uint32_t i = 0; i < n; ++i)
    {
        for (uint32_t j = i + 1; j < n; ++j)
        {
            results.push_back(Evaluate(mobs[i], mobs[j], i, j));
        }
    }

    NS_LOG_DEBUG("EvaluateAll: " << n << " nodes, " << results.size() << " links evaluated");

    return results;
}

}  // namespace mesh_sim
