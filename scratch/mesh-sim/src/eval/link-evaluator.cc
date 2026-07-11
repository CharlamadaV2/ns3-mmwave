/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

/** @file link-evaluator.cc */

#include "src/eval/link-evaluator.h"
#include "src/eval/sinr-capacity.h"

#include "ns3/channel-condition-model.h"
#include "ns3/log.h"

#include <cmath>

NS_LOG_COMPONENT_DEFINE("LinkEvaluator");

namespace {

    inline uint32_t McsIndexForModel(double sinrDb, const std::string& amcModel)
    {
        if (amcModel == "silvus")
        {
            return mesh_sim::SinrToSilvusMcsIndex(sinrDb);
        }
        return mesh_sim::SinrToMcsIndex(sinrDb);
    }

}

namespace mesh_sim
{

// ---------------------------------------------------------------------------
// Configure
// ---------------------------------------------------------------------------

void
LinkEvaluator::Configure(const SimConfig& cfg,
                         ns3::Ptr<ns3::PropagationLossModel> plModel,
                         ns3::Ptr<ns3::ChannelConditionModel> condModel,
                         const std::string& band)
{
    //Model Checks
    if (!plModel || !condModel)
    {
        throw std::runtime_error(
            "[LinkEvaluator] PropagationLossModel and ChannelConditionModel must not be null");
    }

    //Loading Config Params
    m_plModel   = plModel;
    m_condModel = condModel;

    m_txPowerDbm       = cfg.channel.tx_power_dbm;
    m_bandwidthHz      = cfg.channel.bandwidth_mhz * 1e6;
    m_amcModel         = cfg.channel.amc_model;
    m_buildingsEnabled = !cfg.buildings.empty();

    m_noiseFloorDbm = -174.0 + 10.0 * std::log10(m_bandwidthHz) + cfg.channel.noise_figure_db;

    //Clears and loads tx/rx gain values
    m_txGainDbi.clear();
    m_rxGainDbi.clear();
    m_txGainDbi.reserve(cfg.nodes.size());
    m_rxGainDbi.reserve(cfg.nodes.size());
    uint32_t n_overrides = 0;
    for (const auto& spec : cfg.nodes)
    {
        m_txGainDbi.push_back(spec.tx_array_gain_dbi.value_or(cfg.channel.tx_array_gain_dbi));
        m_rxGainDbi.push_back(spec.rx_array_gain_dbi.value_or(cfg.channel.rx_array_gain_dbi));
        if (spec.tx_array_gain_dbi.has_value() || spec.rx_array_gain_dbi.has_value())
        {
            ++n_overrides;
        }
    }

    NS_LOG_DEBUG("Configure: txPower=" << m_txPowerDbm << " dBm, BW="
                 << m_bandwidthHz / 1e6 << " MHz, noiseFloor="
                 << m_noiseFloorDbm << " dBm, gain default tx="
                 << cfg.channel.tx_array_gain_dbi << " rx="
                 << cfg.channel.rx_array_gain_dbi << " dBi, "
                 << n_overrides << "/" << cfg.nodes.size()
                 << " nodes have gain overrides, amc=" << m_amcModel);
}

// ---------------------------------------------------------------------------
// Evaluate (single link) — noise-limited; unchanged.
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

    // Per-link beamforming gain: tx end's array + rx end's array. Either side
    // may be a per-node override from nodes.json; otherwise the channel default.
    const double bfGainDb = m_txGainDbi[txIdx] + m_rxGainDbi[rxIdx];

    // Guard against log10(0) for co-located nodes.
    if (r.distance_m < 1.0)
    {
        r.is_los                   = true;
        r.path_loss_db             = 0.0;
        r.rx_power_dbm             = m_txPowerDbm + bfGainDb;
        r.sinr_db                   = r.rx_power_dbm - m_noiseFloorDbm;
        r.capacity_mbps            = SinrToCapacity(r.sinr_db, m_bandwidthHz, m_amcModel);
        r.mcs_index                = McsIndexForModel(r.sinr_db, m_amcModel);
        r.condition_from_buildings = m_buildingsEnabled;
        NS_LOG_DEBUG("Link " << txIdx << "->" << rxIdx
                     << ": co-located (d<1m), SINR=" << r.sinr_db << " dB");
        return r;
    }

    // LOS / NLOS determination
    auto cond = m_condModel->GetChannelCondition(txMob, rxMob);
    r.is_los = (cond->GetLosCondition() == ns3::ChannelCondition::LosConditionValue::LOS);

    double rxPowerDbm = m_plModel->CalcRxPower(m_txPowerDbm, txMob, rxMob);
    r.path_loss_db = m_txPowerDbm - rxPowerDbm;
    r.rx_power_dbm = rxPowerDbm + bfGainDb;

    // SNR based calculations, refered to as SINR colloquially
    r.sinr_db = r.rx_power_dbm - m_noiseFloorDbm;

    r.capacity_mbps            = SinrToCapacity(r.sinr_db, m_bandwidthHz, m_amcModel);
    r.mcs_index                = McsIndexForModel(r.sinr_db, m_amcModel);
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

    // mmWave: no interference — each pair is independent (original behaviour).
    std::vector<LinkResult> results;
    results.reserve(n * (n - 1) / 2);
    for (uint32_t i = 0; i < n; ++i)
    {
        for (uint32_t j = i + 1; j < n; ++j)
        {
            results.push_back(Evaluate(mobs[i], mobs[j], i, j));
        }
    }
    return results;
}
}  // namespace mesh_sim