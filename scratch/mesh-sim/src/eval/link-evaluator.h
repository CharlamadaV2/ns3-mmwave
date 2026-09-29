/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file link-evaluator.h
 * @brief Per-link path loss, SINR, and capacity computation.
 *
 * @ref LinkEvaluator wraps a pair of ns-3 propagation models — a
 * @c PropagationLossModel and a @c ChannelConditionModel — and applies
 * the configured TX power, per-node array gains, bandwidth, and noise figure
 * to produce a complete @ref LinkResult for each node pair. In the sub-6
 * band it also folds in jammer power via @ref JammerModel.
 *
 * @note @ref Configure must be called before @ref Evaluate or
 *       @ref EvaluateAll; calling either without first configuring produces
 *       undefined behaviour (null model pointers and empty gain vectors).
 */
#pragma once

#include "src/domain/link-result.h"
#include "src/domain/sim-config.h"
#include "src/jammer/jammer-model.h"

#include "ns3/channel-condition-model.h"
#include "ns3/mobility-model.h"
#include "ns3/propagation-loss-model.h"

#include <string>
#include <vector>

namespace mesh_sim
{

/**
 * @brief Computes per-link radio metrics using ns-3 propagation models.
 *
 * Both evaluate methods are const and keep no per-call state (results live
 * in @ref LinkResult). Everything else is set once by @ref Configure.
 */
class LinkEvaluator
{
  public:
    /**
     * @fn LinkEvaluator::Configure
     * @brief Bind ns-3 models and cache the scalar radio parameters from the config.
     *
     * @param cfg         Fully loaded and validated simulation configuration.
     * @param plModel     ns-3 propagation loss model (3GPP or NYU); must not be null.
     * @param condModel   ns-3 channel condition model for LOS/NLOS; must not be null.
     * @param band        Radio band, @c "mmwave" (default) or @c "sub-6".
     *                    Only @c "sub-6" enables jammer interference.
     * @param jammerMobs  Jammer mobility models, same order as @c cfg.jammers.
     *                    Ignored when @c cfg.jammers is empty. Default empty.
     * @return Nothing.
     * @throws std::runtime_error if @p plModel or @p condModel is null.
     *
     * Caches TX power, bandwidth (Hz), carrier frequency (Hz), AMC model name,
     * whether buildings exist, the noise floor
     * (@c −174 + 10·log10(bandwidth_Hz) + noise_figure_dB, in dBm) and one TX and
     * one RX array gain (dBi) per node: the node's own override if set,
     * otherwise the channel default. Configures the internal @ref JammerModel
     * (carrier and seed included) only when @c cfg.jammers is non-empty.
     * Safe to call again; per-node gain vectors are rebuilt.
     */
    void Configure(const SimConfig& cfg,
                   ns3::Ptr<ns3::PropagationLossModel> plModel,
                   ns3::Ptr<ns3::ChannelConditionModel> condModel,
                   const std::string& band = "mmwave",
                   const std::vector<ns3::Ptr<ns3::MobilityModel>>& jammerMobs = {});

    /**
     * @fn LinkEvaluator::Evaluate
     * @brief Evaluate one directed (tx → rx) link at the current node positions.
     *
     * @param txMob  Mobility model of the transmitting node.
     * @param rxMob  Mobility model of the receiving node.
     * @param txIdx  Index of the transmitter in @ref SimConfig::nodes; also
     *               indexes the cached TX gain, so it must be in range.
     * @param rxIdx  Index of the receiver in @ref SimConfig::nodes; indexes the RX gain.
     * @param nowS   Simulation time in seconds, used only for jammer gating. Default 0.0.
     * @return Fully populated @ref LinkResult for this directed pair.
     * @throws std::runtime_error if the AMC model name is unknown (from
     *         @ref SinrToCapacity, when @c amc_model is not shannon, table or silvus).
     *
     * Steps:
     * -# LOS or NLOS comes from the channel condition model.
     * -# Path loss is the propagation model's value, floored at free-space
     *    path loss (FSPL) for the link distance, with distance floored at 1 m.
     *    The model gives implausibly low loss at very short range; the floor
     *    caps SINR at the free-space value. A non-finite model loss (d = 0)
     *    is replaced by FSPL.
     * -# @c rx_power_dbm = TX power − path loss + TX gain(tx) + RX gain(rx).
     * -# Jammer power (sub-6 only, and only if jammers are loaded) is the larger
     *    of the jammer power at the receiver and at the transmitter.
     * -# With no jammer power, @c sinr_db = rx_power_dbm − noise floor (plain
     *    SNR, no inter-node interference). Otherwise
     *    @c sinr_db = 10·log10(signal / (noise + jammer)), clamped to a
     *    minimum of 0 dB.
     * -# Capacity comes from @ref SinrToCapacity. The MCS index comes from
     *    @ref SinrToSilvusMcsIndex when the model is @c "silvus", else
     *    @ref SinrToMcsIndex.
     */
    LinkResult Evaluate(ns3::Ptr<ns3::MobilityModel> txMob,
                        ns3::Ptr<ns3::MobilityModel> rxMob,
                        uint32_t txIdx,
                        uint32_t rxIdx,
			double nowS = 0.0) const;

    /**
     * @fn LinkEvaluator::EvaluateAll
     * @brief Evaluate all N·(N−1)/2 unordered node pairs in index order.
     *
     * @param mobs  Mobility models for all N nodes, in node-index order
     *              (@c mobs[k] is @c SimConfig::nodes[k]).
     * @param nowS  Simulation time in seconds, passed to @ref Evaluate. Default 0.0.
     * @return Vector of N·(N−1)/2 @ref LinkResult, ordered (0,1), (0,2), ..., (N−2,N−1).
     *         Each result has @c tx_id < @c rx_id. Empty for N < 2.
     * @throws std::runtime_error propagated from @ref Evaluate.
     *
     * Only the upper triangle is evaluated; @ref LinkTable::Update mirrors it.
     * This is the order and size @ref LinkTable::Update requires.
     */
    std::vector<LinkResult> EvaluateAll(
        const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs,
       	double nowS = 0.0) const;

  private:
    double      m_txPowerDbm       = 30.0;    ///< TX power in dBm (from @ref ChannelConfig).
    double      m_noiseFloorDbm    = -174.0;  ///< Thermal noise floor in dBm; computed by
                                               ///<   @ref Configure from bandwidth and noise figure.
    double      m_bandwidthHz      = 400e6;   ///< System bandwidth in Hz.
    double      m_frequencyHz      = 2.4e9;   ///< Carrier frequency in Hz (from @ref ChannelConfig);
                                               ///<   used for the free-space path-loss lower bound.
    std::string m_amcModel         = "shannon"; ///< Capacity model: @c "shannon" or @c "table".
    bool        m_buildingsEnabled = false;    ///< @c true when buildings are present;
                                               ///<   written to @ref LinkResult::condition_from_buildings.
    bool        m_computeInterference = false;  ///< true when band == "sub-6";
                                                ///<   adds jammer power to the SINR denominator.
    // Per-node array gains, indexed parallel to cfg.nodes (and to mobs in
    // EvaluateAll). Resolved at Configure() from each NodeSpec's override
    // or the channel default.
    std::vector<double> m_txGainDbi;
    std::vector<double> m_rxGainDbi;

    ns3::Ptr<ns3::PropagationLossModel>  m_plModel;    ///< ns-3 path-loss model (set by @ref Configure).
    ns3::Ptr<ns3::ChannelConditionModel> m_condModel;  ///< ns-3 LOS/NLOS model (set by @ref Configure).
    JammerModel m_jammerModel; //mesh-sim Jammer model
};

}  // namespace mesh_sim
