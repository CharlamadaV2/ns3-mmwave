/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file link-evaluator.h
 * @brief Per-link path loss, SINR, and capacity computation.
 *
 * @ref LinkEvaluator wraps a pair of ns-3 propagation models — a
 * @c PropagationLossModel and a @c ChannelConditionModel — and applies
 * the configured TX power, beamforming gain, bandwidth, and noise figure
 * to produce a complete @ref LinkResult for each node pair.
 *
 * @note @ref Configure must be called before @ref Evaluate or
 *       @ref EvaluateAll; calling either without first configuring produces
 *       undefined behaviour (null pointer dereference on the model pointers).
 */
#pragma once

#include "src/domain/link-result.h"
#include "src/domain/sim-config.h"

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
 * Stateless between calls (all per-call state lives in @ref LinkResult).
 * The ns-3 model pointers are set once by @ref Configure and reused
 * across every subsequent @ref Evaluate / @ref EvaluateAll call.
 */
class LinkEvaluator
{
  public:
    /**
     * @brief Bind ns-3 propagation models and extract scalar radio parameters.
     *
     * Computes and caches:
     * - Thermal noise floor: @c −174 + 10·log₁₀(bandwidth_Hz) + noise_figure_dB.
     * - Combined beamforming gain: @c tx_array_gain_dBi + rx_array_gain_dBi.
     *
     * Must be called once before @ref Evaluate or @ref EvaluateAll.
     *
     * @param cfg       Fully loaded and validated simulation configuration.
     * @param plModel   ns-3 propagation loss model (3GPP or NYU); must not be null.
     * @param condModel ns-3 channel condition model for LOS/NLOS determination;
     *                  must not be null.
     * @throws std::runtime_error if either model pointer is null.
     */
    void Configure(const SimConfig& cfg,
                   ns3::Ptr<ns3::PropagationLossModel> plModel,
                   ns3::Ptr<ns3::ChannelConditionModel> condModel);

    /**
     * @brief Evaluate one directed (tx → rx) link at the current node positions.
     *
     * **Co-location guard**: if the 3-D distance between @c txMob and @c rxMob
     * is less than 1 m, path loss is set to 0 dB and the full TX power plus
     * beamforming gain is returned directly, bypassing the propagation model.
     * This prevents undefined behaviour from @c log10(0) in distance-dependent
     * models.
     *
     * For all other distances:
     * -# The channel condition model determines LOS vs. NLOS.
     * -# The propagation loss model computes @c rx_power_dBm.
     * -# Beamforming gain is added: @c rx_power += bf_gain_dB.
     * -# SINR is computed against the thermal noise floor only (no ICI).
     * -# @ref SinrToCapacity and @ref SinrToMcsIndex translate SINR to capacity
     *    and MCS index using the configured AMC model.
     *
     * @param txMob  Mobility model of the transmitting node.
     * @param rxMob  Mobility model of the receiving node.
     * @param txIdx  Index of the transmitter in @ref SimConfig::nodes.
     * @param rxIdx  Index of the receiver in @ref SimConfig::nodes.
     * @return Fully populated @ref LinkResult for this directed pair.
     */
    LinkResult Evaluate(ns3::Ptr<ns3::MobilityModel> txMob,
                        ns3::Ptr<ns3::MobilityModel> rxMob,
                        uint32_t txIdx,
                        uint32_t rxIdx) const;

    /**
     * @brief Evaluate all N·(N−1)/2 unordered node pairs in index order.
     *
     * Iterates pairs (i, j) with i < j, calling @ref Evaluate for each.
     * The returned vector is in the same order and has the same size as
     * expected by @ref LinkTable::Update.
     *
     * @param mobs  Mobility models for all @c N nodes, in node-index order
     *              (i.e. @c mobs[k] corresponds to @ref SimConfig::nodes[k]).
     * @return Flat vector of N·(N−1)/2 @ref LinkResult objects.
     */
    std::vector<LinkResult> EvaluateAll(
        const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs) const;

  private:
    double      m_txPowerDbm       = 30.0;    ///< TX power in dBm (from @ref ChannelConfig).
    double      m_noiseFloorDbm    = -174.0;  ///< Thermal noise floor in dBm; computed by
                                               ///<   @ref Configure from bandwidth and noise figure.
    double      m_bandwidthHz      = 400e6;   ///< System bandwidth in Hz.
    double      m_bfGainDb         = 0.0;     ///< Combined TX + RX array gain in dB.
    std::string m_amcModel         = "shannon"; ///< Capacity model: @c "shannon" or @c "table".
    bool        m_buildingsEnabled = false;    ///< @c true when buildings are present;
                                               ///<   written to @ref LinkResult::condition_from_buildings.

    ns3::Ptr<ns3::PropagationLossModel>  m_plModel;    ///< ns-3 path-loss model (set by @ref Configure).
    ns3::Ptr<ns3::ChannelConditionModel> m_condModel;  ///< ns-3 LOS/NLOS model (set by @ref Configure).
};

}  // namespace mesh_sim