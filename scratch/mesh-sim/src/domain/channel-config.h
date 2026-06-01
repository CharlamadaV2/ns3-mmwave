/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

/**
 * @file channel-config.h
 * @brief Radio and channel configuration POD types.
 *
 *
 * Two channel models are supported, selected by @ref ChannelConfig::channel_model:
 * - @c "3gpp" — 3GPP TR 38.901 statistical model (default).
 * - @c "nyu"  — NYU WIRELESS mmWave model; requires the additional
 *               parameters in @ref NyuChannelConfig.
 */
#pragma once

#include <cstdint>
#include <string>

namespace mesh_sim
{

/**
 * @brief Parameters specific to the NYU mmWave channel model.
 *
 * These fields are only applied when @ref ChannelConfig::channel_model is
 * @c "nyu". They are silently ignored for the @c "3gpp" model.
 *
 * ns-3 attribute names are shown in comments so they can be cross-referenced
 * with NYUChannelModel / NYUPropagationLossModel documentation.
 */
struct NyuChannelConfig
{
    // ---- NYUChannelModel attributes ----------------------------------------

    double rf_bandwidth_mhz = 800.0;  ///< RF bandwidth passed to NYUChannelModel
                                       ///<   (NYUChannelModel::RfBandwidth).

    // ---- NYUPropagationLossModel attributes --------------------------------

    bool   shadowing_enabled        = true;        ///< Enable shadow-fading component
                                                    ///<   (NYUPropagationLossModel::ShadowingEnabled).
    double pressure_mbar            = 1013.25;     ///< Atmospheric pressure in mbar
                                                    ///<   (NYUPropagationLossModel::Pressure).
    double humidity_pct             = 50.0;        ///< Relative humidity in percent
                                                    ///<   (NYUPropagationLossModel::Humidity).
    double temperature_c            = 20.0;        ///< Air temperature in °C
                                                    ///<   (NYUPropagationLossModel::Temperature).
    double rain_rate_mm_hr          = 0.0;         ///< Rain rate in mm/hr; 0 = dry
                                                    ///<   (NYUPropagationLossModel::RainRate).
    bool   atmospheric_loss_enabled = false;       ///< Enable molecular absorption loss
                                                    ///<   (NYUPropagationLossModel::AtmosphericLossEnabled).
    bool   foliage_loss_enabled     = false;       ///< Enable foliage excess loss
                                                    ///<   (NYUPropagationLossModel::FoliageLossEnabled).
    double foliage_loss_db_m        = 0.4;         ///< Foliage loss rate in dB/m
                                                    ///<   (NYUPropagationLossModel::FoliageLoss).
    std::string o2i_loss_type       = "Low Loss";  ///< Outdoor-to-indoor loss class:
                                                    ///<   @c "Low Loss" or @c "High Loss"
                                                    ///<   (NYUPropagationLossModel::O2ILosstype).
};

/**
 * @brief Complete radio and channel configuration for simulation run.
 *
 * Parameters shared between the 3GPP and NYU models are at the top level.
 * NYU-specific parameters live in the @ref nyu sub-struct.
 *
 * Gain values (@c tx_array_gain_dbi, @c rx_array_gain_dbi) represent peak
 * directional beamforming gain per side and are applied symmetrically; each
 * node uses the same antenna model.
 */
struct ChannelConfig
{
    // Shared between 3gpp and nyu
    double      frequency_ghz    = 28.0;
    double      tx_power_dbm     = 30.0;   // per-node transmit power (dBm)
    std::string scenario         = "UMi";  // "UMi", "UMa", "RMa", "InH", "InF"
    std::string channel_model    = "3gpp"; // "3gpp" or "nyu"
    std::string condition_model  = "auto"; // "auto" or "static_los"
    bool        blockage_enabled = true;   // 3gpp: ThreeGppChannelModel::Blockage
                                           // nyu:  NYUChannelModel::Blockage
    std::string beamforming_model = "svd";    // for reference; simplified in mesh-sim
    std::string amc_model         = "shannon"; // "shannon" or "table"
    double      noise_figure_db   = 5.0;      // receiver noise figure
    double      bandwidth_mhz     = 400.0;    // system bandwidth for noise floor and capacity
    double      tx_array_gain_dbi = 12.0;     // peak directional gain per side
    double      rx_array_gain_dbi = 12.0;

    NyuChannelConfig nyu;  ///< NYU model parameters; ignored when @c channel_model is @c "3gpp".
};

}  // namespace mesh_sim