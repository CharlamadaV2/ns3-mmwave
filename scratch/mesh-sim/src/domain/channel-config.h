/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/** @brief
 * Radio / channel configuration POD types.
 * No ns-3 headers included — keeps compilation fast and allows use in
 * post-run IO code without ns-3 linkage.
 */
#pragma once

#include <cstdint>
#include <string>

/** @brief
*/
namespace mesh_sim
{

// Parameters unique to the NYU channel model.
// Only applied when channel_model = "nyu"; ignored for 3gpp.
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
struct NyuChannelConfig
{
    // NYUChannelModel
    double rf_bandwidth_mhz          = 800.0;      // NYUChannelModel::RfBandwidth

    // NYUPropagationLossModel
    bool        shadowing_enabled        = true;       // ShadowingEnabled
    double      pressure_mbar            = 1013.25;    // Pressure
    double      humidity_pct             = 50.0;       // Humidity
    double      temperature_c            = 20.0;       // Temperature
    double      rain_rate_mm_hr          = 0.0;        // RainRate
    bool        atmospheric_loss_enabled = false;      // AtmosphericLossEnabled
    bool        foliage_loss_enabled     = false;      // FoliageLossEnabled
    double      foliage_loss_db_m        = 0.4;        // FoliageLoss
    std::string o2i_loss_type            = "Low Loss"; // O2ILosstype: "Low Loss" or "High Loss"
};
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
struct ChannelConfig
{
    // Shared between 3gpp and nyu
    double      frequency_ghz    = 28.0;
    double      tx_power_dbm     = 30.0;   // per-node transmit power (dBm)
    std::string scenario         = "UMi";  // "UMi", "UMa", "RMa", "InH", "InF"
    std::string channel_model    = "3gpp"; // "3gpp" or "nyu"
    bool        blockage_enabled = true;   // 3gpp: ThreeGppChannelModel::Blockage
                                           // nyu:  NYUChannelModel::Blockage
    std::string beamforming_model = "svd";    // for reference; simplified in mesh-sim
    std::string amc_model         = "shannon"; // "shannon" or "table"
    double      noise_figure_db   = 5.0;      // receiver noise figure
    double      bandwidth_mhz     = 400.0;    // system bandwidth for noise floor and capacity
    double      tx_array_gain_dbi = 12.0;     // peak directional gain per side
    double      rx_array_gain_dbi = 12.0;

    NyuChannelConfig nyu; // NYU-only params; ignored when channel_model = "3gpp"
};

}  // namespace mesh_sim
