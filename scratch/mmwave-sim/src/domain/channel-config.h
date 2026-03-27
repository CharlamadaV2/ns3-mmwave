/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Radio / channel configuration POD types.
 * No ns-3 headers included — keeps compilation fast and allows use in
 * post-run IO code without ns-3 linkage.
 */
#pragma once

#include <cstdint>
#include <string>

namespace mmwave_sim
{

// Parameters unique to the NYU channel model.
// Only applied when channel_model = "nyu"; ignored for 3gpp.
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

struct ChannelConfig
{
    // Shared between 3gpp and nyu
    double      frequency_ghz    = 28.0;
    double      tx_power_dbm     = 30.0;   // eNB transmit power (dBm)
    std::string scenario         = "UMi";  // "UMi", "UMa", "RMa", "InH", "InF"
    std::string channel_model    = "3gpp"; // "3gpp" or "nyu"
    bool        blockage_enabled = true;   // 3gpp: ThreeGppChannelModel::Blockage
                                           // nyu:  NYUChannelModel::Blockage

    // Performance tuning — controls how often expensive subsystems recompute.
    // 0 = compute once and cache forever (ns3 default, fastest for static scenarios).
    // Non-zero = re-evaluate every N ms (needed for mobile scenarios where the
    // channel physically changes over time).
    uint32_t    channel_update_period_ms   = 0;     // channel matrix recomputation
    uint32_t    condition_update_period_ms = 0;     // LOS/NLOS condition evaluation
    std::string beamforming_model          = "svd";  // "svd", "dft", "codebook"
    uint32_t    cqi_period_slots           = 20;     // CQI reporting period (ns3 default: 10)
    std::string amc_model                  = "error"; // "error" or "shannon"

    NyuChannelConfig nyu; // NYU-only params; ignored when channel_model = "3gpp"
};

}  // namespace mmwave_sim
