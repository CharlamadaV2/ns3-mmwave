/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * SINR-to-capacity mapping functions. Pure math, no ns-3 dependency.
 * Used by LinkEvaluator and testable standalone.
 */
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace mesh_sim
{

// Lowest MCS entry from 3GPP TS 38.214 Table 5.1.3.1-1.
static constexpr double SINR_MIN_DB = -6.7;

// MCS table derived from 3GPP TS 38.214 Table 5.1.3.1-1 (NR CQI mapping).
// SINR thresholds and spectral efficiencies match the 15 CQI indices.
// Same data used by MmWaveAmc::CreateCqiFeedbackWbTdma() in ns3-mmwave.
struct McsEntry
{
    double sinr_min_db;
    double spectral_eff; // bits/s/Hz
};

static const McsEntry MCS_TABLE[] = {
    { -6.7, 0.15},  // CQI 0  – QPSK, code rate ~1/5
    { -4.7, 0.23},  // CQI 1
    { -2.3, 0.38},  // CQI 2
    {  0.2, 0.60},  // CQI 3
    {  2.4, 0.88},  // CQI 4
    {  4.3, 1.18},  // CQI 5
    {  5.9, 1.48},  // CQI 6  – 16QAM starts ~here
    {  8.1, 1.91},  // CQI 7
    { 10.3, 2.41},  // CQI 8
    { 11.7, 2.73},  // CQI 9
    { 14.1, 3.32},  // CQI 10
    { 16.3, 3.90},  // CQI 11 – 64QAM starts ~here
    { 18.7, 4.52},  // CQI 12
    { 21.0, 5.12},  // CQI 13
    { 22.7, 5.55},  // CQI 14
};
static constexpr uint32_t MCS_TABLE_SIZE = 15;

// Walk the MCS table to find the highest entry whose SINR threshold is met.
// Returns the index (0–14). Below SINR_MIN_DB returns 0.
inline uint32_t
SinrToMcsIndex(double sinr_db)
{
    if (sinr_db < SINR_MIN_DB)
    {
        return 0;
    }
    uint32_t idx = 0;
    for (uint32_t k = 0; k < MCS_TABLE_SIZE; ++k)
    {
        if (sinr_db >= MCS_TABLE[k].sinr_min_db)
        {
            idx = k;
        }
        else
        {
            break;
        }
    }
    return idx;
}

inline double
SinrToCapacity(double sinr_db, double bandwidth_hz, const std::string& amc_model)
{
    if (sinr_db < SINR_MIN_DB)
    {
        return 0.0;
    }

    if (amc_model == "shannon")
    {
        // Shannon-Hartley theorem: C = B * log2(1 + SNR).
        // Reference: Shannon, "A Mathematical Theory of Communication", 1948.
        // This gives an upper-bound capacity; real systems achieve ~60-80% of Shannon.
        double sinr_linear = std::pow(10.0, sinr_db / 10.0);
        double capacity_bps = bandwidth_hz * std::log2(1.0 + sinr_linear);
        return capacity_bps / 1e6; // Mbps
    }

    if (amc_model != "table")
    {
        throw std::runtime_error(
            "[LinkEvaluator] Unknown amc_model '" + amc_model +
            "'; expected 'shannon' or 'table'");
    }

    double spectral_eff = MCS_TABLE[SinrToMcsIndex(sinr_db)].spectral_eff;
    return spectral_eff * bandwidth_hz / 1e6; // Mbps
}

}  // namespace mesh_sim
