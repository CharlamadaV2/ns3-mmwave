/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file sinr-capacity.h
 * @brief SINR-to-capacity and SINR-to-MCS mapping functions.
 *
 *
 * | @c amc_model  | Method                                                          |
 * |---------------|-----------------------------------------------------------------|
 * | @c "shannon"  | Shannon-Hartley theorem: @c B·log₂(1+SNR). Theoretical upper   |
 * |               | bound; real systems achieve roughly 60–80% of this value.       |
 * | @c "table"    | 3GPP TS 38.214 NR CQI table (15 entries). Spectral efficiency   |
 * |               | steps match the discrete MCS levels used by the field hardware. |
 *
 * The MCS table is derived from 3GPP TS 38.214 Table 5.1.3.1-1 and uses
 * the same SINR thresholds and spectral efficiencies as
 * @c MmWaveAmc::CreateCqiFeedbackWbTdma() in ns3-mmwave.
 */
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

namespace mesh_sim
{

/**
 * @brief Minimum SINR in dB required for any data transmission (CQI 0).
 *
 * Corresponds to the lowest entry in @ref MCS_TABLE (QPSK, lowest code
 * rate). Links with SINR below this threshold return 0 Mbps capacity and
 * MCS index 0. Also used as the default connectivity threshold in
 * @ref LinkTable::ConnectedLinkCount and @ref LinkTable::IsConnected.
 *
 * Source: 3GPP TS 38.214 Table 5.1.3.1-1, CQI index 0.
 */
static constexpr double SINR_MIN_DB = -6.7;

/**
 * @brief One entry in the NR CQI / MCS table.
 *
 * Used by @ref SinrToMcsIndex and @ref SinrToCapacity (@c "table" mode).
 */
struct McsEntry
{
    double sinr_min_db;    ///< Minimum SINR in dB required to select this entry.
    double spectral_eff;   ///< Spectral efficiency in bits/s/Hz at this CQI index.
};

/**
 * @brief 3GPP NR CQI table (15 entries, indices 0–14).
 *
 * Derived from 3GPP TS 38.214 Table 5.1.3.1-1.
 * Matches @c MmWaveAmc::CreateCqiFeedbackWbTdma() in ns3-mmwave.
 *
 * | Index | Modulation | SINR threshold (dB) | Spectral eff. (bits/s/Hz) |
 * |-------|------------|---------------------|---------------------------|
 * | 0     | QPSK       | −6.7                | 0.15                      |
 * | 1     | QPSK       | −4.7                | 0.23                      |
 * | 2     | QPSK       | −2.3                | 0.38                      |
 * | 3     | QPSK       |  0.2                | 0.60                      |
 * | 4     | QPSK       |  2.4                | 0.88                      |
 * | 5     | QPSK       |  4.3                | 1.18                      |
 * | 6     | 16QAM      |  5.9                | 1.48                      |
 * | 7     | 16QAM      |  8.1                | 1.91                      |
 * | 8     | 16QAM      | 10.3                | 2.41                      |
 * | 9     | 16QAM      | 11.7                | 2.73                      |
 * | 10    | 64QAM      | 14.1                | 3.32                      |
 * | 11    | 64QAM      | 16.3                | 3.90                      |
 * | 12    | 64QAM      | 18.7                | 4.52                      |
 * | 13    | 64QAM      | 21.0                | 5.12                      |
 * | 14    | 64QAM      | 22.7                | 5.55                      |
 */
static const McsEntry MCS_TABLE[] = {
    { -6.7, 0.15},  // CQI  0 – QPSK
    { -4.7, 0.23},  // CQI  1
    { -2.3, 0.38},  // CQI  2
    {  0.2, 0.60},  // CQI  3
    {  2.4, 0.88},  // CQI  4
    {  4.3, 1.18},  // CQI  5
    {  5.9, 1.48},  // CQI  6 – 16QAM starts
    {  8.1, 1.91},  // CQI  7
    { 10.3, 2.41},  // CQI  8
    { 11.7, 2.73},  // CQI  9
    { 14.1, 3.32},  // CQI 10
    { 16.3, 3.90},  // CQI 11 – 64QAM starts
    { 18.7, 4.52},  // CQI 12
    { 21.0, 5.12},  // CQI 13
    { 22.7, 5.55},  // CQI 14
};

/// Number of entries in @ref MCS_TABLE.
static constexpr uint32_t MCS_TABLE_SIZE = 15;

/**
 * @brief Map a SINR value to the highest supportable MCS / CQI index.
 *
 * Walks @ref MCS_TABLE from the lowest entry upward and returns the index
 * of the last entry whose @c sinr_min_db threshold is met.
 *
 * @param sinr_db  Received SINR in dB.
 * @return CQI index in [0, 14]. Returns 0 for any SINR below
 *         @ref SINR_MIN_DB (the link is at minimum viable throughput,
 *         not disconnected — use @ref SinrToCapacity to get 0 Mbps
 *         for truly unusable links).
 */
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

/**
 * @brief Compute link capacity in Mbps from SINR and bandwidth.
 *
 * Returns @c 0.0 for any SINR below @ref SINR_MIN_DB, regardless of model.
 *
 * **Shannon model** (@c "shannon")
 * Applies the Shannon-Hartley theorem:
 * @code
 *   C = bandwidth_hz * log2(1 + 10^(sinr_db / 10))   [bits/s]
 * @endcode
 * This is a theoretical upper bound. Real systems typically achieve
 * 60–80% of the Shannon limit.
 *
 * **Table model** (@c "table")
 * Looks up the spectral efficiency for the highest supportable CQI
 * index (via @ref SinrToMcsIndex) and multiplies by @c bandwidth_hz:
 * @code
 *   C = MCS_TABLE[SinrToMcsIndex(sinr_db)].spectral_eff * bandwidth_hz
 * @endcode
 *
 * @param sinr_db       Received SINR in dB.
 * @param bandwidth_hz  System bandwidth in Hz (e.g. 400e6 for 400 MHz).
 * @param amc_model     Capacity model: @c "shannon" or @c "table".
 * @return Capacity in Mbps, or @c 0.0 if @c sinr_db < @ref SINR_MIN_DB.
 * @throws std::runtime_error if @c amc_model is neither @c "shannon"
 *         nor @c "table".
 */
inline double
SinrToCapacity(double sinr_db, double bandwidth_hz, const std::string& amc_model)
{
    if (sinr_db < SINR_MIN_DB)
    {
        return 0.0;
    }

    if (amc_model == "shannon")
    {
        double sinr_linear  = std::pow(10.0, sinr_db / 10.0);
        double capacity_bps = bandwidth_hz * std::log2(1.0 + sinr_linear);
        return capacity_bps / 1e6;  // → Mbps
    }

    if (amc_model != "table")
    {
        throw std::runtime_error(
            "[LinkEvaluator] Unknown amc_model '" + amc_model +
            "'; expected 'shannon' or 'table'");
    }

    double spectral_eff = MCS_TABLE[SinrToMcsIndex(sinr_db)].spectral_eff;
    return spectral_eff * bandwidth_hz / 1e6;  // → Mbps
}

}  // namespace mesh_sim