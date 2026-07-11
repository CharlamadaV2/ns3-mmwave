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
 * | @c "table"    | 3GPP TS 38.214-style CQI table (16 entries). Spectral efficiency|
 * |               | steps match discrete MCS levels used by ns3-mmwave's AMC.       |
 * | @c "silvus"   | Silvus radio's native single-stream MCS table (MCS 0–6).        |
 * |               | SINR floor derived empirically from field network_status.csv,  |
 * |               | not from a formula or borrowed 3GPP table — the field data      |
 * |               | showed no per-level SINR gradient to fit, only a single usable  |
 * |               | floor. Throughput figures are the radio's published spec sheet  |
 * |               | numbers. Cannot represent the radio's 2-stream MIMO levels       |
 * |               | (MCS 8–14) since the sim has no multi-stream model.              |
 *
 * The MCS table is derived from 3GPP TS 38.214 Table 5.1.3.1-1 and uses
 * the same SINR thresholds and spectral efficiencies as
 * @c MmWaveAmc::CreateCqiFeedbackWbTdma() in ns3-mmwave.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
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
 * | 15    | 256QAM     | 25.0                | 6.23                      |
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
    { 25.0, 6.23},  // CQI 15
};

/// Number of entries in @ref MCS_TABLE.
static constexpr uint32_t MCS_TABLE_SIZE = 16;

/**
 * @brief One entry in the Silvus radio's native MCS table.
 *
 * The Silvus manual (Table 5) publishes MCS, streams, constellation, FEC
 * rate, and PHY throughput — but NO SINR column. @c sinr_min_db is therefore
 * a CALCULATED approximation, not a spec-sheet or field-measured value: each
 * single-stream level is matched to the 3GPP TS 38.214 CQI entry (@ref
 * MCS_TABLE) with the nearest spectral efficiency, and that entry's ~10%-BLER
 * SINR threshold is used. This grounds the ladder in the same link-level
 * curve as the rest of the sim. See @ref SILVUS_MCS_TABLE for the per-level
 * derivation and its limitations.
 */
struct SilvusMcsEntry
{
    double      sinr_min_db;      ///< Calculated SINR threshold for this level (dB); see struct note.
    uint32_t    spatial_streams;  ///< Number of spatial streams (1 or 2, MIMO).
    const char* constellation;    ///< Modulation constellation (e.g. "64QAM").
    const char* fec_rate;         ///< Forward error correction code rate (e.g. "3/4").
    double      phy_mbps_20mhz;   ///< PHY throughput at 20 MHz bandwidth, in Mbps.
};

/**
 * @brief Silvus radio native MCS table ("Table 5. Supported Modulation and
 * Coding Index (MCS)" from the Silvus radio manual).
 *
 * PHY-throughput, streams, constellation, and FEC columns are transcribed
 * verbatim from the manual. The SINR column is NOT in the manual — it is
 * CALCULATED by matching each single-stream level's spectral efficiency
 * (bits/symbol × code rate) to the nearest 3GPP TS 38.214 CQI entry
 * (@ref MCS_TABLE) and taking that entry's ~10%-BLER SINR threshold. MCS 0
 * is pinned to @ref SINR_MIN_DB as the minimum-viable level.
 *
 * | MCS | Str | Const. | FEC | PHY Mbps (20 MHz) | spec.eff | SINR thr (dB) |
 * |-----|-----|--------|-----|-------------------|----------|---------------|
 * | 0   | 1   | BPSK   | 1/2 | 6.5               | 0.50     | −6.7 (floor)  |
 * | 1   | 1   | QPSK   | 1/2 | 13                | 1.00     |  2.4          |
 * | 2   | 1   | QPSK   | 3/4 | 19.5              | 1.50     |  5.9          |
 * | 3   | 1   | 16QAM  | 1/2 | 26                | 2.00     |  8.1          |
 * | 4   | 1   | 16QAM  | 3/4 | 39                | 3.00     | 11.7          |
 * | 5   | 1   | 64QAM  | 2/3 | 52                | 4.00     | 16.3          |
 * | 6   | 1   | 64QAM  | 3/4 | 58.5              | 4.50     | 18.7          |
 * | 8–14| 2   | (MIMO) | ... | 13 … 117          | ...      | (unreachable) |
 *
 * The 2-stream rows (MCS 8–14) are carried for completeness but are NEVER
 * selected: @ref SinrToSilvusMcsIndex returns only 0–6 because the sim has
 * no multi-stream MIMO model. Their @c sinr_min_db values mirror the
 * single-stream modulation counterparts purely as documentation.
 *
 * Accuracy caveats:
 *  - Reported MCS is the *highest SINR-supportable* single-stream level — a
 *    link-capability ceiling under saturating load. Field radios select MCS
 *    by traffic load (rate adaptation), so field MCS often sits far below
 *    this at the same SINR. Sim MCS will over-predict field MCS; matching
 *    requires a rate-adaptation + traffic model the sim does not have.
 *  - Capping at single-stream MCS 6 (58.5 Mbps @ 20 MHz) means the sim
 *    cannot reach the field's 2-stream rates (up to 117 Mbps).
 */
static const SilvusMcsEntry SILVUS_MCS_TABLE[] = {
    {-6.7, 1, "BPSK",   "1/2", 6.5},    // MCS  0  (pinned to SINR floor)
    { 2.4, 1, "QPSK",   "1/2", 13.0},   // MCS  1  (spec.eff 1.00 → 3GPP CQI4)
    { 5.9, 1, "QPSK",   "3/4", 19.5},   // MCS  2  (spec.eff 1.50 → 3GPP CQI6)
    { 8.1, 1, "16QAM",  "1/2", 26.0},   // MCS  3  (spec.eff 2.00 → 3GPP CQI7)
    {11.7, 1, "16QAM",  "3/4", 39.0},   // MCS  4  (spec.eff 3.00 → 3GPP CQI9)
    {16.3, 1, "64QAM",  "2/3", 52.0},   // MCS  5  (spec.eff 4.00 → 3GPP CQI11)
    {18.7, 1, "64QAM",  "3/4", 58.5},   // MCS  6  (spec.eff 4.50 → 3GPP CQI12)
    // MCS index 7 intentionally omitted — not present in the radio's table.
    // MCS 8–14 (2-stream) are never selected (no MIMO model); SINR values
    // mirror the single-stream counterparts for documentation only.
    {-6.7, 2, "BPSK",   "1/2", 13.0},   // MCS  8
    { 2.4, 2, "QPSK",   "1/2", 26.0},   // MCS  9
    { 5.9, 2, "QPSK",   "3/4", 39.0},   // MCS 10
    { 8.1, 2, "16QAM",  "1/2", 52.0},   // MCS 11
    {11.7, 2, "16QAM",  "3/4", 78.0},   // MCS 12
    {16.3, 2, "64QAM",  "2/3", 104.0},  // MCS 13
    {18.7, 2, "64QAM",  "3/4", 117.0},  // MCS 14
};

/// Number of entries in @ref SILVUS_MCS_TABLE.
static constexpr uint32_t SILVUS_MCS_TABLE_SIZE = 14;

/**
 * @brief Look up a Silvus MCS table entry by its native MCS index.
 *
 * @param mcs_index  Silvus MCS index (0–6 or 8–14). Index 7 and any value
 *                   above 14 are invalid.
 * @return Pointer to the matching @ref SilvusMcsEntry, or @c nullptr if
 *         @p mcs_index is 7 or out of range [0, 14].
 */
inline const SilvusMcsEntry*
SilvusMcsLookup(uint32_t mcs_index)
{
    if (mcs_index == 7 || mcs_index > 14)
    {
        return nullptr;
    }
    uint32_t pos = (mcs_index <= 6) ? mcs_index : mcs_index - 1;
    if (pos >= SILVUS_MCS_TABLE_SIZE)
    {
        return nullptr;
    }
    return &SILVUS_MCS_TABLE[pos];
}

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
 * @brief Map a SINR value to the highest supportable single-stream Silvus
 * MCS index (0–6).
 *
 * Walks the single-stream rows of @ref SILVUS_MCS_TABLE (positions 0–6) and
 * returns the index of the last entry whose calculated @c sinr_min_db
 * threshold is met. The thresholds are the single source of truth for both
 * this function and the @c "silvus" branch of @ref SinrToCapacity, so the
 * reported MCS and the reported capacity are always mutually consistent.
 *
 * NOTE: this returns the highest SINR-supportable level (a capability
 * ceiling under load), NOT the field's traffic-driven selection — see the
 * accuracy caveats on @ref SILVUS_MCS_TABLE.
 *
 * @param sinr_db  Received SINR in dB.
 * @return Silvus MCS index in [0, 6]. Returns 0 for any SINR below
 *         @ref SINR_MIN_DB.
 */
inline uint32_t
SinrToSilvusMcsIndex(double sinr_db)
{
    if (sinr_db < SINR_MIN_DB)
    {
        return 0;
    }
    // Single-stream levels occupy positions 0-6 of SILVUS_MCS_TABLE.
    uint32_t idx = 0;
    for (uint32_t k = 0; k <= 6; ++k)
    {
        if (sinr_db >= SILVUS_MCS_TABLE[k].sinr_min_db)
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
 * @param sinr_db       Received SINR in dB.
 * @param bandwidth_hz  System bandwidth in Hz (e.g. 400e6 for 400 MHz).
 * @param amc_model     Capacity model: @c "shannon", @c "table", or @c "silvus".
 * @return Capacity in Mbps, or @c 0.0 if @c sinr_db < @ref SINR_MIN_DB.
 * @throws std::runtime_error if @c amc_model is none of the three above.
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

    else if (amc_model == "silvus")
    {
        // Table-driven: capacity is the PHY rate of the highest SINR-
        // supportable single-stream MCS, scaled from the table's 20 MHz
        // reference to the configured bandwidth. Uses the SAME thresholds as
        // SinrToSilvusMcsIndex, so capacity and reported MCS never disagree.
        // Naturally tops out at MCS 6 (58.5 * BW/20); no separate ceiling
        // needed. Linear BW scaling ignores fixed pilot/guard overhead — a
        // first-order approximation, same one the old ceiling made.
        uint32_t mcs = SinrToSilvusMcsIndex(sinr_db);      // 0..6
        const SilvusMcsEntry* e = SilvusMcsLookup(mcs);
        if (e == nullptr)
        {
            return 0.0;  // unreachable for 0..6, but guards against future edits
        }
        return e->phy_mbps_20mhz * (bandwidth_hz / 20e6);  // → Mbps
    }

    else if (amc_model != "table")
    {
        throw std::runtime_error(
            "[LinkEvaluator] Unknown amc_model '" + amc_model +
            "'; expected 'shannon', 'table', or 'silvus'");
    }

    double spectral_eff = MCS_TABLE[SinrToMcsIndex(sinr_db)].spectral_eff;
    return spectral_eff * bandwidth_hz / 1e6;  // → Mbps
}

}  // namespace mesh_sim