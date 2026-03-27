/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/io/metrics-writer.h"
#include "src/util/string-utils.h"
#include "third_party/json.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <utility>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace mmwave_sim
{

// ---------------------------------------------------------------------------
// Internal parsing types
//
// These structs model rows from ns-3 trace file formats and are private
// implementation details of MetricsWriter::Write().  They live here (not in a
// header) intentionally: no other translation unit needs them, and exposing
// them would add surface area without benefit.
// ---------------------------------------------------------------------------

struct PhyRow
{
    // Columns from RxPacketTrace.txt:
    // DL/UL  time  frame  subF  slot  1stSym  symbol#  cellId  rnti  ccId
    //   tbSize  mcs  rv  SINR(dB)  corrupt  TBler
    std::string direction;
    double      time       = 0.0;
    uint32_t    cell_id    = 0;
    uint32_t    rnti       = 0;
    double      sinr_db    = 0.0;
    int         corrupt    = 0;
    uint32_t    tb_size    = 0;
    int         mcs        = 0;
};

struct RlcRow
{
    // Columns from DlRlcStats.txt / UlRlcStats.txt (aggregated per epoch):
    // % start  end  CellId  IMSI  RNTI  LCID  nTxPDUs  TxBytes  nRxPDUs  RxBytes
    //   delay  stdDev  min  max  PduSize  stdDev  min  max
    double   start     = 0.0;
    double   end       = 0.0;
    uint32_t cell_id   = 0;
    uint64_t imsi      = 0;
    uint32_t rnti      = 0;
    uint64_t rx_bytes  = 0;
    double   delay_mean = 0.0;
};

// ---------------------------------------------------------------------------
// Per-UE accumulated metrics (over all non-warmup data)
// ---------------------------------------------------------------------------

struct UeMetrics
{
    std::string node_id;
    // SINR stats
    double      sinr_sum        = 0.0;
    double      sinr_min        = 1e9;
    double      sinr_max        = -1e9;
    uint64_t    sinr_count      = 0;
    uint64_t    corrupt_count   = 0;
    // Throughput (from RLC stats)
    uint64_t    dl_rx_bytes     = 0;
    double      dl_delay_sum    = 0.0;
    uint64_t    dl_delay_count  = 0;
    // LOS/NLOS from links.csv
    uint64_t    los_count       = 0;
    uint64_t    nlos_count      = 0;
};

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

struct LinkRow
{
    double      time_s  = 0.0;
    uint32_t    node_b  = 0;   // UE ns3 node ID
    std::string condition;     // "LOS" or "NLOS"
};

static std::vector<LinkRow>
parseLinksCSV(const std::string& path, double warmup_s)
{
    std::vector<LinkRow> rows;
    std::ifstream f(path);
    if (!f.is_open())
    {
        return rows;
    }

    std::string line;
    while (std::getline(f, line))
    {
        // Skip comments and header
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        if (line.find("time_s") != std::string::npos)
        {
            continue;
        }

        // time_s,node_a,node_b,dist_m,sinr_db,condition
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> cols;
        while (std::getline(ss, tok, ','))
        {
            cols.push_back(tok);
        }
        if (cols.size() < 6)
        {
            continue;
        }

        LinkRow r;
        r.time_s   = std::stod(cols[0]);
        if (r.time_s <= warmup_s)
        {
            continue;
        }
        r.node_b    = static_cast<uint32_t>(std::stoul(cols[2]));
        // Trim trailing \r / whitespace from last CSV column
        std::string cond = cols[5];
        while (!cond.empty() && (cond.back() == '\r' || cond.back() == '\n' ||
                                  cond.back() == ' '))
        {
            cond.pop_back();
        }
        r.condition = cond;
        rows.push_back(r);
    }
    return rows;
}

static std::vector<PhyRow>
parsePhyTrace(const std::string& path, double warmup_s)
{
    std::vector<PhyRow> rows;
    std::ifstream f(path);
    if (!f.is_open())
    {
        std::cerr << "[MetricsWriter] Warning: cannot open " << path << "\n";
        return rows;
    }

    std::string line;
    bool header_seen = false;

    while (std::getline(f, line))
    {
        if (line.empty())
        {
            continue;
        }
        // Header line starts with "DL/UL"
        if (!header_seen)
        {
            if (line.find("DL/UL") != std::string::npos)
            {
                header_seen = true;
                continue;
            }
            continue;
        }

        auto t = splitTab(line);
        if (t.size() < 16)
        {
            continue;
        }

        PhyRow r;
        r.direction = t[0];
        r.time      = std::stod(t[1]);
        if (r.time <= warmup_s)
        {
            continue;
        }
        r.cell_id   = static_cast<uint32_t>(std::stoul(t[7]));
        r.rnti      = static_cast<uint32_t>(std::stoul(t[8]));
        r.tb_size   = static_cast<uint32_t>(std::stoul(t[10]));
        r.mcs       = std::stoi(t[11]);
        r.sinr_db   = std::stod(t[13]);
        r.corrupt   = std::stoi(t[14]);
        rows.push_back(r);
    }
    return rows;
}

static std::vector<RlcRow>
parseRlcStats(const std::string& path, double warmup_s)
{
    std::vector<RlcRow> rows;
    std::ifstream f(path);
    if (!f.is_open())
    {
        std::cerr << "[MetricsWriter] Warning: cannot open " << path << "\n";
        return rows;
    }

    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '%')
        {
            continue;
        }

        auto t = splitTab(line);
        if (t.size() < 12)
        {
            continue;
        }

        RlcRow r;
        r.start      = std::stod(t[0]);
        r.end        = std::stod(t[1]);
        if (r.end <= warmup_s)
        {
            continue;
        }
        r.cell_id    = static_cast<uint32_t>(std::stoul(t[2]));
        r.imsi       = std::stoull(t[3]);
        r.rnti       = static_cast<uint32_t>(std::stoul(t[4]));
        r.rx_bytes   = std::stoull(t[9]);
        r.delay_mean = std::stod(t[10]);
        rows.push_back(r);
    }
    return rows;
}

// ---------------------------------------------------------------------------
// MetricsWriter::Write
// ---------------------------------------------------------------------------

MetricsWriter::MetricsWriter(const SimConfig& cfg)
    : m_cfg(cfg)
{
}

void
MetricsWriter::SetTiming(const TimingInfo& t)
{
    m_timing = t;
}

void
MetricsWriter::Write() const
{
    const std::string& outDir = m_cfg.output_dir;
    const double       warmup = m_cfg.warmup_s;

    // -----------------------------------------------------------------------
    // Build node-ID lookup tables
    // -----------------------------------------------------------------------
    // UEs are assigned IMSIs starting at 1, in the order they appear in nodes.json.
    // RNTIs in RxPacketTrace are per-eNB and may differ; we join via DlRlcStats.

    std::vector<std::string> ue_ids;  // index = IMSI-1
    for (const auto& n : m_cfg.nodes)
    {
        if (n.role == "ue")
        {
            ue_ids.push_back(n.id);
        }
    }

    // -----------------------------------------------------------------------
    // Parse trace files
    // -----------------------------------------------------------------------
    auto phyRows  = parsePhyTrace(outDir + "/RxPacketTrace.txt", warmup);
    auto dlRows   = parseRlcStats(outDir + "/DlRlcStats.txt",   warmup);
    auto ulRows   = parseRlcStats(outDir + "/UlRlcStats.txt",   warmup);
    auto linkRows = parseLinksCSV(outDir + "/links.csv",         warmup);

    // -----------------------------------------------------------------------
    // Build (cellId, RNTI) → IMSI mapping (from RLC stats, which have both)
    // RNTI is only unique within a cell, so we need the composite key.
    // -----------------------------------------------------------------------
    using CellRnti = std::pair<uint32_t, uint32_t>;
    std::map<CellRnti, uint64_t> rnti_to_imsi;
    for (const auto& r : dlRows)
    {
        rnti_to_imsi[{r.cell_id, r.rnti}] = r.imsi;
    }
    // Also check UL stats in case some (cell, RNTI) pairs only appear there
    for (const auto& r : ulRows)
    {
        auto key = CellRnti{r.cell_id, r.rnti};
        if (rnti_to_imsi.find(key) == rnti_to_imsi.end())
        {
            rnti_to_imsi[key] = r.imsi;
        }
    }

    // -----------------------------------------------------------------------
    // Build UE ns3-node-ID → IMSI mapping from links.csv
    // UE node IDs are assigned in order, matching IMSI 1, 2, …
    // -----------------------------------------------------------------------
    std::set<uint32_t> ueNodeIdSet;
    for (const auto& lr : linkRows)
    {
        ueNodeIdSet.insert(lr.node_b);
    }
    std::vector<uint32_t> ueNodeIds(ueNodeIdSet.begin(), ueNodeIdSet.end());
    // ueNodeIds is already sorted (std::set)
    std::map<uint32_t, uint64_t> nodeId_to_imsi;
    for (size_t i = 0; i < ueNodeIds.size(); i++)
    {
        nodeId_to_imsi[ueNodeIds[i]] = static_cast<uint64_t>(i + 1);
    }

    // -----------------------------------------------------------------------
    // Aggregate per UE (keyed by IMSI)
    // -----------------------------------------------------------------------
    std::map<uint64_t, UeMetrics> ueMap;  // IMSI → UeMetrics

    // Initialize per-IMSI entries
    for (size_t i = 0; i < ue_ids.size(); i++)
    {
        uint64_t imsi      = static_cast<uint64_t>(i + 1);
        ueMap[imsi].node_id = ue_ids[i];
    }

    // PHY trace → SINR stats (DL only)
    for (const auto& r : phyRows)
    {
        if (r.direction != "DL")
        {
            continue;
        }
        auto it = rnti_to_imsi.find({r.cell_id, r.rnti});
        if (it == rnti_to_imsi.end())
        {
            continue;
        }
        uint64_t imsi = it->second;
        if (ueMap.find(imsi) == ueMap.end())
        {
            continue;
        }

        UeMetrics& m = ueMap[imsi];
        m.sinr_sum += r.sinr_db;
        m.sinr_count++;
        m.sinr_min = std::min(m.sinr_min, r.sinr_db);
        m.sinr_max = std::max(m.sinr_max, r.sinr_db);
        if (r.corrupt)
        {
            m.corrupt_count++;
        }
    }

    // DL RLC stats → throughput + delay
    const double active_s = m_cfg.duration_s - warmup;
    for (const auto& r : dlRows)
    {
        if (ueMap.find(r.imsi) == ueMap.end())
        {
            continue;
        }
        UeMetrics& m = ueMap[r.imsi];
        m.dl_rx_bytes += r.rx_bytes;
        if (r.delay_mean > 0.0)
        {
            m.dl_delay_sum   += r.delay_mean;
            m.dl_delay_count++;
        }
    }

    // links.csv → LOS/NLOS fraction per UE
    for (const auto& lr : linkRows)
    {
        auto nit = nodeId_to_imsi.find(lr.node_b);
        if (nit == nodeId_to_imsi.end())
        {
            continue;
        }
        uint64_t imsi = nit->second;
        if (ueMap.find(imsi) == ueMap.end())
        {
            continue;
        }
        UeMetrics& m = ueMap[imsi];
        if (lr.condition == "LOS")
        {
            m.los_count++;
        }
        else
        {
            m.nlos_count++;
        }
    }

    json out;
    out["scenario"]   = m_cfg.scenario_name;
    out["seed"]       = m_cfg.seed;
    out["run_id"]     = m_cfg.run_id;
    out["duration_s"] = m_cfg.duration_s;
    out["warmup_s"]   = m_cfg.warmup_s;

    // Wall-clock timing (populated by sim.cc via SetTiming)
    if (m_timing.elapsed_s > 0.0)
    {
        out["wall_clock_start"] = toIso8601(m_timing.start);
        out["wall_clock_end"]   = toIso8601(m_timing.end);
        out["wall_elapsed_s"]   = m_timing.elapsed_s;
    }

    // Per-UE metrics
    json per_ue = json::object();
    double net_sinr_sum = 0.0;
    double net_sinr_min = 1e9;
    double net_sinr_max = -1e9;
    uint64_t net_sinr_count  = 0;
    uint64_t net_corrupt     = 0;
    uint64_t net_sinr_total  = 0;
    double   net_throughput  = 0.0;
    uint64_t net_los_count   = 0;
    uint64_t net_nlos_count  = 0;

    for (const auto& kv : ueMap)
    {
        const UeMetrics& m = kv.second;
        json u;

        // SINR
        double mean_sinr = (m.sinr_count > 0)
                               ? (m.sinr_sum / static_cast<double>(m.sinr_count))
                               : std::numeric_limits<double>::quiet_NaN();
        double min_sinr  = (m.sinr_count > 0) ? m.sinr_min
                                               : std::numeric_limits<double>::quiet_NaN();
        double max_sinr  = (m.sinr_count > 0) ? m.sinr_max
                                               : std::numeric_limits<double>::quiet_NaN();
        double corrupt_rate = (m.sinr_count > 0)
                                  ? (static_cast<double>(m.corrupt_count) /
                                     static_cast<double>(m.sinr_count))
                                  : std::numeric_limits<double>::quiet_NaN();
        uint64_t link_probes = m.los_count + m.nlos_count;
        double los_frac = (link_probes > 0)
                              ? (static_cast<double>(m.los_count) /
                                 static_cast<double>(link_probes))
                              : std::numeric_limits<double>::quiet_NaN();

        u["mean_sinr_db"]    = std::isnan(mean_sinr)    ? nullptr : json(mean_sinr);
        u["min_sinr_db"]     = std::isnan(min_sinr)     ? nullptr : json(min_sinr);
        u["max_sinr_db"]     = std::isnan(max_sinr)     ? nullptr : json(max_sinr);
        u["corruption_rate"] = std::isnan(corrupt_rate) ? nullptr : json(corrupt_rate);
        u["los_fraction"]    = std::isnan(los_frac)     ? nullptr : json(los_frac);

        // DL throughput
        double dl_tput_mbps = (active_s > 0.0)
                                  ? (static_cast<double>(m.dl_rx_bytes) * 8.0 /
                                     active_s / 1e6)
                                  : 0.0;
        u["dl_throughput_mbps"] = dl_tput_mbps;

        // DL delay
        double dl_delay_ms = (m.dl_delay_count > 0)
                                 ? (m.dl_delay_sum /
                                    static_cast<double>(m.dl_delay_count) * 1e3)
                                 : std::numeric_limits<double>::quiet_NaN();
        u["dl_delay_mean_ms"] = std::isnan(dl_delay_ms) ? nullptr : json(dl_delay_ms);

        per_ue[m.node_id] = u;

        // Network-level accumulation
        if (m.sinr_count > 0)
        {
            net_sinr_sum   += mean_sinr;
            net_sinr_count++;
            net_sinr_min    = std::min(net_sinr_min, min_sinr);
            if (!std::isnan(max_sinr))
            {
                net_sinr_max = std::max(net_sinr_max, max_sinr);
            }
            net_corrupt    += m.corrupt_count;
            net_sinr_total += m.sinr_count;
        }
        net_los_count  += m.los_count;
        net_nlos_count += m.nlos_count;
        net_throughput += dl_tput_mbps;
    }

    out["per_ue"] = per_ue;

    // Network summary
    json net;
    net["mean_sinr_db"]          = (net_sinr_count > 0)
                                       ? json(net_sinr_sum /
                                              static_cast<double>(net_sinr_count))
                                       : nullptr;
    net["min_sinr_db"]            = (net_sinr_count > 0) ? json(net_sinr_min) : nullptr;
    net["max_sinr_db"]            = (net_sinr_count > 0) ? json(net_sinr_max) : nullptr;
    net["corruption_rate"]        = (net_sinr_total > 0)
                                        ? json(static_cast<double>(net_corrupt) /
                                               static_cast<double>(net_sinr_total))
                                        : nullptr;
    uint64_t net_link_probes = net_los_count + net_nlos_count;
    net["los_fraction"]           = (net_link_probes > 0)
                                        ? json(static_cast<double>(net_los_count) /
                                               static_cast<double>(net_link_probes))
                                        : nullptr;
    net["sum_dl_throughput_mbps"] = net_throughput;
    out["network"] = net;

    // -----------------------------------------------------------------------
    // Write to file
    // -----------------------------------------------------------------------
    std::string summary_path = outDir + "/summary.json";
    std::ofstream sf(summary_path);
    if (!sf.is_open())
    {
        std::cerr << "[MetricsWriter] ERROR: cannot write " << summary_path << "\n";
        return;
    }
    sf << out.dump(2) << "\n";
    std::cout << "[MetricsWriter] Summary written to " << summary_path << "\n";
}

}  // namespace mmwave_sim
