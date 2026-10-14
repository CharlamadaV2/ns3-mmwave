/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file eval-jammer-test.cc
 * @brief Unit tests for sinr-capacity.h (Silvus table and table invariants),
 *        LinkTable edge cases, LinkEvaluator and JammerModel.
 *
 * Standalone binary. link-evaluator.cc and jammer-model.cc are compiled
 * against deterministic ns-3 stubs (stubs/ns3-eval-jammer-stub.h): tests set
 * node and jammer positions, the propagation model's received power and the
 * LOS condition directly, so the physics formulas can be checked exactly.
 * Build and run: `make -C tests/unit/eval-jammer test`.
 */

#include "src/eval/link-evaluator.h"
#include "src/eval/link-table.h"
#include "src/eval/sinr-capacity.h"
#include "src/jammer/jammer-model.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace mesh_sim;

// ---- helpers ----

static int g_pass = 0;
static int g_fail = 0;

static void
check(bool cond, const std::string& name)
{
    if (cond)
    {
        ++g_pass;
    }
    else
    {
        ++g_fail;
        std::cerr << "FAIL: " << name << "\n";
    }
}

static bool
approx(double a, double b, double tol = 1e-9)
{
    return std::fabs(a - b) <= tol * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

static std::string
num(double v)
{
    std::ostringstream os;
    os.precision(12);
    os << v;
    return os.str();
}

static double
DbmToW(double dbm)
{
    return std::pow(10.0, (dbm - 30.0) / 10.0);
}

/// Free-space path loss with the exact speed of light, distance floored at 1 m.
static double
Fspl(double d, double fHz)
{
    const double c = 299792458.0;
    const double dd = std::max(d, 1.0);
    return 20.0 * std::log10(4.0 * M_PI * dd * fHz / c);
}

static double
NoiseFloorDbm(double bwMhz, double nfDb)
{
    return -174.0 + 10.0 * std::log10(bwMhz * 1e6) + nfDb;
}

// ---- ns-3 fakes (built on stubs/ns3-eval-jammer-stub.h) ----

using MobPtr = ns3::Ptr<ns3::MobilityModel>;

static MobPtr
Mob(double x, double y, double z = 0.0)
{
    return ns3::CreateObject<ns3::MobilityModel>(ns3::Vector(x, y, z));
}

struct LossCall
{
    double      txDbm;
    ns3::Vector a;
    ns3::Vector b;
};

/// Propagation model: rxPower = tx - lossDb unless fn is set.
class FakeLoss : public ns3::PropagationLossModel
{
  public:
    double lossDb = 100.0;
    std::function<double(double, const ns3::Vector&, const ns3::Vector&)> fn;
    mutable std::vector<LossCall> calls;

    double CalcRxPower(double txPowerDbm, MobPtr a, MobPtr b) const override
    {
        calls.push_back({txPowerDbm, a->RawPosition(), b->RawPosition()});
        if (fn)
        {
            return fn(txPowerDbm, a->RawPosition(), b->RawPosition());
        }
        return txPowerDbm - lossDb;
    }
};

class FakeCond : public ns3::ChannelConditionModel
{
  public:
    ns3::ChannelCondition::LosConditionValue value = ns3::ChannelCondition::LOS;
    mutable std::vector<std::pair<ns3::Vector, ns3::Vector>> calls;

    ns3::Ptr<ns3::ChannelCondition> GetChannelCondition(
        ns3::Ptr<const ns3::MobilityModel> a,
        ns3::Ptr<const ns3::MobilityModel> b) const override
    {
        calls.emplace_back(a->RawPosition(), b->RawPosition());
        return ns3::CreateObject<ns3::ChannelCondition>(value);
    }
};

static bool
SamePos(const ns3::Vector& a, const ns3::Vector& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

/// Base config: 2.4 GHz, 20 MHz, NF 5 dB, 30 dBm, zero default gains.
static SimConfig
BaseCfg(uint32_t numNodes, const std::string& amc = "shannon")
{
    SimConfig cfg;
    cfg.channel.frequency_ghz     = 2.4;
    cfg.channel.bandwidth_mhz     = 20.0;
    cfg.channel.noise_figure_db   = 5.0;
    cfg.channel.tx_power_dbm      = 30.0;
    cfg.channel.tx_array_gain_dbi = 0.0;
    cfg.channel.rx_array_gain_dbi = 0.0;
    cfg.channel.amc_model         = amc;
    for (uint32_t i = 0; i < numNodes; ++i)
    {
        NodeSpec n;
        n.id = "n" + std::to_string(i);
        cfg.nodes.push_back(n);
    }
    return cfg;
}

/// Jammer with the ConfigLoader defaults, but enabled.
static JammerSpec
Jam(const std::string& id = "j")
{
    JammerSpec s{};
    s.enabled           = true;
    s.id                = id;
    s.type              = "constant";
    s.tx_power_dbm      = 25.0;
    s.tx_array_gain_dbi = 12.0;
    s.duty_cycle        = 1.0;
    s.max_range_m       = 0.0;
    s.beamwidth_deg     = 360.0;
    s.azimuth_deg       = 0.0;
    s.zenith_deg        = 0.0;
    return s;
}

/// Jammer power (W) seen by a single receiver from one fresh model.
static double
JamPowerAt(const JammerSpec& spec, const MobPtr& jamMob, const MobPtr& rx, double nowS = 0.0,
           double carrierHz = 0.0, uint32_t seed = 1, double lossDb = 80.0)
{
    auto loss = ns3::CreateObject<FakeLoss>();
    loss->lossDb = lossDb;
    JammerModel jm;
    jm.Configure({spec}, loss, {jamMob}, carrierHz, seed);
    return jm.InterfPowerAtReceiver(rx, nowS);
}

// =====================================================================
// sinr-capacity.h
// =====================================================================

static void
test_silvus_mcs_index_thresholds()
{
    check(SinrToSilvusMcsIndex(-100.0) == 0, "silvus idx far below min -> 0");
    check(SinrToSilvusMcsIndex(std::nextafter(SINR_MIN_DB, -1e9)) == 0,
          "silvus idx just below min -> 0");
    // Exact thresholds select their level; just below selects the previous one.
    for (uint32_t k = 0; k <= 6; ++k)
    {
        const double thr = SILVUS_MCS_TABLE[k].sinr_min_db;
        check(SinrToSilvusMcsIndex(thr) == k, "silvus idx at threshold of MCS " + std::to_string(k));
        if (k > 0)
        {
            check(SinrToSilvusMcsIndex(std::nextafter(thr, -1e9)) == k - 1,
                  "silvus idx just below threshold of MCS " + std::to_string(k));
        }
    }
    // Mid-range: 10 dB is above 8.1 (MCS 3) and below 11.7 (MCS 4).
    check(SinrToSilvusMcsIndex(10.0) == 3, "silvus idx at 10 dB is 3");
    check(SinrToSilvusMcsIndex(0.0) == 0, "silvus idx at 0 dB is 0 (below 2.4)");
}

static void
test_silvus_mcs_index_never_mimo()
{
    // Doc: returns only 0-6; 2-stream MCS 8-14 are never selected.
    bool ok = true;
    uint32_t prev = 0;
    bool mono = true;
    for (double s = -20.0; s <= 200.0; s += 0.05)
    {
        const uint32_t idx = SinrToSilvusMcsIndex(s);
        if (idx > 6)
        {
            ok = false;
        }
        if (idx < prev)
        {
            mono = false;
        }
        prev = idx;
    }
    check(ok, "silvus idx always in [0,6]");
    check(mono, "silvus idx monotonic non-decreasing in SINR");
    check(SinrToSilvusMcsIndex(1e6) == 6, "silvus idx at huge SINR is 6");
}

static void
test_silvus_mcs_lookup()
{
    for (uint32_t k = 0; k <= 6; ++k)
    {
        check(SilvusMcsLookup(k) == &SILVUS_MCS_TABLE[k],
              "SilvusMcsLookup(" + std::to_string(k) + ") is table position " + std::to_string(k));
    }
    check(SilvusMcsLookup(7) == nullptr, "SilvusMcsLookup(7) is null (no MCS 7)");
    for (uint32_t k = 8; k <= 14; ++k)
    {
        check(SilvusMcsLookup(k) == &SILVUS_MCS_TABLE[k - 1],
              "SilvusMcsLookup(" + std::to_string(k) + ") skips MCS 7");
    }
    const SilvusMcsEntry* e8 = SilvusMcsLookup(8);
    check(e8 && e8->spatial_streams == 2 && approx(e8->phy_mbps_20mhz, 13.0),
          "MCS 8 is 2-stream BPSK at 13 Mbps");
    const SilvusMcsEntry* e14 = SilvusMcsLookup(14);
    check(e14 && e14->spatial_streams == 2 && approx(e14->phy_mbps_20mhz, 117.0),
          "MCS 14 is 2-stream 64QAM 3/4 at 117 Mbps");
    check(SilvusMcsLookup(15) == nullptr, "SilvusMcsLookup(15) is null");
    check(SilvusMcsLookup(std::numeric_limits<uint32_t>::max()) == nullptr,
          "SilvusMcsLookup(UINT32_MAX) is null");
}

static void
test_silvus_capacity()
{
    // Values at 20 MHz are the spec-sheet PHY rates (sinr-capacity.h table).
    check(approx(SinrToCapacity(SINR_MIN_DB, 20e6, "silvus"), 6.5), "silvus at SINR_MIN 20MHz = 6.5");
    check(approx(SinrToCapacity(2.4, 20e6, "silvus"), 13.0), "silvus at 2.4 dB 20MHz = 13");
    check(approx(SinrToCapacity(10.0, 20e6, "silvus"), 26.0), "silvus at 10 dB 20MHz = 26");
    check(approx(SinrToCapacity(18.7, 20e6, "silvus"), 58.5), "silvus at 18.7 dB 20MHz = 58.5");
    check(approx(SinrToCapacity(60.0, 20e6, "silvus"), 58.5), "silvus caps at MCS 6 (58.5) at 20MHz");
    // Linear bandwidth scaling.
    check(approx(SinrToCapacity(60.0, 40e6, "silvus"), 117.0), "silvus at 40 MHz doubles MCS 6");
    check(approx(SinrToCapacity(2.4, 5e6, "silvus"), 3.25), "silvus at 5 MHz quarters MCS 1");
    check(SinrToCapacity(std::nextafter(SINR_MIN_DB, -1e9), 20e6, "silvus") == 0.0,
          "silvus just below SINR_MIN is 0");
    // Capacity and reported MCS agree everywhere.
    bool consistent = true;
    for (double s = -6.7; s <= 40.0; s += 0.05)
    {
        const uint32_t idx = SinrToSilvusMcsIndex(s);
        const double expect = SILVUS_MCS_TABLE[idx].phy_mbps_20mhz * (10e6 / 20e6);
        if (!approx(SinrToCapacity(s, 10e6, "silvus"), expect))
        {
            consistent = false;
        }
    }
    check(consistent, "silvus capacity == PHY rate of SinrToSilvusMcsIndex scaled by BW");
}

static void
test_silvus_table_invariants()
{
    check(sizeof(SILVUS_MCS_TABLE) / sizeof(SILVUS_MCS_TABLE[0]) == SILVUS_MCS_TABLE_SIZE,
          "SILVUS_MCS_TABLE_SIZE matches array");
    check(SILVUS_MCS_TABLE[0].sinr_min_db == SINR_MIN_DB, "Silvus MCS 0 pinned to SINR_MIN_DB");

    // Single-stream spectral efficiency (bits/symbol x code rate) from the doc table.
    const double specEff[7] = {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5};
    bool inc = true, rate = true, oneStream = true, nearest = true;
    for (uint32_t k = 0; k <= 6; ++k)
    {
        const auto& e = SILVUS_MCS_TABLE[k];
        oneStream = oneStream && e.spatial_streams == 1;
        if (k > 0)
        {
            inc = inc && e.sinr_min_db > SILVUS_MCS_TABLE[k - 1].sinr_min_db;
            inc = inc && e.phy_mbps_20mhz > SILVUS_MCS_TABLE[k - 1].phy_mbps_20mhz;
            // Doc: threshold = threshold of the 3GPP CQI entry with nearest spectral eff.
            uint32_t best = 0;
            for (uint32_t c = 1; c < MCS_TABLE_SIZE; ++c)
            {
                if (std::fabs(MCS_TABLE[c].spectral_eff - specEff[k]) <
                    std::fabs(MCS_TABLE[best].spectral_eff - specEff[k]))
                {
                    best = c;
                }
            }
            if (MCS_TABLE[best].sinr_min_db != e.sinr_min_db)
            {
                nearest = false;
                std::cerr << "  silvus MCS " << k << " thr " << e.sinr_min_db
                          << " but nearest CQI " << best << " thr " << MCS_TABLE[best].sinr_min_db
                          << "\n";
            }
        }
        // 20 MHz PHY rate is 13 Mbps per bit/s/Hz for one stream.
        rate = rate && approx(e.phy_mbps_20mhz, 13.0 * specEff[k]);
    }
    check(oneStream, "Silvus positions 0-6 are single-stream");
    check(inc, "Silvus single-stream thresholds and rates strictly increasing");
    check(nearest, "Silvus thresholds match nearest-spectral-efficiency CQI entry");
    check(rate, "Silvus single-stream PHY rate == 13 * spectral efficiency");

    // 2-stream rows (positions 7-13) mirror single-stream thresholds, double rate.
    bool mirror = true;
    for (uint32_t k = 0; k <= 6; ++k)
    {
        const auto& s1 = SILVUS_MCS_TABLE[k];
        const auto& s2 = SILVUS_MCS_TABLE[k + 7];
        mirror = mirror && s2.spatial_streams == 2 && s2.sinr_min_db == s1.sinr_min_db &&
                 approx(s2.phy_mbps_20mhz, 2.0 * s1.phy_mbps_20mhz) &&
                 std::string(s2.constellation) == s1.constellation &&
                 std::string(s2.fec_rate) == s1.fec_rate;
    }
    check(mirror, "Silvus 2-stream rows mirror single-stream threshold/modulation at 2x rate");
}

static void
test_capacity_bandwidth_scaling()
{
    for (double s : {-6.7, 0.0, 7.3, 25.0, 40.0})
    {
        check(approx(SinrToCapacity(s, 40e6, "shannon"), 2.0 * SinrToCapacity(s, 20e6, "shannon")),
              "shannon linear in bandwidth at " + num(s));
        check(approx(SinrToCapacity(s, 40e6, "table"), 2.0 * SinrToCapacity(s, 20e6, "table")),
              "table linear in bandwidth at " + num(s));
    }
    // At exactly SINR_MIN_DB every model gives positive capacity.
    check(approx(SinrToCapacity(SINR_MIN_DB, 100e6, "table"), 15.0), "table at SINR_MIN = 0.15*BW");
    const double shannonMin = 100.0 * std::log2(1.0 + std::pow(10.0, -0.67));
    check(approx(SinrToCapacity(SINR_MIN_DB, 100e6, "shannon"), shannonMin),
          "shannon at SINR_MIN = B*log2(1+10^-0.67)");
    check(SinrToCapacity(30.0, 0.0, "shannon") == 0.0, "zero bandwidth -> zero capacity");
}

// =====================================================================
// LinkTable edge cases
// =====================================================================

static LinkResult
LR(uint32_t tx, uint32_t rx, double sinr, double cap)
{
    LinkResult r;
    r.tx_id = tx;
    r.rx_id = rx;
    r.sinr_db = sinr;
    r.capacity_mbps = cap;
    return r;
}

static void
test_link_table_two_nodes_and_mirror()
{
    LinkTable t;
    t.Update(2, {LR(0, 1, 12.5, 77.0)});
    check(t.NumNodes() == 2, "N=2 NumNodes");
    check(t.Get(0, 1).sinr_db == 12.5 && t.Get(1, 0).sinr_db == 12.5, "N=2 mirrored SINR");
    check(t.Get(1, 0).capacity_mbps == 77.0, "N=2 mirrored capacity");
    // Mirror stores the same struct; ids are not swapped.
    check(t.Get(1, 0).tx_id == 0 && t.Get(1, 0).rx_id == 1, "mirror stores the result verbatim");
    check(t.ConnectedLinkCount() == 1, "N=2 one connected link (not 2)");
    check(t.MaxCapacity() == 77.0, "N=2 MaxCapacity");
    check(t.Get(0, 0).sinr_db == -999.0 && t.Get(1, 1).sinr_db == -999.0, "N=2 diagonal default");

    // A result given with tx_id > rx_id is still mirrored to both cells.
    LinkTable u;
    u.Update(3, {LR(1, 0, 1.0, 1.0), LR(2, 0, 2.0, 2.0), LR(2, 1, 3.0, 3.0)});
    check(u.Get(0, 2).sinr_db == 2.0 && u.Get(2, 0).sinr_db == 2.0, "reversed-order result mirrored");
    check(u.Get(0, 1).sinr_db == 1.0 && u.Get(1, 2).sinr_db == 3.0, "reversed-order results upper triangle");
    check(u.MaxCapacity() == 3.0, "MaxCapacity sees reversed-order results");
}

static void
test_link_table_threshold_boundary()
{
    LinkTable t;
    t.Update(3, {LR(0, 1, SINR_MIN_DB, 1), LR(0, 2, std::nextafter(SINR_MIN_DB, -1e9), 1),
                 LR(1, 2, 0.0, 1)});
    check(t.ConnectedLinkCount() == 2, "default threshold equals SINR_MIN_DB (inclusive)");
    check(t.IsConnected(0, 1), "IsConnected default inclusive at SINR_MIN_DB");
    check(t.IsConnected(1, 0), "IsConnected symmetric");
    check(!t.IsConnected(0, 2), "IsConnected false just below SINR_MIN_DB");
    check(!t.IsConnected(1, 1), "diagonal never connected at default threshold");
    check(t.ConnectedLinkCount(0.0) == 1, "custom threshold inclusive");
    check(t.ConnectedLinkCount(-1000.0) == 3, "diagonal not counted at very low threshold");
}

// =====================================================================
// LinkEvaluator
// =====================================================================

struct Rig
{
    ns3::Ptr<FakeLoss> loss = ns3::CreateObject<FakeLoss>();
    ns3::Ptr<FakeCond> cond = ns3::CreateObject<FakeCond>();
    LinkEvaluator ev;
};

static void
test_le_noise_floor_and_snr()
{
    Rig g;
    SimConfig cfg = BaseCfg(2);
    cfg.channel.noise_figure_db   = 7.0;
    cfg.channel.tx_power_dbm      = 23.0;
    cfg.channel.tx_array_gain_dbi = 5.0;
    cfg.channel.rx_array_gain_dbi = 6.0;
    g.loss->lossDb = 120.0;  // well above FSPL(100 m, 2.4 GHz) ~ 80 dB
    g.ev.Configure(cfg, g.loss, g.cond);

    auto a = Mob(0, 0, 0), b = Mob(100, 0, 0);
    LinkResult r = g.ev.Evaluate(a, b, 0, 1);
    const double floor = NoiseFloorDbm(20.0, 7.0);
    check(r.tx_id == 0 && r.rx_id == 1, "Evaluate sets tx_id/rx_id");
    check(approx(r.distance_m, 100.0), "Evaluate distance");
    check(approx(r.path_loss_db, 120.0), "path loss = model loss when above FSPL");
    check(approx(r.rx_power_dbm, 23.0 - 120.0 + 5.0 + 6.0), "rx_power = tx - PL + txGain + rxGain");
    check(approx(r.sinr_db, r.rx_power_dbm - floor),
          "SINR = rx_power - (-174 + 10log10(BW) + NF); got " + num(r.sinr_db) + " expected " +
              num(r.rx_power_dbm - floor));
    check(approx(r.capacity_mbps, SinrToCapacity(r.sinr_db, 20e6, "shannon")), "capacity via SinrToCapacity");
    check(r.mcs_index == SinrToMcsIndex(r.sinr_db), "mcs via SinrToMcsIndex (shannon)");
    check(r.is_los, "LOS condition reported");
    check(!r.condition_from_buildings, "no buildings -> condition_from_buildings false");
    // The model is called with the mesh TX power and (tx, rx) order.
    check(g.loss->calls.size() == 1 && approx(g.loss->calls[0].txDbm, 23.0) &&
              SamePos(g.loss->calls[0].a, a->RawPosition()) &&
              SamePos(g.loss->calls[0].b, b->RawPosition()),
          "CalcRxPower called once with (txPower, tx, rx)");
    check(g.cond->calls.size() == 1 && SamePos(g.cond->calls[0].first, a->RawPosition()),
          "condition model called once with (tx, rx)");
}

static void
test_le_distance_is_3d()
{
    Rig g;
    SimConfig cfg = BaseCfg(2);
    g.ev.Configure(cfg, g.loss, g.cond);
    LinkResult r = g.ev.Evaluate(Mob(1, 2, 3), Mob(4, 6, 15), 0, 1);
    check(approx(r.distance_m, 13.0), "distance is 3-D Euclidean (3-4-12 -> 13)");
}

static void
test_le_per_node_gain_overrides()
{
    Rig g;
    SimConfig cfg = BaseCfg(2);
    cfg.channel.tx_array_gain_dbi = 5.0;
    cfg.channel.rx_array_gain_dbi = 6.0;
    cfg.nodes[0].tx_array_gain_dbi = 3.0;
    cfg.nodes[0].rx_array_gain_dbi = 4.0;
    cfg.nodes[1].rx_array_gain_dbi = 9.0;  // tx gain stays at channel default 5
    g.ev.Configure(cfg, g.loss, g.cond);
    auto a = Mob(0, 0, 0), b = Mob(100, 0, 0);
    const double base = 30.0 - 100.0;
    LinkResult r01 = g.ev.Evaluate(a, b, 0, 1);
    LinkResult r10 = g.ev.Evaluate(b, a, 1, 0);
    check(approx(r01.rx_power_dbm, base + 3.0 + 9.0), "0->1 uses tx override of 0 and rx override of 1");
    check(approx(r10.rx_power_dbm, base + 5.0 + 4.0), "1->0 uses tx default of 1 and rx override of 0");

    // Re-Configure rebuilds the gain vectors rather than appending.
    SimConfig cfg2 = BaseCfg(3);
    cfg2.channel.tx_array_gain_dbi = 1.0;
    cfg2.channel.rx_array_gain_dbi = 2.0;
    g.ev.Configure(cfg2, g.loss, g.cond);
    LinkResult r02 = g.ev.Evaluate(a, Mob(0, 100, 0), 0, 2);
    check(approx(r02.rx_power_dbm, base + 1.0 + 2.0), "re-Configure resets per-node gains");
}

static void
test_le_fspl_floor()
{
    Rig g;
    SimConfig cfg = BaseCfg(2);
    g.ev.Configure(cfg, g.loss, g.cond);
    const double f = 2.4e9;

    g.loss->lossDb = 10.0;  // below free space at 100 m
    LinkResult r = g.ev.Evaluate(Mob(0, 0, 0), Mob(100, 0, 0), 0, 1);
    check(approx(r.path_loss_db, Fspl(100.0, f), 1e-7),
          "PL floored at FSPL(100 m, 2.4 GHz): got " + num(r.path_loss_db) + " expected " +
              num(Fspl(100.0, f)));
    check(approx(r.sinr_db, 30.0 - Fspl(100.0, f) - NoiseFloorDbm(20.0, 5.0), 1e-7),
          "SINR uses floored PL");

    g.loss->lossDb = 0.0;  // model gives no loss at 0.5 m
    r = g.ev.Evaluate(Mob(0, 0, 0), Mob(0.5, 0, 0), 0, 1);
    check(approx(r.distance_m, 0.5), "distance_m is the real distance below 1 m");
    check(approx(r.path_loss_db, Fspl(1.0, f), 1e-7), "PL floored at FSPL(1 m) below 1 m");

    g.loss->lossDb = -20.0;  // negative model loss (gain) at short range
    r = g.ev.Evaluate(Mob(0, 0, 0), Mob(3, 0, 0), 0, 1);
    check(approx(r.path_loss_db, Fspl(3.0, f), 1e-7), "negative model loss floored at FSPL");

    // d = 0: non-finite model output is replaced by FSPL(1 m).
    const double nonFinite[] = {-std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::quiet_NaN()};
    const char* names[] = {"-inf", "+inf", "NaN"};
    for (int k = 0; k < 3; ++k)
    {
        const double v = nonFinite[k];
        g.loss->fn = [v](double, const ns3::Vector&, const ns3::Vector&) { return v; };
        r = g.ev.Evaluate(Mob(5, 5, 5), Mob(5, 5, 5), 0, 1);
        check(approx(r.path_loss_db, Fspl(1.0, f), 1e-7) && std::isfinite(r.sinr_db),
              std::string("d=0 with rxPower ") + names[k] + " -> PL = FSPL(1 m), finite SINR");
    }
    g.loss->fn = nullptr;

    // FSPL depends on the configured carrier.
    SimConfig cfg28 = BaseCfg(2);
    cfg28.channel.frequency_ghz = 28.0;
    g.ev.Configure(cfg28, g.loss, g.cond);
    g.loss->lossDb = 10.0;
    r = g.ev.Evaluate(Mob(0, 0, 0), Mob(100, 0, 0), 0, 1);
    check(approx(r.path_loss_db, Fspl(100.0, 28e9), 1e-7), "FSPL floor uses frequency_ghz (28 GHz)");
}

static void
test_le_los_mapping()
{
    Rig g;
    SimConfig cfg = BaseCfg(2);
    g.ev.Configure(cfg, g.loss, g.cond);
    const ns3::ChannelCondition::LosConditionValue vals[] = {
        ns3::ChannelCondition::LOS, ns3::ChannelCondition::NLOS, ns3::ChannelCondition::NLOSv,
        ns3::ChannelCondition::LC_ND};
    const bool expect[] = {true, false, false, false};
    const char* names[] = {"LOS", "NLOS", "NLOSv", "LC_ND"};
    for (int k = 0; k < 4; ++k)
    {
        g.cond->value = vals[k];
        LinkResult r = g.ev.Evaluate(Mob(0, 0, 0), Mob(100, 0, 0), 0, 1);
        check(r.is_los == expect[k], std::string("is_los for ") + names[k]);
    }
}

static void
test_le_buildings_flag()
{
    Rig g;
    SimConfig cfg = BaseCfg(3);
    cfg.buildings.push_back(BuildingSpec{});
    g.ev.Configure(cfg, g.loss, g.cond);
    auto res = g.ev.EvaluateAll({Mob(0, 0), Mob(100, 0), Mob(0, 100)});
    bool all = res.size() == 3;
    for (const auto& r : res)
    {
        all = all && r.condition_from_buildings;
    }
    check(all, "buildings present -> condition_from_buildings true on every link");
}

static void
test_le_amc_models()
{
    // Fixed loss chosen so SINR ~ 10 dB: table idx 7, silvus idx 3 (they differ).
    const double floor = NoiseFloorDbm(20.0, 5.0);
    const double lossFor10dB = 30.0 - floor - 10.0;
    for (const std::string amc : {"shannon", "table", "silvus"})
    {
        Rig g;
        SimConfig cfg = BaseCfg(2, amc);
        g.loss->lossDb = lossFor10dB;
        g.ev.Configure(cfg, g.loss, g.cond);
        LinkResult r = g.ev.Evaluate(Mob(0, 0, 0), Mob(100, 0, 0), 0, 1);
        check(approx(r.sinr_db, 10.0, 1e-9), amc + ": SINR set to 10 dB by fixed loss");
        check(approx(r.capacity_mbps, SinrToCapacity(r.sinr_db, 20e6, amc)), amc + ": capacity");
        const uint32_t want = (amc == "silvus") ? SinrToSilvusMcsIndex(r.sinr_db) : SinrToMcsIndex(r.sinr_db);
        check(r.mcs_index == want, amc + ": mcs_index from the model's MCS ladder (got " +
                                       std::to_string(r.mcs_index) + ")");
    }
    check(SinrToSilvusMcsIndex(10.0) != SinrToMcsIndex(10.0), "silvus and table ladders differ at 10 dB");
}

static void
test_le_never_writes_sentinel()
{
    Rig g;
    SimConfig cfg = BaseCfg(2);
    g.loss->lossDb = 300.0;
    g.ev.Configure(cfg, g.loss, g.cond);
    LinkResult r = g.ev.Evaluate(Mob(0, 0, 0), Mob(100, 0, 0), 0, 1);
    check(r.sinr_db != -999.0 && r.rx_power_dbm != -999.0, "weak link has real SINR/rx_power, not -999");
    check(r.sinr_db > -900.0, "weak (300 dB) link passes the consumers' > -900 validity filter");
    check(approx(r.rx_power_dbm, -270.0), "weak link rx_power = 30 - 300");
    check(r.capacity_mbps == 0.0 && r.mcs_index == 0, "weak link 0 Mbps, MCS 0");
}

static void
test_le_evaluate_all_order_and_size()
{
    Rig g;
    SimConfig cfg = BaseCfg(4);
    g.ev.Configure(cfg, g.loss, g.cond);

    check(g.ev.EvaluateAll({}).empty(), "EvaluateAll N=0 empty");
    check(g.ev.EvaluateAll({Mob(0, 0)}).empty(), "EvaluateAll N=1 empty");
    check(g.ev.EvaluateAll({Mob(0, 0), Mob(10, 0)}).size() == 1, "EvaluateAll N=2 one link");

    g.cond->calls.clear();
    g.loss->calls.clear();
    std::vector<MobPtr> mobs = {Mob(0, 0), Mob(10, 0), Mob(0, 20), Mob(30, 40)};
    auto res = g.ev.EvaluateAll(mobs);
    check(res.size() == 6, "EvaluateAll N=4 gives 6 results");
    const uint32_t ex[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    bool order = res.size() == 6;
    bool dist = true;
    for (size_t k = 0; order && k < 6; ++k)
    {
        order = order && res[k].tx_id == ex[k][0] && res[k].rx_id == ex[k][1];
        const auto& pa = mobs[ex[k][0]]->RawPosition();
        const auto& pb = mobs[ex[k][1]]->RawPosition();
        dist = dist && approx(res[k].distance_m, std::hypot(pa.x - pb.x, pa.y - pb.y));
    }
    check(order, "EvaluateAll row-major (i<j) order");
    check(dist, "EvaluateAll distances match node pairs");
    // One condition draw and one loss draw per link, in pair order (realization order).
    bool calls = g.cond->calls.size() == 6 && g.loss->calls.size() == 6;
    for (size_t k = 0; calls && k < 6; ++k)
    {
        calls = calls && SamePos(g.cond->calls[k].first, mobs[ex[k][0]]->RawPosition()) &&
                SamePos(g.cond->calls[k].second, mobs[ex[k][1]]->RawPosition()) &&
                SamePos(g.loss->calls[k].a, mobs[ex[k][0]]->RawPosition());
    }
    check(calls, "EvaluateAll queries condition and loss once per pair, in pair order (mmwave)");

    LinkTable t;
    bool threw = false;
    try
    {
        t.Update(4, res);
    }
    catch (const std::exception&)
    {
        threw = true;
    }
    check(!threw && t.Get(3, 1).tx_id == 1 && t.Get(3, 1).rx_id == 3,
          "EvaluateAll output feeds LinkTable::Update");
}

static void
test_le_probe()
{
    Rig g;
    SimConfig cfg = BaseCfg(2);
    cfg.channel.tx_array_gain_dbi = 4.0;
    cfg.channel.rx_array_gain_dbi = 6.0;
    cfg.nodes[1].rx_array_gain_dbi = 2.5;
    g.ev.Configure(cfg, g.loss, g.cond);
    auto tx = Mob(0, 0, 0);
    auto rx = Mob(60, 80, 0);
    LinkResult p = g.ev.EvaluateProbe(tx, rx, 0, 2.5);
    LinkResult n = g.ev.Evaluate(tx, rx, 0, 1);
    check(p.rx_id == LinkEvaluator::kProbeRxId && p.rx_id == std::numeric_limits<uint32_t>::max(),
          "probe rx_id is kProbeRxId (UINT32_MAX)");
    check(p.tx_id == 0, "probe tx_id is txIdx");
    check(approx(p.rx_power_dbm, 30.0 - 100.0 + 4.0 + 2.5), "probe uses tx node gain + request RX gain");
    check(approx(p.distance_m, n.distance_m) && approx(p.path_loss_db, n.path_loss_db) &&
              approx(p.rx_power_dbm, n.rx_power_dbm) && approx(p.sinr_db, n.sinr_db) &&
              approx(p.capacity_mbps, n.capacity_mbps) && p.mcs_index == n.mcs_index &&
              p.is_los == n.is_los,
          "probe equals Evaluate with the same RX gain (shared EvaluateLink)");
    LinkResult p2 = g.ev.EvaluateProbe(tx, rx, 0, 12.5);
    check(approx(p2.rx_power_dbm - p.rx_power_dbm, 10.0), "probe RX gain applied 1:1 in dB");
}

// ---- LinkEvaluator jammer path ----

/// Two nodes 100 m apart, fixed 100 dB loss, 20 MHz, NF 5, 30 dBm, zero gains.
/// Plain SNR = -70 - (-95.99) = 25.99 dB.
struct JamRig
{
    ns3::Ptr<FakeLoss> loss = ns3::CreateObject<FakeLoss>();
    ns3::Ptr<FakeCond> cond = ns3::CreateObject<FakeCond>();
    LinkEvaluator ev;
    SimConfig cfg = BaseCfg(2);
    MobPtr a = Mob(0, 0, 0);
    MobPtr b = Mob(100, 0, 0);
    std::vector<MobPtr> jamMobs;

    void Add(const JammerSpec& s, const MobPtr& m)
    {
        cfg.jammers.push_back(s);
        jamMobs.push_back(m);
    }
    void Configure(const std::string& band) { ev.Configure(cfg, loss, cond, band, jamMobs); }
};

static double
ExpectedJammedSinr(double rxPowerDbm, double noiseDbm, double jamW)
{
    const double s = 10.0 * std::log10(DbmToW(rxPowerDbm) / (DbmToW(noiseDbm) + jamW));
    return s < 0.0 ? 0.0 : s;
}

static void
test_le_jammer_ignored_outside_sub6()
{
    JamRig j;
    j.loss->lossDb = 130.0;  // SNR = -100 + 95.99 = -4.01 dB (negative)
    j.Add(Jam(), Mob(50, 0, 0));  // strong jammer between the nodes
    j.Configure("mmwave");
    LinkResult r = j.ev.Evaluate(j.a, j.b, 0, 1);
    const double snr = -100.0 - NoiseFloorDbm(20.0, 5.0);
    check(approx(r.sinr_db, snr), "mmwave: jammer ignored, negative SNR unclamped (" + num(r.sinr_db) + ")");
    check(j.loss->calls.size() == 1, "mmwave: jammer model not queried");
}

static void
test_le_sub6_without_jammers_plain_snr()
{
    const double snr = -100.0 - NoiseFloorDbm(20.0, 5.0);
    {
        JamRig j;
        j.loss->lossDb = 130.0;
        j.Configure("sub-6");
        LinkResult r = j.ev.Evaluate(j.a, j.b, 0, 1);
        check(approx(r.sinr_db, snr), "sub-6, no jammers: plain negative SNR, no clamp");
    }
    {
        JamRig j;
        j.loss->lossDb = 130.0;
        JammerSpec s = Jam();
        s.enabled = false;
        j.Add(s, Mob(50, 0, 0));
        j.Configure("sub-6");
        LinkResult r = j.ev.Evaluate(j.a, j.b, 0, 1);
        check(approx(r.sinr_db, snr), "sub-6, only disabled jammers: plain negative SNR, no clamp");
    }
    {
        // Jammer loaded but gated off (outside its interval): jamWatt == 0 -> plain SNR.
        JamRig j;
        j.loss->lossDb = 130.0;
        JammerSpec s = Jam();
        s.intervals = {{10.0, 20.0}};
        j.Add(s, Mob(50, 0, 0));
        j.Configure("sub-6");
        LinkResult r = j.ev.Evaluate(j.a, j.b, 0, 1, 5.0);
        check(approx(r.sinr_db, snr), "sub-6, jammer inactive at nowS: plain negative SNR, no clamp");
        LinkResult r2 = j.ev.Evaluate(j.a, j.b, 0, 1, 15.0);
        check(r2.sinr_db == 0.0, "sub-6, jammer active at nowS: strong jammer clamps SINR to 0");
    }
}

static void
test_le_sub6_jammer_formula()
{
    JamRig j;
    j.cfg.channel.tx_array_gain_dbi = 3.0;
    j.cfg.channel.rx_array_gain_dbi = 4.0;
    JammerSpec s = Jam();
    s.tx_power_dbm = -2.0;
    s.tx_array_gain_dbi = 2.0;  // EIRP 0 dBm -> -100 dBm at either node
    j.Add(s, Mob(50, 0, 0));
    j.Configure("sub-6");
    LinkResult r = j.ev.Evaluate(j.a, j.b, 0, 1);
    const double rxDbm = 30.0 - 100.0 + 7.0;
    const double want = ExpectedJammedSinr(rxDbm, NoiseFloorDbm(20.0, 5.0), DbmToW(-100.0));
    check(approx(r.rx_power_dbm, rxDbm), "jammer does not change rx_power");
    check(approx(r.sinr_db, want, 1e-9),
          "sub-6 SINR = 10log10(S/(N+J)), no RX gain on J: got " + num(r.sinr_db) + " want " + num(want));
    check(r.sinr_db < rxDbm - NoiseFloorDbm(20.0, 5.0), "jammer lowers SINR below plain SNR");
    check(approx(r.capacity_mbps, SinrToCapacity(r.sinr_db, 20e6, "shannon")), "jammed capacity via SinrToCapacity");
    // CalcRxPower was called for the link and for the jammer at both ends with EIRP.
    int eirpCalls = 0;
    for (const auto& c : j.loss->calls)
    {
        if (approx(c.txDbm, 0.0) && SamePos(c.a, ns3::Vector(50, 0, 0)))
        {
            ++eirpCalls;
        }
    }
    check(eirpCalls == 2, "jammer power evaluated at RX end and TX end with EIRP");
}

static void
test_le_sub6_jammer_clamp()
{
    JamRig j;
    j.loss->lossDb = 100.0;
    j.Add(Jam(), Mob(50, 0, 0));  // EIRP 37 dBm -> -63 dBm >> signal -70 dBm
    j.Configure("sub-6");
    LinkResult r = j.ev.Evaluate(j.a, j.b, 0, 1);
    check(r.sinr_db == 0.0, "strong jammer clamps SINR to exactly 0 dB (got " + num(r.sinr_db) + ")");
    check(approx(r.capacity_mbps, SinrToCapacity(0.0, 20e6, "shannon")), "clamped link capacity at 0 dB");

    // Documented (README "SINR floor"): with a negative-SNR link, a weak
    // jammer RAISES the reported SINR to 0 dB. Kept as documented behavior.
    JamRig w;
    w.loss->lossDb = 130.0;
    JammerSpec s = Jam();
    s.tx_power_dbm = -100.0;
    s.tx_array_gain_dbi = 0.0;
    w.Add(s, Mob(50, 0, 0));
    w.Configure("sub-6");
    LinkResult rw = w.ev.Evaluate(w.a, w.b, 0, 1);
    check(rw.sinr_db == 0.0, "weak jammer on negative-SNR link reports 0 dB (documented clamp)");
}

static void
test_le_jammer_max_of_both_ends()
{
    const double noise = NoiseFloorDbm(20.0, 5.0);
    const double rxDbm = -70.0;
    // Jammer near the TX end only (range gate excludes the RX end).
    {
        JamRig j;
        JammerSpec s = Jam();
        s.tx_power_dbm = 0.0;
        s.tx_array_gain_dbi = 0.0;
        s.max_range_m = 50.0;
        j.Add(s, Mob(-10, 0, 0));
        j.Configure("sub-6");
        LinkResult r01 = j.ev.Evaluate(j.a, j.b, 0, 1);
        LinkResult r10 = j.ev.Evaluate(j.b, j.a, 1, 0);
        const double want = ExpectedJammedSinr(rxDbm, noise, DbmToW(-100.0));
        check(approx(r01.sinr_db, want), "jammer at TX end only counts (0->1)");
        check(approx(r10.sinr_db, want), "jammer at RX end only counts (1->0)");
    }
    // Different jammers at each end: max, not sum.
    {
        JamRig j;
        JammerSpec near0 = Jam("near0");
        near0.tx_power_dbm = 0.0;
        near0.tx_array_gain_dbi = 0.0;  // -100 dBm at node 0
        near0.max_range_m = 50.0;
        JammerSpec near1 = Jam("near1");
        near1.tx_power_dbm = 5.0;
        near1.tx_array_gain_dbi = 0.0;  // -95 dBm at node 1
        near1.max_range_m = 50.0;
        j.Add(near0, Mob(-10, 0, 0));
        j.Add(near1, Mob(110, 0, 0));
        j.Configure("sub-6");
        LinkResult r = j.ev.Evaluate(j.a, j.b, 0, 1);
        const double wantMax = ExpectedJammedSinr(rxDbm, noise, DbmToW(-95.0));
        const double wantSum = ExpectedJammedSinr(rxDbm, noise, DbmToW(-95.0) + DbmToW(-100.0));
        check(approx(r.sinr_db, wantMax), "jammer power = max(RX end, TX end)");
        check(!approx(r.sinr_db, wantSum), "jammer power is not the sum of both ends");
    }
}

static void
test_le_jammer_uses_cfg_seed_and_carrier()
{
    // Random jammer: evaluator's jam/no-jam pattern must follow cfg.seed.
    JamRig j;
    j.cfg.seed = 7;
    JammerSpec s = Jam("rj");
    s.type = "random";
    s.duty_cycle = 0.5;
    s.tx_power_dbm = 0.0;
    s.tx_array_gain_dbi = 0.0;
    auto jm = Mob(50, 0, 0);
    j.Add(s, jm);
    j.Configure("sub-6");
    const double noise = NoiseFloorDbm(20.0, 5.0);
    const double snr = -70.0 - noise;
    const double jammed = ExpectedJammedSinr(-70.0, noise, DbmToW(-100.0));
    bool match7 = true;
    int on7 = 0, diff8 = 0;
    for (int t = 0; t < 64; ++t)
    {
        const double p7 = JamPowerAt(s, jm, j.b, t, 2.4e9, 7, 100.0);
        const double p8 = JamPowerAt(s, jm, j.b, t, 2.4e9, 8, 100.0);
        on7 += (p7 > 0.0);
        diff8 += ((p7 > 0.0) != (p8 > 0.0));
        LinkResult r = j.ev.Evaluate(j.a, j.b, 0, 1, t + 0.5);
        const double want = (p7 > 0.0) ? jammed : snr;
        match7 = match7 && approx(r.sinr_db, want);
    }
    check(on7 > 0 && on7 < 64, "seed-7 random jammer is on some but not all seconds");
    check(diff8 > 0, "seed 7 and seed 8 burst patterns differ (test is not vacuous)");
    check(match7, "LinkEvaluator passes cfg.seed to JammerModel");

    // Carrier: a spot jammer at 2400 MHz hits a 2.4 GHz link but not 5.8 GHz.
    JamRig c24;
    JammerSpec sp = Jam();
    sp.target_freq = {2400.0};
    c24.Add(sp, Mob(50, 0, 0));
    c24.Configure("sub-6");
    check(c24.ev.Evaluate(c24.a, c24.b, 0, 1).sinr_db == 0.0, "spot jammer at carrier jams (clamped)");
    JamRig c58;
    c58.cfg.channel.frequency_ghz = 5.8;
    c58.Add(sp, Mob(50, 0, 0));
    c58.Configure("sub-6");
    check(approx(c58.ev.Evaluate(c58.a, c58.b, 0, 1).sinr_db, snr),
          "spot jammer off the carrier frequency is ignored (carrier from frequency_ghz)");
}

static void
test_le_reconfigure_drops_jammers()
{
    // Re-Configuring a reused evaluator with a jammer-free config drops the
    // previous config's jammers.
    JamRig j;
    j.loss->lossDb = 130.0;
    j.Add(Jam(), Mob(50, 0, 0));
    j.Configure("sub-6");
    check(j.ev.Evaluate(j.a, j.b, 0, 1).sinr_db == 0.0, "precondition: jammer active after first Configure");
    SimConfig clean = BaseCfg(2);
    j.ev.Configure(clean, j.loss, j.cond, "sub-6", {});
    const double snr = -100.0 - NoiseFloorDbm(20.0, 5.0);
    LinkResult r = j.ev.Evaluate(j.a, j.b, 0, 1);
    check(approx(r.sinr_db, snr),
          "re-Configure with no jammers drops old jammers (got SINR " + num(r.sinr_db) + ", want " +
              num(snr) + ")");
}

// =====================================================================
// JammerModel
// =====================================================================

static void
test_jm_unconfigured_and_empty()
{
    JammerModel jm;
    check(!jm.HasJammers(), "unconfigured JammerModel has no jammers");
    check(jm.InterfPowerAtReceiver(Mob(0, 0), 0.0) == 0.0, "unconfigured power is 0");

    auto loss = ns3::CreateObject<FakeLoss>();
    jm.Configure({}, loss, {});
    check(!jm.HasJammers(), "empty Configure: no jammers");
    check(jm.InterfPowerAtReceiver(Mob(0, 0), 0.0) == 0.0, "empty Configure: power 0");
}

static void
test_jm_configure_filters_disabled()
{
    auto loss = ns3::CreateObject<FakeLoss>();
    loss->lossDb = 80.0;
    JammerSpec off = Jam("off");
    off.enabled = false;
    off.tx_power_dbm = 60.0;
    JammerSpec on = Jam("on");  // EIRP 37 dBm
    JammerModel jm;
    jm.Configure({off, on}, loss, {Mob(0, 0, 0), Mob(50, 0, 0)});
    check(jm.HasJammers(), "one enabled jammer -> HasJammers");
    const double p = jm.InterfPowerAtReceiver(Mob(150, 0, 0), 0.0);
    check(approx(p, DbmToW(37.0 - 80.0)), "only enabled jammer contributes: " + num(p) + " W");
    check(loss->calls.size() == 1 && SamePos(loss->calls[0].a, ns3::Vector(50, 0, 0)),
          "enabled jammer keeps its own mobility model after filtering");

    JammerModel jm2;
    jm2.Configure({off}, loss, {Mob(0, 0, 0)});
    check(!jm2.HasJammers(), "all disabled -> HasJammers false");
    check(jm2.InterfPowerAtReceiver(Mob(10, 0, 0), 0.0) == 0.0, "all disabled -> 0 W");
}

static void
test_jm_reconfigure_replaces()
{
    auto loss = ns3::CreateObject<FakeLoss>();
    loss->lossDb = 80.0;
    JammerSpec a = Jam("a");
    a.tx_power_dbm = 10.0;
    a.tx_array_gain_dbi = 0.0;
    JammerSpec b = Jam("b");
    b.tx_power_dbm = 20.0;
    b.tx_array_gain_dbi = 0.0;
    JammerModel jm;
    jm.Configure({a}, loss, {Mob(0, 0)});
    jm.Configure({b}, loss, {Mob(0, 0)});
    check(approx(jm.InterfPowerAtReceiver(Mob(100, 0), 0.0), DbmToW(20.0 - 80.0)),
          "second Configure replaces jammers");
    jm.Configure({}, loss, {});
    check(!jm.HasJammers(), "Configure({}) clears jammers");
}

static void
test_jm_power_eirp_and_colocated()
{
    auto jamMob = Mob(0, 0, 0);
    auto loss = ns3::CreateObject<FakeLoss>();
    loss->lossDb = 80.0;
    JammerModel jm;
    jm.Configure({Jam()}, loss, {jamMob});
    auto rx = Mob(100, 0, 0);
    const double p = jm.InterfPowerAtReceiver(rx, 0.0);
    check(approx(p, DbmToW(25.0 + 12.0 - 80.0)), "power = W(EIRP - loss), returned in watts");
    check(loss->calls.size() == 1 && approx(loss->calls[0].txDbm, 37.0),
          "CalcRxPower called with EIRP = tx_power + tx_array_gain");
    check(SamePos(loss->calls[0].a, jamMob->RawPosition()) && SamePos(loss->calls[0].b, rx->RawPosition()),
          "CalcRxPower called with (jammer, receiver)");

    loss->calls.clear();
    check(approx(jm.InterfPowerAtReceiver(Mob(0.5, 0, 0), 0.0), DbmToW(37.0)), "under 1 m: EIRP directly");
    check(approx(jm.InterfPowerAtReceiver(Mob(0, 0, 0), 0.0), DbmToW(37.0)), "co-located: EIRP directly");
    check(loss->calls.empty(), "under 1 m: CalcRxPower not called");
    check(approx(jm.InterfPowerAtReceiver(Mob(1.0, 0, 0), 0.0), DbmToW(37.0 - 80.0)),
          "exactly 1 m uses CalcRxPower");
}

static void
test_jm_constant_duty_and_sum()
{
    auto jamMob = Mob(0, 0, 0);
    auto rx = Mob(100, 0, 0);
    const double full = DbmToW(37.0 - 80.0);
    for (double d : {1.0, 0.5, 0.25, 0.0})
    {
        JammerSpec s = Jam();
        s.duty_cycle = d;
        check(approx(JamPowerAt(s, jamMob, rx, 0.0), full * d),
              "constant jammer scaled by duty_cycle " + num(d));
    }
    // Sum over jammers.
    auto loss = ns3::CreateObject<FakeLoss>();
    loss->lossDb = 80.0;
    JammerSpec a = Jam("a");
    JammerSpec b = Jam("b");
    b.tx_power_dbm = 15.0;  // EIRP 27
    b.duty_cycle = 0.5;
    JammerModel jm;
    jm.Configure({a, b}, loss, {Mob(0, 0), Mob(200, 0)});
    check(approx(jm.InterfPowerAtReceiver(rx, 0.0), full + 0.5 * DbmToW(27.0 - 80.0)),
          "power summed over passing jammers");
}

static void
test_jm_interval_gate()
{
    JammerSpec s = Jam();
    s.intervals = {{2.0, 4.0}, {10.0, 11.0}};
    auto jamMob = Mob(0, 0);
    auto rx = Mob(100, 0);
    struct C
    {
        double t;
        bool on;
    } cases[] = {{0.0, false}, {1.999, false}, {2.0, true}, {3.5, true}, {std::nextafter(4.0, 0.0), true},
                 {4.0, false}, {9.99, false}, {10.0, true}, {10.5, true}, {11.0, false}, {100.0, false}};
    for (const auto& c : cases)
    {
        check((JamPowerAt(s, jamMob, rx, c.t) > 0.0) == c.on,
              "interval gate [2,4)+[10,11) at t=" + num(c.t) + (c.on ? " on" : " off"));
    }
    JammerSpec always = Jam();
    check(JamPowerAt(always, jamMob, rx, 0.0) > 0.0 && JamPowerAt(always, jamMob, rx, 1e6) > 0.0,
          "no intervals -> always on");
}

static std::vector<bool>
BurstPattern(const JammerSpec& s, uint32_t seed, int seconds, double offset = 0.0)
{
    std::vector<bool> v;
    auto jamMob = Mob(0, 0);
    auto rx = Mob(100, 0);
    auto loss = ns3::CreateObject<FakeLoss>();
    loss->lossDb = 80.0;
    JammerModel jm;
    jm.Configure({s}, loss, {jamMob}, 0.0, seed);
    for (int t = 0; t < seconds; ++t)
    {
        v.push_back(jm.InterfPowerAtReceiver(rx, t + offset) > 0.0);
    }
    return v;
}

static void
test_jm_random_burst()
{
    const double full = DbmToW(37.0 - 80.0);
    auto jamMob = Mob(0, 0);
    auto rx = Mob(100, 0);

    JammerSpec r0 = Jam("r");
    r0.type = "random";
    r0.duty_cycle = 0.0;
    bool never = true;
    for (int t = 0; t < 300; ++t)
    {
        never = never && JamPowerAt(r0, jamMob, rx, t) == 0.0;
    }
    check(never, "random duty 0 never on");

    JammerSpec r1 = r0;
    r1.duty_cycle = 1.0;
    bool always = true;
    for (int t = 0; t < 300; ++t)
    {
        always = always && approx(JamPowerAt(r1, jamMob, rx, t), full);
    }
    check(always, "random duty 1 always on at full power");

    JammerSpec rh = r0;
    rh.duty_cycle = 0.5;
    auto pat = BurstPattern(rh, 1, 2000);
    int on = 0;
    for (bool b : pat)
    {
        on += b;
    }
    check(on > 900 && on < 1100, "random duty 0.5 on ~half of 2000 seconds (got " + std::to_string(on) + ")");
    // Full power (not scaled by duty) when on.
    bool fullWhenOn = true;
    for (int t = 0; t < 50; ++t)
    {
        const double p = JamPowerAt(rh, jamMob, rx, t);
        fullWhenOn = fullWhenOn && (p == 0.0 || approx(p, full));
    }
    check(fullWhenOn, "random jammer contributes full power when on");

    JammerSpec r2 = r0;
    r2.duty_cycle = 0.2;
    auto pat2 = BurstPattern(r2, 3, 2000);
    on = 0;
    for (bool b : pat2)
    {
        on += b;
    }
    check(on > 320 && on < 480, "random duty 0.2 on ~20% of seconds (got " + std::to_string(on) + ")");

    // One draw per whole second.
    check(BurstPattern(rh, 1, 200, 0.0) == BurstPattern(rh, 1, 200, 0.37) &&
              BurstPattern(rh, 1, 200, 0.0) == BurstPattern(rh, 1, 200, 0.999),
          "random decision constant within each whole second");
    // Deterministic, seed- and id-dependent.
    check(BurstPattern(rh, 1, 200) == BurstPattern(rh, 1, 200), "same seed+id -> same pattern");
    check(BurstPattern(rh, 1, 200) != BurstPattern(rh, 2, 200), "different seed -> different pattern");
    JammerSpec other = rh;
    other.id = "other";
    check(BurstPattern(rh, 1, 200) != BurstPattern(other, 1, 200), "different id -> different pattern");

    // Interval gate still applies to random jammers.
    JammerSpec ri = r1;
    ri.intervals = {{5.0, 6.0}};
    check(JamPowerAt(ri, jamMob, rx, 4.5) == 0.0 && JamPowerAt(ri, jamMob, rx, 5.5) > 0.0,
          "random jammer respects intervals");
}

static void
test_jm_frequency_gate()
{
    auto jamMob = Mob(0, 0);
    auto rx = Mob(100, 0);
    struct C
    {
        std::vector<double> tf;
        double carrierHz;
        bool on;
        const char* name;
    } cases[] = {
        {{}, 2.4e9, true, "empty target_freq -> no filter"},
        {{5800.0}, 0.0, true, "carrier unset (0) -> no filter"},
        {{2400.0}, 2.4e9, true, "spot at carrier"},
        {{2402.5}, 2.4e9, true, "spot +2.5 MHz edge inclusive"},
        {{2397.5}, 2.4e9, true, "spot -2.5 MHz edge inclusive"},
        {{2402.6}, 2.4e9, false, "spot +2.6 MHz excluded"},
        {{2410.0}, 2.4e9, false, "spot 10 MHz away excluded"},
        {{2390.0, 2410.0}, 2.4e9, true, "band contains carrier"},
        {{2410.0, 2390.0}, 2.4e9, true, "unordered band contains carrier"},
        {{2400.0, 2410.0}, 2.4e9, true, "band lower edge inclusive"},
        {{2390.0, 2400.0}, 2.4e9, true, "band upper edge inclusive"},
        {{2401.0, 2410.0}, 2.4e9, false, "band above carrier excluded"},
        {{2210.0, 2215.0}, 2.212e9, true, "make_jammers band 2210-2215 at 2212 MHz"},
        {{2300.0, 2500.0, 2450.0}, 2.4e9, true, "3 values use [min,max]"},
        {{2450.0, 2500.0, 2420.0}, 2.4e9, false, "3 values [min,max] excludes carrier"},
    };
    for (const auto& c : cases)
    {
        JammerSpec s = Jam();
        s.target_freq = c.tf;
        check((JamPowerAt(s, jamMob, rx, 0.0, c.carrierHz) > 0.0) == c.on,
              std::string("frequency gate: ") + c.name);
    }
}

static void
test_jm_range_gate()
{
    auto jamMob = Mob(0, 0);
    JammerSpec s = Jam();
    s.max_range_m = 100.0;
    check(JamPowerAt(s, jamMob, Mob(100, 0)) > 0.0, "range gate inclusive at max_range_m");
    check(JamPowerAt(s, jamMob, Mob(60, 80)) > 0.0, "range gate uses 3-D distance (60,80 -> 100)");
    check(JamPowerAt(s, jamMob, Mob(100.5, 0)) == 0.0, "beyond max_range_m excluded");
    check(JamPowerAt(s, jamMob, Mob(0, 0, 101)) == 0.0, "vertical distance counts toward range");
    s.max_range_m = 0.0;
    check(JamPowerAt(s, jamMob, Mob(1e5, 0)) > 0.0, "max_range_m 0 -> unlimited");
}

static bool
BeamOn(double az, double ze, double bw, double x, double y, double z)
{
    JammerSpec s = Jam();
    s.azimuth_deg = az;
    s.zenith_deg = ze;
    s.beamwidth_deg = bw;
    return JamPowerAt(s, Mob(0, 0, 0), Mob(x, y, z)) > 0.0;
}

static void
test_jm_beam_gate()
{
    const double d = 100.0;
    auto rad = [](double deg) { return deg * M_PI / 180.0; };
    // East-pointing horizontal 60-degree beam (az 90, ze 90). x=East, y=North.
    check(BeamOn(90, 90, 60, d, 0, 0), "east beam hits east receiver");
    check(BeamOn(90, 90, 60, d * std::cos(rad(29)), d * std::sin(rad(29)), 0), "east beam hits 29 deg off-axis");
    check(!BeamOn(90, 90, 60, d * std::cos(rad(31)), d * std::sin(rad(31)), 0), "east beam misses 31 deg off-axis");
    check(!BeamOn(90, 90, 60, 0, d, 0), "east beam misses north receiver");
    check(!BeamOn(90, 90, 60, -d, 0, 0), "east beam misses west receiver");
    check(!BeamOn(90, 90, 60, d * std::cos(rad(31)), 0, d * std::sin(rad(31))), "east beam misses 31 deg above axis");
    // Azimuth 0 = North, 180 = South, 270 = West.
    check(BeamOn(0, 90, 60, 0, d, 0) && !BeamOn(0, 90, 60, d, 0, 0), "azimuth 0 points North");
    check(BeamOn(180, 90, 60, 0, -d, 0), "azimuth 180 points South");
    check(BeamOn(270, 90, 60, -d, 0, 0), "azimuth 270 points West");
    check(BeamOn(450, 90, 60, d, 0, 0), "azimuth 450 wraps to East");
    // Zenith 0 = up (loader default): ground receivers outside a narrow upward cone.
    check(BeamOn(0, 0, 60, 0, 0, d), "zenith 0 points up");
    check(BeamOn(0, 0, 60, 10, 0, d), "zenith 0: 5.7 deg off vertical inside");
    check(!BeamOn(0, 0, 60, d, 0, 0), "zenith 0: ground receiver outside (README warning)");
    check(BeamOn(123, 0, 60, 0, 0, d), "zenith 0 ignores azimuth");
    check(BeamOn(0, 180, 60, 0, 0, -d) && !BeamOn(0, 180, 60, 0, 0, d), "zenith 180 points down");
    // Elevated pointing: az 90, ze 60 -> 30 deg above East horizon; 50-deg cone.
    check(BeamOn(90, 60, 50, d * std::cos(rad(30)), 0, d * std::sin(rad(30))), "elevated beam on-axis");
    check(!BeamOn(90, 60, 50, d, 0, 0), "elevated beam misses horizon receiver 30 deg off");
    // Omni.
    check(BeamOn(90, 90, 360, -d, 0, 0) && BeamOn(0, 0, 360, 0, 0, -d), "beamwidth 360 is omni");
    check(BeamOn(90, 90, 400, -d, 0, 0), "beamwidth > 360 is omni");
    // Co-located receiver is inside any beam (and gets EIRP).
    check(BeamOn(90, 90, 10, 0, 0, 0), "co-located receiver inside a narrow beam");
    // Wide (>180) cone: 270 deg accepts everything except within 45 deg of the back.
    check(BeamOn(90, 90, 270, 0, d, 0), "270-deg beam accepts perpendicular receiver");
    check(!BeamOn(90, 90, 270, -d, 0, 0), "270-deg beam rejects receiver straight behind");
}

// ---- main ----

/**
 * @fn main
 * @brief Run every test_* function and report pass/fail counts.
 * @return 0 if all checks pass, 1 otherwise.
 */
int
main()
{
    // sinr-capacity.h
    test_silvus_mcs_index_thresholds();
    test_silvus_mcs_index_never_mimo();
    test_silvus_mcs_lookup();
    test_silvus_capacity();
    test_silvus_table_invariants();
    test_capacity_bandwidth_scaling();

    // LinkTable
    test_link_table_two_nodes_and_mirror();
    test_link_table_threshold_boundary();

    // LinkEvaluator
    test_le_noise_floor_and_snr();
    test_le_distance_is_3d();
    test_le_per_node_gain_overrides();
    test_le_fspl_floor();
    test_le_los_mapping();
    test_le_buildings_flag();
    test_le_amc_models();
    test_le_never_writes_sentinel();
    test_le_evaluate_all_order_and_size();
    test_le_probe();
    test_le_jammer_ignored_outside_sub6();
    test_le_sub6_without_jammers_plain_snr();
    test_le_sub6_jammer_formula();
    test_le_sub6_jammer_clamp();
    test_le_jammer_max_of_both_ends();
    test_le_jammer_uses_cfg_seed_and_carrier();
    test_le_reconfigure_drops_jammers();

    // JammerModel
    test_jm_unconfigured_and_empty();
    test_jm_configure_filters_disabled();
    test_jm_reconfigure_replaces();
    test_jm_power_eirp_and_colocated();
    test_jm_constant_duty_and_sum();
    test_jm_interval_gate();
    test_jm_random_burst();
    test_jm_frequency_gate();
    test_jm_range_gate();
    test_jm_beam_gate();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.\n";
    if (g_fail > 0)
    {
        std::cout << "SOME TESTS FAILED.\n";
        return 1;
    }
    std::cout << "All tests passed.\n";
    return 0;
}
