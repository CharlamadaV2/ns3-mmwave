/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file traffic-extra-test.cc
 * @brief Extra unit tests for TrafficMatrix: random_pairs, Poisson arrivals,
 *        on/off timing, gateway direction, tick ordering and edge node counts.
 *
 * Complements tests/unit/traffic/traffic-matrix-test.cc. RNG draws come from
 * the scriptable ns3-traffic-extra-stub.h in this directory, so tests can
 * force Poisson arrival counts, src/dst picks and phase durations.
 * ValidateConfig (ns-3-free) is linked only to show that configurations
 * exposing suspected bugs pass validation.
 *
 * Build and run: `make -C tests/unit/traffic-extra test`.
 */

#include "src/config/config-validator.h"
#include "src/domain/sim-config.h"
#include "src/traffic/traffic-matrix.h"

#include <cmath>
#include <csignal>
#include <iostream>
#include <set>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace mesh_sim;
namespace ts = traffic_stub;

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
    return std::fabs(a - b) < tol;
}

/// Uniform value that GetInteger(0, n-1) maps to index k (mid-bucket).
static double
U(uint32_t k, uint32_t n)
{
    return (k + 0.5) / n;
}

static SimConfig
makeCfg(const std::string& model, const std::string& topology,
        uint32_t numNodes = 5, double demandMbps = 10.0,
        double holdingTime = 0.0, double tickS = 0.1)
{
    SimConfig cfg;
    cfg.tick_s = tickS;
    for (uint32_t i = 0; i < numNodes; ++i)
    {
        NodeSpec ns;
        ns.id = "node-" + std::to_string(i);
        ns.role = "peer";
        ns.mobility = "fixed";
        ns.node_type = "drone";
        cfg.nodes.push_back(ns);
    }
    cfg.mesh.traffic.model          = model;
    cfg.mesh.traffic.flow_topology  = topology;
    cfg.mesh.traffic.demand_mbps    = demandMbps;
    cfg.mesh.traffic.holding_time_s = holdingTime;
    return cfg;
}

/// Count flows that are both active and ON (what MeshRouter routes).
static size_t
countRoutable(const TrafficMatrix& tm)
{
    size_t n = 0;
    for (const auto& f : tm.GetActiveFlows())
    {
        if (f.active && f.in_on_phase)
        {
            ++n;
        }
    }
    return n;
}

/// Removal invariant: after Tick no flow is (!active && end_time_s > 0).
static bool
noDeadFlows(const TrafficMatrix& tm)
{
    for (const auto& f : tm.GetActiveFlows())
    {
        if (!f.active && f.end_time_s > 0.0)
        {
            return false;
        }
    }
    return true;
}

/// ValidateConfig(cfg).ok(), printing any errors to stderr.
static bool
validates(const SimConfig& cfg)
{
    auto r = ValidateConfig(cfg);
    for (const auto& e : r.errors)
    {
        std::cerr << "  validator: " << e << "\n";
    }
    return r.ok();
}

using FlowKey = std::vector<double>;

static std::vector<FlowKey>
snapshot(const TrafficMatrix& tm)
{
    std::vector<FlowKey> out;
    for (const auto& f : tm.GetActiveFlows())
    {
        out.push_back({double(f.src), double(f.dst), f.demand_mbps, f.start_time_s,
                       f.end_time_s, double(f.active), double(f.in_on_phase),
                       f.phase_end_s});
    }
    return out;
}

/**
 * Run @p fn in a forked child with a @p seconds alarm.
 * @return true if the child exited normally (did not hang or crash).
 */
template <typename Fn>
static bool
terminatesWithin(unsigned seconds, Fn fn)
{
    std::cout.flush();
    std::cerr.flush();
    pid_t pid = fork();
    if (pid == 0)
    {
        alarm(seconds);
        fn();
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

// ---- all_pairs / constant ----

/// Source: Flow docs (in_on_phase "Always true for non-on-off models"),
/// MakeFlow (no exp draw unless on_off), "constant" needs no per-tick work.
static void
test_constant_flow_defaults_and_no_draws()
{
    ts::Reset();
    auto cfg = makeCfg("constant", "all_pairs", 4, 7.5);
    TrafficMatrix tm(cfg);
    tm.Initialize(4, 0.0);
    for (int i = 0; i <= 20; ++i)
    {
        tm.Tick(i * 0.1);
    }

    bool ok = true;
    for (const auto& f : tm.GetActiveFlows())
    {
        ok = ok && f.active && f.in_on_phase && f.end_time_s == 0.0 &&
             f.start_time_s == 0.0 && f.demand_mbps == 7.5;
    }
    check(ok, "constant: flows active, ON, permanent, start 0, demand 7.5");
    check(tm.GetActiveFlows().size() == 6, "constant: flow count unchanged over ticks");
}

/// Source: MakeFlow "end_time_s = currentTime + holding_time_s", start at currentTime.
static void
test_initialize_nonzero_time_holding()
{
    ts::Reset();
    auto cfg = makeCfg("constant", "all_pairs", 3, 10.0, /*holding=*/2.0);
    TrafficMatrix tm(cfg);
    tm.Initialize(3, 5.0);

    bool ok = true;
    for (const auto& f : tm.GetActiveFlows())
    {
        ok = ok && f.start_time_s == 5.0 && f.end_time_s == 7.0;
    }
    check(ok, "init@5 holding 2: start 5, end 7");
    tm.Tick(6.9);
    check(tm.GetActiveFlows().size() == 3, "init@5 holding 2: alive at t=6.9");
    tm.Tick(7.0);
    check(tm.GetActiveFlows().empty(), "init@5 holding 2: removed at t=7.0 (>= end)");
}

/// Source: mesh-config.h holding_time_s "for the other models every flow lasts
/// exactly this long"; sim.cc uses t = ti * tick_s. A flow with holding k*tick
/// must be routable for exactly k ticks (no float off-by-one).
static void
test_holding_time_exact_tick_lifetime()
{
    const double tick = 0.1;
    int bad = 0;
    for (int k = 1; k <= 50; ++k)
    {
        ts::Reset();
        double holding = std::stod(std::to_string(k) + "e-1");  // as loader parses
        auto cfg = makeCfg("constant", "all_pairs", 2, 10.0, holding, tick);
        TrafficMatrix tm(cfg);
        tm.Initialize(2, 0.0);
        int liveTicks = 0;
        for (int ti = 0; ti <= k + 2; ++ti)
        {
            tm.Tick(ti * tick);
            if (countRoutable(tm) == 1)
            {
                ++liveTicks;
            }
        }
        if (liveTicks != k)
        {
            ++bad;
            std::cerr << "  holding " << holding << " lived " << liveTicks
                      << " ticks, expected " << k << "\n";
        }
    }
    check(bad == 0, "holding k*0.1 on grid t=ti*0.1: flow lives exactly k ticks");
}

// ---- random_pairs ----

/// Source: InitRandomPairs "random_pair_count flows", self-flows rejected;
/// Flow::src/dst are indices into SimConfig::nodes.
static void
test_random_pairs_count_and_range()
{
    ts::Reset(7);
    const uint32_t n = 6;
    auto cfg = makeCfg("constant", "random_pairs", n, 3.0);
    cfg.mesh.traffic.random_pair_count = 50;
    TrafficMatrix tm(cfg);
    tm.Initialize(n, 0.0);

    bool ok = true;
    for (const auto& f : tm.GetActiveFlows())
    {
        ok = ok && f.src < n && f.dst < n && f.src != f.dst && f.demand_mbps == 3.0 &&
             f.active && f.end_time_s == 0.0;
    }
    check(tm.GetActiveFlows().size() == 50, "random_pairs: count = random_pair_count (50)");
    check(ok, "random_pairs: src/dst in range, no self flows, demand, permanent");
}

/// Source: "Each pair is drawn independently" (duplicates allowed) and
/// GetDemand "Sums demand_mbps across all flows" for the directed pair.
static void
test_random_pairs_duplicates_sum_demand()
{
    ts::Reset();
    const uint32_t n = 5;
    auto cfg = makeCfg("constant", "random_pairs", n, 4.0);
    cfg.mesh.traffic.random_pair_count = 3;
    TrafficMatrix tm(cfg);
    ts::g_uniformScript = {U(1, n), U(4, n), U(1, n), U(4, n), U(4, n), U(1, n)};
    tm.Initialize(n, 0.0);

    check(tm.GetActiveFlows().size() == 3, "random_pairs dup: 3 flows kept");
    check(approx(tm.GetDemand(1, 4), 8.0), "random_pairs dup: GetDemand(1,4) sums 2 flows");
    check(approx(tm.GetDemand(4, 1), 4.0), "random_pairs dup: GetDemand(4,1) separate direction");
    check(approx(tm.GetDemand(1, 3), 0.0), "random_pairs dup: unrelated pair 0");
}

/// Source: MakeFlow holding applies to all topologies.
static void
test_random_pairs_with_holding()
{
    ts::Reset(11);
    auto cfg = makeCfg("constant", "random_pairs", 4, 10.0, /*holding=*/0.5);
    cfg.mesh.traffic.random_pair_count = 4;
    TrafficMatrix tm(cfg);
    tm.Initialize(4, 1.0);

    bool ok = true;
    for (const auto& f : tm.GetActiveFlows())
    {
        ok = ok && f.end_time_s == 1.5;
    }
    check(ok, "random_pairs holding: end = 1.0 + 0.5");
    tm.Tick(1.4);
    check(tm.GetActiveFlows().size() == 4, "random_pairs holding: alive at 1.4");
    tm.Tick(1.5);
    check(tm.GetActiveFlows().empty(), "random_pairs holding: gone at 1.5");
}

/// Source: traffic CLAUDE.md "Reproducibility comes from the ns-3 global
/// seed/run"; the class must not add its own nondeterminism.
static void
test_random_pairs_deterministic_by_seed()
{
    auto cfg = makeCfg("constant", "random_pairs", 8);
    cfg.mesh.traffic.random_pair_count = 20;

    ts::Reset(99);
    TrafficMatrix a(cfg);
    a.Initialize(8, 0.0);
    auto sa = snapshot(a);

    ts::Reset(99);
    TrafficMatrix b(cfg);
    b.Initialize(8, 0.0);
    auto sb = snapshot(b);

    ts::Reset(100);
    TrafficMatrix c(cfg);
    c.Initialize(8, 0.0);
    auto sc = snapshot(c);

    check(sa == sb, "random_pairs: same RNG seed -> identical flows");
    check(sa != sc, "random_pairs: different RNG seed -> different flows");
}

// ---- gateway ----

/// Contract: mesh-config.h gateway_node_id is the "Node ID of the gateway ...
/// must match a node ID". Numeric node IDs that differ from indices (e.g.
/// 1-based "1","2","3") must resolve by ID, not be read as an index.
static void
test_gateway_numeric_node_ids()
{
    ts::Reset();
    SimConfig cfg = makeCfg("constant", "gateway", 0);
    for (const char* id : {"1", "2", "3"})
    {
        NodeSpec ns;
        ns.id = id;
        ns.role = "peer";
        ns.mobility = "fixed";
        ns.node_type = "drone";
        cfg.nodes.push_back(ns);
    }
    cfg.mesh.traffic.gateway_node_id = "3";  // ID of index 2
    check(validates(cfg), "gateway numeric ids: config passes ValidateConfig");

    TrafficMatrix tm(cfg);
    tm.Initialize(3, 0.0);
    bool inRange = true;
    bool toIdx2 = true;
    for (const auto& f : tm.GetActiveFlows())
    {
        inRange = inRange && f.dst < 3 && f.src < 3;
        toIdx2 = toIdx2 && f.dst == 2;
    }
    check(inRange, "gateway numeric ids: flow endpoints are valid node indices");
    check(toIdx2 && tm.GetActiveFlows().size() == 2,
          "gateway numeric ids: gateway '3' resolves to the node with id '3' (index 2)");

    // Swapped IDs: index 0 has id "1", index 1 has id "0".
    SimConfig cfg2 = makeCfg("constant", "gateway", 0);
    for (const char* id : {"1", "0"})
    {
        NodeSpec ns;
        ns.id = id;
        ns.role = "peer";
        ns.mobility = "fixed";
        ns.node_type = "drone";
        cfg2.nodes.push_back(ns);
    }
    cfg2.mesh.traffic.gateway_node_id = "0";  // ID of index 1
    TrafficMatrix tm2(cfg2);
    tm2.Initialize(2, 0.0);
    const auto& fl2 = tm2.GetActiveFlows();
    check(fl2.size() == 1 && fl2[0].dst == 1,
          "gateway swapped numeric ids: gateway '0' is index 1 (node with id '0')");
}

// ---- poisson ----

/// Source: TickPoisson Knuth method "multiply uniform(0,1) samples until their
/// product falls below exp(-lambda); count = multiplications - 1".
static void
test_poisson_scripted_arrival_count()
{
    ts::Reset();
    const uint32_t n = 4;
    const double tick = 0.1;
    auto cfg = makeCfg("poisson", "gateway", n, 2.5, 0.0, tick);
    cfg.mesh.traffic.arrival_rate_hz = std::log(2.0) / tick;  // lambda = ln2, L = 0.5
    TrafficMatrix tm(cfg);
    tm.Initialize(n, 0.0);
    const size_t base = tm.GetActiveFlows().size();

    // Tick 1: products 0.9, 0.81, 0.324 -> 2 arrivals: 3->1, 0->2.
    ts::g_uniformScript = {0.9, 0.9, 0.4, U(3, n), U(1, n), U(0, n), U(2, n)};
    tm.Tick(0.3);
    const auto& fl = tm.GetActiveFlows();
    check(fl.size() == base + 2, "poisson knuth: 2 arrivals");
    check(fl.size() == base + 2 && fl[base].src == 3 && fl[base].dst == 1 &&
              fl[base + 1].src == 0 && fl[base + 1].dst == 2,
          "poisson knuth: arrivals use scripted src/dst in draw order");
    check(fl.size() == base + 2 && fl[base].start_time_s == 0.3 &&
              fl[base].demand_mbps == 2.5 && fl[base].end_time_s == 0.0 &&
              fl[base].active && fl[base].in_on_phase,
          "poisson knuth: arrival start=t, demand, permanent, active");
    check(ts::g_uniformScript.empty() && ts::g_uniformFallbackDraws == 0,
          "poisson knuth: consumed exactly the scripted draws");

    // Tick 2: first uniform 0.3 <= 0.5 -> 0 arrivals.
    ts::g_uniformScript = {0.3};
    tm.Tick(0.4);
    check(tm.GetActiveFlows().size() == base + 2, "poisson knuth: first u<=L -> 0 arrivals");
    check(ts::g_uniformFallbackDraws == 0, "poisson knuth: 0 arrivals uses one draw");

    // Tick 3: 0.99^k stays > 0.5 for k<=68 -> many arrivals; just 1 arrival here.
    ts::g_uniformScript = {0.99, 0.1, U(1, n), U(0, n)};
    tm.Tick(0.5);
    check(tm.GetActiveFlows().size() == base + 3, "poisson knuth: 0.99,0.1 -> 1 arrival");
}

/// Source: TickPoisson "end_time_s is overridden with an independent
/// Exponential(holding_time_s) draw".
static void
test_poisson_holding_override_exponential()
{
    ts::Reset();
    const uint32_t n = 3;
    auto cfg = makeCfg("poisson", "gateway", n, 1.0, /*holding=*/2.0, 0.1);
    cfg.mesh.traffic.arrival_rate_hz = std::log(2.0) / 0.1;
    TrafficMatrix tm(cfg);
    tm.Initialize(n, 0.0);
    const size_t base = tm.GetActiveFlows().size();

    ts::g_expMeans.clear();
    ts::g_expScript = {0.375};
    ts::g_uniformScript = {0.9, 0.1, U(0, n), U(2, n)};
    tm.Tick(0.25);
    const auto& fl = tm.GetActiveFlows();
    check(fl.size() == base + 1 && approx(fl[base].end_time_s, 0.625),
          "poisson holding: arrival end = t + exp draw (0.25 + 0.375)");
    check(ts::g_expMeans.size() == 1 && ts::g_expMeans[0] == 2.0,
          "poisson holding: exp drawn with mean holding_time_s");
}

/// Source: Tick expiry "currentTime >= end_time_s" (inclusive boundary).
static void
test_poisson_expiry_boundary()
{
    ts::Reset();
    const uint32_t n = 3;
    auto cfg = makeCfg("poisson", "gateway", n, 1.0, /*holding=*/1.0, 0.125);
    cfg.mesh.traffic.arrival_rate_hz = std::log(2.0) / 0.125;
    TrafficMatrix tm(cfg);
    tm.Initialize(n, 0.0);  // initial flows end at 1.0

    ts::g_expScript = {0.25};
    ts::g_uniformScript = {0.9, 0.1, U(0, n), U(1, n)};
    tm.Tick(0.25);  // arrival 0->1, end 0.5
    check(approx(tm.GetDemand(0, 1), 1.0), "poisson expiry: arrival present at t=0.25");

    ts::g_uniformScript = {0.1};
    tm.Tick(0.375);
    check(approx(tm.GetDemand(0, 1), 1.0), "poisson expiry: alive at t=0.375 < end");

    ts::g_uniformScript = {0.1};
    tm.Tick(0.5);
    check(approx(tm.GetDemand(0, 1), 0.0), "poisson expiry: gone at t=0.5 == end");
    check(tm.GetActiveFlows().size() == 2, "poisson expiry: only initial 2 flows left");
    check(noDeadFlows(tm), "poisson expiry: no dead flows retained");
}

/// Source: "With holding_time_s = 0 arrivals never expire."
static void
test_poisson_holding_zero_permanent()
{
    ts::Reset();
    const uint32_t n = 3;
    auto cfg = makeCfg("poisson", "gateway", n, 1.0, 0.0, 0.1);
    cfg.mesh.traffic.arrival_rate_hz = std::log(2.0) / 0.1;
    TrafficMatrix tm(cfg);
    tm.Initialize(n, 0.0);
    ts::g_uniformScript = {0.9, 0.1, U(0, n), U(1, n)};
    tm.Tick(0.1);
    ts::g_uniformScript = {0.1};
    tm.Tick(1.0e6);
    check(tm.GetActiveFlows().size() == 3 && approx(tm.GetDemand(0, 1), 1.0),
          "poisson holding 0: arrival survives to t=1e6");
    check(ts::g_expDraws == 0, "poisson holding 0: no exponential draws");
}

/// Source: Poisson process with rate arrival_rate_hz over tick_s:
/// per-tick count ~ Poisson(lambda) (mean = var = lambda); Poisson holding
/// times exponential with mean holding_time_s. Uses the real-exponential
/// fallback and a fixed-seed uniform engine.
static void
test_poisson_statistics()
{
    ts::Reset(2024, ts::ExpFallback::Random);
    const uint32_t n = 5;
    const double tick = 0.5;
    auto cfg = makeCfg("poisson", "gateway", n, 1.0, /*holding=*/0.2, tick);
    cfg.mesh.traffic.arrival_rate_hz = 4.0;  // lambda = 2
    TrafficMatrix tm(cfg);
    tm.Initialize(n, 0.0);

    const int ticks = 20000;
    double sum = 0.0;
    double sumSq = 0.0;
    double holdSum = 0.0;
    long holdN = 0;
    bool rangeOk = true;
    bool invariantOk = true;
    for (int ti = 1; ti <= ticks; ++ti)
    {
        double t = ti * tick;
        tm.Tick(t);
        int arrivals = 0;
        for (const auto& f : tm.GetActiveFlows())
        {
            if (f.start_time_s == t)
            {
                ++arrivals;
                holdSum += f.end_time_s - f.start_time_s;
                ++holdN;
                rangeOk = rangeOk && f.src < n && f.dst < n && f.src != f.dst;
            }
        }
        invariantOk = invariantOk && noDeadFlows(tm);
        sum += arrivals;
        sumSq += double(arrivals) * arrivals;
    }
    double mean = sum / ticks;
    double var = sumSq / ticks - mean * mean;
    double holdMean = holdN ? holdSum / holdN : 0.0;
    std::cout << "  poisson stats: mean " << mean << " var " << var
              << " hold mean " << holdMean << " (n=" << holdN << ")\n";
    check(std::fabs(mean - 2.0) < 0.05, "poisson stats: mean arrivals/tick ~ rate*tick_s = 2");
    check(std::fabs(var - 2.0) < 0.15, "poisson stats: variance ~ 2 (Poisson)");
    check(std::fabs(holdMean - 0.2) < 0.01, "poisson stats: mean holding ~ 0.2 s");
    check(rangeOk, "poisson stats: arrivals have valid src != dst");
    check(invariantOk, "poisson stats: no !active && end>0 flow survives a Tick");
}

/// Source: Initialize "Clears any existing flows, sets the node count".
static void
test_poisson_reinitialize_updates_node_count()
{
    ts::Reset(17);
    auto cfg = makeCfg("poisson", "all_pairs", 6, 1.0, 0.0, 0.5);
    cfg.mesh.traffic.arrival_rate_hz = 20.0;  // lambda = 10
    TrafficMatrix tm(cfg);
    tm.Initialize(6, 0.0);
    for (int ti = 1; ti <= 5; ++ti)
    {
        tm.Tick(ti * 0.5);
    }
    check(tm.GetActiveFlows().size() > 15, "poisson reinit: arrivals accumulated");

    tm.Initialize(2, 0.0);
    check(tm.GetActiveFlows().size() == 1, "poisson reinit: arrivals cleared");
    for (int ti = 1; ti <= 5; ++ti)
    {
        tm.Tick(ti * 0.5);
    }
    bool ok = true;
    for (const auto& f : tm.GetActiveFlows())
    {
        ok = ok && f.src < 2 && f.dst < 2 && f.src != f.dst;
    }
    check(tm.GetActiveFlows().size() > 1, "poisson reinit: new arrivals after reinit");
    check(ok, "poisson reinit: arrivals only use the new 2-node range");
}

/// Source: traffic CLAUDE.md reproducibility from the global seed.
static void
test_poisson_deterministic_by_seed()
{
    auto cfg = makeCfg("poisson", "random_pairs", 6, 1.0, 1.0, 0.1);
    cfg.mesh.traffic.arrival_rate_hz = 15.0;
    auto run = [&](uint64_t seed) {
        ts::Reset(seed, ts::ExpFallback::Random);
        TrafficMatrix tm(cfg);
        tm.Initialize(6, 0.0);
        std::vector<std::vector<FlowKey>> hist;
        for (int ti = 0; ti <= 40; ++ti)
        {
            tm.Tick(ti * 0.1);
            hist.push_back(snapshot(tm));
        }
        return hist;
    };
    auto a = run(4);
    auto b = run(4);
    auto c = run(5);
    check(a == b, "poisson: same seed -> identical flow history");
    check(a != c, "poisson: different seed -> different flow history");
}

/// Source: Tick phase order (expiry, model update, removal): flows expiring at
/// t are removed while arrivals at t are kept.
static void
test_tick_order_expiry_and_arrival_same_tick()
{
    ts::Reset();
    const uint32_t n = 3;
    auto cfg = makeCfg("poisson", "all_pairs", n, 1.0, /*holding=*/1.0, 0.25);
    cfg.mesh.traffic.arrival_rate_hz = std::log(2.0) / 0.25;
    TrafficMatrix tm(cfg);
    tm.Initialize(n, 0.0);  // 3 flows, end 1.0

    ts::g_expScript = {0.5};
    ts::g_uniformScript = {0.9, 0.1, U(2, n), U(0, n)};
    tm.Tick(1.0);
    const auto& fl = tm.GetActiveFlows();
    check(fl.size() == 1, "tick order: 3 expired removed, 1 arrival kept");
    check(fl.size() == 1 && fl[0].src == 2 && fl[0].dst == 0 && fl[0].active &&
              approx(fl[0].end_time_s, 1.5),
          "tick order: arrival at t=1.0 active, end 1.5");
}

/// Source: GetDemand sums all active flows on the directed pair, across
/// initial and arrived flows.
static void
test_get_demand_sums_initial_and_arrived()
{
    ts::Reset();
    const uint32_t n = 3;
    auto cfg = makeCfg("poisson", "all_pairs", n, 2.0, 0.0, 0.1);
    cfg.mesh.traffic.arrival_rate_hz = std::log(2.0) / 0.1;
    TrafficMatrix tm(cfg);
    tm.Initialize(n, 0.0);
    ts::g_uniformScript = {0.9, 0.9, 0.4, U(0, n), U(1, n), U(1, n), U(0, n)};
    tm.Tick(0.1);
    check(approx(tm.GetDemand(0, 1), 4.0), "GetDemand: initial 0->1 + arrival 0->1 = 4");
    check(approx(tm.GetDemand(1, 0), 2.0), "GetDemand: arrival 1->0 counted separately");
    check(approx(tm.GetDemand(1, 2), 2.0), "GetDemand: untouched pair 1->2 = 2");
}

// ---- on_off ----

/// Source: MakeFlow "draws the first ON-phase duration from Exponential(on_time_s)".
static void
test_on_off_initial_phase_end_and_mean()
{
    ts::Reset();
    auto cfg = makeCfg("on_off", "all_pairs", 3);
    cfg.mesh.traffic.on_time_s  = 1.5;
    cfg.mesh.traffic.off_time_s = 4.0;
    TrafficMatrix tm(cfg);
    tm.Initialize(3, 2.0);

    bool ok = true;
    for (const auto& f : tm.GetActiveFlows())
    {
        ok = ok && f.in_on_phase && f.active && approx(f.phase_end_s, 3.5) &&
             f.end_time_s == 0.0;
    }
    check(ok, "on_off init@2: ON, phase_end = 2 + on_time");
    bool means = ts::g_expMeans.size() == 3;
    for (double m : ts::g_expMeans)
    {
        means = means && m == 1.5;
    }
    check(means, "on_off init: one Exp(on_time_s) draw per flow");
}

/// Source: CLAUDE.md "on-off OFF flows stay in the list with in_on_phase = false";
/// GetActiveFlows "returns all flows".
static void
test_on_off_off_flows_stay_listed()
{
    ts::Reset();
    auto cfg = makeCfg("on_off", "all_pairs", 4);
    cfg.mesh.traffic.on_time_s  = 1.0;
    cfg.mesh.traffic.off_time_s = 10.0;
    TrafficMatrix tm(cfg);
    tm.Initialize(4, 0.0);
    tm.Tick(1.0);
    bool off = true;
    for (const auto& f : tm.GetActiveFlows())
    {
        off = off && !f.in_on_phase && f.active && approx(f.phase_end_s, 11.0);
    }
    check(tm.GetActiveFlows().size() == 6, "on_off OFF: all 6 flows still listed");
    check(off, "on_off OFF: active=true, in_on_phase=false, phase_end 11");
    check(countRoutable(tm) == 0, "on_off OFF: none routable");
    tm.Tick(100.0);
    check(tm.GetActiveFlows().size() == 6, "on_off: permanent flows never removed");
}

/// Source: TickOnOff "A while loop handles multiple phase transitions within a single tick".
static void
test_on_off_multiple_transitions_one_tick()
{
    ts::Reset();
    auto cfg = makeCfg("on_off", "all_pairs", 2);
    cfg.mesh.traffic.on_time_s  = 1.0;
    cfg.mesh.traffic.off_time_s = 1.0;
    TrafficMatrix tm(cfg);
    tm.Initialize(2, 0.0);
    tm.Tick(3.0);  // transitions at 1 (OFF), 2 (ON), 3 (OFF)
    const auto& f = tm.GetActiveFlows()[0];
    check(!f.in_on_phase, "on_off multi: OFF after 3 transitions");
    check(approx(f.phase_end_s, 4.0), "on_off multi: phase_end accumulates to 4");
    check(ts::g_expDraws == 4, "on_off multi: 1 init + 3 transition draws");

    tm.Tick(3.5);
    check(!tm.GetActiveFlows()[0].in_on_phase && ts::g_expDraws == 4,
          "on_off multi: no transition before phase_end");
}

/// Source: on_off phases ON [t0, t0+on) / OFF [.., +off), transition at
/// phase_end_s <= currentTime. Dyadic times for exact arithmetic.
static void
test_on_off_asymmetric_schedule()
{
    ts::Reset();
    const double tick = 0.125;
    auto cfg = makeCfg("on_off", "all_pairs", 2, 5.0, 0.0, tick);
    cfg.mesh.traffic.on_time_s  = 0.25;
    cfg.mesh.traffic.off_time_s = 0.5;
    TrafficMatrix tm(cfg);
    tm.Initialize(2, 0.0);

    int mismatches = 0;
    for (int ti = 0; ti <= 24; ++ti)
    {
        double t = ti * tick;
        tm.Tick(t);
        double cyc = std::fmod(t, 0.75);
        bool expectOn = cyc < 0.25;
        bool isOn = approx(tm.GetDemand(0, 1), 5.0);
        if (expectOn != isOn)
        {
            ++mismatches;
            std::cerr << "  on_off schedule t=" << t << " expected "
                      << (expectOn ? "ON" : "OFF") << "\n";
        }
    }
    check(mismatches == 0, "on_off schedule: ON 0.25 s / OFF 0.5 s matches ticks");
}

/// Source: holding applies to on_off too; expiry precedes the on/off update and
/// expired flows are removed regardless of phase.
static void
test_on_off_with_holding_removed_in_any_phase()
{
    ts::Reset();
    auto cfg = makeCfg("on_off", "all_pairs", 3, 10.0, /*holding=*/2.0);
    cfg.mesh.traffic.on_time_s  = 0.5;
    cfg.mesh.traffic.off_time_s = 2.0;
    TrafficMatrix tm(cfg);
    tm.Initialize(3, 0.0);
    tm.Tick(0.5);  // OFF until 2.5
    check(tm.GetActiveFlows().size() == 3 && countRoutable(tm) == 0,
          "on_off holding: OFF flows kept before expiry");
    uint64_t drawsBefore = ts::g_expDraws;
    tm.Tick(2.5);  // expiry at 2.0 (<= 2.5) and phase end at 2.5 coincide
    check(tm.GetActiveFlows().empty(), "on_off holding: expired OFF flows removed");
    check(ts::g_expDraws == drawsBefore, "on_off holding: expired flows not toggled");
}

// ---- main ----

/**
 * @fn main
 * @brief Run every traffic-extra test and print a pass/fail count.
 * @return 0 if all checks passed, 1 if any failed.
 */
int
main()
{
    // all_pairs / constant
    test_constant_flow_defaults_and_no_draws();
    test_initialize_nonzero_time_holding();
    test_holding_time_exact_tick_lifetime();

    // random_pairs
    test_random_pairs_count_and_range();
    test_random_pairs_duplicates_sum_demand();
    test_random_pairs_with_holding();
    test_random_pairs_deterministic_by_seed();

    // gateway
    test_gateway_numeric_node_ids();

    // poisson
    test_poisson_scripted_arrival_count();
    test_poisson_holding_override_exponential();
    test_poisson_expiry_boundary();
    test_poisson_holding_zero_permanent();
    test_poisson_statistics();
    test_poisson_reinitialize_updates_node_count();
    test_poisson_deterministic_by_seed();
    test_tick_order_expiry_and_arrival_same_tick();
    test_get_demand_sums_initial_and_arrived();

    // on_off
    test_on_off_initial_phase_end_and_mean();
    test_on_off_off_flows_stay_listed();
    test_on_off_multiple_transitions_one_tick();
    test_on_off_asymmetric_schedule();
    test_on_off_with_holding_removed_in_any_phase();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.\n";
    if (g_fail > 0)
    {
        std::cout << "SOME TESTS FAILED.\n";
        return 1;
    }
    std::cout << "All tests passed.\n";
    return 0;
}
