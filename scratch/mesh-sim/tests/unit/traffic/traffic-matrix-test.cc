/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Unit tests for TrafficMatrix.
 * Standalone binary -- no ns-3 dependency.
 *
 * Build:  make
 * Run:    ./traffic-matrix-test
 */

#include "src/traffic/traffic-matrix.h"
#include "src/domain/sim-config.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

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
approx(double a, double b, double tol = 0.01)
{
    return std::fabs(a - b) < tol;
}

// Build a SimConfig with the given traffic settings.
static SimConfig
makeCfg(const std::string& model, const std::string& topology,
        double demandMbps = 10.0, double holdingTime = 0.0)
{
    SimConfig cfg;
    cfg.tick_s = 0.1;
    cfg.mesh.traffic.model          = model;
    cfg.mesh.traffic.flow_topology  = topology;
    cfg.mesh.traffic.demand_mbps    = demandMbps;
    cfg.mesh.traffic.holding_time_s = holdingTime;
    return cfg;
}

// ---- all_pairs topology ----

static void
test_all_pairs_flow_count()
{
    auto cfg = makeCfg("constant", "all_pairs");
    TrafficMatrix tm(cfg);
    tm.Initialize(4, 0.0);

    // N*(N-1)/2 = 4*3/2 = 6
    check(tm.GetActiveFlows().size() == 6, "all_pairs: 4 nodes -> 6 flows");
}

static void
test_all_pairs_no_self_flows()
{
    auto cfg = makeCfg("constant", "all_pairs");
    TrafficMatrix tm(cfg);
    tm.Initialize(3, 0.0);

    for (const auto& f : tm.GetActiveFlows())
    {
        check(f.src != f.dst, "all_pairs: no self-flow src=" +
              std::to_string(f.src));
    }
}

static void
test_all_pairs_demand()
{
    auto cfg = makeCfg("constant", "all_pairs", 25.0);
    TrafficMatrix tm(cfg);
    tm.Initialize(3, 0.0);

    check(approx(tm.GetDemand(0, 1), 25.0), "all_pairs: demand 0->1");
    check(approx(tm.GetDemand(1, 0), 0.0),  "all_pairs: no reverse flow 1->0");
}

// ---- gateway topology ----

static void
test_gateway_default_node()
{
    auto cfg = makeCfg("constant", "gateway");
    TrafficMatrix tm(cfg);
    tm.Initialize(4, 0.0);

    // Default gateway is node 0, so 3 flows: 1->0, 2->0, 3->0
    check(tm.GetActiveFlows().size() == 3, "gateway: 4 nodes -> 3 flows");

    for (const auto& f : tm.GetActiveFlows())
    {
        check(f.dst == 0, "gateway: all flows go to node 0");
        check(f.src != 0, "gateway: gateway doesn't send to itself");
    }
}

static void
test_gateway_explicit_node()
{
    auto cfg = makeCfg("constant", "gateway");
    cfg.mesh.traffic.gateway_node_id = "2";
    TrafficMatrix tm(cfg);
    tm.Initialize(4, 0.0);

    for (const auto& f : tm.GetActiveFlows())
    {
        check(f.dst == 2, "gateway(2): flow dst=" + std::to_string(f.dst));
    }
}

// ---- unknown topology ----

static void
test_unknown_topology_throws()
{
    auto cfg = makeCfg("constant", "mesh_ring");
    TrafficMatrix tm(cfg);

    bool threw = false;
    try
    {
        tm.Initialize(3, 0.0);
    }
    catch (const std::runtime_error&)
    {
        threw = true;
    }
    check(threw, "unknown topology throws runtime_error");
}

// ---- flow expiration ----

static void
test_flow_expiration()
{
    auto cfg = makeCfg("constant", "all_pairs", 10.0, /*holdingTime=*/1.0);
    TrafficMatrix tm(cfg);
    tm.Initialize(3, 0.0);

    check(tm.GetActiveFlows().size() == 3, "expiration: 3 flows at t=0");

    // All flows have end_time_s = 0.0 + 1.0 = 1.0.
    // Tick at t=0.5: nothing expires.
    tm.Tick(0.5);
    check(tm.GetActiveFlows().size() == 3, "expiration: 3 flows at t=0.5");

    // Tick at t=1.0: all flows expire and get removed.
    tm.Tick(1.0);
    check(tm.GetActiveFlows().size() == 0, "expiration: 0 flows at t=1.0");
}

static void
test_permanent_flows_survive()
{
    auto cfg = makeCfg("constant", "all_pairs", 10.0, /*holdingTime=*/0.0);
    TrafficMatrix tm(cfg);
    tm.Initialize(3, 0.0);

    tm.Tick(100.0);
    tm.Tick(1000.0);
    check(tm.GetActiveFlows().size() == 3, "permanent flows survive long ticks");
}

// ---- GetDemand ----

static void
test_get_demand_nonexistent_pair()
{
    auto cfg = makeCfg("constant", "gateway");
    TrafficMatrix tm(cfg);
    tm.Initialize(3, 0.0);

    // Gateway flows: 1->0, 2->0.  No flow 0->1.
    check(approx(tm.GetDemand(0, 1), 0.0), "GetDemand: no flow returns 0");
}

// ---- on_off model ----

static void
test_on_off_initial_phase_is_on()
{
    auto cfg = makeCfg("on_off", "all_pairs", 10.0);
    cfg.mesh.traffic.on_time_s  = 2.0;
    cfg.mesh.traffic.off_time_s = 2.0;
    TrafficMatrix tm(cfg);
    tm.Initialize(2, 0.0);

    // Initially ON, so demand should be reported.
    check(approx(tm.GetDemand(0, 1), 10.0), "on_off: initial phase is ON");
}

static void
test_on_off_transitions()
{
    auto cfg = makeCfg("on_off", "all_pairs", 10.0);
    cfg.mesh.traffic.on_time_s  = 1.0;
    cfg.mesh.traffic.off_time_s = 1.0;
    TrafficMatrix tm(cfg);
    tm.Initialize(2, 0.0);

    // ExpRng stub returns mean exactly, so:
    //   phase_end_s = 0.0 + 1.0 (on_time mean) = 1.0
    //
    // At t=1.0, phase_end <= currentTime, so it toggles to OFF.
    //   New phase_end = 1.0 + 1.0 (off_time mean) = 2.0
    tm.Tick(1.0);
    check(approx(tm.GetDemand(0, 1), 0.0), "on_off: OFF after first transition");

    // At t=2.0, toggles back to ON.
    tm.Tick(2.0);
    check(approx(tm.GetDemand(0, 1), 10.0), "on_off: ON after second transition");
}

// ---- reinitialize clears state ----

static void
test_reinitialize_clears_flows()
{
    auto cfg = makeCfg("constant", "all_pairs");
    TrafficMatrix tm(cfg);
    tm.Initialize(5, 0.0);
    check(tm.GetActiveFlows().size() == 10, "reinit: 10 flows for 5 nodes");

    tm.Initialize(3, 0.0);
    check(tm.GetActiveFlows().size() == 3, "reinit: 3 flows after re-init with 3 nodes");
}

// ---- main ----

int
main()
{
    // all_pairs
    test_all_pairs_flow_count();
    test_all_pairs_no_self_flows();
    test_all_pairs_demand();

    // gateway
    test_gateway_default_node();
    test_gateway_explicit_node();

    // error handling
    test_unknown_topology_throws();

    // flow lifecycle
    test_flow_expiration();
    test_permanent_flows_survive();

    // GetDemand
    test_get_demand_nonexistent_pair();

    // on_off model
    test_on_off_initial_phase_is_on();
    test_on_off_transitions();

    // reinitialize
    test_reinitialize_clears_flows();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.\n";
    if (g_fail > 0)
    {
        std::cout << "SOME TESTS FAILED.\n";
        return 1;
    }
    std::cout << "All tests passed.\n";
    return 0;
}
