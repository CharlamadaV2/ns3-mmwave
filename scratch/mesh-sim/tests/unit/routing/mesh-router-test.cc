/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Unit tests for MeshRouter.
 * Standalone binary -- no ns-3 dependency.
 *
 * Build:  make
 * Run:    ./mesh-router-test
 */

#include "src/routing/mesh-router.h"
#include "src/eval/link-table.h"
#include "src/domain/link-result.h"
#include "src/domain/mesh-config.h"

#include <cassert>
#include <cmath>
#include <iostream>
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
approx(double a, double b, double tol = 0.01)
{
    return std::fabs(a - b) < tol;
}

// Build a LinkTable from a list of {tx, rx, distance_m, capacity_mbps} tuples.
struct TestLink
{
    uint32_t tx;
    uint32_t rx;
    double   distance_m;
    double   capacity_mbps;
    double   sinr_db;
};

static LinkTable
buildTable(uint32_t numNodes, const std::vector<TestLink>& edges)
{
    std::vector<LinkResult> results;
    for (const auto& e : edges)
    {
        LinkResult r;
        r.tx_id = e.tx;
        r.rx_id = e.rx;
        r.distance_m = e.distance_m;
        r.capacity_mbps = e.capacity_mbps;
        r.sinr_db = e.sinr_db;
        results.push_back(r);
    }
    LinkTable table;
    table.Update(numNodes, results);
    return table;
}

static Flow
makeFlow(uint32_t src, uint32_t dst, double demand)
{
    Flow f;
    f.src = src;
    f.dst = dst;
    f.demand_mbps = demand;
    f.active = true;
    f.in_on_phase = true;
    return f;
}

// ---- path finding tests ----

static void
test_direct_link()
{
    // 2 nodes, direct link
    auto table = buildTable(2, {{0, 1, 100.0, 500.0, 10.0}});
    RoutingConfig cfg;
    cfg.algorithm = "shortest_path";
    MeshRouter router(cfg);

    auto results = router.Route(table, {makeFlow(0, 1, 50.0)}, 2);
    check(results.size() == 1, "direct_link: one result");
    check(results[0].routable, "direct_link: routable");
    check(results[0].path.size() == 2, "direct_link: path length 2");
    check(results[0].path[0] == 0 && results[0].path[1] == 1, "direct_link: path is [0,1]");
    check(results[0].hop_count == 1, "direct_link: hop_count 1");
    check(approx(results[0].delivered_mbps, 50.0), "direct_link: delivered == demand");
}

static void
test_multi_hop()
{
    // 3 nodes: 0--1--2, no direct 0--2 link
    auto table = buildTable(3, {
        {0, 1, 100.0, 500.0, 10.0},
        {0, 2, 1000.0, 0.0, -20.0},
        {1, 2, 200.0, 400.0, 8.0}
    });
    RoutingConfig cfg;
    cfg.algorithm = "shortest_path";
    MeshRouter router(cfg);

    auto results = router.Route(table, {makeFlow(0, 2, 50.0)}, 3);
    check(results.size() == 1, "multi_hop: one result");
    check(results[0].routable, "multi_hop: routable");
    check(results[0].path.size() == 3, "multi_hop: path length 3");
    check(results[0].hop_count == 2, "multi_hop: hop_count 2");
}

static void
test_no_path()
{
    // 3 nodes: 0--1, node 2 disconnected
    auto table = buildTable(3, {
        {0, 1, 100.0, 500.0, 10.0},
        {0, 2, 500.0, 0.0, -20.0},
        {1, 2, 500.0, 0.0, -20.0}
    });
    RoutingConfig cfg;
    cfg.algorithm = "shortest_path";
    MeshRouter router(cfg);

    auto results = router.Route(table, {makeFlow(0, 2, 50.0)}, 3);
    check(results.size() == 1, "no_path: one result");
    check(!results[0].routable, "no_path: not routable");
    check(results[0].path.empty(), "no_path: empty path");
}

static void
test_same_src_dst()
{
    auto table = buildTable(2, {{0, 1, 100.0, 500.0, 10.0}});
    RoutingConfig cfg;
    MeshRouter router(cfg);

    auto results = router.Route(table, {makeFlow(0, 0, 50.0)}, 2);
    check(results.size() == 1, "same_src_dst: one result");
    check(results[0].routable, "same_src_dst: routable (trivial)");
    check(approx(results[0].demand_mbps, 50.0), "same_src_dst: demand preserved");
}

static void
test_inactive_flow()
{
    auto table = buildTable(2, {{0, 1, 100.0, 500.0, 10.0}});
    RoutingConfig cfg;
    MeshRouter router(cfg);

    Flow f = makeFlow(0, 1, 50.0);
    f.active = false;
    auto results = router.Route(table, {f}, 2);
    check(results.size() == 1, "inactive: one result");
    check(!results[0].routable, "inactive: not routable (zero demand)");
}

static void
test_off_phase_flow()
{
    auto table = buildTable(2, {{0, 1, 100.0, 500.0, 10.0}});
    RoutingConfig cfg;
    MeshRouter router(cfg);

    Flow f = makeFlow(0, 1, 50.0);
    f.in_on_phase = false;
    auto results = router.Route(table, {f}, 2);
    check(results.size() == 1, "off_phase: one result");
    check(!results[0].routable, "off_phase: not routable (zero demand)");
}

// ---- algorithm selection tests ----

static void
test_shortest_path_prefers_high_capacity()
{
    // 3 nodes: 0->1 (cap 100), 0->2->1 (cap 1000 each)
    // Shortest path (weight = 1/cap) should prefer the high-capacity path
    auto table = buildTable(3, {
        {0, 1, 100.0, 100.0, 5.0},
        {0, 2, 100.0, 1000.0, 20.0},
        {1, 2, 100.0, 1000.0, 20.0}
    });
    RoutingConfig cfg;
    cfg.algorithm = "shortest_path";
    cfg.max_hops = 0;
    MeshRouter router(cfg);

    auto results = router.Route(table, {makeFlow(0, 1, 10.0)}, 3);
    check(results[0].routable, "shortest_path: routable");
    // weight for direct = 1/100 = 0.01
    // weight for 0->2->1 = 1/1000 + 1/1000 = 0.002
    // So Dijkstra should pick the 2-hop path through node 2
    check(results[0].path.size() == 3, "shortest_path: picks high-cap 2-hop path");
    check(results[0].path[1] == 2, "shortest_path: goes through node 2");
}

static void
test_min_hop_prefers_direct()
{
    // Same topology, but min_hop should prefer the direct 1-hop link
    auto table = buildTable(3, {
        {0, 1, 100.0, 100.0, 5.0},
        {0, 2, 100.0, 1000.0, 20.0},
        {1, 2, 100.0, 1000.0, 20.0}
    });
    RoutingConfig cfg;
    cfg.algorithm = "min_hop";
    cfg.max_hops = 0;
    MeshRouter router(cfg);

    auto results = router.Route(table, {makeFlow(0, 1, 10.0)}, 3);
    check(results[0].routable, "min_hop: routable");
    check(results[0].path.size() == 2, "min_hop: picks direct 1-hop path");
}

static void
test_max_throughput_prefers_widest_bottleneck()
{
    // 4 nodes: two paths from 0 to 3
    //   Path A: 0->1->3  bottleneck = min(200, 200) = 200
    //   Path B: 0->2->3  bottleneck = min(500, 100) = 100
    // Max throughput should pick path A
    auto table = buildTable(4, {
        {0, 1, 100.0, 200.0, 10.0},
        {1, 3, 100.0, 200.0, 10.0},
        {0, 2, 100.0, 500.0, 20.0},
        {2, 3, 100.0, 100.0, 5.0},
        {0, 3, 500.0, 0.0, -20.0},
        {1, 2, 500.0, 0.0, -20.0}
    });
    RoutingConfig cfg;
    cfg.algorithm = "max_throughput";
    cfg.max_hops = 0;
    MeshRouter router(cfg);

    auto results = router.Route(table, {makeFlow(0, 3, 10.0)}, 4);
    check(results[0].routable, "max_throughput: routable");
    check(results[0].path.size() == 3, "max_throughput: 2-hop path");
    check(results[0].path[1] == 1, "max_throughput: picks path through node 1 (wider bottleneck)");
}

// ---- max_hops constraint ----

static void
test_max_hops_rejects_long_path()
{
    // 3 nodes: 0->1->2, only path is 2 hops
    auto table = buildTable(3, {
        {0, 1, 100.0, 500.0, 10.0},
        {1, 2, 100.0, 500.0, 10.0},
        {0, 2, 500.0, 0.0, -20.0}
    });
    RoutingConfig cfg;
    cfg.algorithm = "shortest_path";
    cfg.max_hops = 1; // only allow 1-hop paths
    MeshRouter router(cfg);

    auto results = router.Route(table, {makeFlow(0, 2, 50.0)}, 3);
    check(!results[0].routable, "max_hops: 2-hop path rejected by max_hops=1");
}

// ---- latency tests ----

static void
test_latency_single_hop()
{
    // 300m link: propagation = 300 / 3e8 * 1000 = 0.001 ms, processing = 0.5 ms
    auto table = buildTable(2, {{0, 1, 300.0, 500.0, 10.0}});
    RoutingConfig cfg;
    MeshRouter router(cfg);

    auto results = router.Route(table, {makeFlow(0, 1, 50.0)}, 2);
    double expected = 300.0 / 3e8 * 1000.0 + 0.5;
    check(approx(results[0].latency_ms, expected, 0.001),
          "latency_single_hop: propagation + processing");
}

static void
test_latency_multi_hop()
{
    // 0->1 (100m), 1->2 (200m)
    // hop 1: 100/3e8*1000 + 0.5, hop 2: 200/3e8*1000 + 0.5
    auto table = buildTable(3, {
        {0, 1, 100.0, 500.0, 10.0},
        {1, 2, 200.0, 500.0, 10.0},
        {0, 2, 1000.0, 0.0, -20.0}
    });
    RoutingConfig cfg;
    cfg.max_hops = 0;
    MeshRouter router(cfg);

    auto results = router.Route(table, {makeFlow(0, 2, 50.0)}, 3);
    double expected = (100.0 / 3e8 * 1000.0 + 0.5) + (200.0 / 3e8 * 1000.0 + 0.5);
    check(approx(results[0].latency_ms, expected, 0.001),
          "latency_multi_hop: sums per-hop latency");
}

static void
test_latency_dominated_by_processing()
{
    // At short distances (e.g. 30m), propagation is ~0.0001 ms, so processing (0.5 ms) dominates
    auto table = buildTable(2, {{0, 1, 30.0, 500.0, 10.0}});
    RoutingConfig cfg;
    MeshRouter router(cfg);

    auto results = router.Route(table, {makeFlow(0, 1, 50.0)}, 2);
    check(results[0].latency_ms > 0.49 && results[0].latency_ms < 0.51,
          "latency_short_distance: ~0.5 ms (processing dominates)");
}

// ---- congestion scaling tests ----

static void
test_no_congestion()
{
    // Single flow, 50 Mbps on a 500 Mbps link -- no scaling needed
    auto table = buildTable(2, {{0, 1, 100.0, 500.0, 10.0}});
    RoutingConfig cfg;
    MeshRouter router(cfg);

    auto results = router.Route(table, {makeFlow(0, 1, 50.0)}, 2);
    check(approx(results[0].delivered_mbps, 50.0), "no_congestion: full demand delivered");
}

static void
test_congestion_two_flows()
{
    // Two flows on same link, total demand 600 > capacity 500
    auto table = buildTable(2, {{0, 1, 100.0, 500.0, 10.0}});
    RoutingConfig cfg;
    MeshRouter router(cfg);

    std::vector<Flow> flows = {makeFlow(0, 1, 300.0), makeFlow(0, 1, 300.0)};
    auto results = router.Route(table, flows, 2);

    // scale = 500/600, each flow gets 300 * 500/600 = 250
    check(approx(results[0].delivered_mbps, 250.0, 1.0), "congestion_two: flow 0 scaled");
    check(approx(results[1].delivered_mbps, 250.0, 1.0), "congestion_two: flow 1 scaled");
}

static void
test_congestion_asymmetric()
{
    // Two flows with different demands on same link
    // Flow 0: 100 Mbps, Flow 1: 400 Mbps, capacity: 200 Mbps
    // scale = 200/500 = 0.4
    // Flow 0: min(100, 100*0.4) = 40, Flow 1: min(400, 400*0.4) = 160
    auto table = buildTable(2, {{0, 1, 100.0, 200.0, 10.0}});
    RoutingConfig cfg;
    MeshRouter router(cfg);

    std::vector<Flow> flows = {makeFlow(0, 1, 100.0), makeFlow(0, 1, 400.0)};
    auto results = router.Route(table, flows, 2);

    check(approx(results[0].delivered_mbps, 40.0, 1.0), "congestion_asym: small flow scaled");
    check(approx(results[1].delivered_mbps, 160.0, 1.0), "congestion_asym: large flow scaled");
}

static void
test_congestion_shared_link_in_path()
{
    // 3 nodes: both flows traverse link 1->2
    // Flow 0: 0->1->2 (100 Mbps), Flow 1: 1->2 (100 Mbps)
    // Link 1->2 capacity = 150 Mbps, link 0->1 capacity = 500 Mbps
    // Only link 1->2 is congested: scale = 150/200 = 0.75
    auto table = buildTable(3, {
        {0, 1, 100.0, 500.0, 10.0},
        {1, 2, 100.0, 150.0, 5.0},
        {0, 2, 1000.0, 0.0, -20.0}
    });
    RoutingConfig cfg;
    cfg.max_hops = 0;
    MeshRouter router(cfg);

    std::vector<Flow> flows = {makeFlow(0, 2, 100.0), makeFlow(1, 2, 100.0)};
    auto results = router.Route(table, flows, 3);

    // Both flows use link 1->2, total demand 200 > 150
    check(approx(results[0].delivered_mbps, 75.0, 1.0), "congestion_shared: flow 0 scaled");
    check(approx(results[1].delivered_mbps, 75.0, 1.0), "congestion_shared: flow 1 scaled");
}

// ---- multiple flows ----

static void
test_multiple_independent_flows()
{
    // 4 nodes: 0->1 and 2->3 are independent links
    auto table = buildTable(4, {
        {0, 1, 100.0, 500.0, 10.0},
        {2, 3, 100.0, 400.0, 8.0},
        {0, 2, 500.0, 0.0, -20.0},
        {0, 3, 500.0, 0.0, -20.0},
        {1, 2, 500.0, 0.0, -20.0},
        {1, 3, 500.0, 0.0, -20.0}
    });
    RoutingConfig cfg;
    MeshRouter router(cfg);

    std::vector<Flow> flows = {makeFlow(0, 1, 50.0), makeFlow(2, 3, 80.0)};
    auto results = router.Route(table, flows, 4);

    check(results.size() == 2, "independent: two results");
    check(approx(results[0].delivered_mbps, 50.0), "independent: flow 0 full demand");
    check(approx(results[1].delivered_mbps, 80.0), "independent: flow 1 full demand");
}

// ---- main ----

int
main()
{
    // Path finding
    test_direct_link();
    test_multi_hop();
    test_no_path();
    test_same_src_dst();
    test_inactive_flow();
    test_off_phase_flow();

    // Algorithm selection
    test_shortest_path_prefers_high_capacity();
    test_min_hop_prefers_direct();
    test_max_throughput_prefers_widest_bottleneck();

    // Max hops
    test_max_hops_rejects_long_path();

    // Latency
    test_latency_single_hop();
    test_latency_multi_hop();
    test_latency_dominated_by_processing();

    // Congestion
    test_no_congestion();
    test_congestion_two_flows();
    test_congestion_asymmetric();
    test_congestion_shared_link_in_path();

    // Multiple flows
    test_multiple_independent_flows();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.\n";
    if (g_fail > 0)
    {
        std::cout << "SOME TESTS FAILED.\n";
        return 1;
    }
    std::cout << "All tests passed.\n";
    return 0;
}
