/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file routing-extra-test.cc
 * @brief Additional unit tests for MeshRouter: hop-limit semantics for every
 *        algorithm, edge-existence rules, FlowResult fields, congestion
 *        invariants, determinism, and optimality against brute force.
 *
 * Complements tests/unit/routing/mesh-router-test.cc (not duplicated here).
 * Standalone binary; ns-3 headers are stubbed from tests/common/.
 * Build and run: `make -C tests/unit/routing-extra test`.
 */

#include "src/routing/mesh-router.h"
#include "src/eval/link-table.h"
#include "src/domain/link-result.h"
#include "src/domain/mesh-config.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>
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

/** Full N x N description of a graph; cap[i][j] == cap[j][i]. Unset pairs have capacity 0. */
struct Graph
{
    uint32_t n;
    std::vector<std::vector<double>> cap;
    std::vector<std::vector<double>> dist;

    explicit Graph(uint32_t nodes)
        : n(nodes),
          cap(nodes, std::vector<double>(nodes, 0.0)),
          dist(nodes, std::vector<double>(nodes, 100.0))
    {
    }

    void Set(uint32_t a, uint32_t b, double c, double d = 100.0)
    {
        cap[a][b] = cap[b][a] = c;
        dist[a][b] = dist[b][a] = d;
    }

    LinkTable Table() const
    {
        std::vector<LinkResult> results;
        for (uint32_t i = 0; i < n; ++i)
        {
            for (uint32_t j = i + 1; j < n; ++j)
            {
                LinkResult r;
                r.tx_id = i;
                r.rx_id = j;
                r.distance_m = dist[i][j];
                r.capacity_mbps = cap[i][j];
                r.sinr_db = cap[i][j] > 0.0 ? 10.0 : -20.0;
                results.push_back(r);
            }
        }
        LinkTable t;
        t.Update(n, results);
        return t;
    }
};

static Flow
makeFlow(uint32_t src, uint32_t dst, double demand, bool active = true, bool on = true)
{
    Flow f;
    f.src = src;
    f.dst = dst;
    f.demand_mbps = demand;
    f.active = active;
    f.in_on_phase = on;
    return f;
}

static RoutingConfig
makeCfg(const std::string& algo, uint32_t maxHops)
{
    RoutingConfig cfg;
    cfg.algorithm = algo;
    cfg.max_hops = maxHops;
    return cfg;
}

static const std::vector<std::string> kAlgos = {"shortest_path", "max_throughput", "min_hop"};

static std::vector<uint32_t>
seq(uint32_t from, uint32_t to)
{
    std::vector<uint32_t> v;
    for (uint32_t i = from; i <= to; ++i)
        v.push_back(i);
    return v;
}

static double
expectedLatency(const Graph& g, const std::vector<uint32_t>& path)
{
    double t = 0.0;
    for (size_t i = 0; i + 1 < path.size(); ++i)
        t += g.dist[path[i]][path[i + 1]] / 3e8 * 1000.0 + 0.5;
    return t;
}

/** Fields an unroutable FlowResult must have (mesh-router.h FlowResult doc). */
static bool
isCleanUnroutable(const FlowResult& r)
{
    return !r.routable && r.path.empty() && r.hop_count == 0 && r.delivered_mbps == 0.0 &&
           r.latency_ms == 0.0;
}

/** Deterministic 64-bit LCG so random graphs are reproducible. */
struct Rng
{
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
    uint64_t Next()
    {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return s >> 33;
    }
    double Uniform(double lo, double hi)
    {
        return lo + (hi - lo) * (static_cast<double>(Next() % 1000000) / 1000000.0);
    }
    bool Chance(double p) { return Uniform(0.0, 1.0) < p; }
};

static Graph
randomGraph(Rng& rng, uint32_t n, double density)
{
    Graph g(n);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = i + 1; j < n; ++j)
            if (rng.Chance(density))
                g.Set(i, j, rng.Uniform(1.0, 1000.0), rng.Uniform(10.0, 3000.0));
    return g;
}

/** Brute-force optimum over all simple paths from src. */
struct BruteBest
{
    std::vector<bool>   reach;
    std::vector<double> minCost;   // min sum of 1/cap
    std::vector<double> maxWidth;  // max bottleneck
    std::vector<uint32_t> minHops; // min hop count
};

static void
bruteDfs(const Graph& g, uint32_t u, std::vector<bool>& onPath, double cost, double width,
         uint32_t hops, BruteBest& b)
{
    for (uint32_t v = 0; v < g.n; ++v)
    {
        if (onPath[v] || g.cap[u][v] <= 0.0)
            continue;
        double c = cost + 1.0 / g.cap[u][v];
        double w = std::min(width, g.cap[u][v]);
        uint32_t h = hops + 1;
        b.reach[v] = true;
        b.minCost[v] = std::min(b.minCost[v], c);
        b.maxWidth[v] = std::max(b.maxWidth[v], w);
        b.minHops[v] = std::min(b.minHops[v], h);
        onPath[v] = true;
        bruteDfs(g, v, onPath, c, w, h, b);
        onPath[v] = false;
    }
}

static BruteBest
bruteForce(const Graph& g, uint32_t src)
{
    BruteBest b;
    b.reach.assign(g.n, false);
    b.minCost.assign(g.n, std::numeric_limits<double>::infinity());
    b.maxWidth.assign(g.n, 0.0);
    b.minHops.assign(g.n, std::numeric_limits<uint32_t>::max());
    std::vector<bool> onPath(g.n, false);
    onPath[src] = true;
    bruteDfs(g, src, onPath, 0.0, std::numeric_limits<double>::infinity(), 0, b);
    return b;
}

/** True if path is a simple src->dst path using only edges with capacity > 0. */
static bool
validPath(const Graph& g, const std::vector<uint32_t>& p, uint32_t src, uint32_t dst)
{
    if (p.size() < 2 || p.front() != src || p.back() != dst)
        return false;
    std::vector<bool> seen(g.n, false);
    for (size_t i = 0; i < p.size(); ++i)
    {
        if (p[i] >= g.n || seen[p[i]])
            return false;
        seen[p[i]] = true;
        if (i + 1 < p.size() && !(g.cap[p[i]][p[i + 1]] > 0.0))
            return false;
    }
    return true;
}

static double
pathCost(const Graph& g, const std::vector<uint32_t>& p)
{
    double c = 0.0;
    for (size_t i = 0; i + 1 < p.size(); ++i)
        c += 1.0 / g.cap[p[i]][p[i + 1]];
    return c;
}

static double
pathWidth(const Graph& g, const std::vector<uint32_t>& p)
{
    double w = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 1 < p.size(); ++i)
        w = std::min(w, g.cap[p[i]][p[i + 1]]);
    return w;
}

// ---- empty / single-node / FlowResult field tests ----

static void
test_unroutable_keeps_demand()
{
    // No path: demand_mbps still reports the effective demand, other fields zero.
    Graph g(3);
    g.Set(0, 1, 100.0);
    MeshRouter router(RoutingConfig{});
    auto res = router.Route(g.Table(), {makeFlow(0, 2, 42.0)}, 3);
    check(isCleanUnroutable(res[0]), "unroutable_keeps_demand: clean unroutable fields");
    check(res[0].demand_mbps == 42.0, "unroutable_keeps_demand: demand 42");
}

static void
test_results_order_and_endpoints()
{
    // Mixed flows: results must be one per flow, same order, src/dst copied.
    Graph g(5);
    g.Set(0, 1, 100.0);
    g.Set(1, 2, 100.0);
    g.Set(3, 4, 100.0);
    std::vector<Flow> flows = {makeFlow(2, 0, 10.0), makeFlow(0, 4, 10.0), makeFlow(3, 3, 10.0),
                               makeFlow(4, 3, 0.0),  makeFlow(1, 2, 10.0, false),
                               makeFlow(4, 3, 10.0), makeFlow(0, 1, 10.0)};
    MeshRouter router(RoutingConfig{});
    auto res = router.Route(g.Table(), flows, 5);
    check(res.size() == flows.size(), "results_order: one result per flow");
    bool endpointsOk = true;
    for (size_t i = 0; i < res.size(); ++i)
        endpointsOk = endpointsOk && res[i].src == flows[i].src && res[i].dst == flows[i].dst;
    check(endpointsOk, "results_order: src/dst match input order");
    check(res[0].routable && res[0].path == std::vector<uint32_t>({2, 1, 0}),
          "results_order: reverse multi-hop path [2,1,0]");
    check(!res[1].routable, "results_order: disconnected component unroutable");
    check(res[2].routable && res[2].path.empty(), "results_order: self-flow");
    check(!res[3].routable, "results_order: zero-demand unroutable");
    check(!res[4].routable, "results_order: inactive unroutable");
    check(res[5].routable && res[5].path == std::vector<uint32_t>({4, 3}), "results_order: [4,3]");
    check(res[6].routable && res[6].path == std::vector<uint32_t>({0, 1}), "results_order: [0,1]");
}

// ---- edge-existence rules ----

static void
test_capacity_zero_negative_tiny()
{
    // Edge exists iff capacity_mbps > 0 (routing CLAUDE.md).
    for (const auto& algo : kAlgos)
    {
        const std::string t = "capacity_edges[" + algo + "]: ";
        MeshRouter router(makeCfg(algo, 0));

        Graph zero(2);
        zero.Set(0, 1, 0.0);
        check(isCleanUnroutable(router.Route(zero.Table(), {makeFlow(0, 1, 1.0)}, 2)[0]),
              t + "capacity 0 is no edge");

        Graph neg(2);
        neg.Set(0, 1, -10.0);
        check(isCleanUnroutable(router.Route(neg.Table(), {makeFlow(0, 1, 1.0)}, 2)[0]),
              t + "negative capacity is no edge");

        Graph tiny(2);
        tiny.Set(0, 1, 1e-9);
        auto r = router.Route(tiny.Table(), {makeFlow(0, 1, 10.0)}, 2)[0];
        check(r.routable && r.hop_count == 1, t + "tiny positive capacity is an edge");
        check(approx(r.delivered_mbps, 1e-9, 1e-6), t + "tiny edge delivers its capacity");
    }
}

static void
test_tiny_edge_used_only_when_needed()
{
    // 0-1 tiny (1e-6), 0-2-1 healthy. Shortest/widest avoid the tiny edge;
    // min_hop uses it (fewest hops, ignores quality).
    Graph g(3);
    g.Set(0, 1, 1e-6);
    g.Set(0, 2, 100.0);
    g.Set(2, 1, 100.0);
    auto tbl = g.Table();
    auto sp = MeshRouter(makeCfg("shortest_path", 0)).Route(tbl, {makeFlow(0, 1, 1.0)}, 3)[0];
    auto mt = MeshRouter(makeCfg("max_throughput", 0)).Route(tbl, {makeFlow(0, 1, 1.0)}, 3)[0];
    auto mh = MeshRouter(makeCfg("min_hop", 0)).Route(tbl, {makeFlow(0, 1, 1.0)}, 3)[0];
    check(sp.path == std::vector<uint32_t>({0, 2, 1}), "tiny_edge: shortest avoids tiny edge");
    check(mt.path == std::vector<uint32_t>({0, 2, 1}), "tiny_edge: widest avoids tiny edge");
    check(mh.path == std::vector<uint32_t>({0, 1}), "tiny_edge: min_hop uses tiny edge");
    check(approx(mh.delivered_mbps, 1e-6, 1e-6), "tiny_edge: min_hop delivered limited to 1e-6");
}

// ---- hop limit ----

static void
test_max_hops_zero_unlimited_long_line()
{
    // 20-node line, 19 hops: routable with max_hops = 0 for every algorithm.
    const uint32_t n = 20;
    Graph g(n);
    for (uint32_t i = 0; i + 1 < n; ++i)
        g.Set(i, i + 1, 100.0 + i, 150.0);
    auto tbl = g.Table();
    for (const auto& algo : kAlgos)
    {
        const std::string t = "max_hops_zero_line[" + algo + "]: ";
        MeshRouter router(makeCfg(algo, 0));
        auto r = router.Route(tbl, {makeFlow(0, n - 1, 10.0), makeFlow(n - 1, 0, 10.0)}, n);
        check(r[0].routable && r[0].hop_count == n - 1, t + "19 hops routable");
        check(r[0].path == seq(0, n - 1), t + "path is 0..19");
        check(approx(r[0].latency_ms, 19 * (150.0 / 3e8 * 1000.0 + 0.5)), t + "latency formula");
        auto rev = seq(0, n - 1);
        std::reverse(rev.begin(), rev.end());
        check(r[1].routable && r[1].path == rev, t + "reverse path is 19..0");
    }
}

static void
test_max_hops_boundary_all_algorithms()
{
    // 6-node line (5 hops). max_hops == 5 accepts, max_hops == 4 rejects.
    Graph g(6);
    for (uint32_t i = 0; i + 1 < 6; ++i)
        g.Set(i, i + 1, 100.0);
    auto tbl = g.Table();
    for (const auto& algo : kAlgos)
    {
        const std::string t = "max_hops_boundary[" + algo + "]: ";
        auto ok = MeshRouter(makeCfg(algo, 5)).Route(tbl, {makeFlow(0, 5, 1.0)}, 6)[0];
        auto bad = MeshRouter(makeCfg(algo, 4)).Route(tbl, {makeFlow(0, 5, 1.0)}, 6)[0];
        auto one = MeshRouter(makeCfg(algo, 1)).Route(tbl, {makeFlow(0, 1, 1.0), makeFlow(0, 2, 1.0)}, 6);
        check(ok.routable && ok.hop_count == 5, t + "hops == max_hops routable");
        check(isCleanUnroutable(bad), t + "hops == max_hops+1 unroutable, clean fields");
        check(bad.demand_mbps == 1.0, t + "hop-limited flow keeps demand");
        check(one[0].routable && one[0].hop_count == 1, t + "max_hops 1 allows direct");
        check(!one[1].routable, t + "max_hops 1 rejects 2 hops");
    }
}

static void
test_max_hops_no_fallback_documented()
{
    // Documented (routing README "Conventions", CLAUDE.md): a chosen path
    // longer than max_hops makes the flow unroutable; no other path is tried.
    // Graph: direct 0-3 cap 100; 0-1-2-3 cap 1000 per hop.
    // shortest_path: 3 * 1/1000 < 1/100 -> 3-hop path; widest: 1000 > 100 -> 3 hops.
    Graph g(4);
    g.Set(0, 3, 100.0);
    g.Set(0, 1, 1000.0);
    g.Set(1, 2, 1000.0);
    g.Set(2, 3, 1000.0);
    auto tbl = g.Table();

    auto spFree = MeshRouter(makeCfg("shortest_path", 0)).Route(tbl, {makeFlow(0, 3, 1.0)}, 4)[0];
    auto mtFree = MeshRouter(makeCfg("max_throughput", 0)).Route(tbl, {makeFlow(0, 3, 1.0)}, 4)[0];
    check(spFree.path == std::vector<uint32_t>({0, 1, 2, 3}), "no_fallback: shortest picks 3 hops");
    check(mtFree.path == std::vector<uint32_t>({0, 1, 2, 3}), "no_fallback: widest picks 3 hops");

    auto sp = MeshRouter(makeCfg("shortest_path", 2)).Route(tbl, {makeFlow(0, 3, 1.0)}, 4)[0];
    auto mt = MeshRouter(makeCfg("max_throughput", 2)).Route(tbl, {makeFlow(0, 3, 1.0)}, 4)[0];
    auto mh = MeshRouter(makeCfg("min_hop", 2)).Route(tbl, {makeFlow(0, 3, 1.0)}, 4)[0];
    check(isCleanUnroutable(sp), "no_fallback: shortest unroutable despite 1-hop direct link");
    check(isCleanUnroutable(mt), "no_fallback: widest unroutable despite 1-hop direct link");
    check(mh.routable && mh.path == std::vector<uint32_t>({0, 3}), "no_fallback: min_hop direct");
}

// ---- algorithm dispatch ----

static void
test_algorithms_pick_different_paths()
{
    // Graph where all three algorithms disagree, 0 -> 1:
    //   direct 0-1 cap 10                (cost 0.1,     width 10,  1 hop)
    //   0-2-1 caps 15, 15               (cost 0.1333,  width 15,  2 hops)
    //   0-3-4-1 caps 12, 200, 200       (cost 0.0933,  width 12,  3 hops)
    // shortest_path -> [0,3,4,1], max_throughput -> [0,2,1], min_hop -> [0,1].
    Graph g(5);
    g.Set(0, 1, 10.0);
    g.Set(0, 2, 15.0);
    g.Set(2, 1, 15.0);
    g.Set(0, 3, 12.0);
    g.Set(3, 4, 200.0);
    g.Set(4, 1, 200.0);
    auto tbl = g.Table();
    auto pathFor = [&](const std::string& algo) {
        return MeshRouter(makeCfg(algo, 0)).Route(tbl, {makeFlow(0, 1, 1.0)}, 5)[0].path;
    };
    check(pathFor("shortest_path") == std::vector<uint32_t>({0, 3, 4, 1}),
          "algo_dispatch: shortest_path min sum 1/cap");
    check(pathFor("max_throughput") == std::vector<uint32_t>({0, 2, 1}),
          "algo_dispatch: max_throughput widest");
    check(pathFor("min_hop") == std::vector<uint32_t>({0, 1}), "algo_dispatch: min_hop direct");
}

static void
test_min_hop_tie_lower_index_first()
{
    // FindPathMinHop doc: ties broken by BFS expansion order, lower index first.
    Graph g(4);
    g.Set(0, 2, 1.0);
    g.Set(2, 3, 1.0);
    g.Set(0, 1, 1.0);
    g.Set(1, 3, 1.0);
    auto r = MeshRouter(makeCfg("min_hop", 0)).Route(g.Table(), {makeFlow(0, 3, 0.5)}, 4)[0];
    check(r.path == std::vector<uint32_t>({0, 1, 3}), "min_hop_tie: via lower index 1");
}

static void
test_grid_min_hop_manhattan()
{
    // 5x4 grid (20 nodes), unit-capacity 4-neighbour links: min_hop hop count
    // from corner 0 equals Manhattan distance for every destination.
    const uint32_t W = 5, H = 4, n = W * H;
    Graph g(n);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x)
        {
            uint32_t id = y * W + x;
            if (x + 1 < W)
                g.Set(id, id + 1, 50.0);
            if (y + 1 < H)
                g.Set(id, id + W, 50.0);
        }
    auto tbl = g.Table();
    std::vector<Flow> flows;
    for (uint32_t d = 1; d < n; ++d)
        flows.push_back(makeFlow(0, d, 0.01));
    auto res = MeshRouter(makeCfg("min_hop", 0)).Route(tbl, flows, n);
    bool ok = true;
    for (uint32_t d = 1; d < n; ++d)
    {
        const auto& r = res[d - 1];
        uint32_t manhattan = (d % W) + (d / W);
        ok = ok && r.routable && r.hop_count == manhattan && validPath(g, r.path, 0, d) &&
             r.path.size() == r.hop_count + 1;
    }
    check(ok, "grid_min_hop: hop_count == Manhattan distance for all 19 destinations");

    // Same grid with shortest_path and equal capacities: also minimum hops.
    auto sres = MeshRouter(makeCfg("shortest_path", 0)).Route(tbl, flows, n);
    bool sok = true;
    for (uint32_t d = 1; d < n; ++d)
        sok = sok && sres[d - 1].hop_count == (d % W) + (d / W);
    check(sok, "grid_shortest_equal_caps: hop_count == Manhattan distance");
}

static void
test_bruteforce_optimality()
{
    // Random graphs (seeded): each algorithm must be optimal for its own
    // metric over all simple paths, reachability must match, paths valid.
    int graphs = 0;
    bool reachOk = true, validOk = true, spOk = true, mtOk = true, mhOk = true, fieldsOk = true;
    for (uint64_t seed = 1; seed <= 60; ++seed)
    {
        Rng rng(seed);
        uint32_t n = 3 + static_cast<uint32_t>(rng.Next() % 6);  // 3..8
        double density = rng.Uniform(0.2, 0.7);
        Graph g = randomGraph(rng, n, density);
        auto tbl = g.Table();
        ++graphs;

        std::vector<Flow> flows;
        for (uint32_t s = 0; s < n; ++s)
            for (uint32_t d = 0; d < n; ++d)
                if (s != d)
                    flows.push_back(makeFlow(s, d, 1e-6));  // tiny demand: no congestion

        auto sp = MeshRouter(makeCfg("shortest_path", 0)).Route(tbl, flows, n);
        auto mt = MeshRouter(makeCfg("max_throughput", 0)).Route(tbl, flows, n);
        auto mh = MeshRouter(makeCfg("min_hop", 0)).Route(tbl, flows, n);

        for (size_t i = 0; i < flows.size(); ++i)
        {
            uint32_t s = flows[i].src, d = flows[i].dst;
            BruteBest b = bruteForce(g, s);
            for (const auto* res : {&sp, &mt, &mh})
            {
                const FlowResult& r = (*res)[i];
                reachOk = reachOk && (r.routable == b.reach[d]);
                if (r.routable)
                {
                    validOk = validOk && validPath(g, r.path, s, d);
                    fieldsOk = fieldsOk && r.hop_count + 1 == r.path.size() &&
                               approx(r.latency_ms, expectedLatency(g, r.path)) &&
                               approx(r.delivered_mbps, 1e-6);
                }
                else
                {
                    fieldsOk = fieldsOk && isCleanUnroutable(r);
                }
            }
            if (b.reach[d])
            {
                spOk = spOk && sp[i].routable && approx(pathCost(g, sp[i].path), b.minCost[d], 1e-9);
                mtOk = mtOk && mt[i].routable && pathWidth(g, mt[i].path) == b.maxWidth[d];
                mhOk = mhOk && mh[i].routable && mh[i].hop_count == b.minHops[d];
            }
        }
    }
    check(graphs == 60, "bruteforce: 60 graphs generated");
    check(reachOk, "bruteforce: routable iff reachable (all algorithms)");
    check(validOk, "bruteforce: every path is simple, src->dst, over capacity>0 edges");
    check(fieldsOk, "bruteforce: hop_count/latency/delivered consistent with path");
    check(spOk, "bruteforce: shortest_path minimises sum 1/capacity");
    check(mtOk, "bruteforce: max_throughput maximises bottleneck");
    check(mhOk, "bruteforce: min_hop minimises hop count");
}

static void
test_max_hops_consistent_with_unlimited()
{
    // Documented semantics: with max_hops = k > 0 a flow is routed on exactly
    // the unlimited path if that path has <= k hops, else it is unroutable.
    bool ok = true;
    for (uint64_t seed = 100; seed < 130; ++seed)
    {
        Rng rng(seed);
        uint32_t n = 4 + static_cast<uint32_t>(rng.Next() % 5);
        Graph g = randomGraph(rng, n, 0.35);
        auto tbl = g.Table();
        std::vector<Flow> flows;
        for (uint32_t s = 0; s < n; ++s)
            for (uint32_t d = 0; d < n; ++d)
                if (s != d)
                    flows.push_back(makeFlow(s, d, 1e-6));
        for (const auto& algo : kAlgos)
        {
            auto freeRes = MeshRouter(makeCfg(algo, 0)).Route(tbl, flows, n);
            for (uint32_t k = 1; k <= 3; ++k)
            {
                auto lim = MeshRouter(makeCfg(algo, k)).Route(tbl, flows, n);
                for (size_t i = 0; i < flows.size(); ++i)
                {
                    bool expectRoutable = freeRes[i].routable && freeRes[i].hop_count <= k;
                    ok = ok && lim[i].routable == expectRoutable;
                    if (expectRoutable)
                        ok = ok && lim[i].path == freeRes[i].path;
                    else
                        ok = ok && isCleanUnroutable(lim[i]);
                }
            }
        }
    }
    check(ok, "max_hops_consistency: limited result == unlimited result filtered by hop count");
}

// ---- congestion ----

static void
test_congestion_opposite_directions_share_edge()
{
    // Per undirected edge: 0->1 and 1->0 (60 each) share cap 100 -> 50 each.
    Graph g(2);
    g.Set(0, 1, 100.0);
    auto r = MeshRouter(RoutingConfig{}).Route(g.Table(), {makeFlow(0, 1, 60.0), makeFlow(1, 0, 60.0)}, 2);
    check(approx(r[0].delivered_mbps, 50.0), "congestion_opposite: 0->1 gets 50");
    check(approx(r[1].delivered_mbps, 50.0), "congestion_opposite: 1->0 gets 50");
    check(r[0].demand_mbps == 60.0 && r[1].demand_mbps == 60.0, "congestion_opposite: demand unchanged");
}

static void
test_congestion_ignores_non_contributors()
{
    // Inactive, off-phase, zero-demand, self, and unroutable flows add no
    // demand to the 0-1 edge (cap 100). The single real flow (80) is unscaled.
    Graph g(3);
    g.Set(0, 1, 100.0);
    std::vector<Flow> flows = {makeFlow(0, 1, 80.0),
                               makeFlow(0, 1, 80.0, false, true),
                               makeFlow(1, 0, 80.0, true, false),
                               makeFlow(0, 1, 0.0),
                               makeFlow(0, 0, 500.0),
                               makeFlow(1, 1, 500.0),
                               makeFlow(0, 2, 500.0)};
    auto r = MeshRouter(RoutingConfig{}).Route(g.Table(), flows, 3);
    check(approx(r[0].delivered_mbps, 80.0), "congestion_non_contributors: real flow unscaled");
    bool othersZero = true;
    for (size_t i = 1; i < r.size(); ++i)
        othersZero = othersZero && r[i].delivered_mbps == 0.0;
    check(othersZero, "congestion_non_contributors: others deliver 0");
}

static void
test_congestion_uses_demand_not_delivered()
{
    // A: 0->2 via 0-1-2 (100), B: 1->2 (100), C: 0->1 (100).
    // Edge 0-1 cap 50: A+C demand 200 -> scale 0.25. Edge 1-2 cap 200: A+B
    // demand 200 -> not overloaded. Spare capacity is not redistributed.
    Graph g(3);
    g.Set(0, 1, 50.0);
    g.Set(1, 2, 200.0);
    auto r = MeshRouter(makeCfg("min_hop", 0))
                 .Route(g.Table(), {makeFlow(0, 2, 100.0), makeFlow(1, 2, 100.0), makeFlow(0, 1, 100.0)}, 3);
    check(r[0].path == std::vector<uint32_t>({0, 1, 2}), "congestion_demand: A path 0-1-2");
    check(approx(r[0].delivered_mbps, 25.0), "congestion_demand: A 25");
    check(approx(r[1].delivered_mbps, 100.0), "congestion_demand: B 100 (edge 1-2 not overloaded)");
    check(approx(r[2].delivered_mbps, 25.0), "congestion_demand: C 25");
}

static void
test_congestion_min_over_hops_three_edges()
{
    // Flow 0->3 on line 0-1-2-3 with caps 300, 30, 90 and cross traffic.
    // Edge 0-1: 0->3 (60) alone -> no scale. Edge 1-2: 0->3 (60) + 1->2 (60)
    // = 120 > 30 -> 0.25. Edge 2-3: 0->3 (60) + 3->2 (60) = 120 > 90 -> 0.75.
    Graph g(4);
    g.Set(0, 1, 300.0);
    g.Set(1, 2, 30.0);
    g.Set(2, 3, 90.0);
    auto r = MeshRouter(RoutingConfig{})
                 .Route(g.Table(), {makeFlow(0, 3, 60.0), makeFlow(1, 2, 60.0), makeFlow(3, 2, 60.0)}, 4);
    check(approx(r[0].delivered_mbps, 15.0), "congestion_min_hops: 0->3 takes min factor 0.25");
    check(approx(r[1].delivered_mbps, 15.0), "congestion_min_hops: 1->2 scaled 0.25");
    check(approx(r[2].delivered_mbps, 45.0), "congestion_min_hops: 3->2 scaled 0.75");
}

static void
test_latency_unchanged_by_congestion()
{
    Graph g(3);
    g.Set(0, 1, 10.0, 300000.0);  // 1 ms propagation
    g.Set(1, 2, 10.0, 3000.0);    // 0.01 ms propagation
    auto lightly = MeshRouter(RoutingConfig{}).Route(g.Table(), {makeFlow(0, 2, 1.0)}, 3)[0];
    auto heavy = MeshRouter(RoutingConfig{})
                     .Route(g.Table(), {makeFlow(0, 2, 1000.0), makeFlow(2, 0, 1000.0)}, 3);
    check(approx(lightly.latency_ms, 1.0 + 0.5 + 0.01 + 0.5), "latency_formula: 2 hops = 2.01 ms");
    check(heavy[0].delivered_mbps < 10.0, "latency_congestion: flow was scaled");
    check(heavy[0].latency_ms == lightly.latency_ms, "latency_congestion: latency unchanged");
    check(heavy[0].path == lightly.path, "latency_congestion: path unchanged");
    check(approx(heavy[1].latency_ms, lightly.latency_ms), "latency_symmetric: reverse same latency");
}

static void
test_delivered_invariants_random()
{
    // Random graphs + random flows (incl. inactive, off-phase, self, zero,
    // opposite directions). Checks the documented congestion model exactly:
    //   delivered = demand * min(1, min over hops of cap_e / totalDemand_e)
    // and 0 <= delivered <= demand, delivered <= path bottleneck, and the
    // delivered sum per undirected edge <= capacity.
    bool boundsOk = true, formulaOk = true, edgeOk = true, bottleneckOk = true;
    for (uint64_t seed = 200; seed < 260; ++seed)
    {
        Rng rng(seed);
        uint32_t n = 3 + static_cast<uint32_t>(rng.Next() % 7);
        Graph g = randomGraph(rng, n, 0.5);
        auto tbl = g.Table();
        std::vector<Flow> flows;
        uint32_t nf = 1 + static_cast<uint32_t>(rng.Next() % 12);
        for (uint32_t k = 0; k < nf; ++k)
        {
            uint32_t s = static_cast<uint32_t>(rng.Next() % n);
            uint32_t d = static_cast<uint32_t>(rng.Next() % n);
            double dem = rng.Chance(0.1) ? 0.0 : rng.Uniform(1.0, 800.0);
            flows.push_back(makeFlow(s, d, dem, !rng.Chance(0.15), !rng.Chance(0.15)));
        }
        const std::string algo = kAlgos[seed % 3];
        auto res = MeshRouter(makeCfg(algo, 0)).Route(tbl, flows, n);

        std::map<std::pair<uint32_t, uint32_t>, double> demandSum, deliveredSum;
        for (const auto& r : res)
            for (size_t h = 0; h + 1 < r.path.size(); ++h)
            {
                auto key = std::make_pair(std::min(r.path[h], r.path[h + 1]),
                                          std::max(r.path[h], r.path[h + 1]));
                demandSum[key] += r.demand_mbps;
                deliveredSum[key] += r.delivered_mbps;
            }
        for (const auto& r : res)
        {
            boundsOk = boundsOk && r.delivered_mbps >= 0.0 &&
                       r.delivered_mbps <= r.demand_mbps * (1.0 + 1e-12);
            if (r.path.size() >= 2)
            {
                bottleneckOk = bottleneckOk &&
                               r.delivered_mbps <= pathWidth(g, r.path) * (1.0 + 1e-9);
                double factor = 1.0;
                for (size_t h = 0; h + 1 < r.path.size(); ++h)
                {
                    auto key = std::make_pair(std::min(r.path[h], r.path[h + 1]),
                                              std::max(r.path[h], r.path[h + 1]));
                    double c = g.cap[key.first][key.second];
                    if (demandSum[key] > c)
                        factor = std::min(factor, c / demandSum[key]);
                }
                formulaOk = formulaOk && approx(r.delivered_mbps, r.demand_mbps * factor);
            }
        }
        for (const auto& [key, sum] : deliveredSum)
            edgeOk = edgeOk && sum <= g.cap[key.first][key.second] * (1.0 + 1e-9);
    }
    check(boundsOk, "delivered_invariants: 0 <= delivered <= demand");
    check(bottleneckOk, "delivered_invariants: delivered <= path bottleneck capacity");
    check(formulaOk, "delivered_invariants: delivered == demand * min-over-hops scale");
    check(edgeOk, "delivered_invariants: per-edge delivered sum <= capacity");
}

// ---- determinism ----

static bool
sameResult(const FlowResult& a, const FlowResult& b, bool exact)
{
    bool num = exact ? (a.delivered_mbps == b.delivered_mbps && a.latency_ms == b.latency_ms)
                     : (approx(a.delivered_mbps, b.delivered_mbps) && approx(a.latency_ms, b.latency_ms));
    return num && a.src == b.src && a.dst == b.dst && a.demand_mbps == b.demand_mbps &&
           a.hop_count == b.hop_count && a.path == b.path && a.routable == b.routable;
}

static void
test_determinism_and_order_independence()
{
    bool repeatOk = true, instanceOk = true, permOk = true;
    for (uint64_t seed = 300; seed < 330; ++seed)
    {
        Rng rng(seed);
        uint32_t n = 4 + static_cast<uint32_t>(rng.Next() % 8);
        Graph g = randomGraph(rng, n, 0.4);
        // Add equal-capacity duplicates to create ties.
        for (uint32_t i = 0; i + 2 < n; i += 2)
            g.Set(i, i + 2, 100.0);
        auto tbl = g.Table();
        std::vector<Flow> flows;
        for (uint32_t k = 0; k < 10; ++k)
            flows.push_back(makeFlow(static_cast<uint32_t>(rng.Next() % n),
                                     static_cast<uint32_t>(rng.Next() % n), rng.Uniform(10.0, 500.0)));
        const std::string algo = kAlgos[seed % 3];
        MeshRouter router(makeCfg(algo, 0));
        auto a = router.Route(tbl, flows, n);
        auto b = router.Route(tbl, flows, n);
        auto c = MeshRouter(makeCfg(algo, 0)).Route(tbl, flows, n);

        std::vector<Flow> rev(flows.rbegin(), flows.rend());
        auto d = router.Route(tbl, rev, n);
        for (size_t i = 0; i < flows.size(); ++i)
        {
            repeatOk = repeatOk && sameResult(a[i], b[i], true);
            instanceOk = instanceOk && sameResult(a[i], c[i], true);
            permOk = permOk && sameResult(a[i], d[flows.size() - 1 - i], false);
        }
    }
    check(repeatOk, "determinism: repeated Route() calls identical");
    check(instanceOk, "determinism: fresh router instance identical");
    check(permOk, "determinism: per-flow result independent of flow order");
}

int
main()
{
    // Empty / single node / FlowResult fields
    test_unroutable_keeps_demand();
    test_results_order_and_endpoints();

    // Edge existence
    test_capacity_zero_negative_tiny();
    test_tiny_edge_used_only_when_needed();

    // Hop limit
    test_max_hops_zero_unlimited_long_line();
    test_max_hops_boundary_all_algorithms();
    test_max_hops_no_fallback_documented();
    test_max_hops_consistent_with_unlimited();

    // Algorithms and optimality
    test_algorithms_pick_different_paths();
    test_min_hop_tie_lower_index_first();
    test_grid_min_hop_manhattan();
    test_bruteforce_optimality();

    // Congestion and latency
    test_congestion_opposite_directions_share_edge();
    test_congestion_ignores_non_contributors();
    test_congestion_uses_demand_not_delivered();
    test_congestion_min_over_hops_three_edges();
    test_latency_unchanged_by_congestion();
    test_delivered_invariants_random();

    // Determinism
    test_determinism_and_order_independence();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.\n";
    if (g_fail > 0)
    {
        std::cout << "SOME TESTS FAILED.\n";
        return 1;
    }
    std::cout << "All tests passed.\n";
    return 0;
}
