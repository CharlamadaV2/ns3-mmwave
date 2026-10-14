/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file config-rules-test.cc
 * @brief Tests for ComputeTickCount, ResolveRlControl cadence,
 *        ApplyRlControl, and ApplyLayout that complement tests/unit/config.
 *
 * Standalone binary, no ns-3 dependency.
 * Build and run: `make -C tests/unit/config-rules test`.
 */

#include "src/config/config-loader.h"
#include "src/config/config-validator.h"
#include "src/config/layout-override.h"
#include "src/config/rl-control.h"
#include "src/domain/node-spec.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <unistd.h>

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

static const double kNaN = std::numeric_limits<double>::quiet_NaN();
static const double kInf = std::numeric_limits<double>::infinity();

static NodeSpec
makeNode(const std::string& id, double x = 0.0, double y = 0.0, double z = 0.0)
{
    NodeSpec n;
    n.id        = id;
    n.role      = "peer";
    n.mobility  = "fixed";
    n.node_type = "drone";
    n.position  = Position{x, y, z};
    return n;
}

/// Minimal valid non-RL config with two fixed drones.
static SimConfig
makeValid()
{
    SimConfig cfg;
    cfg.scenario_name = "rules";
    cfg.duration_s    = 10.0;
    cfg.warmup_s      = 0.0;
    cfg.tick_s        = 0.1;

    cfg.channel.frequency_ghz = 28.0;
    cfg.channel.bandwidth_mhz = 400.0;
    cfg.channel.channel_model = "3gpp";
    cfg.channel.scenario      = "UMi";

    cfg.mesh.traffic.model         = "constant";
    cfg.mesh.traffic.demand_mbps   = 10.0;
    cfg.mesh.traffic.flow_topology = "all_pairs";
    cfg.mesh.routing.algorithm     = "shortest_path";

    cfg.nodes.push_back(makeNode("node0", 0.0, 0.0, 10.0));
    cfg.nodes.push_back(makeNode("node1", 100.0, 0.0, 10.0));
    return cfg;
}

/// Centralized RL config ("all") with @p n fixed nodes at (10*i, 20, 10).
static SimConfig
makeRlCfg(size_t n = 3)
{
    SimConfig cfg = makeValid();
    cfg.nodes.clear();
    for (size_t i = 0; i < n; ++i)
    {
        cfg.nodes.push_back(makeNode("node-" + std::to_string(i),
                                     10.0 * static_cast<double>(i), 20.0, 10.0));
    }
    cfg.rl.enabled              = true;
    cfg.rl.controlled_nodes_set = true;
    cfg.rl.controlled_nodes     = "all";
    return cfg;
}

static bool
hasIn(const std::vector<std::string>& errors, const std::string& substr)
{
    for (const auto& e : errors)
    {
        if (e.find(substr) != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

static bool
hasResErr(const RlControlResolution& r, const std::string& substr)
{
    return hasIn(r.errors, substr);
}

// =====================================================================
// ComputeTickCount
// =====================================================================

static void
test_tick_count_boundaries()
{
    check(ComputeTickCount(1.0, 1.0, true) == 1 && ComputeTickCount(1.0, 1.0, false) == 1,
          "ticks: ratio exactly 1 gives 1 in both modes");
    check(ComputeTickCount(0.9999999, 1.0, true) == 0,
          "ticks: ratio below 1 gives 0 even within the robust tolerance (doc)");
    check(ComputeTickCount(0.7, 0.1, false) == 6, "ticks: legacy 0.7/0.1 truncates to 6");
    check(ComputeTickCount(0.7, 0.1, true) == 7, "ticks: robust 0.7/0.1 rounds to 7");

    // 1e-6 tolerance on either side of an integer.
    check(ComputeTickCount(3.0 - 5e-7, 1.0, true) == 3, "ticks: robust 2.9999995 rounds up to 3");
    check(ComputeTickCount(3.0 - 5e-7, 1.0, false) == 2, "ticks: legacy 2.9999995 truncates to 2");
    check(ComputeTickCount(3.0 - 2e-6, 1.0, true) == 2, "ticks: robust 2.999998 floors to 2");
    check(ComputeTickCount(3.0 + 5e-7, 1.0, true) == 3, "ticks: robust 3.0000005 gives 3");
    check(ComputeTickCount(3.0 + 2e-6, 1.0, true) == 3, "ticks: robust 3.000002 floors to 3");
    check(ComputeTickCount(3.5, 1.0, true) == 3, "ticks: robust 3.5 floors to 3 (not rounded)");
    check(ComputeTickCount(3.9, 1.0, true) == 3, "ticks: robust 3.9 floors to 3");

    // Invalid inputs.
    check(ComputeTickCount(-1.0, -0.1, true) == 0 && ComputeTickCount(-1.0, -0.1, false) == 0,
          "ticks: both negative gives 0 (positive ratio not trusted)");
    check(ComputeTickCount(1.0, -0.1, true) == 0, "ticks: negative tick gives 0");
    check(ComputeTickCount(0.0, 0.1, false) == 0, "ticks: zero duration gives 0");
    check(ComputeTickCount(kInf, 1.0, false) == 0, "ticks: infinite duration gives 0");
    check(ComputeTickCount(1.0, kInf, true) == 0, "ticks: infinite tick gives 0");
    check(ComputeTickCount(kNaN, 1.0, true) == 0 && ComputeTickCount(1.0, kNaN, false) == 0,
          "ticks: NaN inputs give 0");
    check(ComputeTickCount(1.0, 1e-320, true) == 0, "ticks: denormal tick (ratio overflows) gives 0");

    // uint32 range.
    check(ComputeTickCount(4294967294.0, 1.0, false) == 4294967294u,
          "ticks: ratio UINT32_MAX-1 accepted");
    check(ComputeTickCount(4294967296.0, 1.0, true) == 0 &&
              ComputeTickCount(4294967296.0, 1.0, false) == 0,
          "ticks: ratio above UINT32_MAX gives 0");
}

// =====================================================================
// ResolveRlControl -- cadence
// =====================================================================

static void
test_resolver_cadence()
{
    auto cfg = makeRlCfg();  // duration 10, tick 0.1, num_ticks 100

    cfg.rl.decision_interval_s = 0.1;
    auto r = ResolveRlControl(cfg);
    check(r.ok() && r.decision_interval_ticks == 1, "cadence: interval == tick gives k = 1");

    cfg.rl.decision_interval_s = 10.0;
    r = ResolveRlControl(cfg);
    check(r.ok() && r.decision_interval_ticks == 100 && r.num_ticks == 100,
          "cadence: interval == duration gives k == num_ticks");

    cfg.rl.decision_interval_s = 0.3;  // 0.3/0.1 = 2.9999999999999996
    r = ResolveRlControl(cfg);
    check(r.ok() && r.decision_interval_ticks == 3, "cadence: fp-noisy 0.3/0.1 rounds to k = 3");

    cfg.rl.decision_interval_s = 0.50000005;  // k = 5.0000005, within 1e-6
    r = ResolveRlControl(cfg);
    check(r.ok() && r.decision_interval_ticks == 5, "cadence: ratio within 1e-6 accepted");

    cfg.rl.decision_interval_s = 0.5000002;  // k = 5.000002, outside 1e-6
    check(hasResErr(ResolveRlControl(cfg), "integer multiple of tick_s"),
          "cadence: ratio 2e-6 off an integer rejected");

    cfg.rl.decision_interval_s = 0.05;
    check(hasResErr(ResolveRlControl(cfg), "integer multiple of tick_s"),
          "cadence: half a tick rejected");

    cfg.rl.decision_interval_s = 1e-9;
    check(hasResErr(ResolveRlControl(cfg), "integer multiple of tick_s"),
          "cadence: tiny positive interval (k rounds to 0) rejected");

    cfg.rl.decision_interval_s = 10.1;
    check(hasResErr(ResolveRlControl(cfg), "rl.decision_interval_s exceeds duration_s"),
          "cadence: k = num_ticks + 1 rejected");

    cfg.rl.decision_interval_s = 1e300;
    check(hasResErr(ResolveRlControl(cfg), "rl.decision_interval_s exceeds duration_s"),
          "cadence: huge interval rejected without overflow");

    cfg.rl.decision_interval_s = -0.0;
    r = ResolveRlControl(cfg);
    check(r.ok() && r.decision_interval_ticks == 1, "cadence: -0.0 treated as 0 (one tick)");

    // Partial final window is allowed (src/rl/README.md "Partial windows").
    cfg = makeRlCfg();
    cfg.duration_s             = 1.0;
    cfg.rl.decision_interval_s = 0.3;
    r = ResolveRlControl(cfg);
    check(r.ok() && r.num_ticks == 10 && r.decision_interval_ticks == 3,
          "cadence: k that does not divide num_ticks accepted");

    // Cadence integer check is skipped when timing is already invalid.
    cfg = makeRlCfg();
    cfg.tick_s                 = 0.0;
    cfg.rl.decision_interval_s = 0.15;
    r = ResolveRlControl(cfg);
    check(hasResErr(r, "rl timing is invalid") && !hasResErr(r, "integer multiple") &&
              !hasResErr(r, "exceeds duration_s"),
          "cadence: invalid timing suppresses the derived cadence errors");

    // The 1e-6 tolerance also applies to the centralized tick count.
    cfg = makeRlCfg();
    cfg.duration_s = 0.7;
    r = ResolveRlControl(cfg);
    check(r.ok() && r.num_ticks == 7, "cadence: centralized 0.7/0.1 gives 7 ticks");
}

// =====================================================================
// ApplyRlControl / ControlledStartPosition / MaxSpeedForType
// =====================================================================

static void
test_apply_rl_control_cases()
{
    auto cfg = makeRlCfg();
    cfg.rl.controlled_nodes_set = false;
    cfg.duration_s              = 0.7;
    auto r = ResolveRlControl(cfg);
    ApplyRlControl(cfg, r);
    check(cfg.rl.control_mode == "legacy" &&
              cfg.rl.controlled_indices == std::vector<uint32_t>({2}) && cfg.rl.num_slots == 1 &&
              cfg.rl.decision_interval_ticks == 1 && cfg.rl.num_ticks == 6,
          "apply: legacy resolution copied (truncated 6 ticks)");

    // A failed apply leaves the previously applied fields untouched.
    RlControlResolution bad = r;
    bad.control_mode        = "centralized";
    bad.controlled_indices  = {0, 1};
    bad.num_slots           = 9;
    bad.num_ticks           = 999;
    bad.errors.push_back("boom");
    bool threw = false;
    try
    {
        ApplyRlControl(cfg, bad);
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }
    check(threw, "apply: errors throw std::invalid_argument");
    check(cfg.rl.control_mode == "legacy" && cfg.rl.num_slots == 1 && cfg.rl.num_ticks == 6 &&
              cfg.rl.controlled_indices == std::vector<uint32_t>({2}),
          "apply: throwing leaves cfg.rl unchanged");

    // Disabled resolution applies defaults.
    auto off = makeRlCfg();
    off.rl.enabled = false;
    off.rl.num_ticks = 42;
    ApplyRlControl(off, ResolveRlControl(off));
    check(off.rl.num_ticks == 0 && off.rl.num_slots == 0 && off.rl.control_mode == "legacy",
          "apply: disabled resolution resets resolved fields");
}

// =====================================================================
// ApplyLayout
// =====================================================================

/// node0 fixed at (1,2,10); node1 random_walk at (10,0,1.5) in x[0,100] y[-50,50].
static SimConfig
makeLayoutCfg()
{
    auto cfg = makeValid();
    cfg.nodes[0].position = {1.0, 2.0, 10.0};
    auto& rn              = cfg.nodes[1];
    rn.mobility           = "random_walk";
    rn.position           = {10.0, 0.0, 1.5};
    rn.random_walk.x_min  = 0.0;
    rn.random_walk.x_max  = 100.0;
    rn.random_walk.y_min  = -50.0;
    rn.random_walk.y_max  = 50.0;
    rn.random_walk.speed_mps = 2.0;
    return cfg;
}

static void
test_layout_random_walk_control_modes()
{
    // Legacy RL control keeps the node's own mobility, so walk bounds still apply.
    auto cfg = makeLayoutCfg();
    cfg.rl.enabled            = true;
    cfg.rl.control_mode       = "legacy";
    cfg.rl.controlled_indices = {1};
    auto errs = ApplyLayout(cfg, {{1.0, 2.0, 10.0}, {500.0, 0.0, 1.5}});
    check(hasIn(errs, "outside its random_walk bounds"),
          "layout-rw: legacy-controlled random walk node still bounded");

    // Resolved fields without rl.enabled do not count as central control.
    cfg = makeLayoutCfg();
    cfg.rl.enabled            = false;
    cfg.rl.control_mode       = "centralized";
    cfg.rl.controlled_indices = {1};
    errs = ApplyLayout(cfg, {{1.0, 2.0, 10.0}, {500.0, 0.0, 1.5}});
    check(hasIn(errs, "outside its random_walk bounds"),
          "layout-rw: disabled RL keeps walk bounds");

    // Via the real resolver: centrally controlled walk node is unbounded.
    cfg = makeLayoutCfg();
    cfg.rl.enabled              = true;
    cfg.rl.controlled_nodes_set = true;
    cfg.rl.controlled_nodes     = "node1";
    auto res = ResolveRlControl(cfg);
    check(res.ok(), "layout-rw: resolver accepts the controlled walk node");
    ApplyRlControl(cfg, res);
    errs = ApplyLayout(cfg, {{1.0, 2.0, 10.0}, {500.0, -300.0, 1.5}});
    check(errs.empty() && cfg.nodes[1].position.x == 500.0 && cfg.nodes[1].position.y == -300.0,
          "layout-rw: resolved centralized control lifts walk bounds");
    check(cfg.nodes[1].mobility == "random_walk" && cfg.nodes[1].random_walk.x_max == 100.0 &&
              cfg.nodes[1].random_walk.speed_mps == 2.0,
          "layout-rw: mobility kind and walk params unchanged");
}

static void
test_layout_waypoint_parity()
{
    // Same arithmetic as scripts/baselines/effective_inputs.py rewrite_nodes:
    // dx = x - start_x; waypoint.x = waypoint.x + dx; position.x = x.
    auto cfg = makeValid();
    auto& wn = cfg.nodes[1];
    wn.mobility  = "waypoint";
    wn.position  = {42.0, 43.0, 7.0};
    wn.waypoints = {{0.0, 0.1, 0.2, 3.0}, {1.5, 1.7, -2.3, 4.0}, {9.0, 1e6 + 0.3, 4.4, 5.0}};
    cfg.nodes[0].velocity = Velocity{1.0, 2.0, 3.0};
    cfg.nodes[0].mobility = "constant_velocity";
    cfg.nodes[0].position = {0.0, 0.0, 10.0};
    const auto orig = wn.waypoints;
    const double tx = 0.3, ty = -0.7;

    auto errs = ApplyLayout(cfg, {{-5.0, 6.0, 10.0}, {tx, ty, 3.0}});
    check(errs.empty(), "layout-wp: waypoint layout applies");
    volatile double dx = tx - orig[0].x;
    volatile double dy = ty - orig[0].y;
    bool same = wn.waypoints.size() == orig.size();
    for (size_t i = 0; same && i < orig.size(); ++i)
    {
        same = wn.waypoints[i].x == orig[i].x + dx && wn.waypoints[i].y == orig[i].y + dy &&
               wn.waypoints[i].z == orig[i].z && wn.waypoints[i].t == orig[i].t;
    }
    check(same, "layout-wp: every waypoint shifted with rewrite_nodes arithmetic, z/t kept");
    check(wn.position.x == tx && wn.position.y == ty && wn.position.z == 7.0,
          "layout-wp: position x/y set to the exact target, z kept");
    check(cfg.nodes[0].velocity.vx == 1.0 && cfg.nodes[0].velocity.vy == 2.0 &&
              cfg.nodes[0].mobility == "constant_velocity" && cfg.nodes[0].position.x == -5.0,
          "layout-wp: constant_velocity node moves, velocity kept");

    // z tolerance uses the first waypoint's z.
    errs = ApplyLayout(cfg, {{-5.0, 6.0, 10.0}, {tx, ty, 3.0 + 2e-6}});
    check(hasIn(errs, "node 'node1': z"), "layout-wp: z 2e-6 off the first waypoint rejected");
    errs = ApplyLayout(cfg, {{-5.0, 6.0, 10.0}, {tx, ty, 3.0 - 9e-7}});
    check(errs.empty() && wn.waypoints[0].z == 3.0,
          "layout-wp: z within 1e-6 accepted, waypoint z unchanged");

    // Centrally controlled waypoint node: start position follows the target.
    auto c2 = makeRlCfg();
    c2.nodes[1].mobility  = "waypoint";
    c2.nodes[1].waypoints = {{0.0, 10.0, 20.0, 10.0}, {1.0, 15.0, 20.0, 10.0}};
    c2.rl.controlled_nodes = "node-1";
    auto res = ResolveRlControl(c2);
    check(res.ok(), "layout-wp: controlled waypoint node resolves");
    ApplyRlControl(c2, res);
    errs = ApplyLayout(c2, {{0.0, 20.0, 10.0}, {300.0, -40.0, 10.0}, {20.0, 20.0, 10.0}});
    Position s = ControlledStartPosition(c2.nodes[1]);
    check(errs.empty() && s.x == 300.0 && s.y == -40.0 && s.z == 10.0,
          "layout-wp: controlled waypoint start follows the layout");
    check(c2.nodes[1].waypoints[1].x == 305.0 && c2.nodes[1].waypoints[1].y == -40.0,
          "layout-wp: controlled waypoint path translated");
}

// ---- main ----

/**
 * @fn main
 * @brief Run every config-rules test and print a pass/fail count.
 * @return 0 when every check passed, 1 otherwise.
 */
int
main()
{

    // ComputeTickCount / ResolveRlControl
    test_tick_count_boundaries();
    test_resolver_cadence();

    // ApplyRlControl / ControlledStartPosition / speed caps
    test_apply_rl_control_cases();

    // ApplyLayout
    test_layout_random_walk_control_modes();
    test_layout_waypoint_parity();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.\n";
    if (g_fail > 0)
    {
        std::cout << "SOME TESTS FAILED.\n";
        return 1;
    }
    std::cout << "All tests passed.\n";
    return 0;
}
