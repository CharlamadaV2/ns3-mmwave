/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file rl-bridge-test.cc
 * @brief Unit tests for RlBridge: init/step messages, masks, facts, window
 *        rewards, joint/legacy action parsing, revalidation, EOF handling,
 *        per-tick bounds clamping, and action-to-velocity mapping.
 *
 * Expectations come from src/rl/README.md (the wire contract), the
 * rl-bridge.h Doxygen, and src/rl/policy-inputs.md. ns-3 mobility is stubbed
 * by ns3-mobility-stub.h; std::cin/cout/cerr are redirected to string
 * streams. Each test_* runs in a forked child so the bridge's once-per-process
 * warning flags start fresh and a crash fails only that test.
 *
 * Build and run: `make -C tests/unit/rl-bridge test`.
 */

#include "src/rl/rl-bridge.h"
#include "src/rl/rl-agent.h"
#include "src/eval/link-table.h"
#include "src/domain/link-result.h"
#include "third_party/json.hpp"

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace mesh_sim;
using json = nlohmann::json;

// ---- helpers ----

static int g_pass = 0;
static int g_fail = 0;

// Writes through C stdio so redirecting std::cerr does not swallow failures.
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
        std::fprintf(stderr, "FAIL: %s\n", name.c_str());
    }
}

static bool
approx(double a, double b, double tol = 1e-9)
{
    return std::fabs(a - b) < tol;
}

// Run @p fn in a forked child; its pass/fail counts come back over a pipe.
// A child that crashes or exits abnormally counts as one failure.
static void
isolated(const std::string& name, const std::function<void()>& fn)
{
    std::cout.flush();
    std::fflush(stdout);
    std::fflush(stderr);
    int fds[2];
    if (pipe(fds) != 0)
    {
        check(false, name + ": pipe() failed");
        return;
    }
    pid_t pid = fork();
    if (pid < 0)
    {
        check(false, name + ": fork() failed");
        return;
    }
    if (pid == 0)
    {
        close(fds[0]);
        g_pass = 0;
        g_fail = 0;
        fn();
        int counts[2] = {g_pass, g_fail};
        ssize_t w = write(fds[1], counts, sizeof(counts));
        (void)w;
        close(fds[1]);
        std::fflush(stderr);
        _exit(0);
    }
    close(fds[1]);
    int counts[2] = {0, 0};
    ssize_t got = read(fds[0], counts, sizeof(counts));
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    if (got != static_cast<ssize_t>(sizeof(counts)) || !WIFEXITED(status) ||
        WEXITSTATUS(status) != 0)
    {
        std::string why = WIFSIGNALED(status)
                              ? "killed by signal " + std::to_string(WTERMSIG(status))
                              : "exited abnormally";
        check(false, name + ": child " + why + " (crash in code under test?)");
        // Keep whatever the child reported before crashing out of the count.
        return;
    }
    g_pass += counts[0];
    g_fail += counts[1];
}

// Redirects std::cin/cout/cerr to string streams for the object's lifetime.
struct Io
{
    std::istringstream in;
    std::ostringstream out;
    std::ostringstream err;
    std::streambuf* oldIn;
    std::streambuf* oldOut;
    std::streambuf* oldErr;

    explicit Io(const std::string& input = "")
        : in(input)
    {
        oldIn  = std::cin.rdbuf(in.rdbuf());
        oldOut = std::cout.rdbuf(out.rdbuf());
        oldErr = std::cerr.rdbuf(err.rdbuf());
        std::cin.clear();
    }

    ~Io()
    {
        std::cin.rdbuf(oldIn);
        std::cout.rdbuf(oldOut);
        std::cerr.rdbuf(oldErr);
        std::cin.clear();
    }

    std::vector<std::string> RawLines() const
    {
        std::vector<std::string> lines;
        std::istringstream s(out.str());
        std::string line;
        while (std::getline(s, line))
        {
            lines.push_back(line);
        }
        return lines;
    }

    std::vector<json> Lines() const
    {
        std::vector<json> v;
        for (const auto& l : RawLines())
        {
            v.push_back(json::parse(l, nullptr, false));
        }
        return v;
    }

    json Last() const
    {
        auto v = Lines();
        return v.empty() ? json() : v.back();
    }

    std::string Unread()
    {
        std::string rest;
        std::string line;
        while (std::getline(in, line))
        {
            rest += line + "\n";
        }
        return rest;
    }
};

static int
countOf(const std::string& hay, const std::string& needle)
{
    int n = 0;
    for (size_t p = hay.find(needle); p != std::string::npos; p = hay.find(needle, p + 1))
    {
        ++n;
    }
    return n;
}

// Nodes and their mobility models (stub objects owned here).
struct World
{
    std::vector<std::unique_ptr<ns3::MobilityModel>> owned;
    std::vector<ns3::Ptr<ns3::MobilityModel>> mobs;

    ns3::ConstantVelocityMobilityModel* AddCvmm(double x, double y, double z)
    {
        auto m = std::make_unique<ns3::ConstantVelocityMobilityModel>();
        m->SetPosition(ns3::Vector(x, y, z));
        auto* raw = m.get();
        mobs.emplace_back(raw);
        owned.push_back(std::move(m));
        return raw;
    }

    ns3::MobilityModel* AddPlain(double x, double y, double z, ns3::Vector v = ns3::Vector())
    {
        auto m = std::make_unique<ns3::MobilityModel>();
        m->SetPosition(ns3::Vector(x, y, z));
        m->SetTestVelocity(v);
        auto* raw = m.get();
        mobs.emplace_back(raw);
        owned.push_back(std::move(m));
        return raw;
    }

    ns3::ConstantVelocityMobilityModel* Cv(uint32_t i)
    {
        return dynamic_cast<ns3::ConstantVelocityMobilityModel*>(owned[i].get());
    }

    ns3::Vector Pos(uint32_t i) const { return owned[i]->GetPosition(); }
    ns3::Vector Vel(uint32_t i) const { return owned[i]->GetVelocity(); }

    // Stand-in for Simulator::Stop(tick)+Run(): constant-velocity models move.
    void Advance(double dt)
    {
        for (auto& m : owned)
        {
            if (auto* cv = dynamic_cast<ns3::ConstantVelocityMobilityModel*>(m.get()))
            {
                cv->AdvanceForTest(dt);
            }
        }
    }
};

static NodeSpec
makeNode(const std::string& id, double x, double y, double z,
         const std::string& type = "drone")
{
    NodeSpec n;
    n.id = id;
    n.node_type = type;
    n.position = Position{x, y, z};
    return n;
}

// Centralized config: N drone nodes "n0".."n{N-1}" at the origin, bounds [0,100]^2.
static SimConfig
centralCfg(uint32_t numNodes, std::vector<uint32_t> controlled, uint32_t slots,
           uint32_t k = 1, uint32_t numTicks = 10, double tickS = 0.1,
           double stepM = 1.0)
{
    SimConfig cfg;
    cfg.tick_s = tickS;
    cfg.duration_s = numTicks * tickS;
    for (uint32_t i = 0; i < numNodes; ++i)
    {
        cfg.nodes.push_back(makeNode("n" + std::to_string(i), 50.0, 50.0, 10.0));
    }
    cfg.rl.enabled = true;
    cfg.rl.control_mode = "centralized";
    cfg.rl.controlled_nodes_set = true;
    cfg.rl.controlled_indices = std::move(controlled);
    cfg.rl.num_slots = slots;
    cfg.rl.decision_interval_ticks = k;
    cfg.rl.num_ticks = numTicks;
    cfg.rl.step_size_m = stepM;
    cfg.rl.x_min = 0.0;
    cfg.rl.x_max = 100.0;
    cfg.rl.y_min = 0.0;
    cfg.rl.y_max = 100.0;
    cfg.rl.z_min = 0.0;
    cfg.rl.z_max = 50.0;
    return cfg;
}

// Legacy config: N drone nodes; controlled_indices empty means "last node".
static SimConfig
legacyCfg(uint32_t numNodes, std::vector<uint32_t> controlled = {},
          double stepM = 50.0, double tickS = 0.1)
{
    SimConfig cfg;
    cfg.tick_s = tickS;
    for (uint32_t i = 0; i < numNodes; ++i)
    {
        cfg.nodes.push_back(makeNode("n" + std::to_string(i), 0.0, 0.0, 10.0));
    }
    cfg.rl.enabled = true;
    cfg.rl.control_mode = "legacy";
    cfg.rl.controlled_indices = std::move(controlled);
    cfg.rl.num_slots = 1;
    cfg.rl.decision_interval_ticks = 1;
    cfg.rl.num_ticks = 10;
    cfg.rl.step_size_m = stepM;
    cfg.rl.x_min = -100.0;
    cfg.rl.x_max = 100.0;
    cfg.rl.y_min = -100.0;
    cfg.rl.y_max = 100.0;
    cfg.rl.z_min = 0.0;
    cfg.rl.z_max = 50.0;
    return cfg;
}

struct L
{
    double sinr;
    double cap;
    bool   los;
};

// Link table where every pair has @p def unless overridden by {i,j} (i<j).
static LinkTable
makeTable(uint32_t n, L def = {10.0, 100.0, true},
          const std::map<std::pair<uint32_t, uint32_t>, L>& over = {})
{
    std::vector<LinkResult> rs;
    for (uint32_t i = 0; i < n; ++i)
    {
        for (uint32_t j = i + 1; j < n; ++j)
        {
            L l = def;
            auto it = over.find({i, j});
            if (it != over.end())
            {
                l = it->second;
            }
            LinkResult r;
            r.tx_id = i;
            r.rx_id = j;
            r.sinr_db = l.sinr;
            r.capacity_mbps = l.cap;
            r.is_los = l.los;
            rs.push_back(r);
        }
    }
    LinkTable t;
    t.Update(n, rs);
    return t;
}

static FlowResult
flow(uint32_t s, uint32_t d, double demand, double delivered, bool routable)
{
    FlowResult f;
    f.src = s;
    f.dst = d;
    f.demand_mbps = demand;
    f.delivered_mbps = delivered;
    f.routable = routable;
    return f;
}

static const std::vector<FlowResult> kNoFlows;

using TableFn = std::function<LinkTable(uint32_t ti)>;
using FlowFn  = std::function<std::vector<FlowResult>(uint32_t ti)>;

// Mirrors the RL part of the sim.cc tick loop (CLAUDE.md "Run flow").
static void
runEpisode(const SimConfig& cfg, World& w, const TableFn& tables,
           const FlowFn& flows = [](uint32_t) { return std::vector<FlowResult>{}; })
{
    RlBridge bridge(cfg);
    const bool central = cfg.rl.control_mode == "centralized";
    if (central)
    {
        bridge.WriteInit();
    }
    const uint32_t numTicks = cfg.rl.num_ticks;
    for (uint32_t ti = 0; ti <= numTicks; ++ti)
    {
        const double t = ti * cfg.tick_s;
        if (ti > 0)
        {
            bridge.BeforeAdvance(w.mobs);
            w.Advance(cfg.tick_s);
        }
        LinkTable table = tables(ti);
        std::vector<FlowResult> fr = flows(ti);
        const bool done = (ti == numTicks);
        if (central)
        {
            bridge.AccumulateTick(table, fr);
            if (done || bridge.IsDecisionTick(ti))
            {
                bridge.Step(ti, t, w.mobs, table, fr, done);
                if (!done)
                {
                    bridge.ApplyAction(w.mobs);
                }
            }
        }
        else
        {
            bridge.Step(ti, t, w.mobs, table, fr, done);
            if (!done)
            {
                bridge.ApplyAction(w.mobs);
            }
        }
    }
}

// One decision in centralized mode: accumulate one tick, Step, ApplyAction.
static void
decide(RlBridge& b, World& w, const LinkTable& t, uint32_t tick, bool done = false,
       const std::vector<FlowResult>& flows = kNoFlows)
{
    b.AccumulateTick(t, flows);
    b.Step(tick, tick * 0.1, w.mobs, t, flows, done);
    if (!done)
    {
        b.ApplyAction(w.mobs);
    }
}

static std::vector<double>
toVec(const json& a)
{
    std::vector<double> v;
    for (const auto& e : a)
    {
        v.push_back(e.is_number() ? e.get<double>() : std::nan(""));
    }
    return v;
}

// =====================================================================
// init message
// =====================================================================

// README "Message fields": init carries the signature and facts metadata.
static void
test_init_fields()
{
    isolated("test_init_fields", [] {
        SimConfig cfg = centralCfg(3, {1, 2}, 3, 5, 10, 0.1, 1.0);
        cfg.band = "mmwave";
        cfg.warmup_s = 0.25;
        cfg.rl.reward_type = "all_links_los";
        Io io;
        RlBridge b(cfg);
        b.WriteInit();
        auto lines = io.Lines();
        check(lines.size() == 1, "init: exactly one line");
        if (lines.size() != 1)
        {
            return;
        }
        const json& m = lines[0];
        check(m["type"] == "init", "init: type");
        check(m["contract"] == "mesh_move_2d_v1", "init: contract");
        check(m["dimensions"] == 2, "init: dimensions 2");
        check(m["action_meanings"] ==
                  json::array({"west", "east", "south", "north", "hold"}),
              "init: action_meanings order");
        check(m["max_controlled_nodes"] == 3, "init: max_controlled_nodes = M");
        check(m["num_controlled"] == 2, "init: num_controlled = active slots");
        check(m["slot_node_ids"] == json::array({"n1", "n2", nullptr}),
              "init: slot_node_ids in resolved order, padding null");
        check(m["slot_speed_mps"].size() == 3 && approx(m["slot_speed_mps"][0], 10.0) &&
                  approx(m["slot_speed_mps"][1], 10.0) && m["slot_speed_mps"][2].is_null(),
              "init: slot_speed_mps = step/tick (1/0.1), padding null");
        check(m["num_mesh_nodes"] == 3, "init: num_mesh_nodes");
        check(m["obs_dim"] == 3 * (4 + 2 * 2), "init: obs_dim = M*(4+2(N-1))");
        check(m["mask_dim"] == 15, "init: mask_dim = 5M");
        check(approx(m["tick_s"], 0.1), "init: tick_s");
        check(approx(m["decision_interval_s"], 0.5), "init: decision_interval_s = k*tick_s");
        check(m["decision_interval_ticks"] == 5, "init: decision_interval_ticks");
        check(m["num_ticks"] == 10, "init: num_ticks");
        check(m["num_decisions"] == 2, "init: num_decisions = ceil(10/5)");
        check(m["reward_type"] == "all_links_los", "init: reward_type");
        check(m["reward_window"] == "mean", "init: reward_window mean");
        check(m["wall_policy"] == "clip", "init: wall_policy clip");
        check(m["facts_schema"] == "mesh_facts_v1", "init: facts_schema");
        json cols = {{"nodes", {"x", "y", "z", "vx", "vy", "vz", "slot"}},
                     {"links", {"sinr_db", "capacity_mbps", "is_los"}}};
        check(m["facts_columns"] == cols, "init: facts_columns");
        check(m["node_ids"] == json::array({"n0", "n1", "n2"}), "init: node_ids file order");
        check(m["num_links"] == 3, "init: num_links = N(N-1)/2");
        json bounds = {{"x_min", 0.0}, {"x_max", 100.0}, {"y_min", 0.0},
                       {"y_max", 100.0}, {"z_min", 0.0}, {"z_max", 50.0}};
        check(m["bounds"] == bounds, "init: bounds");
        check(m["band"] == "mmwave", "init: band");
        check(m["jammer_path_enabled"] == false, "init: jammer_path_enabled false");
        check(approx(m["warmup_s"], 0.25), "init: warmup_s");
        for (const char* key : {"max_controlled_nodes", "num_controlled", "obs_dim",
                                "mask_dim", "decision_interval_ticks", "num_ticks",
                                "num_decisions", "num_links", "num_mesh_nodes"})
        {
            check(m[key].is_number_integer(), std::string("init: integer field ") + key);
        }
        check(io.err.str().empty(), "init: nothing on stderr");
    });
}

// README "Partial windows": num_decisions = ceil(num_ticks / k).
static void
test_init_num_decisions_ceil()
{
    isolated("test_init_num_decisions_ceil", [] {
        struct C { uint32_t ticks, k, expect; };
        for (C c : {C{7, 5, 2}, C{10, 3, 4}, C{10, 1, 10}, C{5, 5, 1}, C{1, 1, 1}, C{11, 5, 3}})
        {
            SimConfig cfg = centralCfg(2, {0}, 1, c.k, c.ticks);
            Io io;
            RlBridge b(cfg);
            b.WriteInit();
            json m = io.Last();
            check(m["num_decisions"] == c.expect,
                  "num_decisions ceil(" + std::to_string(c.ticks) + "/" +
                      std::to_string(c.k) + ")");
        }
    });
}

// README "Speed cap": v = min(step_size_m / tick_s, MaxSpeedForType(node_type)).
static void
test_init_speed_caps()
{
    isolated("test_init_speed_caps", [] {
        SimConfig cfg = centralCfg(5, {0, 1, 2, 3}, 4, 1, 10, 0.1, 5.0);  // 50 m/s nominal
        cfg.nodes[0].node_type = "drone";
        cfg.nodes[1].node_type = "vehicle";
        cfg.nodes[2].node_type = "pedestrian";
        cfg.nodes[3].node_type = "hovercraft";  // unknown -> drone limit
        Io io;
        RlBridge b(cfg);
        b.WriteInit();
        json s = io.Last()["slot_speed_mps"];
        check(approx(s[0], 20.0), "speed cap drone 20");
        check(approx(s[1], 15.0), "speed cap vehicle 15");
        check(approx(s[2], 1.5), "speed cap pedestrian 1.5");
        check(approx(s[3], 20.0), "speed cap unknown type -> drone 20");

        SimConfig slow = centralCfg(3, {0, 1}, 2, 1, 10, 0.1, 0.5);  // 5 m/s nominal
        slow.nodes[1].node_type = "pedestrian";
        Io io2;
        RlBridge b2(slow);
        b2.WriteInit();
        json s2 = io2.Last()["slot_speed_mps"];
        check(approx(s2[0], 5.0), "speed below cap uses step/tick (5)");
        check(approx(s2[1], 1.5), "pedestrian still capped at 1.5");
    });
}

// README "Per-decision facts": jammer_path_enabled = sub-6 and >= 1 enabled jammer.
static void
test_init_jammer_path_flag()
{
    isolated("test_init_jammer_path_flag", [] {
        auto flagFor = [](const std::string& band, std::vector<bool> enabled) {
            SimConfig cfg = centralCfg(2, {0}, 1);
            cfg.band = band;
            for (bool e : enabled)
            {
                JammerSpec j{};
                j.enabled = e;
                j.id = "j";
                cfg.jammers.push_back(j);
            }
            Io io;
            RlBridge b(cfg);
            b.WriteInit();
            return io.Last()["jammer_path_enabled"];
        };
        check(flagFor("sub-6", {true}) == true, "jammer flag: sub-6 + enabled jammer");
        check(flagFor("sub-6", {false, true}) == true, "jammer flag: one of two enabled");
        check(flagFor("sub-6", {false}) == false, "jammer flag: sub-6 + disabled jammer");
        check(flagFor("sub-6", {}) == false, "jammer flag: sub-6, no jammers");
        check(flagFor("mmwave", {true}) == false, "jammer flag: mmwave + enabled jammer");
    });
}

// rl-bridge.h WriteInit: "Does nothing in legacy mode"; README Modes: legacy has no init.
static void
test_init_legacy_writes_nothing()
{
    isolated("test_init_legacy_writes_nothing", [] {
        SimConfig cfg = legacyCfg(3);
        Io io;
        RlBridge b(cfg);
        b.WriteInit();
        check(io.out.str().empty(), "legacy WriteInit: no output");
    });
}

// rl-bridge.h IsDecisionTick: centralized, ti < num_ticks, ti % k == 0.
static void
test_is_decision_tick()
{
    isolated("test_is_decision_tick", [] {
        RlBridge c(centralCfg(2, {0}, 1, 5, 10));
        check(c.IsDecisionTick(0), "decision tick 0");
        check(c.IsDecisionTick(5), "decision tick 5");
        bool anyOther = false;
        for (uint32_t t : {1u, 2u, 3u, 4u, 6u, 9u})
        {
            anyOther = anyOther || c.IsDecisionTick(t);
        }
        check(!anyOther, "non-multiples of k are not decision ticks");
        check(!c.IsDecisionTick(10), "tick == num_ticks is not a decision tick (terminal)");
        check(!c.IsDecisionTick(15), "tick beyond num_ticks is not a decision tick");

        RlBridge k1(centralCfg(2, {0}, 1, 1, 3));
        check(k1.IsDecisionTick(0) && k1.IsDecisionTick(1) && k1.IsDecisionTick(2) &&
                  !k1.IsDecisionTick(3),
              "k=1: every tick below num_ticks");

        RlBridge leg(legacyCfg(2));
        check(!leg.IsDecisionTick(0) && !leg.IsDecisionTick(1), "legacy: never a decision tick");
    });
}

// =====================================================================
// masks
// =====================================================================

// README "Mask order": [W,E,S,N,H] per slot, padded slot [0,0,0,0,1], hold always 1.
static void
test_mask_interior_and_padding()
{
    isolated("test_mask_interior_and_padding", [] {
        SimConfig cfg = centralCfg(3, {0, 1}, 3);
        World w;
        w.AddCvmm(50, 50, 10);
        w.AddCvmm(20, 80, 10);
        w.AddCvmm(10, 10, 10);
        Io io;
        RlBridge b(cfg);
        decide(b, w, makeTable(3), 0, true);
        json m = io.Last();
        check(m["mask"] == json::array({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1}),
              "mask: interior slots all valid, padded slot hold only");
        bool ints = true;
        for (const auto& e : m["mask"])
        {
            ints = ints && e.is_number_integer();
        }
        check(ints, "mask entries are integers");
    });
}

// README "Mask rule (clip-at-wall)": a direction is valid iff room > 1e-6 m.
static void
test_mask_walls_and_corners()
{
    isolated("test_mask_walls_and_corners", [] {
        SimConfig cfg = centralCfg(5, {0, 1, 2, 3, 4}, 5);
        World w;
        w.AddCvmm(0, 50, 10);     // at x_min
        w.AddCvmm(100, 50, 10);   // at x_max
        w.AddCvmm(50, 0, 10);     // at y_min
        w.AddCvmm(50, 100, 10);   // at y_max
        w.AddCvmm(0, 100, 10);    // NW corner
        Io io;
        RlBridge b(cfg);
        decide(b, w, makeTable(5), 0, true);
        json m = io.Last()["mask"];
        auto slot = [&](int i) {
            return json::array({m[5 * i], m[5 * i + 1], m[5 * i + 2], m[5 * i + 3], m[5 * i + 4]});
        };
        check(slot(0) == json::array({0, 1, 1, 1, 1}), "mask at x_min: west masked");
        check(slot(1) == json::array({1, 0, 1, 1, 1}), "mask at x_max: east masked");
        check(slot(2) == json::array({1, 1, 0, 1, 1}), "mask at y_min: south masked");
        check(slot(3) == json::array({1, 1, 1, 0, 1}), "mask at y_max: north masked");
        check(slot(4) == json::array({0, 1, 1, 0, 1}), "mask at NW corner: west+north masked");
    });
}

// =====================================================================
// centralized step message, obs, facts
// =====================================================================

// README "Message fields": step field set and reset values (tick 0, decision 0, ticks 1).
static void
test_step_message_shape()
{
    isolated("test_step_message_shape", [] {
        SimConfig cfg = centralCfg(3, {0, 1}, 2, 1, 5);
        World w;
        w.AddCvmm(10, 10, 5);
        w.AddCvmm(20, 20, 5);
        w.AddCvmm(30, 30, 5);
        Io io("{\"action\":[4,4]}\n");
        RlBridge b(cfg);
        decide(b, w, makeTable(3), 0);
        json m = io.Last();
        for (const char* key : {"type", "tick", "time_s", "decision", "ticks_in_step", "obs",
                                "mask", "reward", "done", "revalidated_slots", "facts"})
        {
            check(m.contains(key), std::string("step has field ") + key);
        }
        check(m["type"] == "step", "step type");
        check(m["tick"] == 0 && m["decision"] == 0 && m["ticks_in_step"] == 1,
              "reset step: tick 0, decision 0, ticks_in_step 1");
        check(approx(m["time_s"], 0.0), "reset step: time_s 0");
        check(m["done"] == false, "reset step: done false");
        check(m["obs"].size() == 2 * (4 + 2 * 2), "obs length = obs_dim");
        check(m["mask"].size() == 10, "mask length = mask_dim");
        check(m["revalidated_slots"].is_array() && m["revalidated_slots"].empty(),
              "reset step: revalidated_slots empty");
        check(m["facts"].contains("nodes") && m["facts"].contains("links") &&
                  m["facts"].contains("window"),
              "facts has nodes/links/window");
        check(!m.contains("action_type"), "centralized step has no legacy action_type");
        check(io.Unread().empty(), "non-terminal step consumed one action line");
    });
}

// README "Observation layout per slot": [active,x,y,z, sinr(i,j0),cap(i,j0),...],
// peers in node-index order skipping self; padded slot all zeros.
static void
test_obs_layout_peer_order()
{
    isolated("test_obs_layout_peer_order", [] {
        // Slot 0 -> node 2, slot 1 -> node 0, slot 2 padding.
        SimConfig cfg = centralCfg(3, {2, 0}, 3);
        World w;
        w.AddCvmm(1, 2, 3);
        w.AddCvmm(4, 5, 6);
        w.AddCvmm(7, 8, 9);
        LinkTable t = makeTable(3, {0, 0, true},
                                {{{0, 1}, {11.0, 101.0, true}},
                                 {{0, 2}, {12.0, 102.0, false}},
                                 {{1, 2}, {21.0, 201.0, true}}});
        Io io;
        RlBridge b(cfg);
        decide(b, w, t, 0, true);
        std::vector<double> o = toVec(io.Last()["obs"]);
        std::vector<double> expect = {
            1, 7, 8, 9, 12, 102, 21, 201,   // slot 0 = node 2: peers 0, 1
            1, 1, 2, 3, 11, 101, 12, 102,   // slot 1 = node 0: peers 1, 2
            0, 0, 0, 0, 0, 0, 0, 0};        // padding
        check(o == expect, "obs layout: slot order, peer order, symmetric link values, padding");
    });
}

// README "facts.nodes": N rows in file order, x,y,z,vx,vy,vz,slot; slot -1 if uncontrolled.
static void
test_facts_nodes_rows()
{
    isolated("test_facts_nodes_rows", [] {
        SimConfig cfg = centralCfg(4, {2, 0}, 3);
        World w;
        w.AddCvmm(1, 2, 3);
        w.AddPlain(4, 5, 6, ns3::Vector(0.5, -0.5, 0.25));  // e.g. random walk
        w.AddCvmm(7, 8, 9);
        w.AddCvmm(10, 11, 12);
        w.Cv(0)->SetVelocity(ns3::Vector(-3, 0, 0));
        Io io;
        RlBridge b(cfg);
        decide(b, w, makeTable(4), 0, true);
        json n = io.Last()["facts"]["nodes"];
        check(n.size() == 4, "facts.nodes has N rows");
        check(toVec(n[0]) == std::vector<double>({1, 2, 3, -3, 0, 0, 1}),
              "facts.nodes[0]: pos, velocity, slot 1");
        check(toVec(n[1]) == std::vector<double>({4, 5, 6, 0.5, -0.5, 0.25, -1}),
              "facts.nodes[1]: uncontrolled plain model velocity, slot -1");
        check(toVec(n[2]) == std::vector<double>({7, 8, 9, 0, 0, 0, 0}),
              "facts.nodes[2]: slot 0");
        check(n[3][6] == -1, "facts.nodes[3]: uncontrolled slot -1");
        check(n[0][6].is_number_integer() && n[1][6].is_number_integer(),
              "facts.nodes slot column is an integer");
    });
}

// README "facts.links": N(N-1)/2 rows, pairs i<j with i outer; is_los 0/1 integer.
static void
test_facts_links_order()
{
    isolated("test_facts_links_order", [] {
        SimConfig cfg = centralCfg(4, {0}, 1);
        World w;
        for (int i = 0; i < 4; ++i)
        {
            w.AddCvmm(i, i, 0);
        }
        std::map<std::pair<uint32_t, uint32_t>, L> over;
        for (uint32_t i = 0; i < 4; ++i)
        {
            for (uint32_t j = i + 1; j < 4; ++j)
            {
                over[{i, j}] = L{10.0 * i + j, 100.0 * i + j, (i + j) % 2 == 0};
            }
        }
        Io io;
        RlBridge b(cfg);
        decide(b, w, makeTable(4, {0, 0, false}, over), 0, true);
        json links = io.Last()["facts"]["links"];
        check(links.size() == 6, "facts.links has N(N-1)/2 rows");
        bool ok = links.size() == 6;
        size_t r = 0;
        for (uint32_t i = 0; i < 4 && ok; ++i)
        {
            for (uint32_t j = i + 1; j < 4 && ok; ++j, ++r)
            {
                const json& row = links[r];
                ok = approx(row[0], 10.0 * i + j) && approx(row[1], 100.0 * i + j) &&
                     row[2].is_number_integer() && row[2] == (((i + j) % 2 == 0) ? 1 : 0);
            }
        }
        check(ok, "facts.links rows in (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) order with 0/1 LOS");
    });
}

// README "facts.window" sums and the reward invariant reward == legacy_reward_sum/ticks.
static void
test_facts_window_sums()
{
    isolated("test_facts_window_sums", [] {
        SimConfig cfg = centralCfg(3, {0}, 1, 3, 6);
        cfg.rl.reward_type = "throughput";
        World w;
        w.AddCvmm(10, 10, 0);
        w.AddCvmm(20, 20, 0);
        w.AddCvmm(30, 30, 0);
        // Tick A: 2 connected (one exactly at -6.7, inclusive), 2 LOS.
        LinkTable a = makeTable(3, {20, 100, true}, {{{1, 2}, {-6.71, 0, false}},
                                                    {{0, 2}, {-6.7, 5, true}}});
        // Tick B: 3 connected, 0 LOS.
        LinkTable bt = makeTable(3, {5, 50, false});
        std::vector<FlowResult> fa = {flow(0, 1, 10, 8, true), flow(1, 2, 5, 0, false),
                                      flow(0, 2, 0, 0, false)};  // zero demand excluded
        std::vector<FlowResult> fb = {flow(0, 1, 10, 10, true), flow(2, 0, 4, 0, false)};
        Io io;
        RlBridge b(cfg);
        b.AccumulateTick(a, fa);
        b.AccumulateTick(bt, fb);
        b.AccumulateTick(bt, {});
        b.Step(3, 0.3, w.mobs, bt, {}, true);
        json m = io.Last();
        json win = m["facts"]["window"];
        check(win["ticks"] == 3 && m["ticks_in_step"] == 3, "window ticks == ticks_in_step == 3");
        check(approx(win["demand_mbps_sum"], 29.0), "demand_mbps_sum = 10+5+0+10+4");
        check(approx(win["delivered_mbps_sum"], 18.0), "delivered_mbps_sum = 8+10");
        check(win["flow_ticks_with_demand"] == 4, "flow_ticks_with_demand counts demand>0 only");
        check(win["unroutable_flow_ticks"] == 2,
              "unroutable_flow_ticks excludes zero-demand unroutable flows");
        check(win["connected_pairs_sum"] == 2 + 3 + 3,
              "connected_pairs_sum uses SINR >= -6.7 inclusive per tick");
        check(win["los_pairs_sum"] == 2, "los_pairs_sum over i<j pairs per tick");
        check(approx(win["legacy_reward_sum"], 18.0), "legacy_reward_sum = sum of throughput");
        check(approx(m["reward"], 6.0), "reward = legacy_reward_sum / ticks");
        bool ints = true;
        for (const char* k : {"ticks", "flow_ticks_with_demand", "unroutable_flow_ticks",
                              "connected_pairs_sum", "los_pairs_sum"})
        {
            ints = ints && win[k].is_number_integer();
        }
        check(ints, "window count fields are integers");
        check(win.size() == 8, "window has exactly the 8 documented keys");
    });
}

// rl-bridge.h Step: "writes a step message ... and clears the window".
static void
test_window_resets_after_step()
{
    isolated("test_window_resets_after_step", [] {
        SimConfig cfg = centralCfg(2, {0}, 1, 2, 4);
        World w;
        w.AddCvmm(10, 10, 0);
        w.AddCvmm(20, 20, 0);
        LinkTable t = makeTable(2);
        std::vector<FlowResult> f = {flow(0, 1, 3, 2, true)};
        Io io("{\"action\":[4]}\n");
        RlBridge b(cfg);
        b.AccumulateTick(t, f);
        b.AccumulateTick(t, f);
        b.Step(1, 0.1, w.mobs, t, f, false);
        b.AccumulateTick(t, f);
        b.Step(2, 0.2, w.mobs, t, f, true);
        auto lines = io.Lines();
        check(lines.size() == 2, "two step lines");
        if (lines.size() == 2)
        {
            json w2 = lines[1]["facts"]["window"];
            check(w2["ticks"] == 1, "second window counts only its own tick");
            check(approx(w2["demand_mbps_sum"], 3.0) && approx(w2["delivered_mbps_sum"], 2.0),
                  "second window sums reset");
            check(approx(lines[1]["reward"], 2.0), "second reward is its own mean");
        }
    });
}

// README invariant: reward is 0.0 when ticks == 0.
static void
test_reward_zero_ticks()
{
    isolated("test_reward_zero_ticks", [] {
        SimConfig cfg = centralCfg(2, {0}, 1);
        World w;
        w.AddCvmm(1, 1, 0);
        w.AddCvmm(2, 2, 0);
        Io io;
        RlBridge b(cfg);
        b.Step(0, 0.0, w.mobs, makeTable(2), {flow(0, 1, 5, 5, true)}, true);
        json m = io.Last();
        check(m["ticks_in_step"] == 0 && approx(m["reward"], 0.0),
              "empty window: ticks 0, reward 0.0");
        check(approx(m["facts"]["window"]["legacy_reward_sum"], 0.0),
              "empty window: legacy_reward_sum 0");
    });
}

// README worked example: tick rewards +1,-1,+1 give decision reward +1/3.
static void
test_reward_mean_all_links_los()
{
    isolated("test_reward_mean_all_links_los", [] {
        SimConfig cfg = centralCfg(2, {0}, 1, 3, 3);
        cfg.rl.reward_type = "all_links_los";
        World w;
        w.AddCvmm(1, 1, 0);
        w.AddCvmm(2, 2, 0);
        LinkTable los = makeTable(2, {10, 10, true});
        LinkTable nlos = makeTable(2, {10, 10, false});
        Io io;
        RlBridge b(cfg);
        b.AccumulateTick(los, {});
        b.AccumulateTick(nlos, {});
        b.AccumulateTick(los, {});
        b.Step(3, 0.3, w.mobs, los, {}, true);
        json m = io.Last();
        check(approx(m["reward"], 1.0 / 3.0), "mean of +1,-1,+1 is 1/3");
        check(approx(m["facts"]["window"]["legacy_reward_sum"], 1.0), "legacy_reward_sum = 1");
    });
}

// sim-config.h RlConfig: throughput = sum of delivered_mbps over flows (legacy, per tick).
static void
test_reward_throughput_legacy()
{
    isolated("test_reward_throughput_legacy", [] {
        SimConfig cfg = legacyCfg(3);
        World w;
        w.AddCvmm(0, 0, 10);
        w.AddCvmm(1, 0, 10);
        w.AddCvmm(2, 0, 10);
        Io io;
        RlBridge b(cfg);
        b.Step(0, 0.0, w.mobs, makeTable(3),
               {flow(0, 1, 5, 3, true), flow(1, 2, 9, 4.5, true), flow(2, 0, 2, 0, false)}, true);
        check(approx(io.Last()["reward"], 7.5), "legacy throughput reward = 3 + 4.5");
        Io io2;
        RlBridge b2(cfg);
        b2.Step(0, 0.0, w.mobs, makeTable(3), {}, true);
        check(approx(io2.Last()["reward"], 0.0), "legacy throughput reward with no flows = 0");
    });
}

// README "Reward and wall behavior": all_links_los +1 iff every controlled node's peer links are LOS.
static void
test_reward_all_links_los_legacy()
{
    isolated("test_reward_all_links_los_legacy", [] {
        SimConfig cfg = legacyCfg(3);  // controlled = last node (2)
        cfg.rl.reward_type = "all_links_los";
        World w;
        w.AddCvmm(0, 0, 10);
        w.AddCvmm(1, 0, 10);
        w.AddCvmm(2, 0, 10);
        auto rewardFor = [&](const LinkTable& t) {
            Io io;
            RlBridge b(cfg);
            b.Step(0, 0.0, w.mobs, t, {}, true);
            return io.Last()["reward"].get<double>();
        };
        check(approx(rewardFor(makeTable(3, {0, 0, true})), 1.0), "all LOS -> +1");
        check(approx(rewardFor(makeTable(3, {0, 0, true}, {{{1, 2}, {0, 0, false}}})), -1.0),
              "one NLOS peer link of the controlled node -> -1");
        check(approx(rewardFor(makeTable(3, {0, 0, true}, {{{0, 1}, {0, 0, false}}})), 1.0),
              "NLOS between uncontrolled nodes does not matter -> +1");
    });
}

// all_links_los across several slots; padding ignored.
static void
test_reward_all_links_los_multi_slot()
{
    isolated("test_reward_all_links_los_multi_slot", [] {
        SimConfig cfg = centralCfg(4, {0, 1}, 3);
        cfg.rl.reward_type = "all_links_los";
        World w;
        for (int i = 0; i < 4; ++i)
        {
            w.AddCvmm(i, i, 0);
        }
        auto rewardFor = [&](const LinkTable& t) {
            Io io;
            RlBridge b(cfg);
            decide(b, w, t, 0, true);
            return io.Last()["reward"].get<double>();
        };
        check(approx(rewardFor(makeTable(4, {0, 0, true})), 1.0), "multi-slot all LOS -> +1");
        check(approx(rewardFor(makeTable(4, {0, 0, true}, {{{1, 3}, {0, 0, false}}})), -1.0),
              "second slot has an NLOS peer -> -1");
        check(approx(rewardFor(makeTable(4, {0, 0, true}, {{{2, 3}, {0, 0, false}}})), 1.0),
              "NLOS only between uncontrolled nodes -> +1");
    });
}

// README "Terminal message": done:true, no action read afterwards; decision counts up.
static void
test_decision_counter_and_terminal()
{
    isolated("test_decision_counter_and_terminal", [] {
        SimConfig cfg = centralCfg(2, {0}, 1, 1, 2);
        World w;
        w.AddCvmm(50, 50, 0);
        w.AddCvmm(10, 10, 0);
        Io io("{\"action\":[4]}\n{\"action\":[4]}\n{\"action\":[1]}\n");
        RlBridge b(cfg);
        decide(b, w, makeTable(2), 0);
        decide(b, w, makeTable(2), 1);
        decide(b, w, makeTable(2), 2, true);
        auto lines = io.Lines();
        check(lines.size() == 3, "three step messages");
        if (lines.size() == 3)
        {
            check(lines[0]["decision"] == 0 && lines[1]["decision"] == 1 &&
                      lines[2]["decision"] == 2,
                  "decision index 0,1,2");
            check(lines[0]["done"] == false && lines[1]["done"] == false &&
                      lines[2]["done"] == true,
                  "only the terminal message is done");
            check(approx(lines[1]["time_s"], 0.1), "time_s passed through");
        }
        check(io.Unread() == "{\"action\":[1]}\n", "no action read after done");
    });
}

// =====================================================================
// joint actions (centralized)
// =====================================================================

// README Action meanings: 0 west, 1 east, 2 south, 3 north, 4 hold; z never changed.
static void
test_joint_action_applied()
{
    isolated("test_joint_action_applied", [] {
        SimConfig cfg = centralCfg(6, {0, 1, 2, 3, 4}, 6);
        cfg.nodes[4].node_type = "pedestrian";
        World w;
        for (int i = 0; i < 6; ++i)
        {
            w.AddCvmm(50, 50, 10);
        }
        w.Cv(4)->SetVelocity(ns3::Vector(0, 0, 3));
        Io io("{\"action\":[0,1,2,3,0,4]}\n");
        RlBridge b(cfg);
        decide(b, w, makeTable(6), 0);
        auto v = [&](int i) { return w.Vel(i); };
        check(approx(v(0).x, -10) && approx(v(0).y, 0) && approx(v(0).z, 0), "action 0 = west");
        check(approx(v(1).x, 10) && approx(v(1).y, 0), "action 1 = east");
        check(approx(v(2).x, 0) && approx(v(2).y, -10), "action 2 = south");
        check(approx(v(3).x, 0) && approx(v(3).y, 10), "action 3 = north");
        check(approx(v(4).x, -1.5) && approx(v(4).z, 0), "pedestrian west at 1.5 m/s, vz zeroed");
        check(w.Cv(5)->SetVelocityCalls() == 0, "uncontrolled node untouched");
        check(io.err.str().empty(), "valid action: no warning");
    });
}

// README "Hold": action 4 commands zero velocity; it does not repeat the previous direction.
static void
test_hold_stops_node()
{
    isolated("test_hold_stops_node", [] {
        SimConfig cfg = centralCfg(2, {0}, 1);
        World w;
        w.AddCvmm(50, 50, 0);
        w.AddCvmm(10, 10, 0);
        Io io("{\"action\":[1]}\n{\"action\":[4]}\n");
        RlBridge b(cfg);
        decide(b, w, makeTable(2), 0);
        check(approx(w.Vel(0).x, 10), "east commanded");
        decide(b, w, makeTable(2), 1);
        check(approx(w.Vel(0).x, 0) && approx(w.Vel(0).y, 0), "hold zeroes velocity");
    });
}

// README: revalidated_slots is empty for a deliberate hold and for wall clipping.
static void
test_revalidated_excludes_hold_and_clipping()
{
    isolated("test_revalidated_excludes_hold_and_clipping", [] {
        SimConfig cfg = centralCfg(3, {0, 1}, 2, 3, 6);
        World w;
        w.AddCvmm(99.5, 50, 0);  // east is valid but will clip at the wall mid-window
        w.AddCvmm(50, 50, 0);
        w.AddCvmm(10, 10, 0);
        Io io("{\"action\":[1,4]}\n{\"action\":[4,4]}\n");
        runEpisode(cfg, w, [](uint32_t) { return makeTable(3); });
        auto lines = io.Lines();
        check(lines.size() == 4, "init + 3 steps");
        if (lines.size() == 4)
        {
            check(lines[2]["revalidated_slots"].empty(),
                  "deliberate hold and wall clip are not revalidated");
            check(approx(lines[2]["facts"]["nodes"][0][0], 100.0, 1e-9),
                  "clipped node lands on x_max");
        }
        check(io.err.str().empty(), "no warnings for hold or clipping");
    });
}

// README EOF (centralized): all slots hold for the rest of the episode, one warning,
// every message still runs.
static void
test_eof_centralized()
{
    isolated("test_eof_centralized", [] {
        SimConfig cfg = centralCfg(2, {0}, 1, 2, 6);
        World w;
        w.AddCvmm(50, 50, 0);
        w.AddCvmm(10, 10, 0);
        w.Cv(0)->SetVelocity(ns3::Vector(5, 0, 0));
        Io io("");
        runEpisode(cfg, w, [](uint32_t) { return makeTable(2); });
        auto lines = io.Lines();
        check(lines.size() == 1 + 4, "init + 4 steps (ticks 0,2,4,6) despite closed stdin");
        check(approx(w.Pos(0).x, 50) && approx(w.Pos(0).y, 50),
              "closed stdin: controlled node holds at start position");
        check(countOf(io.err.str(),
                      "Warning: RL action stream closed; all controlled nodes hold.") == 1,
              "stream-closed warning exactly once");
        bool noneRevalidated = true;
        for (size_t i = 1; i < lines.size(); ++i)
        {
            noneRevalidated = noneRevalidated && lines[i]["revalidated_slots"].empty();
        }
        check(noneRevalidated, "EOF hold does not list revalidated slots");
        if (lines.size() == 5)
        {
            check(lines.back()["done"] == true && lines.back()["tick"] == 6,
                  "terminal message still emitted at num_ticks");
        }
    });
}

// =====================================================================
// per-tick clamp (BeforeAdvance)
// =====================================================================

// README "Clamp": velocity reduced so the node lands exactly on the bound.
static void
test_clamp_upper_partial()
{
    isolated("test_clamp_upper_partial", [] {
        SimConfig cfg = centralCfg(2, {0}, 1);
        World w;
        w.AddCvmm(99.5, 50, 0);
        w.AddCvmm(10, 10, 0);
        w.Cv(0)->SetVelocity(ns3::Vector(10, 0, 0));
        RlBridge b(cfg);
        b.BeforeAdvance(w.mobs);
        check(approx(w.Vel(0).x, 5.0), "vx reduced to (100-99.5)/0.1 = 5");
        w.Advance(0.1);
        check(approx(w.Pos(0).x, 100.0), "lands exactly on x_max");
        b.BeforeAdvance(w.mobs);
        check(approx(w.Vel(0).x, 0.0), "at the wall outward velocity becomes 0");

        w.Cv(0)->SetPosition(ns3::Vector(99.0, 50, 0));
        w.Cv(0)->SetVelocity(ns3::Vector(10, 0, 0));
        b.BeforeAdvance(w.mobs);
        check(approx(w.Vel(0).x, 10.0), "exactly reaching the wall is not reduced");
    });
}

// Inward motion at a wall, lower bounds, y axis, and z untouched.
static void
test_clamp_lower_y_and_z()
{
    isolated("test_clamp_lower_y_and_z", [] {
        SimConfig cfg = centralCfg(2, {0}, 1);
        World w;
        w.AddCvmm(100, 50, 10);
        w.AddCvmm(10, 10, 0);
        w.Cv(0)->SetVelocity(ns3::Vector(-10, 0, 0));
        RlBridge b(cfg);
        b.BeforeAdvance(w.mobs);
        check(approx(w.Vel(0).x, -10), "inward velocity at x_max is not clamped");

        w.Cv(0)->SetPosition(ns3::Vector(0.3, 0.2, 10));
        w.Cv(0)->SetVelocity(ns3::Vector(-10, -10, 3));
        b.BeforeAdvance(w.mobs);
        check(approx(w.Vel(0).x, -3), "x lower clamp: (0-0.3)/0.1 = -3");
        check(approx(w.Vel(0).y, -2), "y lower clamp: (0-0.2)/0.1 = -2");
        check(approx(w.Vel(0).z, 3), "z velocity untouched by clamp");

        w.Cv(0)->SetPosition(ns3::Vector(50, 99.9, 10));
        w.Cv(0)->SetVelocity(ns3::Vector(0, 10, 0));
        b.BeforeAdvance(w.mobs);
        check(approx(w.Vel(0).y, 1.0, 1e-6), "y upper clamp: (100-99.9)/0.1 = 1");
    });
}

// rl-bridge.h BeforeAdvance: controlled slots only; non-CVMM skipped; legacy no-op.
static void
test_clamp_scope()
{
    isolated("test_clamp_scope", [] {
        SimConfig cfg = centralCfg(3, {0, 1}, 3);
        World w;
        w.AddPlain(100, 50, 0, ns3::Vector(10, 0, 0));  // controlled but not a CVMM
        w.AddCvmm(50, 50, 0);
        w.AddCvmm(100, 50, 0);                            // uncontrolled
        w.Cv(2)->SetVelocity(ns3::Vector(10, 0, 0));
        RlBridge b(cfg);
        b.BeforeAdvance(w.mobs);
        check(approx(w.Vel(0).x, 10), "non-CVMM controlled model skipped");
        check(approx(w.Vel(2).x, 10) && w.Cv(2)->SetVelocityCalls() == 1,
              "uncontrolled CVMM is not clamped");

        SimConfig lc = legacyCfg(2);
        World lw;
        lw.AddCvmm(0, 0, 0);
        lw.AddCvmm(100, 0, 10);
        lw.Cv(1)->SetVelocity(ns3::Vector(1000, 0, 0));
        RlBridge lb(lc);
        lb.BeforeAdvance(lw.mobs);
        check(approx(lw.Vel(1).x, 1000) && lw.Cv(1)->SetVelocityCalls() == 1,
              "legacy BeforeAdvance is a no-op");
    });
}

// =====================================================================
// end-to-end centralized episodes
// =====================================================================

// README "Worked three-node example" (centralized-multi-smoke).
static void
test_worked_example_three_node()
{
    isolated("test_worked_example_three_node", [] {
        SimConfig cfg;
        cfg.tick_s = 0.1;
        cfg.duration_s = 1.0;
        cfg.nodes = {makeNode("node-a", 50, 20, 10), makeNode("node-b", 100, 0, 10),
                     makeNode("node-c", 97, 50, 10)};
        cfg.rl.enabled = true;
        cfg.rl.control_mode = "centralized";
        cfg.rl.controlled_indices = {1, 2};
        cfg.rl.num_slots = 3;
        cfg.rl.decision_interval_ticks = 5;
        cfg.rl.num_ticks = 10;
        cfg.rl.step_size_m = 1.0;
        cfg.rl.reward_type = "all_links_los";
        cfg.rl.x_min = 0;
        cfg.rl.x_max = 100;
        cfg.rl.y_min = -50;
        cfg.rl.y_max = 100;
        cfg.rl.z_min = 0;
        cfg.rl.z_max = 50;
        World w;
        w.AddPlain(50, 20, 10, ns3::Vector(1.5, 0, 0));  // node-a random walk
        w.AddCvmm(100, 0, 10);
        w.AddCvmm(97, 50, 10);
        Io io("{\"action\":[2,1,4]}\n{\"action\":[4,0,4]}\n");
        runEpisode(cfg, w, [](uint32_t) { return makeTable(3); });
        auto lines = io.Lines();
        check(lines.size() == 4, "worked example: init + steps at ticks 0,5,10");
        if (lines.size() != 4)
        {
            return;
        }
        json s1 = lines[2];
        json n1 = s1["facts"]["nodes"];
        check(s1["tick"] == 5 && s1["ticks_in_step"] == 5 && s1["decision"] == 1,
              "first decision at tick 5 over 5 ticks");
        check(approx(n1[1][0], 100, 1e-9) && approx(n1[1][1], -5, 1e-9),
              "b moves (100,0) -> (100,-5)");
        check(approx(n1[2][0], 100, 1e-9) && approx(n1[2][1], 50, 1e-9),
              "c moves (97,50) -> (100,50), clipped at x_max");
        check(approx(n1[2][3], 0.0) && approx(n1[1][4], -10.0),
              "facts velocity is the post-clamp velocity (c 0, b -10)");
        check(approx(n1[1][2], 10) && approx(n1[2][2], 10), "z unchanged");
        // b (100,-5) and c (100,50) are both on x_max: east masked for both.
        check(s1["mask"] == json::array({1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1}),
              "b and c east masked at x_max; padded slot hold only");
        check(s1["revalidated_slots"].empty(), "padding hold is not revalidated");
        check(approx(s1["reward"], 1.0), "all LOS -> mean reward +1");
        json s2 = lines[3];
        json n2 = s2["facts"]["nodes"];
        check(approx(n2[1][0], 100, 1e-9) && approx(n2[1][1], -5, 1e-9), "b holds at (100,-5)");
        check(approx(n2[2][0], 95, 1e-9) && approx(n2[2][1], 50, 1e-9), "c reaches (95,50)");
        check(s2["done"] == true && s2["tick"] == 10 && s2["decision"] == 2,
              "terminal step at tick 10, decision 2");
        check(n2[0][6] == -1 && n2[1][6] == 0 && n2[2][6] == 1, "slot column: a -1, b 0, c 1");
    });
}

// README "Partial windows": last window shorter, ticks_in_step reports it.
static void
test_partial_final_window()
{
    isolated("test_partial_final_window", [] {
        SimConfig cfg = centralCfg(2, {0}, 1, 5, 7);
        cfg.rl.reward_type = "throughput";
        World w;
        w.AddCvmm(50, 50, 0);
        w.AddCvmm(10, 10, 0);
        Io io("{\"action\":[4]}\n{\"action\":[4]}\n");
        runEpisode(cfg, w, [](uint32_t) { return makeTable(2); },
                   [](uint32_t ti) {
                       return std::vector<FlowResult>{flow(0, 1, 100, ti, true)};
                   });
        auto lines = io.Lines();
        check(lines.size() == 4, "init + steps at ticks 0, 5, 7");
        if (lines.size() != 4)
        {
            return;
        }
        check(lines[1]["tick"] == 0 && lines[2]["tick"] == 5 && lines[3]["tick"] == 7,
              "steps at ticks 0,5,7");
        check(lines[1]["ticks_in_step"] == 1 && lines[2]["ticks_in_step"] == 5 &&
                  lines[3]["ticks_in_step"] == 2,
              "ticks_in_step 1,5,2");
        check(approx(lines[1]["reward"], 0.0), "reset reward = tick-0 reward (0)");
        check(approx(lines[2]["reward"], (1 + 2 + 3 + 4 + 5) / 5.0), "mean over ticks 1..5 = 3");
        check(approx(lines[3]["reward"], (6 + 7) / 2.0), "mean over ticks 6..7 = 6.5");
        check(lines[3]["done"] == true, "partial window ends with done");
    });
}

// README: identical inputs give identical stdout (time_s is simulated time).
static void
test_determinism_same_stdout()
{
    isolated("test_determinism_same_stdout", [] {
        auto run = [] {
            SimConfig cfg = centralCfg(3, {0, 2}, 3, 2, 6);
            World w;
            w.AddCvmm(99, 1, 0);
            w.AddPlain(30, 30, 0);
            w.AddCvmm(40, 60, 5);
            Io io("{\"action\":[1,2,4]}\n{\"action\":[0,3,1]}\n{\"action\":[3,0,4]}\n");
            runEpisode(cfg, w,
                       [](uint32_t ti) { return makeTable(3, {ti * 1.5 - 3, ti * 10.0, ti % 2 == 0}); },
                       [](uint32_t ti) {
                           return std::vector<FlowResult>{flow(0, 2, 7, ti * 0.5, ti % 3 != 0)};
                       });
            return io.out.str();
        };
        std::string a = run();
        std::string b = run();
        check(!a.empty() && a == b, "two identical runs produce identical stdout");
    });
}

// =====================================================================
// legacy mode
// =====================================================================

// README "Legacy step remains unchanged": type,tick,time_s,obs.{...},reward,done,action_type.
static void
test_legacy_step_shape()
{
    isolated("test_legacy_step_shape", [] {
        SimConfig cfg = legacyCfg(3);
        World w;
        w.AddCvmm(1, 2, 3);
        w.AddCvmm(4, 5, 6);
        w.AddCvmm(7, 8, 9);
        LinkTable t = makeTable(3, {0, 0, true},
                                {{{0, 1}, {11, 101, true}}, {{0, 2}, {12, 102, true}},
                                 {{1, 2}, {21, 201, true}}});
        Io io("{\"action\":6}\n");
        RlBridge b(cfg);
        b.WriteInit();
        b.Step(4, 0.4, w.mobs, t, {}, false);
        auto lines = io.Lines();
        check(lines.size() == 1, "legacy: one step line, no init");
        if (lines.size() != 1)
        {
            return;
        }
        json m = lines[0];
        check(m["type"] == "step" && m["tick"] == 4 && approx(m["time_s"], 0.4),
              "legacy step: type, tick, time_s");
        check(toVec(m["obs"]["controlled_pos"]) == std::vector<double>({7, 8, 9}),
              "legacy obs.controlled_pos = last node xyz");
        check(toVec(m["obs"]["link_sinrs"]) == std::vector<double>({12, 21}),
              "legacy link_sinrs to peers in index order");
        check(toVec(m["obs"]["link_capacities"]) == std::vector<double>({102, 201}),
              "legacy link_capacities to peers in index order");
        check(m["done"] == false && m["action_type"] == "discrete", "legacy done, action_type");
        check(!m.contains("facts") && !m.contains("mask") && !m.contains("decision"),
              "legacy step carries no facts/mask/decision");
        check(m.size() == 7, "legacy step has exactly 7 top-level fields");
    });
}

// rl-bridge.h constructor: legacy uses the first controlled index, else the last node.
static void
test_legacy_controlled_node_selection()
{
    isolated("test_legacy_controlled_node_selection", [] {
        World w;
        w.AddCvmm(1, 1, 1);
        w.AddCvmm(2, 2, 2);
        w.AddCvmm(3, 3, 3);
        Io io("{\"action\":1}\n");
        RlBridge b(legacyCfg(3, {0}));
        b.Step(0, 0, w.mobs, makeTable(3), {}, false);
        b.ApplyAction(w.mobs);
        check(toVec(io.Last()["obs"]["controlled_pos"]) == std::vector<double>({1, 1, 1}),
              "controlled_indices[0] selects node 0");
        check(approx(w.Vel(0).x, 20) && w.Cv(2)->SetVelocityCalls() == 0,
              "action applied to node 0 only");
    });
}

// sim-config.h action table: 0:-X 1:+X 2:-Y 3:+Y 4:-Z 5:+Z 6:Stay; speed min(dist/tick, cap).
static void
test_legacy_discrete_velocities()
{
    isolated("test_legacy_discrete_velocities", [] {
        struct C { int a; double vx, vy, vz; };
        for (C c : {C{0, -20, 0, 0}, C{1, 20, 0, 0}, C{2, 0, -20, 0}, C{3, 0, 20, 0},
                    C{4, 0, 0, -20}, C{5, 0, 0, 20}, C{6, 0, 0, 0}})
        {
            World w;
            w.AddCvmm(0, 0, 0);
            w.AddCvmm(0, 0, 25);
            Io io("{\"action\":" + std::to_string(c.a) + "}\n");
            RlBridge b(legacyCfg(2));  // step 50 m, tick 0.1 -> 500 m/s capped at drone 20
            b.Step(0, 0, w.mobs, makeTable(2), {}, false);
            b.ApplyAction(w.mobs);
            ns3::Vector v = w.Vel(1);
            check(approx(v.x, c.vx) && approx(v.y, c.vy) && approx(v.z, c.vz),
                  "legacy action " + std::to_string(c.a) + " velocity");
        }
        // Small step: speed = dist / tick = 1 / 0.1 = 10 < cap.
        World w;
        w.AddCvmm(0, 0, 0);
        w.AddCvmm(0, 0, 25);
        Io io("{\"action\":1}\n");
        RlBridge b(legacyCfg(2, {}, 1.0));
        b.Step(0, 0, w.mobs, makeTable(2), {}, false);
        b.ApplyAction(w.mobs);
        check(approx(w.Vel(1).x, 10), "legacy speed = step/tick when below cap");
        // Vehicle cap.
        SimConfig vc = legacyCfg(2);
        vc.nodes[1].node_type = "vehicle";
        World vw;
        vw.AddCvmm(0, 0, 0);
        vw.AddCvmm(0, 0, 25);
        Io vio("{\"action\":2}\n");
        RlBridge vb(vc);
        vb.Step(0, 0, vw.mobs, makeTable(2), {}, false);
        vb.ApplyAction(vw.mobs);
        check(approx(vw.Vel(1).y, -15), "legacy vehicle capped at 15 m/s");
    });
}

// sim-config.h: legacy target is clamped to the bounding box (x/y/z).
static void
test_legacy_target_clamp()
{
    isolated("test_legacy_target_clamp", [] {
        struct C { double x, y, z; int a; double vx, vy, vz; const char* name; };
        for (C c : {C{100, 0, 25, 1, 0, 0, 0, "at x_max, +X -> 0"},
                    C{99.5, 0, 25, 1, 5, 0, 0, "0.5 m from x_max, +X -> 5 m/s"},
                    C{0, -100, 25, 2, 0, 0, 0, "at y_min, -Y -> 0"},
                    C{0, 0, 50, 5, 0, 0, 0, "at z_max, +Z -> 0"},
                    C{0, 0, 0.2, 4, 0, 0, -2, "0.2 m above z_min, -Z -> -2 m/s"}})
        {
            World w;
            w.AddCvmm(0, 0, 0);
            w.AddCvmm(c.x, c.y, c.z);
            Io io("{\"action\":" + std::to_string(c.a) + "}\n");
            RlBridge b(legacyCfg(2));
            b.Step(0, 0, w.mobs, makeTable(2), {}, false);
            b.ApplyAction(w.mobs);
            ns3::Vector v = w.Vel(1);
            check(approx(v.x, c.vx, 1e-6) && approx(v.y, c.vy, 1e-6) && approx(v.z, c.vz, 1e-6),
                  std::string("legacy clamp: ") + c.name);
        }
    });
}

// README legacy EOF: Stay, one "stream closed" warning; Test 9: steps continue.
static void
test_legacy_eof()
{
    isolated("test_legacy_eof", [] {
        SimConfig cfg = legacyCfg(2);
        cfg.rl.num_ticks = 3;
        World w;
        w.AddCvmm(0, 0, 0);
        w.AddCvmm(0, 0, 25);
        w.Cv(1)->SetVelocity(ns3::Vector(5, 5, 5));
        Io io("");
        runEpisode(cfg, w, [](uint32_t) { return makeTable(2); });
        auto lines = io.Lines();
        check(lines.size() == 4, "legacy closed stdin: steps for ticks 0..3, no init");
        bool allSteps = true;
        for (const auto& l : lines)
        {
            allSteps = allSteps && l["type"] == "step";
        }
        check(allSteps, "legacy closed stdin: only step messages");
        check(approx(w.Pos(1).x, 0) && approx(w.Pos(1).z, 25), "legacy closed stdin: Stay");
        check(countOf(io.err.str(),
                      "Warning: RL action stream closed; holding position (Stay).") == 1,
              "legacy stream-closed warning exactly once");
        check(!lines.empty() && lines.back()["done"] == true, "legacy last step done");
    });
}

// README legacy: no action read after done.
static void
test_legacy_done_no_read()
{
    isolated("test_legacy_done_no_read", [] {
        World w;
        w.AddCvmm(0, 0, 0);
        w.AddCvmm(0, 0, 25);
        Io io("{\"action\":1}\n");
        RlBridge b(legacyCfg(2));
        b.Step(9, 0.9, w.mobs, makeTable(2), {}, true);
        check(io.Last()["done"] == true, "legacy terminal step done:true");
        check(io.Unread() == "{\"action\":1}\n", "legacy: no action read after done");
    });
}

// sim-config.h: continuous action is an absolute (x, y[, z]) target.
static void
test_legacy_continuous_target()
{
    isolated("test_legacy_continuous_target", [] {
        SimConfig cfg = legacyCfg(2);
        cfg.rl.action_type = "continuous";
        cfg.rl.arrival_threshold_m = 1.0;
        {
            World w;
            w.AddCvmm(0, 0, 0);
            w.AddCvmm(0, 0, 10);
            Io io("{\"action\":[30,40,10]}\n");
            RlBridge b(cfg);
            b.Step(0, 0, w.mobs, makeTable(2), {}, false);
            b.ApplyAction(w.mobs);
            check(approx(w.Vel(1).x, 12) && approx(w.Vel(1).y, 16) && approx(w.Vel(1).z, 0),
                  "continuous: toward (30,40) at capped 20 m/s");
            check(io.Last()["action_type"] == "continuous", "continuous: action_type reported");
        }
        {
            World w;
            w.AddCvmm(0, 0, 0);
            w.AddCvmm(0, 0, 10);
            Io io("{\"action\":[0.5,0,10]}\n");
            RlBridge b(cfg);
            b.Step(0, 0, w.mobs, makeTable(2), {}, false);
            b.ApplyAction(w.mobs);
            check(approx(w.Vel(1).x, 0), "continuous: within arrival threshold -> stop");
        }
        {
            World w;
            w.AddCvmm(0, 0, 0);
            w.AddCvmm(0, 0, 10);
            Io io("{\"action\":[500,0,10]}\n");
            RlBridge b(cfg);
            b.Step(0, 0, w.mobs, makeTable(2), {}, false);
            b.ApplyAction(w.mobs);
            check(approx(w.Vel(1).x, 20) && approx(w.Vel(1).y, 0),
                  "continuous: target beyond x_max clamped, heads east");
        }
    });
}

// rl-bridge.cc:447 comment: "keep current z if absent" for a 2-component target.
static void
test_legacy_continuous_2d_keeps_z()
{
    isolated("test_legacy_continuous_2d_keeps_z", [] {
        // A 2-D target keeps the node's current z.
        SimConfig cfg = legacyCfg(2);
        cfg.rl.action_type = "continuous";
        World w;
        w.AddCvmm(0, 0, 0);
        w.AddCvmm(0, 0, 10);
        Io io("{\"action\":[30,0]}\n");
        RlBridge b(cfg);
        b.Step(0, 0, w.mobs, makeTable(2), {}, false);
        b.ApplyAction(w.mobs);
        check(approx(w.Vel(1).z, 0), "continuous 2-D target keeps current z (vz == 0)");
        check(approx(w.Vel(1).x, 20), "continuous 2-D target heads +x at full speed");
    });
}

// ---- main ----

int
main()
{
    test_init_fields();
    test_init_num_decisions_ceil();
    test_init_speed_caps();
    test_init_jammer_path_flag();
    test_init_legacy_writes_nothing();
    test_is_decision_tick();
    test_mask_interior_and_padding();
    test_mask_walls_and_corners();
    test_step_message_shape();
    test_obs_layout_peer_order();
    test_facts_nodes_rows();
    test_facts_links_order();
    test_facts_window_sums();
    test_window_resets_after_step();
    test_reward_zero_ticks();
    test_reward_mean_all_links_los();
    test_reward_throughput_legacy();
    test_reward_all_links_los_legacy();
    test_reward_all_links_los_multi_slot();
    test_decision_counter_and_terminal();
    test_joint_action_applied();
    test_hold_stops_node();
    test_revalidated_excludes_hold_and_clipping();
    test_eof_centralized();
    test_clamp_upper_partial();
    test_clamp_lower_y_and_z();
    test_clamp_scope();
    test_worked_example_three_node();
    test_partial_final_window();
    test_determinism_same_stdout();
    test_legacy_step_shape();
    test_legacy_controlled_node_selection();
    test_legacy_discrete_velocities();
    test_legacy_target_clamp();
    test_legacy_eof();
    test_legacy_done_no_read();
    test_legacy_continuous_target();
    test_legacy_continuous_2d_keeps_z();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
