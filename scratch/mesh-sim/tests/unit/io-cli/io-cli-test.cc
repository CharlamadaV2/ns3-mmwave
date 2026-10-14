/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file io-cli-test.cc
 * @brief Unit tests for src/io (MetricsWriter, WriteRunLog, VizWriter) and
 *        src/cli (ParseCommandLine, ResolveSeeds, ResolveQuerySeed,
 *        ArchiveScenarioInputs).
 *
 * Standalone binary; ns-3 headers are stubbed (common/ log + random stubs,
 * stubs/ mobility + command-line stubs symlinked into a local ns3/ dir).
 * Functions that call std::exit(1) run in a fork()ed child. Every file is
 * written under one unique directory in std::filesystem::temp_directory_path()
 * that is removed at exit.
 * Build and run: `make -C tests/unit/io-cli test`.
 */

#include "src/cli/cli-parser.h"
#include "src/eval/link-table.h"
#include "src/io/metrics-writer.h"
#include "src/io/progress-logger.h"
#include "src/io/run-logger.h"
#include "src/io/viz-writer.h"
#include "third_party/json.hpp"

#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace mesh_sim;
using json = nlohmann::json;
namespace fs = std::filesystem;

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

/// Root temp dir for this process; removed at the end of main().
static fs::path g_root;

/// Return a fresh, empty subdirectory of g_root named after the test.
static fs::path
freshDir(const std::string& name)
{
    fs::path p = g_root / name;
    fs::remove_all(p);
    fs::create_directories(p);
    return p;
}

static std::string
readFile(const fs::path& p)
{
    std::ifstream in(p);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void
writeFile(const fs::path& p, const std::string& content)
{
    std::ofstream out(p);
    out << content;
}

static std::vector<std::string>
readLines(const fs::path& p)
{
    std::vector<std::string> lines;
    std::ifstream in(p);
    std::string line;
    while (std::getline(in, line))
    {
        lines.push_back(line);
    }
    return lines;
}

static std::vector<std::string>
splitCsv(const std::string& line)
{
    std::vector<std::string> out;
    std::string cur;
    for (char c : line)
    {
        if (c == ',')
        {
            out.push_back(cur);
            cur.clear();
        }
        else
        {
            cur += c;
        }
    }
    out.push_back(cur);
    return out;
}

static bool
contains(const std::string& hay, const std::string& needle)
{
    return hay.find(needle) != std::string::npos;
}

/// Redirect std::cerr into a string for the lifetime of the object.
struct CerrCapture
{
    std::ostringstream buf;
    std::streambuf* old;
    std::ios::fmtflags oldFlags;
    std::streamsize oldPrec;
    CerrCapture()
        : old(std::cerr.rdbuf(buf.rdbuf())),
          oldFlags(std::cerr.flags()),
          oldPrec(std::cerr.precision())
    {
    }
    ~CerrCapture()
    {
        std::cerr.rdbuf(old);
        std::cerr.flags(oldFlags);
        std::cerr.precision(oldPrec);
    }
    std::string str() const { return buf.str(); }
};

/// Result of running a function in a forked child.
struct ChildResult
{
    int code = -1;      ///< Exit code, or -signal if killed, or 77 if fn returned.
    std::string err;    ///< Captured stderr of the child.
    std::string out;    ///< Text the child wrote to its result stream.
};

/// Run @p fn in a forked child with stderr captured. If @p fn returns, the
/// child exits 77. @p fn may write results to the provided stream.
static ChildResult
runChild(const std::string& tag, const std::function<void(std::ostream&)>& fn)
{
    fs::path errPath = g_root / (tag + ".child.err");
    fs::path outPath = g_root / (tag + ".child.out");
    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);

    pid_t pid = fork();
    if (pid == 0)
    {
        int fd = ::open(errPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0)
        {
            ::dup2(fd, 2);
            ::close(fd);
        }
        {
            std::ofstream out(outPath);
            fn(out);
        }
        std::cerr.flush();
        std::fflush(nullptr);
        _exit(77);
    }

    ChildResult r;
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status))
    {
        r.code = WEXITSTATUS(status);
    }
    else if (WIFSIGNALED(status))
    {
        r.code = -WTERMSIG(status);
    }
    r.err = readFile(errPath);
    r.out = readFile(outPath);
    return r;
}

/// Parse "  key   = value" lines in the section starting at @p header
/// (until the next blank line) into a map.
static std::map<std::string, std::string>
parseSection(const std::string& text, const std::string& header)
{
    std::map<std::string, std::string> kv;
    std::istringstream in(text);
    std::string line;
    bool inSection = false;
    while (std::getline(in, line))
    {
        if (!inSection)
        {
            if (line == header)
            {
                inSection = true;
            }
            continue;
        }
        if (line.empty())
        {
            break;
        }
        auto pos = line.find("= ");
        if (pos == std::string::npos)
        {
            continue;
        }
        kv[trimStr(line.substr(0, pos))] = trimStr(line.substr(pos + 2));
    }
    return kv;
}

static NodeSpec
mkNode(const std::string& id, const std::string& role = "peer")
{
    NodeSpec n;
    n.id = id;
    n.role = role;
    return n;
}

static LinkResult
mkLink(uint32_t a, uint32_t b, double sinr, bool los,
       double cap = 0.0, double dist = 0.0, double rx = -999.0,
       uint32_t mcs = 0, bool fromBuildings = false)
{
    LinkResult r;
    r.tx_id = a;
    r.rx_id = b;
    r.sinr_db = sinr;
    r.is_los = los;
    r.capacity_mbps = cap;
    r.distance_m = dist;
    r.rx_power_dbm = rx;
    r.mcs_index = mcs;
    r.condition_from_buildings = fromBuildings;
    return r;
}

static FlowResult
mkFlow(uint32_t src, uint32_t dst, double demand, double delivered,
       double latency, uint32_t hops, bool routable,
       std::vector<uint32_t> path = {})
{
    FlowResult f;
    f.src = src;
    f.dst = dst;
    f.demand_mbps = demand;
    f.delivered_mbps = delivered;
    f.latency_ms = latency;
    f.hop_count = hops;
    f.routable = routable;
    f.path = std::move(path);
    return f;
}

/// LinkTable for 3 nodes from the three upper-triangle (sinr, los) values.
static LinkTable
table3(double s01, bool l01, double s02, bool l02, double s12, bool l12)
{
    LinkTable t;
    t.Update(3, {mkLink(0, 1, s01, l01), mkLink(0, 2, s02, l02), mkLink(1, 2, s12, l12)});
    return t;
}

static SimConfig
cfg3(const fs::path& dir)
{
    SimConfig c;
    c.scenario_name = "unit-scn";
    c.seed = 7;
    c.duration_s = 2.0;
    c.warmup_s = 0.5;
    c.output_dir = dir.string();
    c.nodes = {mkNode("nA"), mkNode("nB"), mkNode("nC")};
    return c;
}

static json
readJson(const fs::path& p)
{
    try
    {
        return json::parse(readFile(p));
    }
    catch (const std::exception& e)
    {
        std::cerr << "  (json parse failed for " << p << ": " << e.what() << ")\n";
        return json();
    }
}

static double
num(const json& j, const std::string& a, const std::string& b)
{
    if (!j.is_object() || !j.contains(a) || !j[a].contains(b) || !j[a][b].is_number())
    {
        return std::nan("");
    }
    return j[a][b].get<double>();
}

static double
num3(const json& j, const std::string& a, const std::string& b, const std::string& c)
{
    if (!j.is_object() || !j.contains(a) || !j[a].contains(b) || !j[a][b].contains(c) ||
        !j[a][b][c].is_number())
    {
        return std::nan("");
    }
    return j[a][b][c].get<double>();
}

// =========================================================================
// MetricsWriter
// =========================================================================

/// Full worked example: 3 nodes, a pre-warmup tick, two counted ticks.
/// Expected values are hand-computed from the schema in metrics-writer.h.
static void
test_metrics_averages_worked_example()
{
    fs::path dir = freshDir("metrics-avg");
    SimConfig c = cfg3(dir);
    MetricsWriter mw(c);

    // Pre-warmup tick (t < warmup_s = 0.5): must not affect anything.
    mw.AccumulateTick(0.0, table3(50, true, 50, true, 50, true),
                      {mkFlow(0, 1, 999, 999, 99, 4, true)}, 3);

    // Tick 1 at t == warmup_s (inclusive, since only t < warmup_s is skipped).
    // (0,1)=10 LOS, (0,2)=-6.7 NLOS (connected: threshold inclusive),
    // (1,2)=-900 exactly (sentinel; excluded from SINR, not connected).
    mw.AccumulateTick(0.5, table3(10.0, true, -6.7, false, -900.0, false),
                      {mkFlow(0, 1, 10, 8, 1.0, 1, true, {0, 1}),
                       mkFlow(2, 0, 5, 0, 0, 0, false)},
                      3);
    // Tick 2: (0,1)=20 NLOS, (0,2)=-10 NLOS, (1,2)=-899 LOS (valid sample).
    mw.AccumulateTick(1.0, table3(20.0, false, -10.0, false, -899.0, true),
                      {mkFlow(0, 1, 10, 6, 3.0, 2, true, {0, 2, 1}),
                       mkFlow(1, 1, 4, 0, 0, 0, true)},
                      3);
    {
        CerrCapture cap;
        mw.Write();
    }

    json j = readJson(dir / "summary.json");
    check(j.is_object(), "metrics: summary.json parses as a JSON object");
    if (!j.is_object())
    {
        return;
    }
    for (const char* k : {"scenario", "seed", "duration_s", "warmup_s", "per_node", "per_flow", "network"})
    {
        check(j.contains(k), std::string("metrics: top-level key present: ") + k);
    }
    check(j.value("scenario", "") == "unit-scn", "metrics: scenario = cfg.scenario_name");
    check(j.value("seed", 0u) == 7u, "metrics: seed = cfg.seed");
    check(approx(j.value("duration_s", -1.0), 2.0), "metrics: duration_s = cfg.duration_s");
    check(approx(j.value("warmup_s", -1.0), 0.5), "metrics: warmup_s = cfg.warmup_s");
    check(!j.contains("wall_clock_start") && !j.contains("wall_elapsed_s"),
          "metrics: wall_clock_* omitted when SetTiming not called");

    // per_node keyed by nodes.json ids
    check(j["per_node"].size() == 3 && j["per_node"].contains("nA") &&
              j["per_node"].contains("nB") && j["per_node"].contains("nC"),
          "metrics: per_node keyed by node ids nA/nB/nC");

    // node 0: samples 10,-6.7,20,-10
    check(approx(num3(j, "per_node", "nA", "mean_sinr_db"), 13.3 / 4), "metrics: nA mean_sinr_db = 3.325");
    check(approx(num3(j, "per_node", "nA", "min_sinr_db"), -10.0), "metrics: nA min_sinr_db = -10");
    check(approx(num3(j, "per_node", "nA", "max_sinr_db"), 20.0), "metrics: nA max_sinr_db = 20");
    check(approx(num3(j, "per_node", "nA", "num_links"), 1.5), "metrics: nA num_links = (2+1)/2");
    check(approx(num3(j, "per_node", "nA", "num_los_links"), 0.5), "metrics: nA num_los_links = (1+0)/2");
    check(approx(num3(j, "per_node", "nA", "tx_throughput_mbps"), 7.0), "metrics: nA tx_throughput = (8+6)/2");
    check(approx(num3(j, "per_node", "nA", "rx_throughput_mbps"), 0.0), "metrics: nA rx_throughput = 0");
    // node 1: samples 10,20,-899 (-900 excluded)
    check(approx(num3(j, "per_node", "nB", "mean_sinr_db"), -869.0 / 3), "metrics: nB mean_sinr_db excludes -900, includes -899");
    check(approx(num3(j, "per_node", "nB", "min_sinr_db"), -899.0), "metrics: nB min_sinr_db = -899");
    check(approx(num3(j, "per_node", "nB", "num_links"), 1.0), "metrics: nB num_links = 1");
    check(approx(num3(j, "per_node", "nB", "num_los_links"), 1.0), "metrics: nB num_los_links = (1+1)/2");
    check(approx(num3(j, "per_node", "nB", "rx_throughput_mbps"), 7.0), "metrics: nB rx_throughput = (8+6)/2");
    check(approx(num3(j, "per_node", "nB", "tx_throughput_mbps"), 0.0), "metrics: nB tx_throughput = 0 (self-flow delivers 0)");
    // node 2: samples -6.7,-10,-899
    check(approx(num3(j, "per_node", "nC", "mean_sinr_db"), -915.7 / 3), "metrics: nC mean_sinr_db");
    check(approx(num3(j, "per_node", "nC", "max_sinr_db"), -6.7), "metrics: nC max_sinr_db = -6.7");
    check(approx(num3(j, "per_node", "nC", "num_links"), 0.5), "metrics: nC num_links = (1+0)/2");
    check(approx(num3(j, "per_node", "nC", "num_los_links"), 0.5), "metrics: nC num_los_links = (0+1)/2");

    // per_flow: averaged over the ticks each flow was reported
    check(j["per_flow"].size() == 3, "metrics: per_flow has 3 src->dst keys");
    check(approx(num3(j, "per_flow", "nA->nB", "demand_mbps"), 10.0), "metrics: nA->nB demand mean");
    check(approx(num3(j, "per_flow", "nA->nB", "delivered_mbps"), 7.0), "metrics: nA->nB delivered mean");
    check(approx(num3(j, "per_flow", "nA->nB", "latency_ms"), 2.0), "metrics: nA->nB latency mean");
    check(approx(num3(j, "per_flow", "nA->nB", "hop_count"), 1.5), "metrics: nA->nB hop_count mean");
    check(approx(num3(j, "per_flow", "nC->nA", "demand_mbps"), 5.0), "metrics: nC->nA averaged over its 1 tick");
    check(approx(num3(j, "per_flow", "nB->nB", "demand_mbps"), 4.0), "metrics: self-flow nB->nB reported");

    // network
    check(approx(num(j, "network", "sum_throughput_mbps"), 7.0), "metrics: network.sum_throughput = 14/2 ticks");
    check(approx(num(j, "network", "mean_sinr_db"), -1771.4 / 10), "metrics: network.mean_sinr_db over 10 valid directed samples");
    check(approx(num(j, "network", "connectivity"), 0.5), "metrics: network.connectivity = (2+1)/(3+3)");
    check(approx(num(j, "network", "mean_hop_count"), 1.0), "metrics: network.mean_hop_count over routable obs (1,2,0)");
    check(approx(num(j, "network", "flows_routed"), 1.5), "metrics: network.flows_routed = 3/2");
    check(approx(num(j, "network", "flows_unroutable"), 0.5), "metrics: network.flows_unroutable = 1/2");

}

/// warmup_s boundary: a tick just below warmup is skipped, at warmup counted.
static void
test_metrics_warmup_boundary()
{
    fs::path dir = freshDir("metrics-warmup");
    SimConfig c = cfg3(dir);
    c.warmup_s = 1.0;
    MetricsWriter mw(c);
    auto lt = table3(5, true, 5, true, 5, true);
    mw.AccumulateTick(0.9999, lt, {mkFlow(0, 1, 100, 100, 1, 1, true)}, 3);
    mw.AccumulateTick(1.0, lt, {mkFlow(0, 1, 4, 4, 1, 1, true)}, 3);
    {
        CerrCapture cap;
        mw.Write();
    }
    json j = readJson(dir / "summary.json");
    check(approx(num(j, "network", "sum_throughput_mbps"), 4.0),
          "metrics: tick at 0.9999 < warmup skipped, tick at warmup counted");
    check(approx(num3(j, "per_flow", "nA->nB", "delivered_mbps"), 4.0),
          "metrics: per_flow excludes pre-warmup tick");
}

/// Connectivity threshold -6.7 dB is inclusive.
static void
test_metrics_connectivity_threshold()
{
    fs::path dir = freshDir("metrics-conn");
    SimConfig c = cfg3(dir);
    c.warmup_s = 0.0;
    c.nodes = {mkNode("a"), mkNode("b")};
    MetricsWriter mw(c);
    LinkTable at, below;
    at.Update(2, {mkLink(0, 1, -6.7, false)});
    below.Update(2, {mkLink(0, 1, -6.71, false)});
    mw.AccumulateTick(0.0, at, {}, 2);
    mw.AccumulateTick(0.1, at, {}, 2);
    mw.AccumulateTick(0.2, at, {}, 2);
    mw.AccumulateTick(0.3, below, {}, 2);
    {
        CerrCapture cap;
        mw.Write();
    }
    json j = readJson(dir / "summary.json");
    check(approx(num(j, "network", "connectivity"), 0.75), "metrics: connectivity uses inclusive -6.7 dB threshold (3/4)");
    check(approx(num3(j, "per_node", "a", "num_links"), 0.75), "metrics: per-node num_links inclusive threshold");
}

/// Write overwrites an existing summary.json and confirms on stderr.
static void
test_metrics_overwrites()
{
    fs::path dir = freshDir("metrics-overwrite");
    writeFile(dir / "summary.json", "garbage garbage garbage garbage garbage garbage garbage garbage garbage garbage garbage");
    SimConfig c = cfg3(dir);
    MetricsWriter mw(c);
    std::string err;
    {
        CerrCapture cap;
        mw.Write();
        err = cap.str();
    }
    json j = readJson(dir / "summary.json");
    check(j.is_object() && j.value("scenario", "") == "unit-scn", "metrics: existing summary.json overwritten (truncated)");
}

// =========================================================================
// WriteRunLog
// =========================================================================

static CliArgs
baseArgs()
{
    CliArgs a;
    a.run_config_path = "/scn/run.ini";
    return a;
}

static SimConfig
runLogCfg()
{
    SimConfig c;
    c.scenario_name = "rl-scn";
    c.duration_s = 12.5;
    c.tick_s = 0.25;
    c.warmup_s = 1.5;
    c.run_id = 4;
    c.nodes = {mkNode("nA"), mkNode("nB"), mkNode("nC")};
    c.buildings = {BuildingSpec{}, BuildingSpec{}};
    c.channel.channel_model = "nyu";
    c.channel.scenario = "UMa";
    c.channel.frequency_ghz = 2.4;
    c.channel.bandwidth_mhz = 20.0;
    c.channel.tx_power_dbm = 23.5;
    c.mesh.traffic.model = "on_off";
    c.mesh.traffic.flow_topology = "gateway";
    c.mesh.traffic.demand_mbps = 2.5;
    c.mesh.routing.algorithm = "min_hop";
    c.mesh.routing.max_hops = 3;
    c.band = "sub-6";
    c.band_source = "cli";
    return c;
}

static JammerSpec
mkJammer(const std::string& id, bool enabled)
{
    JammerSpec j;
    j.enabled = enabled;
    j.id = id;
    j.type = "constant";
    j.target_freq = {};
    j.tx_power_dbm = 25.0;
    j.tx_array_gain_dbi = 12.0;
    j.duty_cycle = 1.0;
    j.max_range_m = 0.0;
    j.beamwidth_deg = 360.0;
    j.azimuth_deg = 0.0;
    j.zenith_deg = 0.0;
    return j;
}

/// Creates nested dirs; header block; seeds list.
static void
test_runlog_header_and_dirs()
{
    fs::path base = g_root / "runlog-hdr" / "nested" / "deeper";
    fs::remove_all(g_root / "runlog-hdr");
    WriteRunLog(base.string(), baseArgs(), runLogCfg(), {1, 2, 3});
    check(fs::exists(base / "run.log"), "runlog: creates intermediate dirs and run.log");
    auto lines = readLines(base / "run.log");
    check(lines.size() > 6, "runlog: has content");
    if (lines.size() <= 6)
    {
        return;
    }
    check(lines[0] == "mesh-sim run log", "runlog: title line");
    check(std::regex_match(lines[2], std::regex("timestamp: +\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z")),
          "runlog: ISO-8601 timestamp line");
    check(std::regex_match(lines[3], std::regex("scenario: +rl-scn")), "runlog: scenario line = cfg.scenario_name");
    check(std::regex_match(lines[4], std::regex("run_config: +/scn/run\\.ini")), "runlog: run_config line");
    check(std::regex_match(lines[5], std::regex("seeds: +\\[1, 2, 3\\]")), "runlog: seeds list '[1, 2, 3]'");

    fs::path b2 = freshDir("runlog-empty-seeds");
    WriteRunLog(b2.string(), baseArgs(), runLogCfg(), {});
    check(contains(readFile(b2 / "run.log"), "seeds:               []\n"), "runlog: empty seed list renders '[]'");
}

/// seed_source label follows --seeds > --seed > config default.
static void
test_runlog_seed_source()
{
    auto sourceOf = [](const CliArgs& a, const std::string& tag) {
        fs::path d = freshDir("runlog-src-" + tag);
        WriteRunLog(d.string(), a, runLogCfg(), {1});
        std::smatch m;
        std::string text = readFile(d / "run.log");
        std::regex re("seed_source: +([^\n]*)\n");
        return std::regex_search(text, m, re) ? m[1].str() : std::string("<missing>");
    };
    CliArgs a = baseArgs();
    check(sourceOf(a, "default") == "config default", "runlog: seed_source config default");
    a.seed_override = 0;
    check(sourceOf(a, "seed0") == "--seed CLI override", "runlog: --seed=0 counts as set (>= 0)");
    a.seeds_arg = "1,2";
    check(sourceOf(a, "both") == "--seeds CLI argument", "runlog: --seeds wins over --seed");
}

/// CLI overrides block: "(not set)" sentinels and set values.
static void
test_runlog_cli_overrides()
{
    fs::path d1 = freshDir("runlog-cli-unset");
    WriteRunLog(d1.string(), baseArgs(), runLogCfg(), {1});
    auto kv = parseSection(readFile(d1 / "run.log"), "CLI overrides:");
    check(kv["--run-config"] == "/scn/run.ini", "runlog cli: --run-config value");
    check(kv["--seeds"] == "(not set)", "runlog cli: --seeds (not set)");
    check(kv["--seed"] == "(not set)", "runlog cli: --seed (not set) for -1");
    check(kv["--run-id"] == "(not set)", "runlog cli: --run-id (not set) for -1");
    check(kv["--positions-override"] == "(not set)", "runlog cli: --positions-override (not set)");
    check(kv["--output-dir"] == "(not set)", "runlog cli: --output-dir (not set)");
    check(kv["--debug-links"] == "false", "runlog cli: --debug-links false");
    check(kv["--rl-mode"] == "false", "runlog cli: --rl-mode false");
    check(kv["--band"] == "(not set)", "runlog cli: --band (not set)");

    CliArgs a = baseArgs();
    a.seeds_arg = "4,5";
    a.seed_override = 9;
    a.run_id_override = 0;
    a.positions_override_path = "/p.json";
    a.output_dir = "/out";
    a.debug_links = true;
    a.rl_mode = true;
    a.band = "sub-6";
    fs::path d2 = freshDir("runlog-cli-set");
    WriteRunLog(d2.string(), a, runLogCfg(), {4, 5});
    kv = parseSection(readFile(d2 / "run.log"), "CLI overrides:");
    check(kv["--seeds"] == "4,5", "runlog cli: --seeds value");
    check(kv["--seed"] == "9", "runlog cli: --seed value");
    check(kv["--run-id"] == "0", "runlog cli: --run-id=0 is set");
    check(kv["--positions-override"] == "/p.json", "runlog cli: --positions-override value");
    check(kv["--output-dir"] == "/out", "runlog cli: --output-dir value");
    check(kv["--debug-links"] == "true", "runlog cli: --debug-links true");
    check(kv["--rl-mode"] == "true", "runlog cli: --rl-mode true");
    check(kv["--band"] == "sub-6", "runlog cli: --band value");
}

/// Resolved config block mirrors the SimConfig scalar fields.
static void
test_runlog_resolved_config()
{
    fs::path d = freshDir("runlog-resolved");
    WriteRunLog(d.string(), baseArgs(), runLogCfg(), {1});
    auto kv = parseSection(readFile(d / "run.log"), "Resolved config:");
    check(kv["duration_s"] == "12.5", "runlog cfg: duration_s");
    check(kv["tick_s"] == "0.25", "runlog cfg: tick_s");
    check(kv["warmup_s"] == "1.5", "runlog cfg: warmup_s");
    check(kv["run_id"] == "4", "runlog cfg: run_id");
    check(kv["nodes"] == "3", "runlog cfg: nodes count");
    check(kv["buildings"] == "2", "runlog cfg: buildings count");
    check(kv["channel_model"] == "nyu", "runlog cfg: channel_model");
    check(kv["scenario"] == "UMa", "runlog cfg: channel scenario");
    check(kv["frequency_ghz"] == "2.4", "runlog cfg: frequency_ghz");
    check(kv["bandwidth_mhz"] == "20", "runlog cfg: bandwidth_mhz");
    check(kv["tx_power_dbm"] == "23.5", "runlog cfg: tx_power_dbm");
    check(kv["traffic.model"] == "on_off", "runlog cfg: traffic.model");
    check(kv["traffic.topology"] == "gateway", "runlog cfg: traffic.topology");
    check(kv["traffic.demand_mbps"] == "2.5", "runlog cfg: traffic.demand_mbps");
    check(kv["routing.algorithm"] == "min_hop", "runlog cfg: routing.algorithm");
    check(kv["routing.max_hops"] == "3", "runlog cfg: routing.max_hops");
    check(kv["band"] == "sub-6", "runlog cfg: band");
    check(kv["band_source"] == "cli", "runlog cfg: band_source");
    check(kv["rl.reward_type"] == "throughput", "runlog cfg: rl.reward_type default");
    check(kv["rl.reward_alias"] == "none", "runlog cfg: rl.reward_alias none when empty");
    check(kv["rl.control_mode"] == "legacy", "runlog cfg: rl.control_mode default legacy");
    check(kv["rl.contract"] == "legacy_discrete7", "runlog cfg: rl.contract legacy_discrete7 by default");
    check(kv["rl.controlled_nodes"] == "(none)", "runlog cfg: rl.controlled_nodes (none)");
    check(kv["jammers.configured"] == "0" && kv["jammers.enabled"] == "0", "runlog cfg: zero jammers");
    check(kv["jammer_path_enabled"] == "false", "runlog cfg: jammer_path_enabled false with no jammers");

    SimConfig c = runLogCfg();
    c.rl.reward_type = "all_links_los";
    c.rl.reward_type_alias = "mean_sinr";
    fs::path d2 = freshDir("runlog-alias");
    WriteRunLog(d2.string(), baseArgs(), c, {1});
    kv = parseSection(readFile(d2 / "run.log"), "Resolved config:");
    check(kv["rl.reward_type"] == "all_links_los", "runlog cfg: rl.reward_type value");
    check(kv["rl.reward_alias"] == "mean_sinr", "runlog cfg: rl.reward_alias value");
}

/// rl.contract derivation, controlled ids, slot count, decision interval.
static void
test_runlog_rl_block()
{
    auto block = [](const SimConfig& c, const std::string& tag) {
        fs::path d = freshDir("runlog-rl-" + tag);
        WriteRunLog(d.string(), baseArgs(), c, {1});
        return parseSection(readFile(d / "run.log"), "Resolved config:");
    };
    SimConfig c = runLogCfg();
    c.rl.control_mode = "centralized";
    c.rl.action_type = "continuous";  // centralized wins regardless
    c.rl.controlled_indices = {2, 0, 9};  // 9 out of range -> skipped
    c.rl.num_slots = 4;
    c.rl.decision_interval_ticks = 3;
    auto kv = block(c, "central");
    check(kv["rl.control_mode"] == "centralized", "runlog rl: control_mode centralized");
    check(kv["rl.contract"] == "mesh_move_2d_v1", "runlog rl: centralized -> mesh_move_2d_v1");
    check(kv["rl.controlled_nodes"] == "nC,nA", "runlog rl: controlled ids in slot order, out-of-range skipped");
    check(kv["rl.max_controlled_nodes"] == "4", "runlog rl: max_controlled_nodes = num_slots");
    check(kv["rl.decision_interval_s"] == "0.75", "runlog rl: decision_interval_s = ticks * tick_s");
    check(kv["rl.decision_interval_ticks"] == "3", "runlog rl: decision_interval_ticks");

    c = runLogCfg();
    c.rl.action_type = "continuous";
    c.rl.controlled_indices = {1};
    kv = block(c, "cont");
    check(kv["rl.contract"] == "legacy_continuous", "runlog rl: legacy continuous -> legacy_continuous");
    check(kv["rl.controlled_nodes"] == "nB", "runlog rl: single controlled id");

    c = runLogCfg();
    c.rl.action_type = "discrete";
    kv = block(c, "disc");
    check(kv["rl.contract"] == "legacy_discrete7", "runlog rl: legacy discrete -> legacy_discrete7");
    check(kv["rl.decision_interval_s"] == "0.25", "runlog rl: default 1 tick -> tick_s");
}

/// Jammer counts, jammer_path_enabled rule, and per-jammer lines.
static void
test_runlog_jammers()
{
    // sub-6 with one enabled + one disabled jammer -> path enabled.
    SimConfig c = runLogCfg();
    JammerSpec j1 = mkJammer("jam1", true);
    j1.target_freq = {2400.0, 2480.0};
    j1.duty_cycle = 0.5;
    j1.azimuth_deg = 90.0;
    j1.zenith_deg = 90.0;
    JammerSpec j2 = mkJammer("", false);
    j2.type = "random";
    j2.waypoints = {Waypoint{0, 0, 0, 0}, Waypoint{1, 1, 1, 1}};
    j2.velocity.vx = 3.0;  // waypoints take precedence over velocity
    JammerSpec j3 = mkJammer("jam3", false);
    j3.velocity.vz = -1.0;
    c.jammers = {j1, j2, j3};
    fs::path d = freshDir("runlog-jam");
    WriteRunLog(d.string(), baseArgs(), c, {1});
    std::string text = readFile(d / "run.log");
    auto kv = parseSection(text, "Resolved config:");
    check(kv["jammers.configured"] == "3", "runlog jam: configured = 3");
    check(kv["jammers.enabled"] == "1", "runlog jam: enabled = 1");
    check(kv["jammer_path_enabled"] == "true", "runlog jam: sub-6 + enabled jammer -> path enabled");
    check(contains(text, "\nJammers:\n"), "runlog jam: Jammers section present");
    check(contains(text,
                   "  jam1: enabled=true type=constant target_freq_mhz=[2400, 2480] tx_power_dbm=25 "
                   "tx_array_gain_dbi=12 duty_cycle=0.5 max_range_m=0 beamwidth_deg=360 "
                   "azimuth_deg=90 zenith_deg=90 motion=static\n"),
          "runlog jam: jam1 line exact");
    check(contains(text, "  (unnamed): enabled=false type=random target_freq_mhz=[] "),
          "runlog jam: empty id -> (unnamed), empty target list");
    check(contains(text, "motion=waypoints\n"), "runlog jam: waypoints motion (precedence over velocity)");
    check(contains(text, "  jam3: ") && contains(text, "motion=velocity\n"), "runlog jam: velocity motion");

    // mmwave with enabled jammer -> path disabled.
    c.band = "mmwave";
    fs::path d2 = freshDir("runlog-jam-mmw");
    WriteRunLog(d2.string(), baseArgs(), c, {1});
    kv = parseSection(readFile(d2 / "run.log"), "Resolved config:");
    check(kv["jammer_path_enabled"] == "false", "runlog jam: mmwave -> path disabled");

    // sub-6 with only disabled jammers -> path disabled.
    c.band = "sub-6";
    c.jammers = {j2, j3};
    fs::path d3 = freshDir("runlog-jam-dis");
    WriteRunLog(d3.string(), baseArgs(), c, {1});
    kv = parseSection(readFile(d3 / "run.log"), "Resolved config:");
    check(kv["jammers.enabled"] == "0" && kv["jammer_path_enabled"] == "false",
          "runlog jam: sub-6 with no enabled jammer -> path disabled");

    // No jammers -> no Jammers section.
    c.jammers.clear();
    fs::path d4 = freshDir("runlog-jam-none");
    WriteRunLog(d4.string(), baseArgs(), c, {1});
    check(!contains(readFile(d4 / "run.log"), "Jammers:"), "runlog jam: section omitted when none");
}


// =========================================================================
// VizWriter
// =========================================================================

static SimConfig
vizCfg(const fs::path& dir)
{
    SimConfig c;
    c.scenario_name = "viz-scn";
    c.output_dir = dir.string();
    c.duration_s = 5.0;
    c.viz_tick_ms = 100;
    c.channel.frequency_ghz = 28.0;
    c.channel.tx_power_dbm = 30.0;
    c.channel.channel_model = "nyu";
    c.channel.nyu.rain_rate_mm_hr = 2.5;
    c.mesh.traffic.flow_topology = "gateway";
    c.mesh.traffic.model = "on_off";
    c.nodes = {mkNode("a", "relay"), mkNode("b", "peer"), mkNode("c", "gw")};
    return c;
}

static const char* kVizFiles[] = {"positions.csv", "links.csv", "rx-power.csv",
                                  "mcs.csv", "flows.csv", "routes.csv"};

/// Headers, metadata block in documented order, overwrite of old files.
static void
test_viz_open_headers_and_meta()
{
    fs::path dir = freshDir("viz-open");
    writeFile(dir / "links.csv", "stale,content\n1,2\n");
    SimConfig c = vizCfg(dir);
    {
        VizWriter vw(c);
        vw.Open();
        vw.Close();
    }
    for (const char* f : kVizFiles)
    {
        check(fs::exists(dir / f), std::string("viz: Open creates ") + f);
    }
    auto pos = readLines(dir / "positions.csv");
    std::vector<std::string> expectPos = {
        "# scenario=viz-scn",       "# frequency=28000000000", "# txPower=30",
        "# numNodes=3",             "# simDuration=5000",      "# tickMs=100",
        "# dimensions=3",           "# rainRate=2.5",          "# channelModel=nyu",
        "# flowTopology=gateway",   "# trafficModel=on_off",
        "time_s,node_id,x,y,z,node_type,active"};
    check(pos == expectPos, "viz: positions.csv metadata block (order/values) + header");
    check(readLines(dir / "links.csv") ==
              std::vector<std::string>{"time_s,node_a,node_b,dist_m,sinr_db,condition,"
                                       "condition_reason,capacity_mbps,delivered_mbps,hop_count"},
          "viz: links.csv header only, old content truncated");
    check(readLines(dir / "rx-power.csv") == std::vector<std::string>{"time_s,node_a,node_b,rx_power_dbm"},
          "viz: rx-power.csv header");
    check(readLines(dir / "mcs.csv") == std::vector<std::string>{"time_s,node_a,node_b,mcs_index,spectral_eff"},
          "viz: mcs.csv header");
    check(readLines(dir / "flows.csv") ==
              std::vector<std::string>{"time_s,src,dst,demand_mbps,delivered_mbps,latency_ms,hop_count,routable"},
          "viz: flows.csv header");
    check(readLines(dir / "routes.csv") ==
              std::vector<std::string>{"time_s,src,dst,path,bottleneck_mbps,hop_count,routable"},
          "viz: routes.csv header");
}

/// One snapshot: exact rows in all six files with documented formatting.
static void
test_viz_tick_rows()
{
    fs::path dir = freshDir("viz-rows");
    SimConfig c = vizCfg(dir);
    ns3::MobilityModel m0(1.5, 2.25, 3.0), m1(10.0, 0.0, -1.0), m2(100.123456789, 0.0, 50.0);
    std::vector<ns3::Ptr<ns3::MobilityModel>> mobs = {&m0, &m1, &m2};
    LinkTable lt;
    lt.Update(3, {mkLink(0, 1, 12.345678, true, 123.26, 10.5, -60.5, 7, true),
                  mkLink(0, 2, -3.0, false, 50.04, 20.0, -70.25, 2, false),
                  mkLink(1, 2, 5.5, false, 40.0, 30.0, -80.0, 4, false)});
    std::vector<FlowResult> flows = {
        mkFlow(0, 2, 10.04, 8.06, 1.23456, 2, true, {0, 1, 2}),
        mkFlow(1, 2, 3.0, 3.0, 0.5, 1, true, {1, 2}),
        mkFlow(2, 0, 5.0, 0.0, 0.0, 0, false),
        mkFlow(1, 1, 0.0, 0.0, 0.0, 0, true)};  // self-flow
    {
        VizWriter vw(c);
        vw.Open();
        vw.WriteTick(0.0, mobs, lt, flows);
        vw.WriteTick(0.1, mobs, lt, flows);  // second snapshot: precision must not leak
    }

    auto pos = readLines(dir / "positions.csv");
    std::vector<std::string> posRows(pos.begin() + 12, pos.end());
    check(posRows.size() == 6, "viz rows: positions 3 rows per snapshot x 2");
    if (posRows.size() == 6)
    {
        check(posRows[0] == "0.000000,0,1.500000,2.250000,3.000000,relay,1", "viz rows: positions row 0 (got '" + posRows[0] + "')");
        check(posRows[1] == "0.000000,1,10.000000,0.000000,-1.000000,peer,1", "viz rows: positions row 1");
        check(posRows[2] == "0.000000,2,100.123457,0.000000,50.000000,gw,1", "viz rows: positions row 2 (role, 6 dp)");
        check(posRows[3].rfind("0.100000,0,", 0) == 0, "viz rows: second snapshot time_s 6 dp");
    }

    auto links = readLines(dir / "links.csv");
    check(links.size() == 1 + 6, "viz rows: links one row per unordered pair x 2 snapshots");
    if (links.size() == 7)
    {
        check(links[1] == "0.000000,0,1,10.500000,12.345678,LOS,building,123.3,8.1,2",
              "viz rows: links (0,1) LOS/building, cap 1dp, delivered summed, max hops (got '" + links[1] + "')");
        check(links[2] == "0.000000,0,2,20.000000,-3.000000,NLOS,probabilistic,50.0,0.0,0",
              "viz rows: links (0,2) unused edge 0.0/0 (got '" + links[2] + "')");
        check(links[3] == "0.000000,1,2,30.000000,5.500000,NLOS,probabilistic,40.0,11.1,2",
              "viz rows: links (1,2) delivered 8.06+3.0, max hop 2 (got '" + links[3] + "')");
        check(links[4] == "0.100000,0,1,10.500000,12.345678,LOS,building,123.3,8.1,2",
              "viz rows: links second snapshot precision reset (got '" + links[4] + "')");
    }

    auto rx = readLines(dir / "rx-power.csv");
    check(rx.size() == 7 && rx[1] == "0.000000,0,1,-60.500000" && rx[2] == "0.000000,0,2,-70.250000" &&
              rx[3] == "0.000000,1,2,-80.000000",
          "viz rows: rx-power rows node_a<node_b, 6 dp");

    auto mcs = readLines(dir / "mcs.csv");
    check(mcs.size() == 7 && mcs[1] == "0.000000,0,1,7,1.91" && mcs[2] == "0.000000,0,2,2,0.38" &&
              mcs[3] == "0.000000,1,2,4,0.88" && mcs[4] == "0.100000,0,1,7,1.91",
          "viz rows: mcs index + MCS_TABLE spectral_eff 2 dp");

    auto fl = readLines(dir / "flows.csv");
    check(fl.size() == 1 + 8, "viz rows: flows one row per flow x 2");
    if (fl.size() == 9)
    {
        check(fl[1] == "0.000000,0,2,10.0,8.1,1.235,2,1", "viz rows: flows routable row (got '" + fl[1] + "')");
        check(fl[3] == "0.000000,2,0,5.0,0.0,0.000,0,0", "viz rows: flows unroutable row (got '" + fl[3] + "')");
        check(fl[4] == "0.000000,1,1,0.0,0.0,0.000,0,1", "viz rows: flows self-flow row");
        check(fl[5].rfind("0.100000,0,2,", 0) == 0, "viz rows: flows precision reset next row");
    }

    auto rt = readLines(dir / "routes.csv");
    check(rt.size() == 1 + 8, "viz rows: routes one row per flow x 2");
    if (rt.size() == 9)
    {
        check(rt[1] == "0.000000,0,2,0;1;2,40.0,2,1", "viz rows: routes path ';' and bottleneck=min cap (got '" + rt[1] + "')");
        check(rt[2] == "0.000000,1,2,1;2,40.0,1,1", "viz rows: routes 1-hop");
        check(rt[3] == "0.000000,2,0,,0.0,0,0", "viz rows: routes unroutable empty path, bottleneck 0");
        check(rt[4] == "0.000000,1,1,,0.0,0,1", "viz rows: routes self-flow empty path, bottleneck 0");
        check(rt[5].rfind("0.100000,0,2,", 0) == 0, "viz rows: routes precision reset next row");
    }
}

static std::vector<std::string>
snapshotTimes(const fs::path& posCsv)
{
    std::vector<std::string> times;
    std::set<std::string> seen;
    for (const auto& l : readLines(posCsv))
    {
        if (l.empty() || l[0] == '#' || l.rfind("time_s", 0) == 0)
        {
            continue;
        }
        std::string t = splitCsv(l)[0];
        if (seen.insert(t).second)
        {
            times.push_back(t);
        }
    }
    return times;
}

/// viz_tick_ms rate limiting: first tick at/after each boundary is written.
static void
test_viz_rate_limit()
{
    fs::path dir = freshDir("viz-rate");
    SimConfig c = vizCfg(dir);
    c.viz_tick_ms = 250;
    ns3::MobilityModel m0;
    std::vector<ns3::Ptr<ns3::MobilityModel>> mobs = {&m0};
    LinkTable lt;
    lt.Update(1, {});
    {
        VizWriter vw(c);
        vw.Open();
        for (uint32_t ti = 0; ti <= 10; ++ti)
        {
            vw.WriteTick(ti * 0.1, mobs, lt, {});
        }
    }
    auto times = snapshotTimes(dir / "positions.csv");
    check(times == std::vector<std::string>({"0.000000", "0.300000", "0.500000", "0.800000", "1.000000"}),
          "viz: 250 ms rate limit with 100 ms ticks -> 0, 0.3, 0.5, 0.8, 1.0");

    // No drift over many ticks: viz 100 ms == tick 0.1 s -> every tick written.
    fs::path dir2 = freshDir("viz-rate-nodrift");
    SimConfig c2 = vizCfg(dir2);
    c2.viz_tick_ms = 100;
    {
        VizWriter vw(c2);
        vw.Open();
        for (uint32_t ti = 0; ti <= 1000; ++ti)
        {
            vw.WriteTick(ti * 0.1, mobs, lt, {});
        }
    }
    check(snapshotTimes(dir2 / "positions.csv").size() == 1001,
          "viz: viz_tick_ms == tick_s writes all 1001 ticks (epsilon prevents drift skips)");

    // 1 s viz with 0.1 s ticks over 10 s -> 11 snapshots at whole seconds.
    fs::path dir3 = freshDir("viz-rate-1s");
    SimConfig c3 = vizCfg(dir3);
    c3.viz_tick_ms = 1000;
    {
        VizWriter vw(c3);
        vw.Open();
        for (uint32_t ti = 0; ti <= 100; ++ti)
        {
            vw.WriteTick(ti * 0.1, mobs, lt, {});
        }
    }
    auto t3 = snapshotTimes(dir3 / "positions.csv");
    bool whole = t3.size() == 11;
    for (size_t k = 0; whole && k < t3.size(); ++k)
    {
        whole = std::fabs(std::stod(t3[k]) - static_cast<double>(k)) < 1e-6;
    }
    check(whole, "viz: 1000 ms rate limit -> 11 snapshots at 0..10 s");
}

// =========================================================================
// cli: ResolveSeeds / ResolveQuerySeed
// =========================================================================

static std::string
seedsToString(const std::vector<uint32_t>& v)
{
    std::string s;
    for (size_t i = 0; i < v.size(); ++i)
    {
        s += (i ? "," : "") + std::to_string(v[i]);
    }
    return s;
}

/// Run ResolveSeeds in a child; on return, child writes the seed list.
static ChildResult
resolveInChild(const std::string& tag, const CliArgs& a, uint32_t cfgSeed)
{
    return runChild(tag, [&](std::ostream& out) {
        SimConfig c;
        c.seed = cfgSeed;
        out << seedsToString(ResolveSeeds(a, c));
    });
}

static void
test_resolve_seeds_precedence()
{
    CliArgs a;
    auto r = resolveInChild("rs-default", a, 42);
    check(r.code == 77 && r.out == "42", "seeds: no flags -> cfg.seed");
    a.seed_override = 0;
    r = resolveInChild("rs-seed0", a, 42);
    check(r.code == 77 && r.out == "0", "seeds: --seed=0 overrides cfg.seed");
    a.seed_override = 5;
    a.seeds_arg = "8,9";
    r = resolveInChild("rs-both", a, 42);
    check(r.code == 77 && r.out == "8,9", "seeds: --seeds wins over --seed and cfg");
}

static void
test_resolve_seeds_list_parsing()
{
    CliArgs a;
    a.seeds_arg = "3,1,,2,3,";
    auto r = resolveInChild("rs-list", a, 42);
    check(r.code == 77 && r.out == "3,1,2,3", "seeds: order kept, empty items skipped, no de-dup (got '" + r.out + "')");
    a.seeds_arg = "4294967295";
    r = resolveInChild("rs-max", a, 42);
    check(r.code == 77 && r.out == "4294967295", "seeds: UINT32_MAX accepted");
}

static void
test_resolve_query_seed()
{
    auto q = [](const std::string& tag, const CliArgs& a, uint32_t cfgSeed) {
        return runChild(tag, [&](std::ostream& out) {
            SimConfig c;
            c.seed = cfgSeed;
            out << ResolveQuerySeed(a, c);
        });
    };
    CliArgs a;
    auto r = q("qs-default", a, 11);
    check(r.code == 77 && r.out == "11", "query seed: cfg.seed when no flags");
    a.seed_override = 6;
    r = q("qs-seed", a, 11);
    check(r.code == 77 && r.out == "6", "query seed: --seed");
    a.seeds_arg = "9,";
    r = q("qs-one", a, 11);
    check(r.code == 77 && r.out == "9", "query seed: --seeds with one non-empty item");
    a.seeds_arg = "1,2";
    r = q("qs-two", a, 11);
    check(r.code == 1 && contains(r.err, "Error: --channel-query needs exactly one seed"),
          "query seed: two seeds -> exit 1 with message");
    a.seeds_arg = "5,5";
    r = q("qs-dup", a, 11);
    check(r.code == 1, "query seed: duplicate seeds are not de-duplicated -> exit 1");
    a.seeds_arg = ",";
    r = q("qs-empty", a, 11);
    check(r.code == 1 && contains(r.err, "no seeds to run"), "query seed: empty list -> ResolveSeeds error exit 1");
}

// =========================================================================
// cli: ArchiveScenarioInputs
// =========================================================================

static void
test_archive_copies_regular_files()
{
    fs::path scn = freshDir("arch-scn");
    writeFile(scn / "run.ini", "[scenario]\nname = x\n");
    writeFile(scn / "nodes.json", "{\"nodes\": []}");
    writeFile(scn / "extra.txt", "extra");
    fs::create_directories(scn / "subdir");
    writeFile(scn / "subdir" / "deep.json", "{}");
    fs::path out = g_root / "arch-out" / "batch";
    fs::remove_all(g_root / "arch-out");

    ArchiveScenarioInputs(out.string(), (scn / "run.ini").string());
    check(fs::is_directory(out / "inputs"), "archive: creates <out>/inputs (and parents)");
    check(readFile(out / "inputs" / "run.ini") == "[scenario]\nname = x\n", "archive: run.ini copied byte-exact");
    check(readFile(out / "inputs" / "nodes.json") == "{\"nodes\": []}", "archive: nodes.json copied");
    check(fs::exists(out / "inputs" / "extra.txt"), "archive: every regular file copied");
    check(!fs::exists(out / "inputs" / "subdir"), "archive: subdirectories not traversed");

    // Overwrite existing archive files.
    writeFile(scn / "nodes.json", "{\"nodes\": [1]}");
    ArchiveScenarioInputs(out.string(), (scn / "run.ini").string());
    check(readFile(out / "inputs" / "nodes.json") == "{\"nodes\": [1]}", "archive: existing files overwritten");
}

// =========================================================================
// cli: ParseCommandLine (ns3::CommandLine stubbed)
// =========================================================================

static ChildResult
parseInChild(const std::string& tag, std::vector<std::string> argvStr)
{
    return runChild(tag, [argvStr](std::ostream& out) mutable {
        argvStr.insert(argvStr.begin(), "sim");
        std::vector<char*> argv;
        for (auto& s : argvStr)
        {
            argv.push_back(&s[0]);
        }
        argv.push_back(nullptr);
        CliArgs a = ParseCommandLine(static_cast<int>(argvStr.size()), argv.data());
        out << "run_config=" << a.run_config_path << "\n"
            << "positions=" << a.positions_override_path << "\n"
            << "output_dir=" << a.output_dir << "\n"
            << "seeds=" << a.seeds_arg << "\n"
            << "seed=" << a.seed_override << "\n"
            << "run_id=" << a.run_id_override << "\n"
            << "debug_links=" << a.debug_links << "\n"
            << "rl_mode=" << a.rl_mode << "\n"
            << "band=" << a.band << "\n"
            << "channel_query=" << a.channel_query << "\n";
    });
}

static void
test_parse_cli_defaults_and_values()
{
    fs::path scn = freshDir("cli-scn");
    std::string ini = (scn / "run.ini").string();
    std::string pos = (scn / "pos.json").string();
    writeFile(ini, "x");
    writeFile(pos, "{}");

    auto r = parseInChild("cli-min", {"--run-config=" + ini});
    check(r.code == 77, "cli: only --run-config parses (code " + std::to_string(r.code) + ")");
    check(r.out == "run_config=" + ini + "\npositions=\noutput_dir=\nseeds=\nseed=-1\nrun_id=-1\n"
                   "debug_links=0\nrl_mode=0\nband=\nchannel_query=0\n",
          "cli: not-set sentinels (empty strings, -1, false)");
    check(r.err.empty(), "cli: no stderr output on valid minimal args");

    r = parseInChild("cli-all", {"--run-config=" + ini, "--positions-override=" + pos, "--output-dir=/o",
                                 "--seeds=1,2", "--seed=3", "--run-id=4", "--debug-links", "--rl-mode",
                                 "--band=sub-6"});
    check(r.code == 77 && r.out == "run_config=" + ini + "\npositions=" + pos +
                                       "\noutput_dir=/o\nseeds=1,2\nseed=3\nrun_id=4\n"
                                       "debug_links=1\nrl_mode=1\nband=sub-6\nchannel_query=0\n",
          "cli: all flags parsed; bare bool flags set true");

    r = parseInChild("cli-mmw", {"--run-config=" + ini, "--band=mmwave"});
    check(r.code == 77 && contains(r.out, "band=mmwave\n"), "cli: --band=mmwave accepted");
}

// ---- main ----

/**
 * @fn main
 * @brief Run all io + cli tests and print a pass/fail summary.
 * @return 0 if all checks passed, 1 if any failed.
 */
int
main()
{
    std::random_device rd;
    g_root = fs::temp_directory_path() /
             ("io-cli-test-" + std::to_string(::getpid()) + "-" + std::to_string(rd()));
    fs::create_directories(g_root);

    // MetricsWriter
    test_metrics_averages_worked_example();
    test_metrics_warmup_boundary();
    test_metrics_connectivity_threshold();
    test_metrics_overwrites();

    // WriteRunLog
    test_runlog_header_and_dirs();
    test_runlog_seed_source();
    test_runlog_cli_overrides();
    test_runlog_resolved_config();
    test_runlog_rl_block();
    test_runlog_jammers();

    // VizWriter
    test_viz_open_headers_and_meta();
    test_viz_tick_rows();
    test_viz_rate_limit();

    // cli
    test_resolve_seeds_precedence();
    test_resolve_seeds_list_parsing();
    test_resolve_query_seed();
    test_archive_copies_regular_files();
    test_parse_cli_defaults_and_values();

    std::error_code ec;
    fs::remove_all(g_root, ec);

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.\n";
    if (g_fail > 0)
    {
        std::cout << "SOME TESTS FAILED.\n";
        return 1;
    }
    std::cout << "All tests passed.\n";
    return 0;
}
