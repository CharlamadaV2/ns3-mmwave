/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file channel-query.cc
 * @brief Implements @ref mesh_sim::RunChannelQuery; contract in @ref src_query.
 */

#include "src/query/channel-query.h"

#include "src/config/layout-override.h"
#include "src/config/rl-control.h"
#include "src/domain/probe-spec.h"
#include "src/eval/link-evaluator.h"
#include "src/eval/sinr-capacity.h"
#include "src/setup/topology-builder.h"

#include "third_party/json.hpp"

#include "ns3/rng-seed-manager.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace mesh_sim
{

namespace
{

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

constexpr const char* kContract = "mesh_channel_query_v1";
constexpr std::size_t kMaxRequestLineBytes = 16u * 1024u * 1024u;
constexpr std::size_t kMaxChildResponseBytes = 16u * 1024u * 1024u;
constexpr std::size_t kMaxLayouts = 1024;
constexpr std::size_t kMaxProbes = 10000;
constexpr double kChildDeadlineS = 60.0;
constexpr double kTerminateGraceS = 1.0;
// Child-side backstop so a child outlives neither its deadline nor a killed parent for long.
constexpr unsigned kChildAlarmS = 65;

volatile std::sig_atomic_t g_activeChild = 0;

void
ForwardTermination(int sig)
{
    const pid_t child = static_cast<pid_t>(g_activeChild);
    if (child > 0)
    {
        kill(child, SIGKILL);
    }
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

void
InstallTerminationForwarding()
{
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = ForwardTermination;
    sigemptyset(&sa.sa_mask);
    for (int sig : {SIGTERM, SIGINT, SIGHUP})
    {
        sigaction(sig, &sa, nullptr);
    }
    // An inherited SIG_IGN would auto-reap children and break waitpid.
    std::signal(SIGCHLD, SIG_DFL);
}

double
SecondsSince(Clock::time_point t0)
{
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

void
Emit(std::ostream& out, const ojson& msg)
{
    out << msg.dump(-1, ' ', false, ojson::error_handler_t::replace) << '\n';
    out.flush();
}

void
EmitError(std::ostream& out, const ojson& requestId, const std::string& message)
{
    ojson msg;
    msg["type"] = "error";
    msg["request_id"] = requestId;
    msg["message"] = message;
    Emit(out, msg);
}

ojson
LayoutError(const std::string& message)
{
    ojson e;
    e["error"] = message;
    return e;
}

enum class LineStatus
{
    kLine,
    kTooLong,
    kEof
};

// Never buffers more than maxBytes; the rest of an oversized line is consumed and dropped.
LineStatus
ReadBoundedLine(std::istream& in, std::string& line, std::size_t maxBytes)
{
    line.clear();
    std::streambuf* sb = in.rdbuf();
    bool any = false;
    bool tooLong = false;
    for (;;)
    {
        const int c = sb->sbumpc();
        if (c == std::char_traits<char>::eof())
        {
            in.setstate(std::ios::eofbit);
            if (!any)
            {
                return LineStatus::kEof;
            }
            return tooLong ? LineStatus::kTooLong : LineStatus::kLine;
        }
        any = true;
        if (c == '\n')
        {
            return tooLong ? LineStatus::kTooLong : LineStatus::kLine;
        }
        if (tooLong)
        {
            continue;
        }
        if (line.size() >= maxBytes)
        {
            tooLong = true;
            std::string().swap(line);
            continue;
        }
        line.push_back(static_cast<char>(c));
    }
}

bool
IsBlank(const std::string& s)
{
    return s.find_first_not_of(" \t\r") == std::string::npos;
}

bool
FiniteNumber(const json& v, double& out)
{
    if (!v.is_number())
    {
        return false;
    }
    out = v.get<double>();
    return std::isfinite(out);
}

struct EvaluateRequest
{
    std::vector<std::vector<Position>> layouts;
    bool hasProbes = false;
    ProbeGrid probes;
};

bool
ParseEvaluate(const json& req, std::size_t numNodes, EvaluateRequest& out, std::string& err)
{
    if (!req.contains("layouts") || !req["layouts"].is_array() || req["layouts"].empty())
    {
        err = "'layouts' must be a non-empty array";
        return false;
    }
    const json& layouts = req["layouts"];
    if (layouts.size() > kMaxLayouts)
    {
        err = "'layouts' has " + std::to_string(layouts.size()) + " entries; limit is " +
              std::to_string(kMaxLayouts);
        return false;
    }
    out.layouts.reserve(layouts.size());
    for (std::size_t l = 0; l < layouts.size(); ++l)
    {
        const json& layout = layouts[l];
        const std::string where = "layouts[" + std::to_string(l) + "]";
        if (!layout.is_array() || layout.size() != numNodes)
        {
            err = where + " must be an array of " + std::to_string(numNodes) + " [x, y, z] points";
            return false;
        }
        std::vector<Position> positions(numNodes);
        for (std::size_t i = 0; i < numNodes; ++i)
        {
            const json& p = layout[i];
            if (!p.is_array() || p.size() != 3 || !FiniteNumber(p[0], positions[i].x) ||
                !FiniteNumber(p[1], positions[i].y) || !FiniteNumber(p[2], positions[i].z))
            {
                err = where + "[" + std::to_string(i) + "] must be [x, y, z] finite numbers";
                return false;
            }
        }
        out.layouts.push_back(std::move(positions));
    }

    if (!req.contains("probes") || req["probes"].is_null())
    {
        return true;
    }
    const json& probes = req["probes"];
    if (!probes.is_object())
    {
        err = "'probes' must be an object";
        return false;
    }
    ProbeGrid& grid = out.probes;
    for (const auto& [key, field] :
         {std::pair<const char*, double*>{"height_m", &grid.height_m},
          std::pair<const char*, double*>{"rx_gain_dbi", &grid.rx_gain_dbi},
          std::pair<const char*, double*>{"sinr_db", &grid.sinr_db}})
    {
        if (!probes.contains(key) || !FiniteNumber(probes[key], *field))
        {
            err = std::string("'probes.") + key + "' must be a finite number";
            return false;
        }
    }
    if (grid.height_m < 0.0)
    {
        err = "'probes.height_m' must be >= 0";
        return false;
    }
    if (!probes.contains("points") || !probes["points"].is_array())
    {
        err = "'probes.points' must be an array of [x, y] points";
        return false;
    }
    const json& points = probes["points"];
    if (points.size() > kMaxProbes)
    {
        err = "'probes.points' has " + std::to_string(points.size()) + " entries; limit is " +
              std::to_string(kMaxProbes);
        return false;
    }
    grid.points.resize(points.size());
    for (std::size_t k = 0; k < points.size(); ++k)
    {
        const json& p = points[k];
        Position& pos = grid.points[k];
        if (!p.is_array() || p.size() != 2 || !FiniteNumber(p[0], pos.x) ||
            !FiniteNumber(p[1], pos.y))
        {
            err = "probes.points[" + std::to_string(k) + "] must be [x, y] finite numbers";
            return false;
        }
        pos.z = grid.height_m;
    }
    out.hasProbes = true;
    return true;
}

ojson
BuildInit(const SimConfig& cfg, uint32_t planningSeed)
{
    ojson ids = ojson::array();
    ojson types = ojson::array();
    ojson mobility = ojson::array();
    ojson starts = ojson::array();
    for (const auto& spec : cfg.nodes)
    {
        ids.push_back(spec.id);
        types.push_back(spec.node_type);
        mobility.push_back(spec.mobility);
        const Position p = ControlledStartPosition(spec);
        starts.push_back(ojson::array({p.x, p.y, p.z}));
    }
    std::size_t jammersEnabled = 0;
    for (const auto& j : cfg.jammers)
    {
        if (j.enabled)
        {
            ++jammersEnabled;
        }
    }

    const auto& ch = cfg.channel;
    ojson channel;
    channel["frequency_ghz"] = ch.frequency_ghz;
    channel["tx_power_dbm"] = ch.tx_power_dbm;
    channel["bandwidth_mhz"] = ch.bandwidth_mhz;
    channel["noise_figure_db"] = ch.noise_figure_db;
    channel["amc_model"] = ch.amc_model;
    channel["channel_model"] = ch.channel_model;
    channel["scenario"] = ch.scenario;
    channel["condition_model"] = ch.condition_model;
    channel["tx_array_gain_dbi"] = ch.tx_array_gain_dbi;
    channel["rx_array_gain_dbi"] = ch.rx_array_gain_dbi;

    ojson limits;
    limits["max_layouts"] = kMaxLayouts;
    limits["max_probes"] = kMaxProbes;
    limits["max_request_line_bytes"] = kMaxRequestLineBytes;
    limits["max_child_response_bytes"] = kMaxChildResponseBytes;
    limits["child_deadline_s"] = kChildDeadlineS;

    ojson init;
    init["type"] = "init";
    init["contract"] = kContract;
    init["isolation"] = "fork_per_layout";
    init["node_ids"] = ids;
    init["node_types"] = types;
    init["mobility"] = mobility;
    init["start_positions"] = starts;
    init["controlled_indices"] = cfg.rl.controlled_indices;
    init["rl_enabled"] = cfg.rl.enabled;
    init["band"] = cfg.band;
    init["band_source"] = cfg.band_source;
    init["seed"] = planningSeed;
    init["run_id"] = cfg.run_id;
    init["jammer_seed"] = planningSeed;
    init["sinr_threshold_db"] = SINR_MIN_DB;
    init["jammer_path_enabled"] = (cfg.band == "sub-6" && jammersEnabled > 0);
    init["num_buildings"] = cfg.buildings.size();
    init["channel"] = channel;
    init["limits"] = limits;
    init["time_s"] = 0.0;
    return init;
}

// Runs only in a forked child; mirrors sim.cc's per-seed setup and tick-0 evaluation.
ojson
EvaluateInChild(SimConfig cfg, uint32_t planningSeed, const ProbeGrid* probes)
{
    // As in sim.cc, the resolved run seed also reaches JammerModel through cfg.seed.
    cfg.seed = planningSeed;
    ns3::RngSeedManager::SetSeed(planningSeed);
    ns3::RngSeedManager::SetRun(cfg.run_id);

    TopologyBuilder topo(cfg);
    if (probes != nullptr)
    {
        topo.SetProbes(*probes);
    }
    topo.Build();
    const auto mobs = topo.GetMobilityModels();
    const auto jammerMobs = topo.GetJammerMobilityModels();

    LinkEvaluator linkEval;
    linkEval.Configure(cfg, topo.GetPropagationModel(), topo.GetConditionModel(), cfg.band, jammerMobs);

    // Draw order is part of the realization: all mesh pairs first, exactly as tick 0 of a run.
    const auto links = linkEval.EvaluateAll(mobs, 0.0);
    ojson rows = ojson::array();
    for (const auto& r : links)
    {
        if (!std::isfinite(r.sinr_db) || !std::isfinite(r.capacity_mbps))
        {
            return LayoutError("non-finite channel result for link " + std::to_string(r.tx_id) +
                               "-" + std::to_string(r.rx_id));
        }
        rows.push_back(ojson::array(
            {r.tx_id, r.rx_id, r.sinr_db, r.capacity_mbps, r.is_los, r.sinr_db >= SINR_MIN_DB}));
    }

    ojson coverage = nullptr;
    if (probes != nullptr)
    {
        const auto probeMobs = topo.GetProbeMobilityModels();
        coverage = ojson::array();
        for (uint32_t i = 0; i < mobs.size(); ++i)
        {
            ojson covered = ojson::array();
            for (std::size_t k = 0; k < probeMobs.size(); ++k)
            {
                const LinkResult r =
                    linkEval.EvaluateProbe(mobs[i], probeMobs[k], i, probes->rx_gain_dbi, 0.0);
                if (!std::isfinite(r.sinr_db))
                {
                    return LayoutError("non-finite channel result for node " + std::to_string(i) +
                                       " probe " + std::to_string(k));
                }
                if (r.sinr_db >= probes->sinr_db)
                {
                    covered.push_back(k);
                }
            }
            coverage.push_back(std::move(covered));
        }
    }

    ojson result;
    result["links"] = std::move(rows);
    result["coverage"] = std::move(coverage);
    return result;
}

bool
WriteAll(int fd, const std::string& data)
{
    std::size_t off = 0;
    while (off < data.size())
    {
        const ssize_t n = write(fd, data.data() + off, data.size() - off);
        if (n < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            return false;
        }
        off += static_cast<std::size_t>(n);
    }
    return true;
}

[[noreturn]] void
ChildMain(const SimConfig& cfg, uint32_t planningSeed, const ProbeGrid* probes, int fd)
{
    for (int sig : {SIGTERM, SIGINT, SIGHUP, SIGPIPE})
    {
        std::signal(sig, SIG_DFL);
    }
    alarm(kChildAlarmS);
    const int devNull = open("/dev/null", O_RDONLY);
    if (devNull >= 0)
    {
        dup2(devNull, STDIN_FILENO);
        if (devNull != STDIN_FILENO)
        {
            close(devNull);
        }
    }
    // Stray stdout writes must never reach the parent's NDJSON stream.
    dup2(STDERR_FILENO, STDOUT_FILENO);

    std::string body;
    try
    {
        body = EvaluateInChild(cfg, planningSeed, probes)
                   .dump(-1, ' ', false, ojson::error_handler_t::replace);
    }
    catch (const std::exception& e)
    {
        body = LayoutError(std::string("child evaluation failed: ") + e.what())
                   .dump(-1, ' ', false, ojson::error_handler_t::replace);
    }
    const bool written = WriteAll(fd, body);
    close(fd);
    std::clog.flush();
    std::cout.flush();
    std::fflush(nullptr);
    _exit(written ? 0 : 3);
}

bool
ReapUntil(pid_t pid, Clock::time_point deadline, int& status)
{
    for (;;)
    {
        const pid_t r = waitpid(pid, &status, WNOHANG);
        if (r == pid)
        {
            return true;
        }
        if (r < 0 && errno != EINTR)
        {
            return false;
        }
        if (Clock::now() >= deadline)
        {
            return false;
        }
        usleep(1000);
    }
}

bool
TerminateAndReap(pid_t pid, int& status)
{
    kill(pid, SIGTERM);
    const auto grace = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                          std::chrono::duration<double>(kTerminateGraceS));
    if (ReapUntil(pid, grace, status))
    {
        return true;
    }
    kill(pid, SIGKILL);
    for (;;)
    {
        const pid_t r = waitpid(pid, &status, 0);
        if (r == pid)
        {
            return true;
        }
        if (r < 0 && errno != EINTR)
        {
            return false;
        }
    }
}

// Forks one child, drains its pipe while it runs, and always reaps it.
bool
RunChild(const SimConfig& layoutCfg,
         uint32_t planningSeed,
         const ProbeGrid* probes,
         std::ostream& out,
         std::string& body,
         std::string& failure)
{
    out.flush();
    std::cout.flush();
    std::clog.flush();
    std::cerr.flush();
    std::fflush(nullptr);

    int fds[2];
    if (pipe(fds) != 0)
    {
        failure = std::string("pipe failed: ") + std::strerror(errno);
        return false;
    }
    const pid_t pid = fork();
    if (pid < 0)
    {
        failure = std::string("fork failed: ") + std::strerror(errno);
        close(fds[0]);
        close(fds[1]);
        return false;
    }
    if (pid == 0)
    {
        close(fds[0]);
        ChildMain(layoutCfg, planningSeed, probes, fds[1]);
    }
    close(fds[1]);
    g_activeChild = static_cast<std::sig_atomic_t>(pid);

    const auto deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                             std::chrono::duration<double>(kChildDeadlineS));
    bool timedOut = false;
    bool oversize = false;
    std::string readError;
    char buf[65536];
    for (;;)
    {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   deadline - Clock::now()).count();
        if (remaining <= 0)
        {
            timedOut = true;
            break;
        }
        pollfd pfd{fds[0], POLLIN, 0};
        const int rc = poll(&pfd, 1, static_cast<int>(std::min<long long>(remaining, 1000)));
        if (rc < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            readError = std::strerror(errno);
            break;
        }
        if (rc == 0)
        {
            continue;
        }
        const ssize_t n = read(fds[0], buf, sizeof(buf));
        if (n < 0)
        {
            if (errno == EINTR || errno == EAGAIN)
            {
                continue;
            }
            readError = std::strerror(errno);
            break;
        }
        if (n == 0)
        {
            break;
        }
        if (body.size() + static_cast<std::size_t>(n) > kMaxChildResponseBytes)
        {
            oversize = true;
            break;
        }
        body.append(buf, static_cast<std::size_t>(n));
    }
    close(fds[0]);

    int status = 0;
    const bool aborted = timedOut || oversize || !readError.empty();
    bool reaped = !aborted && ReapUntil(pid, deadline, status);
    if (!reaped)
    {
        timedOut = timedOut || (!oversize && readError.empty());
        reaped = TerminateAndReap(pid, status);
    }
    g_activeChild = 0;

    if (timedOut)
    {
        failure = "child exceeded the " + std::to_string(static_cast<int>(kChildDeadlineS)) +
                  " s deadline";
    }
    else if (oversize)
    {
        failure = "child response exceeds " + std::to_string(kMaxChildResponseBytes) + " bytes";
    }
    else if (!readError.empty())
    {
        failure = "reading child output failed: " + readError;
    }
    else if (!reaped)
    {
        failure = "could not reap child process";
    }
    else if (WIFSIGNALED(status))
    {
        failure = "child terminated by signal " + std::to_string(WTERMSIG(status));
    }
    else if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    {
        failure = "child exited with status " +
                  std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    }
    return failure.empty();
}

ojson
EvaluateLayout(const SimConfig& cfg,
               uint32_t planningSeed,
               const std::vector<Position>& layout,
               const ProbeGrid* probes,
               std::ostream& out)
{
    const auto t0 = Clock::now();
    SimConfig layoutCfg = cfg;
    const auto errors = ApplyLayout(layoutCfg, layout);
    if (!errors.empty())
    {
        std::string msg = "invalid layout:";
        for (const auto& e : errors)
        {
            msg += " " + e + ";";
        }
        msg.pop_back();
        return LayoutError(msg);
    }

    std::string body;
    std::string failure;
    if (!RunChild(layoutCfg, planningSeed, probes, out, body, failure))
    {
        return LayoutError(failure);
    }
    ojson result = ojson::parse(body, nullptr, false);
    if (result.is_discarded() || !result.is_object())
    {
        return LayoutError("unparsable child output");
    }
    if (result.contains("error"))
    {
        return LayoutError(result["error"].is_string() ? result["error"].get<std::string>()
                                                       : "child reported an error");
    }
    if (!result.contains("links") || !result["links"].is_array() || !result.contains("coverage"))
    {
        return LayoutError("unparsable child output");
    }
    result["wall_s"] = SecondsSince(t0);
    return result;
}

}  // namespace

int
RunChannelQuery(const SimConfig& cfg, const CliArgs& args, std::istream& in, std::ostream& out)
{
    const uint32_t planningSeed = ResolveQuerySeed(args, cfg);

    ojson init;
    try
    {
        init = BuildInit(cfg, planningSeed);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: channel query init failed: " << e.what() << "\n";
        return 1;
    }
    Emit(out, init);
    InstallTerminationForwarding();

    std::string line;
    for (;;)
    {
        const LineStatus status = ReadBoundedLine(in, line, kMaxRequestLineBytes);
        if (status == LineStatus::kEof)
        {
            return 0;
        }
        if (status == LineStatus::kTooLong)
        {
            EmitError(out, nullptr,
                      "request line exceeds " + std::to_string(kMaxRequestLineBytes) + " bytes");
            continue;
        }
        if (IsBlank(line))
        {
            continue;
        }

        const json req = json::parse(line, nullptr, false);
        if (req.is_discarded())
        {
            EmitError(out, nullptr, "malformed JSON request");
            continue;
        }
        if (!req.is_object())
        {
            EmitError(out, nullptr, "request must be a JSON object");
            continue;
        }
        ojson requestId = nullptr;
        if (req.contains("request_id") && !req["request_id"].is_null())
        {
            const json& rid = req["request_id"];
            if (!rid.is_number_integer())
            {
                EmitError(out, nullptr, "'request_id' must be an integer");
                continue;
            }
            requestId = rid.is_number_unsigned() ? ojson(rid.get<std::uint64_t>())
                                                 : ojson(rid.get<std::int64_t>());
        }
        if (!req.contains("type") || !req["type"].is_string())
        {
            EmitError(out, requestId, "request needs a string 'type'");
            continue;
        }
        const std::string type = req["type"].get<std::string>();
        if (type == "shutdown")
        {
            return 0;
        }
        if (type != "evaluate")
        {
            EmitError(out, requestId, "unknown request type '" + type + "'");
            continue;
        }

        EvaluateRequest evalReq;
        std::string err;
        if (!ParseEvaluate(req, cfg.nodes.size(), evalReq, err))
        {
            EmitError(out, requestId, err);
            continue;
        }

        const auto t0 = Clock::now();
        const ProbeGrid* probes = evalReq.hasProbes ? &evalReq.probes : nullptr;
        ojson layouts = ojson::array();
        for (const auto& layout : evalReq.layouts)
        {
            layouts.push_back(EvaluateLayout(cfg, planningSeed, layout, probes, out));
        }
        ojson response;
        response["type"] = "result";
        response["request_id"] = requestId;
        response["layouts"] = std::move(layouts);
        response["wall_s"] = SecondsSince(t0);
        Emit(out, response);
        if (!out)
        {
            return 1;
        }
    }
}

}  // namespace mesh_sim
