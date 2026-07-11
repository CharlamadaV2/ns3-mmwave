/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/** @file sim
*/

#include "src/cli/cli-parser.h"
#include "src/config/config-loader.h"
#include "src/config/config-validator.h"
#include "src/eval/link-evaluator.h"
#include "src/eval/link-table.h"
#include "src/io/metrics-writer.h"
#include "src/io/progress-logger.h"
#include "src/io/run-logger.h"
#include "src/io/viz-writer.h"
#include "src/rl/rl-bridge.h"
#include "src/routing/mesh-router.h"
#include "src/setup/topology-builder.h"
#include "src/traffic/traffic-matrix.h"

#include "ns3/core-module.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>

namespace fs = std::filesystem;

NS_LOG_COMPONENT_DEFINE("MeshSim");

/// @brief Entry point: loads config, builds topology, runs the per-tick
///        simulation loop for each seed, and writes metrics/run logs.
/// @param argc CLI argument count.
/// @param argv CLI argument values.
/// @return 0 on success, 1 on config/load/output error.
int
main(int argc, char* argv[]) ///< Takes params for cli at start
{
    auto args = mesh_sim::ParseCommandLine(argc, argv); //< Passes params to cli
    ns3::LogComponentEnable("MeshSim", ns3::LOG_LEVEL_INFO);

    // Enable with: NS_LOG="LinkEvaluator=debug:LinkTable=debug" or --debug-links
    if (args.debug_links)
    {
        ns3::LogComponentEnable("LinkEvaluator", ns3::LOG_LEVEL_DEBUG);
        ns3::LogComponentEnable("LinkTable", ns3::LOG_LEVEL_ALL);
    }

    /// @brief Load run.ini + nodes.json/buildings.json into a SimConfig.
    NS_LOG_INFO("Loading config: " << args.run_config_path);
    mesh_sim::SimConfig cfg;
    try
    {
        cfg = mesh_sim::ConfigLoader::Load(args.run_config_path,
                                           args.positions_override_path);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error loading config: " << e.what() << "\n";
        return 1;
    }

    /// @brief Apply CLI overrides on top of the loaded config.
    if (args.run_id_override >= 0)
    {
        cfg.run_id = static_cast<uint32_t>(args.run_id_override);
    }

    if (!args.output_dir.empty())
    {
        cfg.output_dir = args.output_dir;
    }

    if (args.rl_mode)
    {
        cfg.rl.enabled = true;
    }

    /// @brief Validate the fully-resolved config before running.
    auto vr = mesh_sim::ValidateConfig(cfg);
    if (!vr.ok())
    {
        for (const auto& e : vr.errors)
        {
            std::cerr << "Config error: " << e << "\n";
        }
        return 1;
    }

    auto seeds = mesh_sim::ResolveSeeds(args, cfg); //< From cli-parser, manages seeds
    NS_LOG_INFO("Scenario '" << cfg.scenario_name << "'"
                             << " seeds=" << seeds.size()
                             << " duration=" << cfg.duration_s << "s"
                             << " tick=" << cfg.tick_s << "s"
                             << " nodes=" << cfg.nodes.size()
                             << " buildings=" << cfg.buildings.size());

    const std::string baseOutputDir = cfg.output_dir;

    /// @brief Snapshot the run.ini/nodes.json inputs into the output dir
    ///        for reproducibility, then write the run.log summary.
    try
    {
        mesh_sim::ArchiveScenarioInputs(baseOutputDir, args.run_config_path);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error archiving inputs: " << e.what() << "\n";
        return 1;
    }

    mesh_sim::WriteRunLog(baseOutputDir, args, cfg, seeds);

    /// @brief Per-seed simulation loop — each seed gets its own RNG stream
    ///        and output subdirectory (seed-<N>/).
    for (size_t si = 0; si < seeds.size(); ++si)
    {
        uint32_t seed = seeds[si];
        cfg.seed = seed;
        cfg.output_dir = baseOutputDir + "/seed-" + std::to_string(seed);

        NS_LOG_INFO("=== Seed " << seed << " (" << (si + 1) << "/" << seeds.size() << ") ===");

        try
        {
            fs::create_directories(cfg.output_dir);
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error creating output directory '" << cfg.output_dir
                      << "': " << e.what() << "\n";
            return 1;
        }

        ns3::RngSeedManager::SetSeed(seed);
        ns3::RngSeedManager::SetRun(cfg.run_id);

        auto wallStart = std::chrono::system_clock::now(); //<Start time

        /// @brief Build ns-3 nodes, mobility models, buildings, and the
        ///        propagation/condition models for this seed's topology.
        mesh_sim::TopologyBuilder topo(cfg);
        topo.Build();
        auto mobs = topo.GetMobilityModels();
        uint32_t N = static_cast<uint32_t>(mobs.size());

        /// @brief Bind the propagation/condition models to a LinkEvaluator
        ///        that will compute per-tick SINR/capacity/MCS.
        mesh_sim::LinkEvaluator linkEval;
        linkEval.Configure(cfg, topo.GetPropagationModel(), topo.GetConditionModel(), args.band);

        /// @brief Per-tick components: link table, traffic generator, router.
        mesh_sim::LinkTable      linkTable;
        mesh_sim::TrafficMatrix  trafficMatrix(cfg);
        mesh_sim::MeshRouter     router(cfg.mesh.routing);

        trafficMatrix.Initialize(N, 0.0);

        /// @brief Output writers: live viz stream and accumulated metrics.
        mesh_sim::VizWriter vizWriter(cfg);
        vizWriter.Open();

        mesh_sim::MetricsWriter metricsWriter(cfg);

        /// @brief Optional RL bridge — picks the controlled node and steps
        ///        the obs/action loop alongside the main tick loop.
        std::unique_ptr<mesh_sim::RlBridge> rlBridge;
        uint32_t rlControlledIdx = N - 1; // default: last node
        if (cfg.rl.enabled)
        {
            if (!cfg.rl.controlled_node_id.empty())
            {
                for (uint32_t i = 0; i < cfg.nodes.size(); ++i)
                {
                    if (cfg.nodes[i].id == cfg.rl.controlled_node_id)
                    {
                        rlControlledIdx = i;
                        break;
                    }
                }
            }
            rlBridge = std::make_unique<mesh_sim::RlBridge>(cfg, rlControlledIdx);
        }

        /// @brief Main per-tick loop: advance sim time, evaluate links,
        ///        route flows, log progress, write metrics/viz.
        uint32_t numTicks = static_cast<uint32_t>(cfg.duration_s / cfg.tick_s);

        uint32_t progressInterval = std::max(1u, numTicks / 20);
        mesh_sim::ProgressLogger progress{numTicks, progressInterval, seed,
                                           cfg.duration_s, cfg.tick_s,
                                           std::chrono::steady_clock::now()};
        for (uint32_t ti = 0; ti <= numTicks; ++ti)
        {
            double t = ti * cfg.tick_s;

            // Advance the ns-3 simulator clock so that Simulator::Now() == t.
            // Required for ConstantVelocityMobilityModel, RandomWalk2dMobilityModel,
            // and ThreeGpp/NYU channel-condition cache expiry.
            if (ti > 0)
            {
                ns3::Simulator::Stop(ns3::Seconds(cfg.tick_s));
                ns3::Simulator::Run();
            }

            /// @brief Evaluate all N*(N-1)/2 links at the current positions.
            linkTable.Update(N, linkEval.EvaluateAll(mobs));

            /// @brief Advance traffic state and route active flows over
            ///        the freshly-evaluated link table.
            trafficMatrix.Tick(t);

            auto flowResults = router.Route(linkTable,
                                            trafficMatrix.GetActiveFlows(),
                                            N);

            /// @brief Aggregate per-tick summary stats for the progress log.
            uint32_t routableCount = 0;
            double totalDemand     = 0.0;
            double totalDelivered  = 0.0;
            for (const auto& fr : flowResults)
            {
                if (fr.routable)
                    ++routableCount;
                totalDemand    += fr.demand_mbps;
                totalDelivered += fr.delivered_mbps;
            }

            NS_LOG_INFO("  t=" << std::fixed << std::setprecision(3) << t
                        << "s  links=" << linkTable.ConnectedLinkCount()
                        << "  flows=" << flowResults.size()
                        << "  routable=" << routableCount
                        << "  demand=" << std::setprecision(1) << totalDemand
                        << "  delivered=" << totalDelivered << " Mbps");

            vizWriter.WriteTick(t, mobs, linkTable, flowResults);
            metricsWriter.AccumulateTick(t, linkTable, flowResults, N);
            progress.Tick(ti);

            /// @brief RL interaction: send obs+reward, receive action,
            ///        apply velocity to the controlled node.
            if (rlBridge)
            {
                bool done = (ti == numTicks);
                rlBridge->Step(ti, t, mobs, linkTable, flowResults, done);
                if (!done)
                {
                    rlBridge->ApplyAction(mobs[rlControlledIdx]);
                }
            }
        }

        vizWriter.Close();

        /// @brief Record wall-clock timing and flush accumulated metrics
        ///        for this seed before moving to the next.
        auto wallEnd = std::chrono::system_clock::now();
        double wallElapsed = std::chrono::duration<double>(wallEnd - wallStart).count();

        mesh_sim::TimingInfo timing{wallStart, wallEnd, wallElapsed};
        metricsWriter.SetTiming(timing);
        metricsWriter.Write();

        NS_LOG_INFO("Seed " << seed << " complete (wall=" << std::fixed
                            << std::setprecision(3) << wallElapsed << "s).");

        ns3::Simulator::Destroy();
    }

    if (seeds.size() > 1)
    {
        NS_LOG_INFO("All " << seeds.size() << " seeds complete. Output: " << baseOutputDir);
    }

    return 0;
}