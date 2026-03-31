/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

#include "src/cli/cli-parser.h"
#include "src/config/config-loader.h"
#include "src/config/config-validator.h"
#include "src/eval/link-evaluator.h"
#include "src/eval/link-table.h"
#include "src/io/metrics-writer.h"
#include "src/io/progress-logger.h"
#include "src/io/run-logger.h"
#include "src/io/viz-writer.h"
#include "src/routing/mesh-router.h"
#include "src/setup/topology-builder.h"
#include "src/traffic/traffic-matrix.h"

#include "ns3/core-module.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>

namespace fs = std::filesystem;

NS_LOG_COMPONENT_DEFINE("MeshSim");

int
main(int argc, char* argv[])
{
    auto args = mesh_sim::ParseCommandLine(argc, argv);
    ns3::LogComponentEnable("MeshSim", ns3::LOG_LEVEL_INFO);

    // Enable with: NS_LOG="LinkEvaluator=debug:LinkTable=debug" or --debug-links
    if (args.debug_links)
    {
        ns3::LogComponentEnable("LinkEvaluator", ns3::LOG_LEVEL_DEBUG);
        ns3::LogComponentEnable("LinkTable", ns3::LOG_LEVEL_ALL);
    }

    // Load configuration + apply CLI overrides
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

    if (args.run_id_override >= 0)
    {
        cfg.run_id = static_cast<uint32_t>(args.run_id_override);
    }

    auto vr = mesh_sim::ValidateConfig(cfg);
    if (!vr.ok())
    {
        for (const auto& e : vr.errors)
        {
            std::cerr << "Config error: " << e << "\n";
        }
        return 1;
    }

    auto seeds = mesh_sim::ResolveSeeds(args, cfg);
    NS_LOG_INFO("Scenario '" << cfg.scenario_name << "'"
                             << " seeds=" << seeds.size()
                             << " duration=" << cfg.duration_s << "s"
                             << " tick=" << cfg.tick_s << "s"
                             << " nodes=" << cfg.nodes.size()
                             << " buildings=" << cfg.buildings.size());

    const std::string baseOutputDir = cfg.output_dir;

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

    // Per-seed simulation loop
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

        auto wallStart = std::chrono::system_clock::now();

        // Build topology (ns-3 nodes, mobility, propagation models)
        mesh_sim::TopologyBuilder topo(cfg);
        topo.Build();
        auto mobs = topo.GetMobilityModels();
        uint32_t N = static_cast<uint32_t>(mobs.size());

        // Configure link evaluator
        mesh_sim::LinkEvaluator linkEval;
        linkEval.Configure(cfg, topo.GetPropagationModel(), topo.GetConditionModel());

        // Create per-tick components
        mesh_sim::LinkTable      linkTable;
        mesh_sim::TrafficMatrix  trafficMatrix(cfg);
        mesh_sim::MeshRouter     router(cfg.mesh.routing);

        trafficMatrix.Initialize(N, 0.0);

        // Output components
        mesh_sim::VizWriter vizWriter(cfg);
        vizWriter.Open();

        mesh_sim::MetricsWriter metricsWriter(cfg);

        // Step loop
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

            // Evaluate all links
            linkTable.Update(N, linkEval.EvaluateAll(mobs));

            // Advance traffic state
            trafficMatrix.Tick(t);

            // Route flows over the mesh
            auto flowResults = router.Route(linkTable,
                                            trafficMatrix.GetActiveFlows(),
                                            N);

            // Per-tick summary
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
        }

        vizWriter.Close();

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
