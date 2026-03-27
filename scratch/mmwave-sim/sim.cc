/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

#include "src/config/config-loader.h"
#include "src/io/metrics-writer.h"
#include "src/io/progress-logger.h"
#include "src/io/viz-writer.h"
#include "src/cli/cli-parser.h"
#include "src/setup/ns3-defaults.h"
#include "src/setup/topology-builder.h"
#include "src/setup/traffic-setup.h"

#include "ns3/core-module.h"
#include "ns3/mmwave-mac-trace.h"
#include "ns3/mmwave-phy-trace.h"
#include "ns3/simulator.h"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>

using namespace ns3;
using namespace ns3::mmwave;
namespace fs = std::filesystem;

NS_LOG_COMPONENT_DEFINE("MmWaveSim");

int
main(int argc, char* argv[])
{
    auto args = mmwave_sim::ParseCommandLine(argc, argv);
    LogComponentEnable("MmWaveSim", LOG_LEVEL_INFO);

    // Load configuration + apply CLI overrides
    NS_LOG_INFO("Loading config: " << args.run_config_path);
    mmwave_sim::SimConfig cfg;
    try
    {
        cfg = mmwave_sim::ConfigLoader::Load(args.run_config_path, args.positions_override_path);
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

    auto seeds = mmwave_sim::ResolveSeeds(args, cfg);
    NS_LOG_INFO("Scenario '" << cfg.scenario_name << "' seeds=" << seeds.size()
                             << " duration=" << cfg.duration_s << "s"
                             << " nodes=" << cfg.nodes.size()
                             << " buildings=" << cfg.buildings.size());

    const std::string baseOutputDir = cfg.output_dir;

    try
    {
        mmwave_sim::ArchiveScenarioInputs(baseOutputDir, args.run_config_path);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error archiving inputs: " << e.what() << "\n";
        return 1;
    }

    // One-time ns3 defaults (must precede per-seed loop)
    mmwave_sim::ApplyProtocolDefaults();
    mmwave_sim::ApplyTuningDefaults(cfg);

    // Per-seed simulation loop
    double progressInterval = std::max(1.0, cfg.duration_s / 10.0);

    for (size_t si = 0; si < seeds.size(); ++si)
    {
        uint32_t seed = seeds[si];
        cfg.seed = seed;
        cfg.output_dir = baseOutputDir + "/seed-" + std::to_string(seed);

        NS_LOG_INFO("=== Seed " << seed << " (" << (si + 1) << "/" << seeds.size() << ") ===");

        try
        {
            fs::create_directories(cfg.output_dir);
            if (cfg.pcap_enabled)
            {
                fs::create_directories(cfg.output_dir + "/pcap");
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error creating output directory '" << cfg.output_dir << "': " << e.what()
                      << "\n";
            return 1;
        }

        RngSeedManager::SetSeed(seed);
        RngSeedManager::SetRun(cfg.run_id);

        mmwave_sim::ApplyTraceFileDefaults(cfg.output_dir);
        mmwave_sim::TopologyBuilder::ConfigureChannelDefaults(cfg);
        auto [mmwH, epcH] = mmwave_sim::TopologyBuilder::CreateHelpers(cfg);

        mmwave_sim::TopologyBuilder topology(cfg, mmwH, epcH);
        topology.Build();

        mmwave_sim::TrafficSetup traffic(cfg, topology);
        traffic.Install();

        mmwave_sim::VizWriter vizWriter(cfg,
                                        topology.GetEnbNodes(),
                                        topology.GetUeNodes(),
                                        topology.GetChannelConditionModel());
        vizWriter.Start();

        mmwave_sim::ProgressLogger logger{cfg.duration_s,
                                          progressInterval,
                                          seed,
                                          std::chrono::steady_clock::now()};
        Simulator::Schedule(Seconds(progressInterval), &mmwave_sim::ProgressLogger::Tick, &logger);

        if (cfg.trace_level == "full")
        {
            mmwH->EnableTraces();
        }
        else if (cfg.trace_level == "minimal")
        {
            mmwH->EnableDlPhyTrace();
            mmwH->EnableUlPhyTrace();
            mmwH->EnableRlcTraces();
        }

        auto wallStart = std::chrono::system_clock::now();
        Simulator::Stop(Seconds(cfg.duration_s));
        Simulator::Run();
        auto wallEnd = std::chrono::system_clock::now();
        double wallElapsed = std::chrono::duration<double>(wallEnd - wallStart).count();

        NS_LOG_INFO("Simulation complete (seed=" << seed << ", wall=" << std::fixed
                                                 << std::setprecision(1) << wallElapsed << "s).");

        // Post-run: flush viz, write metrics, reset
        vizWriter.Flush();

        mmwave_sim::TimingInfo timing{wallStart, wallEnd, wallElapsed};
        mmwave_sim::MetricsWriter writer(cfg);
        writer.SetTiming(timing);
        writer.Write();

        Simulator::Destroy();
        MmWavePhyTrace::ResetTraceFiles();
        MmWaveMacTrace::ResetTraceFiles();
    }

    if (seeds.size() > 1)
    {
        NS_LOG_INFO("All " << seeds.size() << " seeds complete. "
                           << "Output: " << baseOutputDir);
    }

    return 0;
}
