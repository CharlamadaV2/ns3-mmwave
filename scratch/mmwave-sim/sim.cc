/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */

#include "ns3/core-module.h"
#include "ns3/simulator.h"
#include "ns3/string.h"
#include "ns3/boolean.h"

#include "src/config/config-loader.h"
#include "src/sim/topology-builder.h"
#include "src/sim/traffic-setup.h"
#include "src/io/metrics-writer.h"
#include "src/io/viz-writer.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace ns3;
using namespace ns3::mmwave;
namespace fs = std::filesystem;

NS_LOG_COMPONENT_DEFINE("MmWaveSim");

int
main(int argc, char* argv[])
{
    std::string runConfigPath;
    std::string positionsOverridePath;

    CommandLine cmd;
    cmd.AddValue("run-config",
                 "Path to run.ini scenario configuration file",
                 runConfigPath);
    cmd.AddValue("positions-override",
                 "Optional JSON file overriding node positions (RL extension point). "
                 "NOTE: positions are applied before the simulation starts only — "
                 "real-time trajectory control is not yet supported. "
                 "TODO (RL): implement waypoint mobility or step-based co-simulation "
                 "for mid-run control. See docs/rl-extension.md.",
                 positionsOverridePath);
    cmd.Parse(argc, argv);

    if (runConfigPath.empty())
    {
        std::cerr << "Error: --run-config=<path> is required.\n";
        return 1;
    }

    LogComponentEnable("MmWaveSim", LOG_LEVEL_INFO);

    // -----------------------------------------------------------------------
    // Load configuration
    // -----------------------------------------------------------------------
    NS_LOG_INFO("Loading config: " << runConfigPath);
    mmwave_sim::SimConfig cfg;
    try
    {
        cfg = mmwave_sim::ConfigLoader::Load(runConfigPath, positionsOverridePath);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error loading config: " << e.what() << "\n";
        return 1;
    }
    NS_LOG_INFO("Scenario '" << cfg.scenario_name << "' seed=" << cfg.seed
                             << " duration=" << cfg.duration_s << "s"
                             << " nodes=" << cfg.nodes.size()
                             << " buildings=" << cfg.buildings.size());

    // -----------------------------------------------------------------------
    // Ensure output directory exists (and pcap subdir if requested)
    // -----------------------------------------------------------------------
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
        std::cerr << "Error creating output directory '" << cfg.output_dir
                  << "': " << e.what() << "\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // RNG seed / run ID
    // -----------------------------------------------------------------------
    RngSeedManager::SetSeed(cfg.seed);
    RngSeedManager::SetRun(cfg.run_id);

    // -----------------------------------------------------------------------
    // Protocol defaults (not exposed to config — sensible fixed values)
    // -----------------------------------------------------------------------
    Config::SetDefault("ns3::MmWaveHelper::RlcAmEnabled",          BooleanValue(false));
    Config::SetDefault("ns3::MmWaveHelper::HarqEnabled",           BooleanValue(true));
    Config::SetDefault("ns3::MmWaveFlexTtiMacScheduler::HarqEnabled", BooleanValue(true));
    Config::SetDefault("ns3::LteRlcUmLowLat::ReportBufferStatusTimer",
                       TimeValue(MicroSeconds(100.0)));

    // -----------------------------------------------------------------------
    // Trace output file redirection (must be before EnableTraces())
    // -----------------------------------------------------------------------
    Config::SetDefault("ns3::MmWavePhyTrace::OutputFilename",
                       StringValue(cfg.output_dir + "/RxPacketTrace.txt"));
    Config::SetDefault("ns3::MmWavePhyTrace::UlPhyTransmissionFilename",
                       StringValue(cfg.output_dir + "/UlPhyTransmissionTrace.txt"));
    Config::SetDefault("ns3::MmWavePhyTrace::DlPhyTransmissionFilename",
                       StringValue(cfg.output_dir + "/DlPhyTransmissionTrace.txt"));
    Config::SetDefault("ns3::MmWaveBearerStatsCalculator::DlRlcOutputFilename",
                       StringValue(cfg.output_dir + "/DlRlcStats.txt"));
    Config::SetDefault("ns3::MmWaveBearerStatsCalculator::UlRlcOutputFilename",
                       StringValue(cfg.output_dir + "/UlRlcStats.txt"));
    Config::SetDefault("ns3::MmWaveBearerStatsCalculator::DlPdcpOutputFilename",
                       StringValue(cfg.output_dir + "/DlPdcpStats.txt"));
    Config::SetDefault("ns3::MmWaveBearerStatsCalculator::UlPdcpOutputFilename",
                       StringValue(cfg.output_dir + "/UlPdcpStats.txt"));
    Config::SetDefault("ns3::MmWaveMacTrace::SchedInfoOutputFilename",
                       StringValue(cfg.output_dir + "/EnbSchedAllocTraces.txt"));
    Config::SetDefault("ns3::MmWaveBearerStatsConnector::MmWaveSinrOutputFilename",
                       StringValue(cfg.output_dir + "/MmWaveSinrTime.txt"));

    // -----------------------------------------------------------------------
    // Create helpers
    // -----------------------------------------------------------------------
    Ptr<MmWaveHelper> mmwH = CreateObject<MmWaveHelper>();
    mmwH->SetSchedulerType("ns3::MmWaveFlexTtiMacScheduler");

    Ptr<MmWavePointToPointEpcHelper> epcHelper =
        CreateObject<MmWavePointToPointEpcHelper>();
    mmwH->SetEpcHelper(epcHelper);

    // -----------------------------------------------------------------------
    // Build topology
    // -----------------------------------------------------------------------
    NS_LOG_INFO("Building topology...");
    mmwave_sim::TopologyBuilder topology(cfg, mmwH, epcHelper);
    topology.Build();

    // -----------------------------------------------------------------------
    // Install traffic
    // -----------------------------------------------------------------------
    NS_LOG_INFO("Installing traffic...");
    mmwave_sim::TrafficSetup traffic(cfg, topology);
    traffic.Install();

    // -----------------------------------------------------------------------
    // Start viz writer (schedules periodic CSV snapshots from t=0)
    // -----------------------------------------------------------------------
    NS_LOG_INFO("Starting viz writer (tick=" << cfg.viz_tick_ms << " ms)...");
    mmwave_sim::VizWriter vizWriter(cfg, topology.GetEnbNodes(), topology.GetUeNodes(),
                                    topology.GetChannelConditionModel());
    vizWriter.Start();

    // -----------------------------------------------------------------------
    // Enable traces and run
    // -----------------------------------------------------------------------
    NS_LOG_INFO("Starting simulation (duration=" << cfg.duration_s << "s)...");
    mmwH->EnableTraces();

    Simulator::Stop(Seconds(cfg.duration_s));
    Simulator::Run();
    NS_LOG_INFO("Simulation complete.");

    // -----------------------------------------------------------------------
    // Write metrics summary
    // -----------------------------------------------------------------------
    NS_LOG_INFO("Writing metrics to " << cfg.output_dir);
    mmwave_sim::MetricsWriter writer(cfg);
    writer.Write();

    Simulator::Destroy();
    return 0;
}
