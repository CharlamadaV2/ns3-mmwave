/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/setup/ns3-defaults.h"

#include "ns3/boolean.h"
#include "ns3/enum.h"
#include "ns3/mmwave-amc.h"
#include "ns3/string.h"
#include "ns3/uinteger.h"
#include "ns3/config.h"
#include "ns3/nstime.h"

namespace mmwave_sim
{

void
ApplyProtocolDefaults()
{
    ns3::Config::SetDefault("ns3::MmWaveHelper::RlcAmEnabled",
                            ns3::BooleanValue(false));
    ns3::Config::SetDefault("ns3::MmWaveHelper::HarqEnabled",
                            ns3::BooleanValue(true));
    ns3::Config::SetDefault("ns3::MmWaveFlexTtiMacScheduler::HarqEnabled",
                            ns3::BooleanValue(true));
    ns3::Config::SetDefault("ns3::LteRlcUmLowLat::ReportBufferStatusTimer",
                            ns3::TimeValue(ns3::MicroSeconds(100.0)));
}

void
ApplyTuningDefaults(const SimConfig& cfg)
{
    ns3::Config::SetDefault(
        "ns3::NYUChannelModel::UpdatePeriod",
        ns3::TimeValue(ns3::MilliSeconds(cfg.channel.channel_update_period_ms)));
    ns3::Config::SetDefault(
        "ns3::ThreeGppChannelModel::UpdatePeriod",
        ns3::TimeValue(ns3::MilliSeconds(cfg.channel.channel_update_period_ms)));
    ns3::Config::SetDefault(
        "ns3::NYUChannelConditionModel::UpdatePeriod",
        ns3::TimeValue(ns3::MilliSeconds(cfg.channel.condition_update_period_ms)));
    ns3::Config::SetDefault(
        "ns3::ThreeGppChannelConditionModel::UpdatePeriod",
        ns3::TimeValue(ns3::MilliSeconds(cfg.channel.condition_update_period_ms)));
    ns3::Config::SetDefault(
        "ns3::MmWaveUePhy::CqiReportPeriod",
        ns3::UintegerValue(cfg.channel.cqi_period_slots));

    // The default AMC is not shannon (I forget the method name - More computationally complex but more realistic)
    if (cfg.channel.amc_model == "shannon")
    {
        ns3::Config::SetDefault(
            "ns3::MmWaveAmc::AmcModel",
            ns3::EnumValue(ns3::mmwave::MmWaveAmc::ShannonModel));
    }
}

void
ApplyTraceFileDefaults(const std::string& output_dir)
{
    ns3::Config::SetDefault(
        "ns3::MmWavePhyTrace::OutputFilename",
        ns3::StringValue(output_dir + "/RxPacketTrace.txt"));
    ns3::Config::SetDefault(
        "ns3::MmWavePhyTrace::UlPhyTransmissionFilename",
        ns3::StringValue(output_dir + "/UlPhyTransmissionTrace.txt"));
    ns3::Config::SetDefault(
        "ns3::MmWavePhyTrace::DlPhyTransmissionFilename",
        ns3::StringValue(output_dir + "/DlPhyTransmissionTrace.txt"));
    ns3::Config::SetDefault(
        "ns3::MmWaveBearerStatsCalculator::DlRlcOutputFilename",
        ns3::StringValue(output_dir + "/DlRlcStats.txt"));
    ns3::Config::SetDefault(
        "ns3::MmWaveBearerStatsCalculator::UlRlcOutputFilename",
        ns3::StringValue(output_dir + "/UlRlcStats.txt"));
    ns3::Config::SetDefault(
        "ns3::MmWaveBearerStatsCalculator::DlPdcpOutputFilename",
        ns3::StringValue(output_dir + "/DlPdcpStats.txt"));
    ns3::Config::SetDefault(
        "ns3::MmWaveBearerStatsCalculator::UlPdcpOutputFilename",
        ns3::StringValue(output_dir + "/UlPdcpStats.txt"));
    ns3::Config::SetDefault(
        "ns3::MmWaveMacTrace::SchedInfoOutputFilename",
        ns3::StringValue(output_dir + "/EnbSchedAllocTraces.txt"));
    ns3::Config::SetDefault(
        "ns3::MmWaveBearerStatsConnector::MmWaveSinrOutputFilename",
        ns3::StringValue(output_dir + "/MmWaveSinrTime.txt"));
}

}  // namespace mmwave_sim
