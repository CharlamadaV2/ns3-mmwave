/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * MetricsWriter: runs after Simulator::Run() completes.
 * Reads ns-3 trace files from output_dir, computes per-UE aggregates,
 * and writes summary.json.
 *
 * No ns-3 simulation dependencies — only stdlib and nlohmann/json.
 */
#pragma once

#include "src/domain/sim-config.h"

namespace mmwave_sim
{

class MetricsWriter
{
  public:
    explicit MetricsWriter(const SimConfig& cfg);

    /**
     * Set wall-clock timing metadata to include in summary.json.
     * Must be called before Write().
     */
    void SetTiming(const TimingInfo& t);

    /**
     * Parse RxPacketTrace.txt and DlRlcStats.txt from cfg.output_dir,
     * compute per-UE and network-level metrics, write summary.json.
     *
     * Rows with time <= cfg.warmup_s are excluded from all statistics.
     */
    void Write() const;

  private:
    const SimConfig& m_cfg;
    TimingInfo       m_timing;
};

}  // namespace mmwave_sim
