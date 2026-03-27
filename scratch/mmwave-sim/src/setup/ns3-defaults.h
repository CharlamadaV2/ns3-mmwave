/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Centralized ns3::Config::SetDefault calls.
 * Keeps sim.cc free of low-level parameter wiring.
 */
#pragma once

#include "src/domain/sim-config.h"

#include <string>

namespace mmwave_sim
{

/**
 * Fixed protocol defaults (RLC, HARQ, buffer timers).
 * Call once before the per-seed loop.
 */
void ApplyProtocolDefaults();

/**
 * Performance-tuning defaults derived from SimConfig
 * (channel update periods, CQI, AMC model).
 * Call once before the per-seed loop.
 */
void ApplyTuningDefaults(const SimConfig& cfg);

/**
 * Per-seed trace file path redirection.
 * Must be called before MmWaveHelper construction so that
 * trace sinks open files in the correct output directory.
 */
void ApplyTraceFileDefaults(const std::string& output_dir);

}  // namespace mmwave_sim
