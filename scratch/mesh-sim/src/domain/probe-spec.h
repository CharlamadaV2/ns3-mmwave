/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file probe-spec.h
 * @brief Coverage probe receivers scored by the channel query (no ns-3).
 */
#pragma once

#include "node-spec.h"

#include <vector>

namespace mesh_sim
{

/**
 * @brief Receive-only probe points evaluated against every mesh node.
 *
 * Probes are not mesh nodes: they get no traffic, routing or RL slot, and
 * exist only in a channel-query child process.
 */
struct ProbeGrid
{
    std::vector<Position> points;  ///< Probe positions (m); the producer sets each z to @c height_m.
    double height_m    = 1.5;      ///< Probe receiver height above the ground plane (m).
    double rx_gain_dbi = 0.0;      ///< RX array gain applied at every probe (dBi).
    double sinr_db     = -6.7;     ///< A probe is covered by a node when its SINR is >= this (dB).
};

}  // namespace mesh_sim
