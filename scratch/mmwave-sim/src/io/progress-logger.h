/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * ProgressLogger: periodically prints simulation progress to stderr.
 * Scheduled inside the ns-3 event loop via Simulator::Schedule.
 *
 * IMPORTANT: The instance must live on the stack of the per-seed loop
 * body (or otherwise outlive Simulator::Run()), because the event
 * scheduler holds a raw pointer to it.
 */
#pragma once

#include "ns3/simulator.h"

#include <chrono>
#include <iomanip>
#include <iostream>

namespace mmwave_sim
{

struct ProgressLogger
{
    double   duration_s;
    double   interval_s;
    uint32_t seed;
    std::chrono::time_point<std::chrono::steady_clock> wall_start;

    void Tick()
    {
        double sim_t = ns3::Simulator::Now().GetSeconds();
        double pct   = (sim_t / duration_s) * 100.0;
        double wall_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start).count();
        double rate  = (sim_t > 0) ? wall_s / sim_t : 0;
        double eta_s = (duration_s - sim_t) * rate;

        std::cerr << std::fixed
                  << "[seed " << seed << "] "
                  << std::setprecision(1) << sim_t << "s / " << duration_s << "s"
                  << " (" << std::setprecision(0) << pct << "%)"
                  << "  wall=" << std::setprecision(1) << wall_s << "s"
                  << "  ETA~" << eta_s << "s"
                  << std::endl;

        if (sim_t + interval_s <= duration_s)
        {
            ns3::Simulator::Schedule(ns3::Seconds(interval_s),
                                     &ProgressLogger::Tick, this);
        }
    }
};

}  // namespace mmwave_sim
