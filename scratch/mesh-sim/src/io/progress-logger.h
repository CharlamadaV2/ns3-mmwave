/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/** @brief
 * ProgressLogger: prints simulation progress to stderr.
 * Called directly from the step loop (no Simulator::Schedule).
 * Header-only.
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

  /** @brief
  */
namespace mesh_sim
{
  /** @brief
  */
struct ProgressLogger
{
    uint32_t total_ticks;
    uint32_t interval_ticks;
    uint32_t seed;
    double   duration_s;
    double   tick_s;
    std::chrono::time_point<std::chrono::steady_clock> wall_start;
    /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
    void Tick(uint32_t tick_index)
    {
        if (tick_index % interval_ticks != 0 && tick_index != total_ticks)
        {
            return;
        }

        double sim_t  = tick_index * tick_s;
        double pct    = (static_cast<double>(tick_index) / total_ticks) * 100.0;
        double wall_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - wall_start).count();
        double rate   = (sim_t > 0) ? wall_s / sim_t : 0;
        double eta_s  = (duration_s - sim_t) * rate;

        std::cerr << std::fixed
                  << "[seed " << seed << "] "
                  << std::setprecision(1) << sim_t << "s / " << duration_s << "s"
                  << " (" << std::setprecision(0) << pct << "%)"
                  << "  wall=" << std::setprecision(1) << wall_s << "s"
                  << "  ETA~" << eta_s << "s"
                  << std::endl;
    }
};

}  // namespace mesh_sim
