/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file progress-logger.h
 * @brief Lightweight simulation progress reporter.
 *
 *
 * @ref ProgressLogger is called directly from the step loop — not via
 * @c ns3::Simulator::Schedule — so it has zero overhead on non-reporting
 * ticks (a single modulo check per tick).
 *
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace mesh_sim
{

/**
 * @brief Prints sim-time progress, wall-clock elapsed time, and ETA to stderr.
 *
 * All fields must be set before the first call to @ref Tick. The struct has
 * no constructor — use aggregate initialisation or set fields directly.
 */
struct ProgressLogger
{
    uint32_t total_ticks    = 0;  ///< Total number of simulation ticks (duration_s / tick_s).
    uint32_t interval_ticks = 1;  ///< Print every @c interval_ticks ticks. Set to
                                   ///<   @c total_ticks/N for roughly N progress lines.
    uint32_t seed       = 0;      ///< Seed shown in the log prefix (@c "[seed N]").
    double   duration_s = 0.0;    ///< Total simulated duration in seconds.
    double   tick_s     = 0.1;    ///< Simulation time step in seconds; used to convert
                                   ///<   @c tick_index to a sim-time value.

    /// Wall-clock reference point captured before the step loop starts.
    std::chrono::time_point<std::chrono::steady_clock> wall_start;

    /**
     * @brief Report progress if this tick falls on a reporting boundary.
     *
     * Prints to @c stderr when @c tick_index is a multiple of
     * @c interval_ticks *or* when @c tick_index equals @c total_ticks
     * (ensuring the 100% line is always printed).  All other ticks are
     * a single modulo check and return immediately.
     *
     * The ETA is estimated as:
     * @code
     *   rate  = wall_elapsed / sim_elapsed    (wall seconds per sim second)
     *   eta_s = (duration_s - sim_t) * rate
     * @endcode
     * ETA is 0 when @c sim_t == 0 (avoids division by zero at tick 0).
     *
     * @param tick_index  Zero-based index of the current tick.
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