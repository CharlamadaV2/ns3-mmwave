/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file viz-writer.h
 * @brief Writes per-tick CSV snapshots for GUI and offline plotting.
 *
 * @ref VizWriter maintains six open CSV files for the duration of the
 * simulation and appends one row (or one row per node/link/flow) each time
 * @ref WriteTick is called.  Writes are rate-limited to @c viz_tick_ms
 * intervals so a small tick step (@c tick_s) does not inflate file sizes
 * unnecessarily.
 *
 *
 * **Output files** (all written to @c cfg.output_dir)
 *
 * | File             | Columns                                                                  |
 * |------------------|--------------------------------------------------------------------------|
 * | @c positions.csv | @c time_s, @c node_id, @c x, @c y, @c z, @c node_type, @c active        |
 * | @c links.csv     | @c time_s, @c node_a, @c node_b, @c dist_m, @c sinr_db, @c condition, @c condition_reason, @c capacity_mbps, @c delivered_mbps, @c hop_count |
 * | @c rx-power.csv  | @c time_s, @c node_a, @c node_b, @c rx_power_dbm                        |
 * | @c mcs.csv       | @c time_s, @c node_a, @c node_b, @c mcs_index, @c spectral_eff          |
 * | @c flows.csv     | @c time_s, @c src, @c dst, @c demand_mbps, @c delivered_mbps, @c latency_ms, @c hop_count, @c routable |
 * | @c routes.csv    | @c time_s, @c src, @c dst, @c path, @c bottleneck_mbps, @c hop_count, @c routable |
 *
 */
#pragma once

#include "src/domain/sim-config.h"
#include "src/eval/link-table.h"
#include "src/routing/mesh-router.h"

#include "ns3/mobility-model.h"

#include <cstdint>
#include <fstream>
#include <vector>

namespace mesh_sim
{

/**
 * @brief Streams per-tick simulation state to six CSV files.
 *
 * Holds a const reference to @ref SimConfig (not a copy); the config must
 * remain valid for the lifetime of the writer. The destructor calls
 * @ref Close so files are always flushed even if the step loop exits early.
 */
class VizWriter
{
  public:
    /**
     * @brief Construct a writer for the given simulation configuration.
     *
     * Computes @c m_vizTickS from @c cfg.viz_tick_ms and initialises
     * @c m_nextWriteS to 0.0 so the very first tick is always written.
     * Does not open any files; call @ref Open before the step loop.
     *
     * @param cfg  Simulation configuration. Must outlive this object.
     */
    explicit VizWriter(const SimConfig& cfg);

    /**
     * @brief Destructor — calls @ref Close to flush and close all files.
     */
    ~VizWriter();

    /**
     * @brief Open all six output CSV files and write their headers.
     *
     * Creates the files in @c cfg.output_dir (the directory must already
     * exist). Writes the @c # metadata comment block to @c positions.csv
     * and a column-header row to every file.
     *
     * Must be called once before @ref WriteTick.
     *
     * @throws std::runtime_error if any output file cannot be opened.
     */
    void Open();

    /**
     * @brief Append one snapshot to all six CSV files, rate-limited by @c viz_tick_ms.
     *
     * Skips the write if @c time_s < @c m_nextWriteS (using a 1 ns epsilon
     * for floating-point safety). When a write does occur, advances
     * @c m_nextWriteS by @c m_vizTickS for the next boundary.
     *
     * Internally calls the six private @c Write* helpers in order:
     * positions → links → rx-power → mcs → flows → routes.
     *
     * @param time_s  Current simulation time in seconds.
     * @param mobs    Mobility models for all nodes, in node-index order.
     * @param links   Current @ref LinkTable from the link evaluator.
     * @param flows   Flow results from the routing engine for this tick.
     */
    void WriteTick(double time_s,
                   const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs,
                   const LinkTable& links,
                   const std::vector<FlowResult>& flows);

    /**
     * @brief Flush and close all six output files.
     *
     * Safe to call multiple times; subsequent calls are no-ops for already-closed
     * files. Called automatically by the destructor.
     */
    void Close();

  private:
    /**
     * @brief Write one position row per node to @c positions.csv.
     *
     * Queries each @c MobilityModel for its current (x, y, z) and writes
     * the node's @c role from @c cfg.nodes (falling back to @c "peer").
     * The @c active column is always @c 1 — nodes are never deactivated
     * mid-simulation in mesh-sim.
     */
    void WritePositions(double time_s,
                        const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs);

    /**
     * @brief Write one row per unordered node pair to @c links.csv.
     *
     * Joins flow data to physical edges: for each flow whose path is
     * routable and contains at least two hops, the delivered throughput is
     * accumulated per edge and the maximum hop count is tracked.  These
     * aggregated values appear in @c delivered_mbps and @c hop_count.
     */
    void WriteLinks(double time_s,
                    const LinkTable& links,
                    const std::vector<FlowResult>& flows);

    /**
     * @brief Write one RX-power row per unordered node pair to @c rx-power.csv.
     */
    void WriteRxPower(double time_s, const LinkTable& links);

    /**
     * @brief Write one MCS row per unordered node pair to @c mcs.csv.
     *
     * Looks up @c spectral_eff from @ref MCS_TABLE using the stored
     * @c mcs_index so consumers do not need to re-derive it.
     */
    void WriteMcs(double time_s, const LinkTable& links);

    /**
     * @brief Write one row per active flow to @c flows.csv.
     */
    void WriteFlows(double time_s, const std::vector<FlowResult>& flows);

    /**
     * @brief Write one row per active flow to @c routes.csv.
     *
     * The @c path column is a semicolon-separated list of node indices
     * (e.g. @c "0;2;3"). The @c bottleneck_mbps column is the minimum
     * @c capacity_mbps across all hops; 0 for unroutable flows.
     */
    void WriteRoutes(double time_s,
                     const std::vector<FlowResult>& flows,
                     const LinkTable& links);

    const SimConfig& m_cfg;          ///< Simulation config (const reference; not owned).
    std::ofstream    m_posFile;      ///< @c positions.csv file stream.
    std::ofstream    m_linkFile;     ///< @c links.csv file stream.
    std::ofstream    m_rxPowerFile;  ///< @c rx-power.csv file stream.
    std::ofstream    m_mcsFile;      ///< @c mcs.csv file stream.
    std::ofstream    m_flowFile;     ///< @c flows.csv file stream.
    std::ofstream    m_routeFile;    ///< @c routes.csv file stream.
    double           m_vizTickS;     ///< Write interval in seconds (viz_tick_ms / 1000.0).
    double           m_nextWriteS;   ///< Sim time of the next scheduled write.
};

}  // namespace mesh_sim