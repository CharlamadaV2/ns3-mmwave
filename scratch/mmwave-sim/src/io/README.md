# io/

Output writers for simulation metrics, visualization data, and progress reporting.

## Files

### metrics-writer.h / metrics-writer.cc

Post-simulation metric aggregation. Runs after `Simulator::Run()` completes.

**Key functions:**
- `MetricsWriter(cfg)` -- constructor, takes `SimConfig` reference
- `SetTiming(timing)` -- sets wall-clock metadata (must call before `Write()`)
- `Write()` -- parses `RxPacketTrace.txt` and `DlRlcStats.txt` from `cfg.output_dir`, computes per-UE aggregates (throughput, delay, SINR, LOS fraction, corruption rate), writes `summary.json`

Rows with `time <= cfg.warmup_s` are excluded from all statistics. No ns-3 simulation dependencies.

### viz-writer.h / viz-writer.cc

Periodic CSV snapshots during simulation. Fires at `cfg.viz_tick_ms` intervals.

**Key functions:**
- `VizWriter(cfg, enbNodes, ueNodes, condModel)` -- constructor
- `Start()` -- opens CSV files, connects SINR trace callback, schedules first tick
- `Flush()` -- flushes output streams after simulation

**Output files:**
- `positions.csv` -- columns: `time_s, node_id, x, y, z, node_type, active`
- `links.csv` -- columns: `time_s, node_a, node_b, dist_m, sinr_db, condition`

SINR values come from `MmWaveUePhy/ReportCurrentCellRsrpSinr`. Missing reports output `-999`.

### progress-logger.h

Header-only. Prints simulation progress to stderr with ETA estimates.

**Key type:**
- `ProgressLogger` -- struct with `Tick()` method, self-reschedules via `Simulator::Schedule`

## Usage

In `sim.cc`'s per-seed loop:
```cpp
mmwave_sim::VizWriter vizWriter(cfg, topology.GetEnbNodes(), topology.GetUeNodes(), topology.GetChannelConditionModel());
vizWriter.Start();
// ... Simulator::Run() ...
vizWriter.Flush();

mmwave_sim::MetricsWriter writer(cfg);
writer.SetTiming(timing);
writer.Write();
```

## Notes

- `ProgressLogger` instance must outlive `Simulator::Run()` because the scheduler holds a raw pointer to it.
