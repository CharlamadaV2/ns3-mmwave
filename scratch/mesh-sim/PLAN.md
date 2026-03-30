# mesh-sim: Lightweight mmWave Mesh Simulator -- Implementation Plan

> Target: implementable in one focused day.
> Sits next to `mmwave-sim` in `scratch/`.
> Follows the same directory structure and I/O conventions so it works with the existing GUI.

---

## Table of Contents

1. [What This Simulator Is](#1-what-this-simulator-is)
2. [What We Reuse vs. What We Write](#2-what-we-reuse-vs-what-we-write)
3. [Directory Structure](#3-directory-structure)
4. [Domain Layer -- Every POD Type](#4-domain-layer----every-pod-type)
5. [The Link Evaluator -- How Wireless Links Work](#5-the-link-evaluator----how-wireless-links-work)
6. [Traffic Generation](#6-traffic-generation)
7. [Multi-Node Connections and Time-Varying Links](#7-multi-node-connections-and-time-varying-links)
8. [Routing](#8-routing)
9. [Uplink vs. Downlink](#9-uplink-vs-downlink)
10. [The Step Loop -- How One Tick Works](#10-the-step-loop----how-one-tick-works)
11. [Beamforming, AMC, SINR -- In Depth](#11-beamforming-amc-sinr----in-depth)
12. [Output -- GUI Compatibility](#12-output----gui-compatibility)
13. [Metrics Writer](#13-metrics-writer)
14. [RL Placeholder](#14-rl-placeholder)
15. [Class/Object Inventory](#15-classobject-inventory)
16. [Build System](#16-build-system)
17. [Implementation Order](#17-implementation-order)
18. [Equation Reference Sheet](#18-equation-reference-sheet)

---

## 1. What This Simulator Is

A **time-stepped, link-level mmWave mesh simulator** that:

- Models N **peer nodes** (no eNB/UE distinction -- all nodes are identical).
- At each time step, evaluates **all N*(N-1)/2 directional links** between every pair of nodes.
- Uses the **exact same 3GPP / NYU channel models** from ns3-mmwave for path loss, LOS/NLOS, and shadow fading.
- Generates **traffic demands** between node pairs and routes them over the mesh.
- Computes **per-link and per-flow metrics** (SINR, throughput capacity, delivered traffic, latency proxy).
- Runs **~1,000-10,000x faster** than the full cellular stack because it skips RRC, MAC scheduling, HARQ, RLC, PDCP, EPC, and the ns-3 event scheduler.

### What it is NOT

- Not a packet-level simulator. We don't track individual packets.
- Not a cellular simulator. There is no base station, no UE, no centralized scheduler.
- Not a full ns-3 simulation. We call ns-3 propagation models as library functions, but we don't use `Simulator::Run()`.

### Why this is realistic enough

The key claim: **for a node-positioning problem, the geometry (path loss + LOS/NLOS + beamforming) dominates the link quality.** MAC scheduling, HARQ, and RLC add 5-15% throughput variation that does not change which position is better. We validate this claim by comparing our SINR values against the full-stack `mmwave-sim` at identical positions (see Section 12).

---

## 2. What We Reuse vs. What We Write

### Reused from ns-3 / ns3-mmwave (we call these as library functions)

| Component | ns-3 Class | File | What it does | How we call it |
|-----------|-----------|------|-------------|----------------|
| **Path loss (3GPP)** | `ThreeGppUmiStreetCanyonPropagationLossModel` (+ UMa, RMa, InH, V2V variants) | `src/propagation/model/three-gpp-propagation-loss-model.h` | Computes received power given TX power and two node positions. Implements 3GPP TR 38.901 equations. Internally queries `ChannelConditionModel` for LOS/NLOS. | `plModel->CalcRxPower(txPowerDbm, mobA, mobB)` returns rx power in dBm. Pure function -- no scheduler needed. |
| **Path loss (NYU)** | `NYUUmiPropagationLossModel` (+ UMa, RMa, InH, InF) | `src/propagation/model/nyu-propagation-loss-model.h` | Same interface as 3GPP. Adds atmospheric absorption, foliage loss, rain attenuation. | Same `CalcRxPower()` call. Also pure function. |
| **LOS/NLOS condition** | `BuildingsChannelConditionModel` | `src/propagation/model/channel-condition-model.h` | Given two mobility models, checks if the line between them intersects any `Building` box. Returns LOS, NLOS, or NLOSv. | `condModel->GetChannelCondition(mobA, mobB)->GetLosCondition()`. The propagation loss model calls this internally, but we also call it directly for the link table. |
| **Buildings** | `Building`, `BuildingsHelper` | `src/buildings/model/building.h`, `src/buildings/helper/buildings-helper.h` | Axis-aligned boxes with material type and penetration loss. `BuildingsHelper::Install()` tags each node as indoor/outdoor. | We create `Building` objects from our `BuildingSpec` config, call `BuildingsHelper::Install()` once, then `MakeMobilityModelConsistent()` after position changes. |
| **Mobility models** | `ConstantPositionMobilityModel` | `src/mobility/model/constant-position-mobility-model.h` | Stores an (x,y,z) position. `SetPosition()` / `GetPosition()`. Used by all propagation models to get node locations. | One per node. Updated via `SetPosition()` at each time step. |
| **Noise PSD** | `MmWaveSpectrumValueHelper` | `src/mmwave/model/mmwave-spectrum-value-helper.h` | Creates thermal noise power spectral density for a given bandwidth and noise figure. | `CreateNoisePowerSpectralDensity(config, noiseFigure)` -- used once to compute noise floor. |
| **PHY/MAC config** | `MmWavePhyMacCommon` | `src/mmwave/model/mmwave-phy-mac-common.h` | Container for numerology: bandwidth, subcarrier spacing, number of RBs, symbols per slot. Needed by `MmWaveSpectrumValueHelper` and `MmWaveAmc`. | `CreateObject<MmWavePhyMacCommon>()`, `SetCentreFrequency()`, `SetNumRb()`. |
| **AMC table** | `MmWaveAmc` | `src/mmwave/model/mmwave-amc.h` | Maps SINR to MCS (modulation and coding scheme) to transport block size. | `GetMcsFromSpectralEfficiency(se)` or Shannon formula. See Section 11. |
| **RNG** | `RngSeedManager` | `src/core/model/rng-seed-manager.h` | Deterministic random number seeding for reproducibility. | `RngSeedManager::SetSeed(seed)` at startup. |

### What we write ourselves (new code in mesh-sim)

| Component | Class | File | Lines (est.) | What it does | Reference for equations |
|-----------|-------|------|-------------|-------------|------------------------|
| **LinkEvaluator** | `LinkEvaluator` | `src/eval/link-evaluator.h/cc` | ~200 | Wraps ns-3 propagation models. For a node pair, returns: path loss, LOS/NLOS, SINR, capacity. | Equations from ns3-mmwave (Section 18). |
| **LinkTable** | `LinkTable` | `src/eval/link-table.h/cc` | ~150 | Stores the NxN link quality matrix. Updated each tick. Exposes per-link and aggregate queries. | Data structure only. |
| **TrafficMatrix** | `TrafficMatrix` | `src/traffic/traffic-matrix.h/cc` | ~200 | Generates and holds traffic demands between node pairs. Supports constant-rate, Poisson, and on-off models. Tracks holding times. | ns-3 `UdpClient` pattern (Section 6). |
| **MeshRouter** | `MeshRouter` | `src/routing/mesh-router.h/cc` | ~250 | Given a link table and traffic matrix, routes flows over the mesh. Computes delivered throughput and latency proxy per flow. | Dijkstra / shortest-path-first (Section 8). |
| **TopologyBuilder** | `TopologyBuilder` | `src/setup/topology-builder.h/cc` | ~250 | Creates ns-3 node objects, mobility models, buildings, configures propagation models. Similar to mmwave-sim but no EPC/RRC/MAC. | Pattern from `mmwave-sim/src/setup/topology-builder.cc`. |
| **VizWriter** | `VizWriter` | `src/io/viz-writer.h/cc` | ~150 | Writes `positions.csv` and `links.csv` in the same format as mmwave-sim. | Same CSV format as `mmwave-sim/src/io/viz-writer.cc`. |
| **MetricsWriter** | `MetricsWriter` | `src/io/metrics-writer.h/cc` | ~200 | Writes `summary.json` with per-node and network-level metrics. | Same JSON schema as `mmwave-sim/src/io/metrics-writer.cc`. |
| **ConfigLoader** | `ConfigLoader` | `src/config/config-loader.h/cc` | ~200 | Loads `run.ini` + `nodes.json` + `buildings.json`. Reuses mmwave-sim's INI/JSON parsing. | Same format as `mmwave-sim/src/config/`. |
| **Entry point** | `main()` | `sim.cc` | ~120 | CLI parsing, config loading, per-seed loop, step loop. | Pattern from `mmwave-sim/sim.cc`. |
| **RL placeholder** | `RlAgent` | `src/rl/rl-agent.h` | ~30 | Empty interface. `GetActions()` returns current positions unchanged. | N/A -- placeholder. |

---

## 3. Directory Structure

```
scratch/mesh-sim/
+-- PLAN.md                      <-- this file
+-- CLAUDE.md                    <-- code guidelines (same conventions as mmwave-sim)
+-- CMakeLists.txt               <-- build config
+-- sim.cc                       <-- entry point (~120 lines)
+-- third_party/
|   +-- json.hpp                 <-- nlohmann/json (copy from mmwave-sim)
+-- inputs/
|   +-- scenarios/
|       +-- triangle/            <-- 3-node mesh scenario
|       |   +-- run.ini
|       |   +-- nodes.json
|       +-- five-node-mesh/      <-- 5-node scenario with buildings
|           +-- run.ini
|           +-- nodes.json
|           +-- buildings.json
+-- src/
    +-- domain/                  <-- Pure POD types (NO ns-3 headers)
    |   +-- CLAUDE.md
    |   +-- sim-config.h
    |   +-- node-spec.h          <-- reuse from mmwave-sim (identical)
    |   +-- channel-config.h     <-- reuse from mmwave-sim (identical)
    |   +-- mesh-config.h        <-- NEW: traffic + routing config
    |   +-- link-result.h        <-- NEW: per-link evaluation result
    +-- config/                  <-- Config loading (no ns-3)
    |   +-- CLAUDE.md
    |   +-- config-loader.h/cc
    +-- util/                    <-- Generic helpers (no ns-3)
    |   +-- CLAUDE.md
    |   +-- ini-parser.h/cc      <-- reuse from mmwave-sim (symlink or copy)
    |   +-- string-utils.h/cc    <-- reuse from mmwave-sim (symlink or copy)
    +-- cli/                     <-- CLI parsing
    |   +-- CLAUDE.md
    |   +-- cli-parser.h/cc
    +-- eval/                    <-- Link evaluation (ns-3 propagation models)
    |   +-- CLAUDE.md
    |   +-- link-evaluator.h/cc
    |   +-- link-table.h/cc
    +-- traffic/                 <-- Traffic generation
    |   +-- CLAUDE.md
    |   +-- traffic-matrix.h/cc
    +-- routing/                 <-- Mesh routing
    |   +-- CLAUDE.md
    |   +-- mesh-router.h/cc
    +-- setup/                   <-- ns-3 object creation
    |   +-- CLAUDE.md
    |   +-- topology-builder.h/cc
    +-- io/                      <-- Output writers
    |   +-- CLAUDE.md
    |   +-- viz-writer.h/cc
    |   +-- metrics-writer.h/cc
    |   +-- progress-logger.h
    +-- rl/                      <-- RL placeholder
        +-- CLAUDE.md
        +-- rl-agent.h
```

---

## 4. Domain Layer -- Every POD Type

All in `src/domain/`. No ns-3 headers. No logic. Pure data.

### node-spec.h (reuse from mmwave-sim, identical)

```cpp
struct Position { double x = 0.0, y = 0.0, z = 0.0; };
struct Velocity { double vx = 0.0, vy = 0.0, vz = 0.0; };
struct RandomWalkParams { double x_min, x_max, y_min, y_max, speed_mps; };

struct NodeSpec {
    std::string id;          // e.g. "node0", "node1"
    std::string role;        // "peer" (all nodes are peers in mesh-sim)
    std::string mobility;    // "fixed", "constant_velocity", "random_walk"
    Position    position;
    Velocity    velocity;
    RandomWalkParams random_walk;
};

struct BuildingSpec {
    std::string id;
    double x_min, x_max, y_min, y_max, z_min, z_max;
    std::string type;       // "Residential", "Office", "Commercial"
    std::string ext_walls;  // "ConcreteWithWindows", etc.
    int n_floors;
};
```

### channel-config.h (reuse from mmwave-sim, identical)

```cpp
struct NyuChannelConfig {
    double rf_bandwidth_mhz = 800.0;
    bool   shadowing_enabled = true;
    double pressure_mbar = 1013.25;
    double humidity_pct = 50.0;
    double temperature_c = 20.0;
    double rain_rate_mm_hr = 0.0;
    bool   atmospheric_loss_enabled = false;
    bool   foliage_loss_enabled = false;
    double foliage_loss_db_m = 0.4;
    std::string o2i_loss_type = "Low Loss";
};

struct ChannelConfig {
    double      frequency_ghz    = 28.0;
    double      tx_power_dbm     = 30.0;
    std::string scenario         = "UMi";
    std::string channel_model    = "3gpp";  // "3gpp" or "nyu"
    bool        blockage_enabled = true;
    std::string beamforming_model = "svd";  // for reference; simplified in mesh-sim
    std::string amc_model         = "shannon";  // "shannon" or "table"
    double      noise_figure_db   = 5.0;    // receiver noise figure
    NyuChannelConfig nyu;
};
```

### mesh-config.h (NEW)

```cpp
// How traffic demands are generated between node pairs.
struct TrafficConfig {
    std::string model = "constant";  // "constant", "poisson", "on_off"

    // -- constant model --
    // Each active flow demands this many Mbps.
    double demand_mbps = 10.0;

    // -- poisson model --
    // Flow arrivals per second (network-wide).
    double arrival_rate_hz = 1.0;

    // -- on_off model (bursty traffic) --
    // Modeled after ns-3 OnOffApplication:
    //   src/applications/helper/on-off-helper.h
    // During ON: transmit at demand_mbps.
    // During OFF: transmit nothing.
    double on_time_s  = 1.0;   // mean ON duration
    double off_time_s = 1.0;   // mean OFF duration

    // -- holding time (applies to all models) --
    // How long a flow persists before it ends.
    // 0 = flow lasts the entire simulation (always-on).
    // > 0 = flow ends after this many seconds; new flows may arrive (Poisson).
    double holding_time_s = 0.0;

    // -- flow topology --
    // "all_pairs": every node sends to every other node.
    // "random_pairs": each tick, randomly select K source-dest pairs.
    // "gateway": all nodes send to/from a designated gateway node.
    std::string flow_topology = "all_pairs";
    uint32_t    random_pair_count = 3;       // for "random_pairs"
    std::string gateway_node_id   = "";       // for "gateway"
};

struct RoutingConfig {
    // "shortest_path":  Dijkstra on link capacity graph.
    // "max_throughput": widest-path (maximize bottleneck capacity).
    // "min_hop":        fewest hops (ignore link quality).
    std::string algorithm = "shortest_path";

    // Maximum number of hops a flow can traverse. 0 = unlimited.
    uint32_t max_hops = 5;
};

struct MeshConfig {
    TrafficConfig traffic;
    RoutingConfig routing;
};
```

### link-result.h (NEW)

```cpp
// Result of evaluating a single directional link between two nodes.
// One instance per ordered pair (tx, rx) per time step.
struct LinkResult {
    uint32_t tx_id = 0;          // index into node list
    uint32_t rx_id = 0;
    double   distance_m = 0.0;   // 3D Euclidean distance
    bool     is_los = false;     // true = Line of Sight, false = NLOS
    double   path_loss_db = 0.0; // total path loss including shadow fading
    double   rx_power_dbm = -999.0;
    double   sinr_db = -999.0;   // Signal to Interference+Noise Ratio
    double   capacity_mbps = 0.0; // Shannon or AMC-based capacity
};
```

### sim-config.h (adapted from mmwave-sim)

```cpp
struct TimingInfo {
    std::chrono::system_clock::time_point start, end;
    double elapsed_s = 0.0;
};

struct SimConfig {
    std::string scenario_name;
    uint32_t    seed       = 42;
    uint32_t    run_id     = 1;
    double      duration_s = 10.0;   // total simulated time
    double      warmup_s   = 0.0;    // skip this many seconds for metrics
    double      tick_s     = 0.1;    // time step interval (100 ms)
    std::string output_dir;

    ChannelConfig channel;
    MeshConfig    mesh;

    uint32_t viz_tick_ms = 100;  // how often to write CSV snapshots

    std::vector<NodeSpec>     nodes;
    std::vector<BuildingSpec> buildings;
};
```

---

## 5. The Link Evaluator -- How Wireless Links Work

This is the core physics engine. It replaces the entire cellular stack (MmWaveHelper, EPC, RRC, MAC, HARQ, RLC, PDCP) with direct calls to the propagation model.

### What the ns-3 cellular stack does (and what we replace)

In `mmwave-sim`, the signal path is:

```
TX MAC creates Transport Block
  -> TX PHY modulates and transmits via MmWaveSpectrumPhy
    -> SpectrumChannel delivers signal to all RX PHYs
      -> ThreeGppSpectrumPropagationLossModel applies:
           path loss + beamforming gain + small-scale fading
        -> RX PHY's mmWaveInterference model computes:
             SINR = rxPower / (interference + noise)
          -> Error model determines if TB decoded correctly
            -> RLC reassembles, PDCP delivers to IP
```

**Reference**: `src/mmwave/model/mmwave-spectrum-phy.cc` lines 339-579 (StartRx through EndRxData), `src/spectrum/model/three-gpp-spectrum-propagation-loss-model.cc` lines 155-427 (long-term beamforming + frequency-domain channel).

We replace all of that with:

```
LinkEvaluator::Evaluate(nodeA, nodeB)
  1. Get positions from MobilityModel
  2. Call PropagationLossModel::CalcRxPower(txPower, mobA, mobB)
     -> Internally queries ChannelConditionModel for LOS/NLOS
     -> Applies 3GPP TR 38.901 or NYU path loss equation
     -> Applies shadow fading
     -> Returns rx power in dBm
  3. Compute SINR = rxPower - noiseFloor  (+ beamforming gain, if enabled)
  4. Compute capacity via Shannon formula or AMC table
  5. Return LinkResult
```

### Why this is valid

The `PropagationLossModel::CalcRxPower()` interface is a **pure function** of two positions. It does not depend on MAC scheduling, RRC state, HARQ, or any protocol layer. This is proven by its signature:

```cpp
// src/propagation/model/propagation-loss-model.h
double CalcRxPower(double txPowerDbm,
                   Ptr<MobilityModel> a,
                   Ptr<MobilityModel> b) const;
```

The `ChannelConditionModel` is also a pure function of positions (with optional random state for probabilistic LOS):

```cpp
// src/propagation/model/channel-condition-model.h
Ptr<ChannelCondition> GetChannelCondition(Ptr<const MobilityModel> a,
                                          Ptr<const MobilityModel> b) const;
```

### LinkEvaluator class

```cpp
// src/eval/link-evaluator.h
#pragma once
#include "src/domain/link-result.h"
#include "src/domain/sim-config.h"
#include <ns3/propagation-loss-model.h>
#include <ns3/channel-condition-model.h>
#include <ns3/mobility-model.h>

namespace mesh_sim {

class LinkEvaluator {
public:
    // Configure propagation models based on SimConfig.
    // Must be called after ns-3 nodes and buildings exist.
    void Configure(const SimConfig& cfg,
                   ns3::Ptr<ns3::PropagationLossModel> plModel,
                   ns3::Ptr<ns3::ChannelConditionModel> condModel);

    // Evaluate a single directed link.  tx_power_dbm comes from config.
    LinkResult Evaluate(ns3::Ptr<ns3::MobilityModel> txMob,
                        ns3::Ptr<ns3::MobilityModel> rxMob,
                        uint32_t txIdx, uint32_t rxIdx) const;

    // Evaluate all N*(N-1)/2 undirected pairs.  Returns vector of LinkResults.
    // For each pair (i,j) where i < j, evaluates i->j.  The link is symmetric
    // for path loss (same equation both directions).
    std::vector<LinkResult> EvaluateAll(
        const std::vector<ns3::Ptr<ns3::MobilityModel>>& mobs) const;

private:
    double m_txPowerDbm = 30.0;
    double m_noiseFloorDbm = -174.0;  // computed from kTB + noise figure
    double m_bandwidthHz = 1e9;       // for Shannon capacity
    std::string m_amcModel = "shannon";
    ns3::Ptr<ns3::PropagationLossModel> m_plModel;
    ns3::Ptr<ns3::ChannelConditionModel> m_condModel;
};

} // namespace mesh_sim
```

### Noise floor computation

**Reference**: Thermal noise is `N = k_B * T * B` where k_B = 1.38e-23 J/K, T = 290 K, B = bandwidth in Hz. Plus a receiver noise figure (typically 5-9 dB for mmWave).

In ns3-mmwave, this is computed by `MmWaveSpectrumValueHelper::CreateNoisePowerSpectralDensity()` (`src/mmwave/model/mmwave-spectrum-value-helper.h`). We compute it ourselves:

```cpp
// Noise floor in dBm:
// N_dBm = 10*log10(k_B * T * B * 1000)  + noiseFigure_dB
// At 28 GHz with 800 MHz BW and 5 dB NF:
// N_dBm = -174 dBm/Hz + 10*log10(800e6) + 5 = -174 + 89 + 5 = -80 dBm
double noiseFloorDbm = -174.0 + 10.0 * std::log10(bandwidthHz) + noiseFigureDb;
```

**Reference for -174 dBm/Hz**: This is the standard thermal noise density at 290 K. Used universally in wireless engineering. ns-3 uses this in `src/spectrum/model/wifi-spectrum-value-helper.cc` and implicitly in mmwave via `MmWaveSpectrumValueHelper`.

### SINR computation

In the full stack, SINR is per-subcarrier with interference from other transmitters:

```cpp
// src/mmwave/model/mmwave-interference.cc (ConditionallyEvaluateChunk, ~line 197)
SINR[rb] = rxSignal[rb] / (allSignals[rb] - rxSignal[rb] + noise[rb])
```

In our simplified model, we compute a **wideband SINR** with aggregate interference:

```cpp
// For link (i -> j):
double rxPower_dBm = plModel->CalcRxPower(txPowerDbm, mob_i, mob_j);
// Interference from all other nodes transmitting simultaneously:
double interference_linear = 0.0;
for (each other node k != i, k != j, if k is transmitting):
    double interfPower_dBm = plModel->CalcRxPower(txPowerDbm, mob_k, mob_j);
    interference_linear += pow(10.0, interfPower_dBm / 10.0);
double noise_linear = pow(10.0, noiseFloorDbm / 10.0);
double sinr_linear = pow(10.0, rxPower_dBm / 10.0) / (interference_linear + noise_linear);
double sinr_dB = 10.0 * log10(sinr_linear);
```

**Simplification for v1**: In the first implementation, we can **ignore inter-node interference** (assume orthogonal channels or TDMA where only one link is active at a time). This makes SINR = rxPower - noiseFloor. This is the same simplification used in `mmwave-sim/src/io/viz-writer.cc` line 186, where the cached SINR comes from the single attached eNB-UE link.

We can add interference in v2 by iterating over all simultaneously-active transmitters.

### Beamforming gain -- simplified model

In the full stack, beamforming is applied inside `ThreeGppSpectrumPropagationLossModel::DoCalcRxPowerSpectralDensity()` (`src/spectrum/model/three-gpp-spectrum-propagation-loss-model.cc` lines 155-427). The gain comes from the steering vector pointing toward the target:

```
Long_Term[cluster] = w_rx^H * H[cluster] * w_tx
```

where `w_tx` and `w_rx` are beamforming weight vectors computed by `MmWaveDftBeamforming` (`src/mmwave/model/mmwave-beamforming-model.cc` lines 142-171).

**For our simplified model**, we use a **directional antenna gain approximation**:

```cpp
// Approximation: ideal beam with half-power beamwidth (HPBW) based on array size.
// For an N_row x N_col array, the DFT beamforming gain is approximately:
//   G_max = 10 * log10(N_row * N_col)  dBi
// Reference: antenna theory, also matches ns-3 PhasedArrayModel normalization
//   in src/antenna/model/phased-array-model.cc lines 97-112 where
//   the beamforming vector is normalized by sqrt(NumPorts).
//
// For a 4x4 array: G_max = 10*log10(16) = 12.0 dBi (per side)
// TX + RX combined: up to 24 dBi total beamforming gain.
double bfGainDb = 10.0 * std::log10(numAntennaElements);  // per node
double totalBfGainDb = txBfGainDb + rxBfGainDb;            // both sides
```

This is a **best-case (beam perfectly aligned)** approximation. In v1 this is fine because we assume each node steers toward its communication partner. In v2 we could add misalignment loss.

The full `CalcRxPower()` from the propagation loss model does **NOT** include beamforming gain -- it only computes path loss. So we add it:

```cpp
double rxPower_dBm = plModel->CalcRxPower(txPowerDbm, mobA, mobB);
double effectiveRxPower_dBm = rxPower_dBm + totalBfGainDb;
double sinr_dB = effectiveRxPower_dBm - noiseFloorDbm;
```

---

## 6. Traffic Generation

### What mmwave-sim does (reference)

In `mmwave-sim/src/setup/traffic-setup.cc` (lines 33-63), traffic is generated by `ns3::UdpClient`:
- Sends fixed-size packets at fixed intervals.
- Always from remote host to UE (downlink) or UE to remote host (uplink).
- The `UdpClient` class is defined in `src/applications/model/udp-client.h`. Key attributes: `m_interval` (inter-packet time), `m_size` (packet size bytes), `m_count` (max packets).

### What we do instead

We don't generate actual packets. Instead, we model **traffic demands** as a matrix: "node A wants to send X Mbps to node B."

```cpp
// src/traffic/traffic-matrix.h
struct Flow {
    uint32_t src;              // source node index
    uint32_t dst;              // destination node index
    double   demand_mbps;      // how much bandwidth this flow wants
    double   start_time_s;     // when the flow begins
    double   end_time_s;       // when the flow ends (0 = never)
    bool     active;           // is this flow currently active?

    // On-off state (if using on_off model)
    bool     in_on_phase;
    double   phase_end_s;      // when current ON/OFF phase ends
};

class TrafficMatrix {
public:
    TrafficMatrix(const SimConfig& cfg);

    // Initialize flows based on traffic config.
    void Initialize(uint32_t numNodes, double currentTime);

    // Advance to next time step.  Updates on/off states, creates/removes
    // flows (for Poisson arrivals), handles holding times.
    void Tick(double currentTime);

    // Get all currently active flows.
    const std::vector<Flow>& GetActiveFlows() const;

    // Get total demand from node src to node dst (summed over all flows).
    double GetDemand(uint32_t src, uint32_t dst) const;

private:
    SimConfig m_cfg;
    std::vector<Flow> m_flows;
    ns3::Ptr<ns3::UniformRandomVariable> m_rng;
    ns3::Ptr<ns3::ExponentialRandomVariable> m_expRng;
};
```

### Traffic models in detail

#### Constant rate (simplest, start here)

Every pair of nodes has a fixed demand. This is analogous to `mmwave-sim`'s `UdpClient` with constant inter-packet interval.

```
demand_mbps = config.traffic.demand_mbps  (e.g. 10 Mbps per flow)
For all_pairs with N=5 nodes: 5*4/2 = 10 bidirectional flows, each 10 Mbps.
```

**Reference**: `mmwave-sim/src/setup/traffic-setup.cc` line 54 sets `client.SetAttribute("Interval", ...)` which produces constant-rate UDP.

#### Poisson arrivals with holding times

Flows arrive randomly and last for a finite time. This is the standard teletraffic model (Erlang model).

```
- New flows arrive at rate lambda (arrival_rate_hz)
- Each flow lasts for holding_time_s (exponentially distributed)
- Source and destination are chosen uniformly at random
```

**Reference**: This is the standard M/M/c queuing model used in telecom capacity planning. Not directly in ns-3 but the exponential RNG is: `src/core/model/random-variable-stream.h` (`ExponentialRandomVariable`).

#### On-off (bursty)

Each flow alternates between ON (transmitting at `demand_mbps`) and OFF (idle). This models bursty applications like video or web browsing.

**Reference**: ns-3's `OnOffApplication` in `src/applications/model/onoff-application.h` does exactly this. We reimplement the state machine without the packet-level machinery.

```
During ON period:  flow.active = true,  demand = demand_mbps
During OFF period: flow.active = false, demand = 0
ON duration:  exponentially distributed with mean on_time_s
OFF duration: exponentially distributed with mean off_time_s
```

### Holding times

**What is a holding time?** How long a communication session (flow) lasts before it ends. A phone call might last 3 minutes. A video stream might last 30 minutes. A sensor report might last 0.1 seconds.

```cpp
// In TrafficMatrix::Tick():
for (auto& flow : m_flows) {
    // Check if flow has expired
    if (flow.end_time_s > 0.0 && currentTime >= flow.end_time_s) {
        flow.active = false;
        // Mark for removal or replacement
    }
}
```

When `holding_time_s = 0`, flows are permanent (always-on). This is the default and matches `mmwave-sim`'s behavior where `UdpClient` sends for the entire simulation.

---

## 7. Multi-Node Connections and Time-Varying Links

### How connections work in cellular (mmwave-sim)

In mmwave-sim, each UE connects to **exactly one eNB** at a time. This is hardwired:

```cpp
// src/mmwave/model/mmwave-ue-net-device.h
void SetTargetEnb(Ptr<MmWaveEnbNetDevice> enb);  // exactly one target
Ptr<MmWaveEnbNetDevice> GetTargetEnb();
```

The eNB's centralized scheduler (`MmWaveFlexTtiMacScheduler` in `src/mmwave/model/mmwave-flex-tti-mac-scheduler.h`) allocates time-frequency resources to all attached UEs. A UE transmits/receives **only when the scheduler grants it resources**.

### How connections work in our mesh model

**There is no "connection" in the cellular sense.** Instead:

1. **Every node can potentially communicate with every other node.** Whether the link is usable depends on whether the SINR is above a threshold (e.g., -5 dB, meaning the signal is still decodable).

2. **The link table evaluates all pairs every tick.** If a link's capacity drops to zero (SINR below threshold), it's effectively "disconnected" -- the router won't use it.

3. **Connections change over time** because:
   - Nodes move (position changes -> path loss changes -> SINR changes).
   - LOS/NLOS condition changes (a node moves behind a building).
   - Shadow fading realizations change (random component of path loss).
   - Traffic demands change (flows start/stop).

### One node connected to five others

In the cellular model, one eNB connects to multiple UEs by **time-division multiplexing** (TDMA). The scheduler gives each UE a fraction of the available time slots. Each UE gets `capacity / N_ues` throughput.

In our mesh model, a node with links to 5 neighbors faces the same sharing problem. We model this as:

```
Node A has links to B, C, D, E, F.
Link capacities: A-B = 100 Mbps, A-C = 50 Mbps, etc.

Traffic demands:
  A->B: 20 Mbps, A->C: 30 Mbps, A->D: 10 Mbps

Node A's total TX capacity is its bandwidth / number of active TX links.
(Alternatively: A can transmit to only one neighbor at a time,
 and time-shares among them.)
```

**How we model this (TDMA sharing)**:

```cpp
// In MeshRouter, after routing all flows:
// For each node, compute the total demand it must transmit.
// If total demand exceeds the node's TX budget, scale down proportionally.

double nodeTxBudget = maxSingleLinkCapacity;  // e.g., 500 Mbps
double totalDemand = sum of all flows where this node is TX;
if (totalDemand > nodeTxBudget) {
    double scale = nodeTxBudget / totalDemand;
    for each flow through this node:
        flow.delivered_mbps *= scale;
}
```

**Reference for TDMA sharing**: This is how `MmWaveFlexTtiMacScheduler` works (`src/mmwave/model/mmwave-flex-tti-mac-scheduler.h`) -- it divides slot time among UEs. We approximate the same effect with proportional scaling.

### What measurements look like with 5 connections

Each tick, the output for node A would include:

```
links.csv:
time_s, node_a, node_b, dist_m, sinr_db, condition, capacity_mbps
1.0,    0,      1,      150.0,  15.3,    LOS,       200.5
1.0,    0,      2,      300.0,  5.1,     NLOS,      50.2
1.0,    0,      3,      100.0,  20.1,    LOS,       350.0
1.0,    0,      4,      500.0,  -2.3,    NLOS,      0.0    <-- below threshold
1.0,    0,      5,      200.0,  10.5,    LOS,       120.0
```

Node 0 has 4 usable links (SINR > threshold) and 1 dead link (node 4 is too far / blocked).

---

## 8. Routing

### Why routing matters and how to not punish RL

The RL agent controls **node positions**. It does NOT control routing. So the routing algorithm must be **deterministic and optimal given the current link state**. If routing makes a bad choice, the RL agent gets a bad reward for something it can't control. That's unfair.

**Solution**: Use the shortest-path algorithm on the current link graph. This is the optimal single-path routing for the current topology. The RL agent can then learn that moving nodes to create better links leads to better routing outcomes.

### MeshRouter class

```cpp
// src/routing/mesh-router.h
#pragma once
#include "src/domain/link-result.h"
#include "src/domain/mesh-config.h"
#include "src/eval/link-table.h"
#include "src/traffic/traffic-matrix.h"
#include <vector>

namespace mesh_sim {

struct FlowResult {
    uint32_t src;
    uint32_t dst;
    double   demand_mbps;       // what the flow wanted
    double   delivered_mbps;    // what it actually got (after congestion)
    double   latency_ms;        // sum of per-hop latency proxy
    uint32_t hop_count;         // number of wireless hops
    std::vector<uint32_t> path; // node indices along the route
    bool     routable;          // false if no path exists (disconnected)
};

class MeshRouter {
public:
    MeshRouter(const RoutingConfig& cfg);

    // Given current link table and traffic demands, compute routes and
    // delivered throughput for all flows.
    std::vector<FlowResult> Route(const LinkTable& links,
                                  const std::vector<Flow>& flows,
                                  uint32_t numNodes) const;
private:
    RoutingConfig m_cfg;

    // Dijkstra shortest path on the link graph.
    // Edge weight depends on m_cfg.algorithm:
    //   "shortest_path":  weight = 1 / capacity_mbps  (minimize total cost)
    //   "max_throughput":  weight = -capacity_mbps    (widest path via negation)
    //   "min_hop":         weight = 1                 (fewest hops)
    std::vector<uint32_t> FindPath(const LinkTable& links,
                                   uint32_t src, uint32_t dst,
                                   uint32_t numNodes) const;
};

} // namespace mesh_sim
```

### Routing algorithms

#### Shortest-path (Dijkstra) -- default

Edge weight = `1.0 / capacity_mbps`. Finds the path that minimizes total "cost" (inverse capacity). This naturally prefers high-capacity links.

**Reference**: Dijkstra's algorithm is standard. The weight choice mirrors OSPF's link cost = reference_bandwidth / link_bandwidth, used in real networks.

#### Max-throughput (widest path)

Finds the path where the **minimum-capacity link** (bottleneck) is maximized. Implemented as a modified Dijkstra where we track the bottleneck capacity instead of summing weights.

#### Min-hop

Weight = 1 for all links. Finds the path with fewest hops. Useful when you want to minimize latency and don't care about capacity.

### Latency proxy

We don't simulate packet queuing. Instead:

```cpp
// Per-hop latency = propagation delay + fixed processing delay
double propagation_delay_ms = distance_m / 3e8 * 1000.0;  // speed of light
double processing_delay_ms = 0.5;  // ~ half-slot at 60 kHz SCS (0.25 ms) + overhead
double per_hop_latency_ms = propagation_delay_ms + processing_delay_ms;

// Flow latency = sum over all hops
flow.latency_ms = sum of per_hop_latency for each hop in path;
```

**Reference for processing delay**: One subframe at numerology-2 (60 kHz SCS) is 0.25 ms. With DL+UL + processing, ~0.5 ms per hop is a reasonable lower bound. See `MmWavePhyMacCommon` in `src/mmwave/model/mmwave-phy-mac-common.h` for symbol/slot timing.

### Congestion / capacity sharing

After routing all flows, some links may be overloaded (total demand > capacity). We handle this with **proportional fairness**:

```cpp
for each link (i,j):
    total_demand = sum of demand_mbps for all flows using link (i,j)
    if total_demand > link.capacity_mbps:
        scale = link.capacity_mbps / total_demand
        for each flow using this link:
            flow.delivered_mbps = min(flow.delivered_mbps, flow.demand_mbps * scale)
```

This is analogous to how `MmWaveFlexTtiMacScheduler` divides resources. The RL agent is not punished for routing -- it's punished for creating topologies where links get congested.

---

## 9. Uplink vs. Downlink

### In cellular (mmwave-sim)

Uplink (UE -> eNB) and downlink (eNB -> UE) are fundamentally different:
- **Downlink**: eNB transmits at high power (30 dBm), UE receives.
- **Uplink**: UE transmits at lower power (23 dBm), eNB receives.
- Time is divided into DL slots and UL slots by the scheduler (`src/mmwave/model/mmwave-phy-mac-common.h`: `GetDlCtrlSymbols()`, `GetUlCtrlSymbols()`).
- `mmwave-sim/src/setup/traffic-setup.cc` installs separate DL and UL applications.

### In our mesh model

**There is no uplink/downlink distinction.** All nodes are peers with identical radios. Every node transmits and receives at the same power.

However, we model **directional flows**. A flow from node A to node B is different from B to A:
- They may take different routes (asymmetric link quality due to different interference environments).
- They share the node's TX budget separately.

For the traffic matrix, `flow_topology = "all_pairs"` creates **bidirectional** flows (A->B and B->A). `flow_topology = "gateway"` creates flows in both directions (to gateway and from gateway), modeling a scenario where one node aggregates/distributes data.

**If you later need UL/DL asymmetry** (e.g., one node has a bigger antenna or higher TX power), we can add a per-node `tx_power_dbm` field to `NodeSpec`. The `LinkEvaluator` would then use different TX powers for different source nodes.

---

## 10. The Step Loop -- How One Tick Works

```
for t = 0.0 to duration_s, step tick_s:

    1. [RL agent moves nodes]  (placeholder: no-op in v1)
       positions = rlAgent.GetActions(currentState);
       for each node i:
           mob[i]->SetPosition(positions[i]);
       BuildingsHelper::MakeMobilityModelConsistent();

    2. [Evaluate all links]
       linkTable.Update(linkEvaluator.EvaluateAll(mobs));

    3. [Update traffic]
       trafficMatrix.Tick(t);

    4. [Route flows]
       flowResults = meshRouter.Route(linkTable, trafficMatrix.GetActiveFlows(), N);

    5. [Compute metrics]
       For each flow: throughput, latency, hop count
       Network-level: sum throughput, min throughput, connectivity, outage count

    6. [Write output]
       if (t >= warmup_s):
           vizWriter.WriteTick(t, nodes, linkTable);
           accumulateMetrics(flowResults);

    7. [Advance time]
       t += tick_s;

After loop:
    metricsWriter.Write(summary.json);
```

### Why we don't need Simulator::Run()

The ns-3 event scheduler (`Simulator::Run()`) is needed when you have asynchronous events: packets arriving, timers firing, protocol state machines advancing. We have none of that. Our model is **synchronous**: every tick, we evaluate the world state, route traffic, and record metrics. A simple `for` loop suffices.

The only ns-3 functionality we use is:
- `MobilityModel::SetPosition()` / `GetPosition()` -- no scheduler needed.
- `PropagationLossModel::CalcRxPower()` -- pure function, no scheduler needed.
- `ChannelConditionModel::GetChannelCondition()` -- pure function.
- `BuildingsHelper::MakeMobilityModelConsistent()` -- updates building tags, no scheduler.
- `RngSeedManager` -- sets seed, no scheduler.

---

## 11. Beamforming, AMC, SINR -- In Depth

### Beamforming: What Happens in ns3-mmwave

When a signal is transmitted in the full stack, the `ThreeGppSpectrumPropagationLossModel` applies beamforming gain as part of the spectrum-level channel computation:

1. **Beamforming vectors are set** by `MmWaveDftBeamforming::SetBeamformingVectorForDevice()` (`src/mmwave/model/mmwave-beamforming-model.cc` lines 142-171):
   ```cpp
   Angles completeAngle(bPos, aPos);  // angle from A to B
   auto antennaWeights = antenna->GetBeamformingVector(completeAngle);
   antenna->SetBeamformingVector(antennaWeights);
   ```

2. **The steering vector** is computed by `PhasedArrayModel::GetSteeringVector()` (`src/antenna/model/phased-array-model.cc` lines 115-129):
   ```
   For each antenna element i at position (x_i, y_i, z_i) in wavelengths:
     phase = -2*pi * (sin(elev)*cos(azim)*x_i + sin(elev)*sin(azim)*y_i + cos(elev)*z_i)
     steeringVector[i] = exp(j * phase)
   ```

3. **The long-term component** combines beamforming with the channel matrix (`src/spectrum/model/three-gpp-spectrum-propagation-loss-model.cc` lines 155-206):
   ```
   LT[cluster] = w_rx^H * H[cluster] * w_tx
   ```

4. **The received PSD** is computed from the long-term, delays, and Doppler (lines 341-427):
   ```
   H_freq[rb] = sqrt(PSD_in[rb]) * sum_c( LT[c] * exp(-j*2*pi*f_rb*delay[c]) * doppler[c] )
   PSD_out[rb] = |H_freq[rb]|^2
   ```

### What we implement (simplified)

For v1, we use the **ideal beamforming gain approximation**:

```cpp
// Maximum array gain for an N-element array with DFT beamforming:
// G = N (in linear), or 10*log10(N) dB.
// This is the theoretical maximum when the beam is perfectly steered.
// Reference: phased-array-model.cc lines 97-112 where the BF vector
// is normalized by sqrt(NumPorts), giving total array power = N.

struct AntennaConfig {
    uint32_t rows = 4;     // antenna rows per node
    uint32_t cols = 4;     // antenna columns per node
    // Total elements = rows * cols = 16
    // Max gain = 10*log10(16) = 12.04 dBi
};

double ComputeBfGain(const AntennaConfig& ant) {
    return 10.0 * std::log10(static_cast<double>(ant.rows * ant.cols));
}

// Applied in LinkEvaluator::Evaluate():
// Both TX and RX steer toward each other (ideal alignment).
double totalBfGain_dB = ComputeBfGain(txAntenna) + ComputeBfGain(rxAntenna);
double effectiveRxPower = rxPower_dBm + totalBfGain_dB;
```

**When to upgrade**: If results are too optimistic (all links look great), add a **misalignment penalty** of 3-5 dB, or use the actual ns-3 `PhasedArrayModel` to compute steering vectors and evaluate gain at the actual angle (more accurate but slower).

### AMC: SINR to Throughput

**What ns3-mmwave does**: `MmWaveAmc` (`src/mmwave/model/mmwave-amc.h`) maps SINR to MCS to transport block size. Two modes:
- `ShannonModel`: throughput = BW * log2(1 + SINR_linear) (theoretical upper bound)
- `ErrorModel`: uses actual MCS table with TBLER threshold

**What we implement**:

```cpp
double SinrToCapacity(double sinr_dB, double bandwidth_hz, const std::string& model) {
    if (model == "shannon") {
        // Shannon capacity: C = B * log2(1 + SNR)
        // Reference: Shannon-Hartley theorem.
        // Also used by MmWaveAmc::ShannonModel in src/mmwave/model/mmwave-amc.h
        double sinr_linear = std::pow(10.0, sinr_dB / 10.0);
        double capacity_bps = bandwidth_hz * std::log2(1.0 + sinr_linear);
        return capacity_bps / 1e6;  // Mbps
    }
    else {  // "table" -- simplified MCS lookup
        // 3GPP NR MCS table (Table 5.1.3.1-1 from TS 38.214)
        // Maps SINR thresholds to spectral efficiency.
        // Reference: the same table is embedded in MmWaveAmc::CreateCqiFeedbackWbTdma
        // in src/mmwave/model/mmwave-amc.cc
        struct McsEntry { double sinr_min_dB; double se_bps_hz; };
        static const McsEntry table[] = {
            {-6.7, 0.15},   // QPSK, code rate ~1/5
            {-4.7, 0.23},
            {-2.3, 0.38},
            { 0.2, 0.60},
            { 2.4, 0.88},
            { 4.3, 1.18},
            { 5.9, 1.48},   // 16QAM starts ~here
            { 8.1, 1.91},
            {10.3, 2.41},
            {11.7, 2.73},
            {14.1, 3.32},
            {16.3, 3.90},   // 64QAM starts ~here
            {18.7, 4.52},
            {21.0, 5.12},
            {22.7, 5.55},
        };
        double se = 0.0;
        for (const auto& e : table) {
            if (sinr_dB >= e.sinr_min_dB) se = e.se_bps_hz;
            else break;
        }
        return se * bandwidth_hz / 1e6;  // Mbps
    }
}
```

### SINR threshold for link viability

Below a certain SINR, no MCS can decode reliably. We set a minimum:

```cpp
static constexpr double SINR_MIN_DB = -6.7;  // lowest MCS entry from table above
// If sinr_dB < SINR_MIN_DB, link capacity = 0 (link is dead).
```

---

## 12. Output -- GUI Compatibility

We produce the **exact same CSV format** as `mmwave-sim` so the existing GUI works unchanged.

### positions.csv

```
# scenario=five-node-mesh
# frequency=28000000000
# txPower=30
# numNodes=5
# simDuration=10000
# tickMs=100
# dimensions=3
time_s,node_id,x,y,z,node_type,active
0.100000,0,0.000000,0.000000,60.000000,peer,1
0.100000,1,100.000000,0.000000,60.000000,peer,1
...
```

**Difference from mmwave-sim**: `node_type` is "peer" instead of "bs" or "air"/"ground". The GUI should handle this gracefully (or we can map to "air" if all nodes are UAVs).

### links.csv

```
time_s,node_a,node_b,dist_m,sinr_db,condition
0.100000,0,1,100.000000,15.300000,LOS
0.100000,0,2,300.000000,5.100000,NLOS
...
```

**Difference from mmwave-sim**: We write **all pairs**, not just eNB-UE pairs. For N=5 nodes, that's 10 rows per tick instead of mmwave-sim's 1-2 rows. The GUI will see a richer link graph.

**Additional columns** (we add these, GUI can ignore unknown columns):

```
time_s,node_a,node_b,dist_m,sinr_db,condition,capacity_mbps,delivered_mbps,hop_count
```

### summary.json

```json
{
  "scenario": "five-node-mesh",
  "seed": 42,
  "duration_s": 10.0,
  "warmup_s": 0.0,
  "wall_clock_start": "2026-03-27T14:00:00Z",
  "wall_clock_end": "2026-03-27T14:00:02Z",
  "wall_elapsed_s": 1.8,
  "per_node": {
    "node0": {
      "mean_sinr_db": 12.5,
      "min_sinr_db": -2.1,
      "max_sinr_db": 22.3,
      "num_links": 4,
      "num_los_links": 3,
      "tx_throughput_mbps": 45.2,
      "rx_throughput_mbps": 52.1
    }
  },
  "per_flow": {
    "node0->node1": {
      "demand_mbps": 10.0,
      "delivered_mbps": 10.0,
      "latency_ms": 0.83,
      "hop_count": 1
    },
    "node0->node3": {
      "demand_mbps": 10.0,
      "delivered_mbps": 8.5,
      "latency_ms": 1.66,
      "hop_count": 2
    }
  },
  "network": {
    "sum_throughput_mbps": 85.3,
    "mean_sinr_db": 10.2,
    "connectivity": 0.9,
    "mean_hop_count": 1.4,
    "flows_routed": 10,
    "flows_unroutable": 0
  }
}
```

---

## 13. Metrics Writer

```cpp
// src/io/metrics-writer.h  (NO ns-3 headers -- same rule as mmwave-sim)
class MetricsWriter {
public:
    explicit MetricsWriter(const SimConfig& cfg);
    void SetTiming(const TimingInfo& t);

    // Called each tick to accumulate metrics.
    void AccumulateTick(double time_s,
                        const std::vector<LinkResult>& links,
                        const std::vector<FlowResult>& flows,
                        uint32_t numNodes);

    // Write summary.json after simulation completes.
    void Write() const;

private:
    SimConfig m_cfg;
    TimingInfo m_timing;
    // Per-node accumulators (keyed by node index)
    struct NodeStats { double sinr_sum, sinr_min, sinr_max; uint64_t sinr_count; ... };
    std::map<uint32_t, NodeStats> m_nodeStats;
    // Per-flow accumulators
    struct FlowStats { double delivered_sum; uint64_t tick_count; ... };
    std::map<std::pair<uint32_t,uint32_t>, FlowStats> m_flowStats;
};
```

Unlike mmwave-sim's `MetricsWriter` which parses trace files post-hoc (`mmwave-sim/src/io/metrics-writer.cc` lines 149-243), ours accumulates data in-memory during the step loop. Same JSON output format.

---

## 14. RL Placeholder

```cpp
// src/rl/rl-agent.h
#pragma once
#include "src/domain/node-spec.h"
#include <vector>

namespace mesh_sim {

// Placeholder. Returns unchanged positions.
// Replace with actual RL agent (Python IPC via ZMQ, stdin/stdout, or shared memory).
class RlAgent {
public:
    // Called each tick with current state.  Returns new positions for all nodes.
    // Default implementation: return current positions unchanged.
    std::vector<Position> GetActions(
        const std::vector<Position>& currentPositions,
        const std::vector<LinkResult>& links,
        const std::vector<FlowResult>& flows)
    {
        return currentPositions;  // no-op
    }
};

} // namespace mesh_sim
```

---

## 15. Class/Object Inventory

### Complete list: every class, where it lives, implement vs. reuse

| # | Class | Header | Implement or Reuse | ns-3 dependency | Lines (est.) |
|---|-------|--------|-------------------|----------------|-------------|
| 1 | `Position` | `domain/node-spec.h` | **Reuse** from mmwave-sim | None | 5 |
| 2 | `Velocity` | `domain/node-spec.h` | **Reuse** from mmwave-sim | None | 5 |
| 3 | `RandomWalkParams` | `domain/node-spec.h` | **Reuse** from mmwave-sim | None | 5 |
| 4 | `NodeSpec` | `domain/node-spec.h` | **Reuse** (change `role` to "peer") | None | 10 |
| 5 | `BuildingSpec` | `domain/node-spec.h` | **Reuse** from mmwave-sim | None | 10 |
| 6 | `ChannelConfig` | `domain/channel-config.h` | **Reuse** from mmwave-sim | None | 20 |
| 7 | `NyuChannelConfig` | `domain/channel-config.h` | **Reuse** from mmwave-sim | None | 15 |
| 8 | `MeshConfig` | `domain/mesh-config.h` | **Implement** | None | 30 |
| 9 | `TrafficConfig` | `domain/mesh-config.h` | **Implement** | None | 20 |
| 10 | `RoutingConfig` | `domain/mesh-config.h` | **Implement** | None | 10 |
| 11 | `LinkResult` | `domain/link-result.h` | **Implement** | None | 15 |
| 12 | `Flow` | `traffic/traffic-matrix.h` | **Implement** | None | 12 |
| 13 | `FlowResult` | `routing/mesh-router.h` | **Implement** | None | 12 |
| 14 | `SimConfig` | `domain/sim-config.h` | **Adapt** from mmwave-sim | None | 25 |
| 15 | `TimingInfo` | `domain/sim-config.h` | **Reuse** from mmwave-sim | None | 5 |
| 16 | `LinkEvaluator` | `eval/link-evaluator.h/cc` | **Implement** | Yes: PropagationLossModel, ChannelConditionModel, MobilityModel | 200 |
| 17 | `LinkTable` | `eval/link-table.h/cc` | **Implement** | None | 150 |
| 18 | `TrafficMatrix` | `traffic/traffic-matrix.h/cc` | **Implement** | Yes: RandomVariableStream (for Poisson/on-off) | 200 |
| 19 | `MeshRouter` | `routing/mesh-router.h/cc` | **Implement** | None | 250 |
| 20 | `TopologyBuilder` | `setup/topology-builder.h/cc` | **Implement** (similar to mmwave-sim, no EPC) | Yes: MobilityHelper, Building, BuildingsHelper, PropagationLossModel, ChannelConditionModel | 250 |
| 21 | `VizWriter` | `io/viz-writer.h/cc` | **Implement** (simpler than mmwave-sim, no trace hooks) | None (reads from LinkTable directly) | 150 |
| 22 | `MetricsWriter` | `io/metrics-writer.h/cc` | **Implement** | None | 200 |
| 23 | `ProgressLogger` | `io/progress-logger.h` | **Reuse** from mmwave-sim | None | 20 |
| 24 | `ConfigLoader` | `config/config-loader.h/cc` | **Adapt** from mmwave-sim | None | 200 |
| 25 | `CliParser` | `cli/cli-parser.h/cc` | **Adapt** from mmwave-sim | Yes: ns3::CommandLine | 80 |
| 26 | `RlAgent` | `rl/rl-agent.h` | **Implement** (placeholder) | None | 30 |
| 27 | `IniParser` | `util/ini-parser.h/cc` | **Reuse** from mmwave-sim | None | 60 |
| 28 | `StringUtils` | `util/string-utils.h/cc` | **Reuse** from mmwave-sim | None | 40 |
| | **ns3::PropagationLossModel** | ns-3 core | **Call** | -- | -- |
| | **ns3::ChannelConditionModel** | ns-3 core | **Call** | -- | -- |
| | **ns3::ConstantPositionMobilityModel** | ns-3 core | **Call** | -- | -- |
| | **ns3::Building** | ns-3 buildings module | **Call** | -- | -- |
| | **ns3::BuildingsHelper** | ns-3 buildings module | **Call** | -- | -- |
| | **ns3::MmWavePhyMacCommon** | ns3-mmwave | **Call** (for config) | -- | -- |
| | **ns3::RngSeedManager** | ns-3 core | **Call** | -- | -- |

**Total new code to write**: ~1,500 lines of C++ (headers + implementations).
**Reused from mmwave-sim**: ~200 lines (domain types, util, progress logger).
**Called from ns-3**: ~10 classes used as library functions.

---

## 16. Build System

```cmake
# scratch/mesh-sim/CMakeLists.txt

# --- Source list ---
set(MESH_SIM_SOURCES
    sim.cc
    src/config/config-loader.cc
    src/util/string-utils.cc
    src/util/ini-parser.cc
    src/cli/cli-parser.cc
    src/setup/topology-builder.cc
    src/eval/link-evaluator.cc
    src/eval/link-table.cc
    src/traffic/traffic-matrix.cc
    src/routing/mesh-router.cc
    src/io/viz-writer.cc
    src/io/metrics-writer.cc
)

# --- Target ---
# Follow mmwave-sim convention: scratch_<dir>_<main-file>
set(target_name scratch_mesh-sim_sim)
add_executable(${target_name} ${MESH_SIM_SOURCES})

target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link ns-3 modules we actually need (subset of what mmwave-sim links):
#   core, network, mobility, propagation, buildings, spectrum, mmwave
# (mmwave only for MmWavePhyMacCommon and NYU models)
target_link_libraries(${target_name}
    ${ns3-core}
    ${ns3-network}
    ${ns3-mobility}
    ${ns3-propagation}
    ${ns3-buildings}
    ${ns3-spectrum}
    ${ns3-antenna}
    ${ns3-mmwave}      # for NYU models and MmWavePhyMacCommon
)
```

---

## 17. Implementation Order

### Phase 1: Skeleton (~1 hour)

1. Create directory structure (all folders, CLAUDE.md files).
2. Copy `third_party/json.hpp`, `util/ini-parser.h/cc`, `util/string-utils.h/cc` from mmwave-sim.
3. Copy `domain/node-spec.h`, `domain/channel-config.h` from mmwave-sim.
4. Write `domain/mesh-config.h`, `domain/link-result.h`, `domain/sim-config.h`.
5. Write `CMakeLists.txt`.
6. Write empty `sim.cc` that just parses CLI and loads config.
7. Verify it compiles.

### Phase 2: Link Evaluation (~2 hours)

8. Write `setup/topology-builder.h/cc` -- create nodes, mobility, buildings, propagation model. No EPC.
9. Write `eval/link-evaluator.h/cc` -- call `CalcRxPower()`, compute SINR, capacity.
10. Write `eval/link-table.h/cc` -- store NxN results.
11. Test: print link table for a 3-node scenario, verify SINR values make sense.

### Phase 3: Traffic & Routing (~2 hours)

12. Write `traffic/traffic-matrix.h/cc` -- start with constant-rate all-pairs.
13. Write `routing/mesh-router.h/cc` -- Dijkstra shortest path.
14. Integrate into step loop in `sim.cc`.
15. Test: verify flows are routed, multi-hop works for blocked links.

### Phase 4: Output (~1.5 hours)

16. Write `io/viz-writer.h/cc` -- positions.csv + links.csv.
17. Write `io/metrics-writer.h/cc` -- summary.json.
18. Copy `io/progress-logger.h` from mmwave-sim.
19. Write `config/config-loader.h/cc` -- adapted from mmwave-sim.
20. Write `cli/cli-parser.h/cc` -- adapted from mmwave-sim.
21. Write `rl/rl-agent.h` -- placeholder.

### Phase 5: Scenarios & Validation (~1.5 hours)

22. Create `inputs/scenarios/triangle/` (3 nodes, no buildings).
23. Create `inputs/scenarios/five-node-mesh/` (5 nodes, 1 building).
24. Run mesh-sim, examine CSV and JSON output.
25. Compare SINR at same distance/scenario against mmwave-sim's `links.csv`.

---

## 18. Equation Reference Sheet

Every equation we implement, with its source.

### Path Loss (3GPP TR 38.901, implemented in ns-3)

We **call** these -- we don't reimplement them. But for reference:

**UMi Street Canyon LOS** (d_2D <= d_BP):
```
PL = 32.4 + 21.0*log10(d_3D) + 20.0*log10(f_c)    [dB]
```
**Source**: `src/propagation/model/three-gpp-propagation-loss-model.cc`, `ThreeGppUmiStreetCanyonPropagationLossModel::GetLossLos()`

**UMi Street Canyon NLOS**:
```
PL = max(PL_LOS, 22.4 + 35.3*log10(d_3D) + 21.3*log10(f_c) - 0.3*(h_UT - 1.5))    [dB]
```
**Source**: same file, `GetLossNlos()`

**Shadow fading**: log-normal, sigma depends on LOS/NLOS and scenario.
- UMi LOS: sigma = 4.0 dB
- UMi NLOS: sigma = 7.82 dB
**Source**: `three-gpp-propagation-loss-model.cc`, `GetShadowingStd()` and `GetShadowingCorrelationDistance()`

### Noise Floor (our implementation)

```
N_dBm = -174 + 10*log10(B_Hz) + NF_dB
```
- -174 dBm/Hz = thermal noise density at T=290K = 10*log10(k_B * T * 1000)
- B_Hz = bandwidth in Hz (e.g., 800 MHz = 800e6)
- NF_dB = noise figure (e.g., 5 dB)
- Example: -174 + 89.0 + 5.0 = -80.0 dBm

**Source for -174**: Standard thermal noise, also in `src/spectrum/model/wifi-spectrum-value-helper.cc` and mmwave's noise PSD creation.

### SINR (our implementation, simplified from ns3-mmwave)

**Without interference** (v1):
```
SINR_dB = rxPower_dBm + bfGain_dB - noiseFloor_dBm
```

**With interference** (v2):
```
SINR = P_rx / (sum_k(P_interferer_k) + N)
SINR_dB = 10*log10(SINR)
```

**Source**: `src/mmwave/model/mmwave-interference.cc` line ~197:
```cpp
// SINR = m_rxSignal / ((m_allSignals - m_rxSignal) + m_noise)
```

### Beamforming Gain (our implementation, simplified)

```
G_dB = 10*log10(N_elements)    per antenna array
Total = G_tx + G_rx
```

**Source**: `src/antenna/model/phased-array-model.cc` lines 97-112. The BF vector normalization divides by `norm/sqrt(NumPorts)`, so the total array power gain equals `NumPorts` (which equals `N_elements` for single-panel arrays).

### Shannon Capacity (our implementation)

```
C = B * log2(1 + SNR_linear)    [bps]
SNR_linear = 10^(SINR_dB / 10)
```

**Source**: Shannon-Hartley theorem. Also the `ShannonModel` branch in `MmWaveAmc` (`src/mmwave/model/mmwave-amc.h`).

### MCS Table (our implementation, derived from 3GPP)

See the table in Section 11. Derived from 3GPP TS 38.214 Table 5.1.3.1-1, same data used by `MmWaveAmc::CreateCqiFeedbackWbTdma()` in `src/mmwave/model/mmwave-amc.cc`.

### Latency Proxy (our implementation)

```
latency_per_hop = distance / c + T_proc
c = 3e8 m/s
T_proc = 0.5 ms  (one subframe at 60 kHz SCS + processing)
```

**Source for T_proc**: `MmWavePhyMacCommon::GetSymbolPeriod()` at numerology-2 gives ~4.17 us per symbol, 14 symbols per slot = 0.25 ms per slot. A TX-RX turnaround is ~2 slots = 0.5 ms. See `src/mmwave/model/mmwave-phy-mac-common.h`.

### Routing: Dijkstra Weight

```
weight(i,j) = 1.0 / capacity_mbps(i,j)
```

**Source**: OSPF link cost = reference_bandwidth / interface_bandwidth. Standard network engineering practice.
