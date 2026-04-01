# Baseline Validation Scenarios

16 scenarios that progressively test each layer of the simulator.  Every
scenario changes **one knob** from scenario 01 (the baseline) so results are
easy to interpret and compare.

## Baseline configuration (scenario 01)

| Parameter | Value | Why |
|-----------|-------|-----|
| Nodes | 2, fixed, 100m apart, z=10m | Simplest possible topology |
| Scenario | RMa (Rural Macrocell) | Near-100% LOS probability at 100m, lowest shadow fading |
| Channel model | 3GPP | Standard reference model |
| Traffic | constant, 10 Mbps, all_pairs | Simple, predictable demand |
| Routing | shortest_path, max_hops=5 | Default routing |
| Frequency | 28 GHz | Standard mmWave band |
| TX power | 30 dBm | Standard |
| Bandwidth | 400 MHz | Standard NR bandwidth |

---

## Scenario table

| # | Name | Knob changed | Nodes | Expected output shape |
|---|------|-------------|-------|-----------------------|
| 01 | static-los-baseline | (baseline) | 2 | All flat lines |
| 02 | distance-sweep | mobility (node moves away) | 2 | Smooth SINR decline, MCS staircase |
| 03 | building-blockage | buildings added | 2 | Sharp SINR step at LOS transition |
| 04 | nyu-vs-3gpp | channel_model = nyu | 2 | Flat lines (diff vs 05 for rain) |
| 05 | nyu-atmosphere | atmospheric + rain loss | 2 | Flat lines, lower SINR than 04 |
| 06 | mcs-table-mode | amc_model = table | 2 | Flat lines, lower capacity than 01 |
| 07 | three-node-relay | 3 nodes (multi-hop) | 3 | Flat, latency stacking visible |
| 08 | max-throughput-routing | algorithm = max_throughput | 4 | Flat, 2-hop path preferred |
| 09 | congestion-bottleneck | buildings + high demand | 4 | Flat, flows equally capped |
| 10 | on-off-traffic | traffic model = on_off | 2 | SINR flat, throughput square-wave |
| 11 | poisson-arrivals | traffic model = poisson | 3 | Flows appear/disappear over time |
| 12 | gateway-topology | flow_topology = gateway | 4 | 3 flat lines converging on gateway |
| 13 | colocation-guard | distance = 0.5m | 2 | Flat, absurdly high SINR (~137 dB) |
| 14 | max-hops-limit | max_hops=3, low TX power | 5 | Some flows unroutable |
| 15 | umi-los-probability | scenario = UMi | 2 | Seed-dependent (teaching scenario) |
| 16 | random-walk-urban | everything (integration) | 3 | Noisy, NLOS/LOS transitions |

---

## How to interpret the output graphs

### SINR time-series (dB)

Signal-to-interference-plus-noise ratio for each link at each tick.

| Shape | Meaning |
|-------|---------|
| Flat line | Static geometry, stable channel (expected for 01, 04-06, 10, 12-13) |
| Smooth decline | Node moving away, path loss increasing (expected for 02) |
| Sharp step | LOS/NLOS transition from building blockage (expected for 03) |
| Noisy/jittery | Probabilistic channel or random mobility (expected for 15, 16) |

Higher SINR = better link.  Below about -6.7 dB the link is dead (no usable
MCS).  RMa LOS at 100m gives ~42-50 dB SINR with the baseline TX power.

### Capacity time-series (Mbps)

Maximum data rate the link can support at the current SINR.

- **Shannon mode** (`amc_model = shannon`): theoretical upper bound,
  `C = B * log2(1 + SNR)`.  At SINR ~46 dB with 400 MHz BW: ~6100 Mbps.
- **Table mode** (`amc_model = table`): uses 3GPP MCS table.  At CQI 14
  (highest): 5.55 bits/s/Hz * 400 MHz = 2220 Mbps.  ~60% of Shannon.

Capacity follows the same shape as SINR (they are directly related).

### MCS / CQI index

Modulation and coding scheme index (0-14).  Higher = faster but requires
better SINR.

| Shape | Meaning |
|-------|---------|
| Flat at 14 | Link is strong, using highest MCS (expected for most static scenarios) |
| Staircase descent | SINR falling through CQI thresholds (expected for 02) |
| Drops to 0 | Link at minimum MCS, barely alive |

### Flow throughput (Mbps)

Actual data delivered per flow per tick.  Throughput = min(demand, capacity
share).

| Shape | Meaning |
|-------|---------|
| Flat at demand level | Link has plenty of capacity, demand fully met |
| Flat below demand | Congestion — shared bottleneck limits throughput (expected for 09) |
| Square wave | ON/OFF traffic toggling (expected for 10) |
| Lines appearing/disappearing | Poisson arrivals with holding times (expected for 11) |
| Zero | Flow unroutable or link dead |

### Latency (ms)

Per-flow latency.  In this simulator latency is modeled as 0.5 ms per hop
(processing delay) plus negligible propagation delay.

- 1-hop: ~0.5 ms
- 2-hop: ~1.0 ms
- 3-hop: ~1.5 ms

The latency bar chart is most interesting for multi-hop scenarios (07, 08, 14).

### Geometry (distance, azimuth, elevation)

Spatial relationship between node pairs over time.

| Shape | Meaning |
|-------|---------|
| Flat | Static topology, no mobility |
| Linear increase | Node moving away at constant speed (expected for 02) |
| Smooth curves | Random walk or constant velocity at an angle (expected for 03, 16) |

---

## Scenario categories

### A. Physics (01-06)
Test the propagation and link-budget pipeline in isolation.  All use 2 nodes,
constant traffic, and shortest-path routing.

### B. Routing (07-09)
Test multi-hop routing, algorithm selection, and congestion.  Physics are kept
simple (RMa, stable links) so routing behavior is the only variable.

### C. Traffic (10-12)
Test traffic models and flow topologies.  Physics and routing are kept simple
so throughput patterns reflect only the traffic model.

### D. Edge cases (13-14)
Test boundary conditions: co-location guard and hop-limit rejection.

### E. Environment and integration (15-16)
Scenario 15 teaches about probabilistic LOS in UMi.  Scenario 16 combines
mobility, buildings, gateway traffic, and relay into one integration test.

---

## Tips

- **Start with 01**.  If 01 doesn't produce flat lines, something is wrong
  with the build or the simulator.
- **Diff scenarios**.  Compare 04 vs 05 to see rain penalty.  Compare 01 vs
  15 to see how UMi differs from RMa.  Compare 06 vs 01 to see Shannon vs
  table capacity.
- **Vary the seed**.  Most scenarios use `seed = 1`.  Try different seeds to
  see how shadow fading affects results.  Scenario 15 is specifically designed
  to show seed-dependent behavior.
- **Read the run.ini header**.  Every scenario has a detailed header comment
  explaining the purpose, geometry, link budget math, and expected behavior.
