@page src_eval src/eval

@brief Computes radio link quality for every node pair each simulation tick.

Given the current node positions and ns-3 propagation models, this module
produces path loss, SINR (signal-to-interference-plus-noise ratio), capacity,
and MCS (modulation and coding scheme) index for each link. It stores the
results in a symmetric matrix for O(1) lookup by the routing, metrics, and RL
layers. In the sub-6 band it also adds jammer power (see @ref src_jammer).

## Module Layout

| File | Role |
|------|------|
| `link-evaluator.h` / `.cc` | `LinkEvaluator`: `Configure`, `Evaluate` (one link), `EvaluateAll` (all pairs). Wraps ns-3 propagation and channel-condition models. |
| `sinr-capacity.h` | Header-only, no ns-3: MCS tables (3GPP-style and Silvus), `SinrToMcsIndex`, `SinrToSilvusMcsIndex`, `SinrToCapacity`. |
| `link-table.h` / `.cc` | `LinkTable`: symmetric N×N matrix with `Update`, `Get`, `MaxCapacity`, `ConnectedLinkCount`, `IsConnected`. |
| `CLAUDE.md` | Scope notes and invariants for contributors. |

## Run

This module is a library used by `sim.cc`; it has no executable of its own.
Its standalone unit tests (no ns-3 needed) run from `scratch/mesh-sim/`:

```bash
make -C tests/unit/eval test
```

The tests cover `sinr-capacity.h` and `link-table.cc`. `link-evaluator.cc`
needs ns-3 and is exercised only by the full simulator.

## Output

No files are written. Two in-memory structures are produced each tick.

| Step | Where |
|------|-------|
| `LinkEvaluator::EvaluateAll` returns `std::vector<LinkResult>`: one entry per unordered pair, N·(N−1)/2 total, in (i < j) order. | Passed to `LinkTable::Update` in `sim.cc`. |
| `LinkTable::Update` builds a symmetric N×N matrix; `Get(i, j)` equals `Get(j, i)`. | Read by `routing/`, `io/`, and `rl/`. |

`LinkResult` (defined in `src/domain/link-result.h`) fields:

| Field | Description |
|-------|-------------|
| `tx_id`, `rx_id` | Node indices for this link (`tx_id < rx_id` from `EvaluateAll`). |
| `distance_m` | 3-D Euclidean distance (m). |
| `is_los` | `true` if Line-of-Sight. |
| `path_loss_db` | Path loss (dB): model value, floored at free-space loss. |
| `rx_power_dbm` | Received power after TX power, both array gains, and path loss (dBm). |
| `sinr_db` | SINR (dB). Default `-999.0` only for a default-constructed result (for example the `LinkTable` diagonal). |
| `capacity_mbps` | Capacity from the AMC (adaptive modulation and coding) model (Mbps). |
| `mcs_index` | MCS index chosen by the AMC model. |
| `condition_from_buildings` | `true` when buildings are configured (set for every link in that case). |

## Conventions

### Physics
- **Noise floor.** `-174 + 10·log10(bandwidth_Hz) + noise_figure_dB` (dBm).
- **No inter-node interference.** SINR is signal vs. noise floor. Only jammer power is added, and only in sub-6.
- **Array gains.** `rx_power_dbm` adds the TX node's `tx_array_gain_dbi` and the RX node's `rx_array_gain_dbi`. A per-node override in `nodes.json` wins over the channel default.
- **Free-space floor.** Path loss is never below free-space path loss at `max(distance, 1 m)`. This removes SINR spikes at short range and at `d = 0`.
- **Jammer handling.** Uses the larger of the jammer power at the RX and TX ends. When any jammer power is present, SINR is clamped to a minimum of 0 dB.

### Capacity models (`[channel] amc_model`)
- `shannon`: `B·log2(1+SINR)`.
- `table`: CQI spectral efficiency times bandwidth (16 entries, indices 0–15).
- `silvus`: Silvus single-stream MCS 0–6, scaled from 20 MHz to the configured bandwidth.
- Any SINR below `SINR_MIN_DB` (−6.7 dB) gives 0 Mbps.
- An unknown model name throws `std::runtime_error` from `SinrToCapacity`.

### Connectivity
- `LinkTable::IsConnected` and `ConnectedLinkCount` default to `sinr_db >= -6.7`, a literal that matches `SINR_MIN_DB`.
- `metrics-writer.cc` skips links with `sinr_db <= -900` as invalid (the `-999` sentinel).

### Call order
- `LinkEvaluator::Configure` must run before `Evaluate` or `EvaluateAll`. It throws `std::runtime_error` on a null model pointer.
- `EvaluateAll` evaluates only i < j; `LinkTable::Update` mirrors each result and throws `std::runtime_error` if the result count is not N·(N−1)/2.

## Dependencies

| Dependency | Used by | Reason |
|------------|---------|--------|
| `ns3/propagation-loss-model.h` | `link-evaluator` | `CalcRxPower` per link. |
| `ns3/channel-condition-model.h` | `link-evaluator` | LOS/NLOS per link. |
| `ns3/mobility-model.h` | `link-evaluator` | Distance and positions. |
| `ns3/log.h` | `.cc` files | `NS_LOG` debug output. |
| `src/domain/` (`link-result.h`, `sim-config.h`) | all | Shared POD types. |
| `src/jammer/jammer-model.h` | `link-evaluator` | Jammer interference. |
| g++ with C++17 | unit tests | `make -C tests/unit/eval test`. |
