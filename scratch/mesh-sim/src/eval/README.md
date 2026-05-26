@page src_eval src/eval

@brief Computes radio link quality for every node pair each simulation tick.  

Given the current node positions and ns-3 propagation models, this module
produces path loss, SINR, capacity, and MCS for each link, then stores the
results in a symmetric matrix for O(1) lookup by the routing and metrics layers.


## Output

The eval module produces no files. It outputs two in-memory data structures
each tick:

**`std::vector<LinkResult>`** — returned by `LinkEvaluator::EvaluateAll`.
One entry per unordered node pair (N·(N−1)/2 total), in (i < j) index order.

**`LinkTable`** — populated by `LinkTable::Update` from the vector above.
A symmetric N×N matrix; `Get(i, j)` and `Get(j, i)` return the same result.

| `LinkResult` field | Description |
|--------------------|-------------|
| `tx_id`, `rx_id` | Node indices for this directed link. |
| `distance_m` | 3-D Euclidean distance between the two nodes (m). |
| `is_los` | `true` if the link is Line-of-Sight. |
| `path_loss_db` | Total path loss including shadow fading (dB). |
| `rx_power_dbm` | Received power after TX power and beamforming gain (dBm). |
| `sinr_db` | Signal quality vs thermal noise floor (dB); −999.0 sentinel if invalid. |
| `capacity_mbps` | Link capacity from Shannon or MCS table (Mbps). |
| `mcs_index` | CQI index [0–14] selected by the AMC model. |
| `condition_from_buildings` | `true` if LOS/NLOS was determined deterministically by buildings. |


## Module Layout

| File | Description |
|------|-------------|
| `link-evaluator.h` | `LinkEvaluator` class: wraps ns-3 propagation models, computes per-link SINR and capacity. |
| `link-evaluator.cc` | `Configure`, `Evaluate` (single link), and `EvaluateAll` (all pairs) implementations. |
| `sinr-capacity.h` | Header-only: MCS table (3GPP TS 38.214), `SinrToMcsIndex`, `SinrToCapacity`. No ns-3 dependency. |
| `link-table.h` | `LinkTable` class: symmetric N×N matrix with O(1) lookup and aggregate queries. |
| `link-table.cc` | `Update`, `Get`, `MaxCapacity`, `ConnectedLinkCount`, `IsConnected` implementations. |


## Conventions

- **No inter-node interference.** mmWave beams are assumed orthogonal so SINR
  is computed as received signal power vs thermal noise floor only — no ICI term.
  The noise floor is: `−174 + 10·log₂(bandwidth_Hz) + noise_figure_dB`.

- **Beamforming gain is applied symmetrically.** `tx_array_gain_dBi + rx_array_gain_dBi`
  is added to `rx_power_dbm` as a single combined gain. Both sides use the
  same antenna model.

- **Co-location guard at 1 m.** Links with `distance_m < 1.0` bypass the
  propagation model entirely (full TX power, zero path loss) to prevent
  `log10(0)` in distance-dependent models.

- **`EvaluateAll` evaluates upper triangle only.** Only pairs (i, j) with
  i < j are evaluated. `LinkTable::Update` then mirrors each result to (j, i)
  so all lookups are symmetric. This halves the number of propagation model
  calls per tick.

- **SINR sentinel is −999.0.** Links that are unevaluated or below the noise
  floor carry `sinr_db = −999.0`. Consumers check `sinr_db > −900.0` to
  filter these out rather than using a boolean flag.

- **Connectivity threshold is −6.7 dB.** This matches `SINR_MIN_DB` in
  `sinr-capacity.h` — the minimum SINR for any MCS entry. Links below this
  have zero capacity and are treated as disconnected by the routing layer.

- **`LinkEvaluator::Configure` must be called before any evaluation.** It
  caches the noise floor and beamforming gain derived from `SimConfig` and
  stores the propagation model pointers. Calling `Evaluate` or `EvaluateAll`
  without configuring first is a null pointer dereference.


## Dependencies

| Dependency | Reason |
|------------|--------|
| `ns3/propagation-loss-model.h` | `CalcRxPower` called per link to compute received signal strength. |
| `ns3/channel-condition-model.h` | LOS/NLOS condition queried per link for `is_los` and shadow fading. |
| `ns3/mobility-model.h` | `GetDistanceFrom` used for distance; positions read for co-location guard. |