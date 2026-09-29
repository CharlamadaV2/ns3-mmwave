@page src_jammer src/jammer

@brief Adds interference from jammer emitters to per-link SINR in the sub-6 band.

Without jammers, SINR is signal power minus the thermal noise floor. With
jammers, each active jammer's received power is added to the noise in the
SINR (signal-to-interference-plus-noise ratio) denominator:
`SINR = 10·log10(signalWatt / (noiseWatt + jamWatt))`. The model is active
only when `band == sub-6` and `jammers.json` defines at least one enabled
jammer. Jammers are looked up through `SimConfig::jammers`.

## Module Layout

| File | Role |
|------|------|
| @c src/jammer/jammer-spec.h | @ref mesh_sim::JammerSpec and `Interval`: one jammer's parameters (plain data, loaded from `jammers.json`). |
| @c src/jammer/jammer-model.h / .cc | @ref mesh_sim::JammerModel: total received jammer power (W) at a receiver, applying the gates below. |
| @c src/eval/link-evaluator.cc | Calls `JammerModel::InterfPowerAtReceiver` at both link ends and folds the larger into SINR. |
| @c src/config/config-loader.cc | Reads `jammers.json` into `cfg.jammers` (`parseJammerSpec`). |
| @c src/config/config-validator.cc | Validates `type`, `duty_cycle`, `beamwidth_deg`, and `intervals`. |
| @c src/setup/topology-builder.cc | Builds one ns-3 mobility model per jammer, in `cfg.jammers` order. |
| @c scripts/validation/make_jammers.py | Host-side generator: EW (electronic warfare) trials CSV to `jammers.json`. |

## Run

The model has no executable. It runs inside the simulator when the scenario
points at a jammers file and the band is sub-6. From `scratch/mesh-sim/`,
generate a `jammers.json` from field data (writes the file given by `-o`):

```bash
python3 -m scripts.validation.make_jammers \
    --trials <ew_trials.csv> --trace <gps_all_nodes_trace.csv> \
    -o <scenario_dir>/jammers.json --run-ini <scenario_dir>/run.ini
```

`--run-ini` is optional and adds `jammers_file = <name>` under `[scenario]`.
Then run the simulator with `--band=sub-6` (or `[channel] band = sub-6`).
Example scenarios with jammers: `inputs/baselines/p0-jammer-smoke/`, `inputs/calfex/1555-1559/`.

## Conventions

### Gates
Each tick, for each receiver, the model sums the received power of every
jammer that passes all gates, in this order:

-# **Time**: sim time is inside one of the jammer's `intervals` (half-open, empty means always on).
-# **Random burst**: for `type=random`, a seed-dependent on/off draw per whole second (probability `duty_cycle`). `type=constant` is always on.
-# **Frequency**: the link carrier is inside `target_freq` (empty means no filtering; also skipped if the carrier is unset).
-# **Range**: the receiver is within `max_range_m` (0 means unlimited).
-# **Beam**: the receiver is inside the 3-D cone from `azimuth_deg`, `zenith_deg`, `beamwidth_deg` (omni when `beamwidth_deg >= 360`).

A passing jammer contributes `CalcRxPower(tx_power_dbm + tx_array_gain_dbi)`
through the same propagation model as mesh links. Under 1 m it contributes its EIRP
(effective isotropic radiated power) directly. No receive-side gain is applied.

### duty_cycle
- `random`: at each whole second a seed-dependent draw decides whether the jammer is on **at full power**. At 0.5 it is on about half the seconds.
- `constant`: always on, but its power is multiplied by `duty_cycle`. At 0.5 it adds half the watts (about 3 dB less).
- At 0 neither type contributes; at 1 both contribute full power.

### jammers.json fields
Defaults are those applied by `ConfigLoader` when a key is absent.

| Field | Type | Meaning |
|-------|------|---------|
| `id` | string | Label (logs, validation, random-burst seed). Default empty. |
| `enabled` | bool | `false` drops the jammer in `JammerModel::Configure`. **Default `false`.** |
| `type` | string | `constant` (default) or `random`. |
| `target_freq` | number[] | Target band (MHz). Empty means all. 2 values = `[lo,hi]`; 1 value = spot ±2.5 MHz. |
| `tx_power_dbm` | number | Transmit power (dBm). Default 25. |
| `tx_array_gain_dbi` | number | Antenna gain (dBi), added to EIRP. Default 12. |
| `duty_cycle` | number | 0 to 1; meaning depends on `type` (see above). Default 1. |
| `max_range_m` | number | Range cutoff (m); 0 disables. Default 0. |
| `beamwidth_deg` | number | Full cone angle (deg), in (0, 360]; 360 is omni. Default 360. |
| `azimuth_deg` | number | Horizontal pointing, degrees from North (0=N, 90=E). Default 0. |
| `zenith_deg` | number | Vertical pointing (0=up, 90=horizontal, 180=down). Default 0. |
| `position` | {x,y,z} | Location (ENU metres, same frame as nodes). |
| `velocity`, `waypoints`, `random_walk` | object / array | Optional jammer motion, same shapes as node mobility. |
| `intervals` | {start,end}[] | Active windows in sim seconds from scenario start. |

@warning The loader default `zenith_deg = 0` points a directional beam straight
**up**; ground receivers are outside a narrow upward cone. A ground emitter
normally uses `zenith_deg = 90`. `make_jammers.py` defaults to 90. An omni
beam ignores pointing angles.

The beam is a hard cone: a 60-degree beam accepts receivers within 30 degrees
of the pointing vector; there is no sidelobe roll-off. `max_range_m = 0`
disables only the cutoff; path loss still reduces power.

### EW trials CSV mapping (make_jammers.py)

| CSV column | Jammer field | Conversion |
|------------|--------------|------------|
| `start` / `end` (epoch) | `intervals` | epoch minus scenario-start epoch, clipped to `[0,duration]` |
| `ew_strength (W)` | `tx_power_dbm` | `10*log10(W)+30` |
| `ew_approx_lat` / `lon` | `position.x/y` | ENU, fit from the trace's paired lat/lon and east/north |
| `ew_approx_heading (deg from N)` | `azimuth_deg` | direct |
| `ew_band_range` | `target_freq` | `"2210-2215"` becomes `[2210,2215]` |
| `ew_type` | `beamwidth_deg` | `directional` gives `--beamwidth` (default 60); else 360 |
| `ew_type = none` | none | no jammer (baseline trial) |

### Limitations
- **Beamwidth is an assumption.** The CSV logs heading but not beamwidth; widen `--beamwidth` (up to 360) if nodes are jammed outside a narrow cone.
- **Hard-edged cone.** Full power just inside the beam, none just outside.
- **One power and heading per spec.** A sweep is modeled as several specs (one per trial); `make_jammers.py` does this when `--trial` is omitted.
- **SINR floor.** `LinkEvaluator` clamps SINR to 0 dB whenever jammer power is nonzero, so jammed SINR is never negative. Whether to change this is an unresolved team decision.

## Dependencies

| Dependency | Reason |
|------------|--------|
| `ns3/propagation-loss-model.h`, `ns3/mobility-model.h` | Path loss and jammer/receiver positions. |
| `ns3/log.h` | `NS_LOG` debug output in `jammer-model.cc`. |
| `src/domain/node-spec.h` | `Position`, `Velocity`, `Waypoint`, `RandomWalkParams` used by `JammerSpec`. |
| Python 3 (repo `.venv`) | `scripts/validation/make_jammers.py` only. |
