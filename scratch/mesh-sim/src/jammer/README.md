@page src_jammer src/jammer

The jammer model adds interference from one or more jammer emitters to the
per-link SINR, so links under active jamming degrade. Without it the sim defaults to noise floor
(SINR = signal − noise floor). With the jammer, each active jammer's received power is
added to the SINR denominator (SINR = 10.0 * std::log10(signalWatt / (noiseWatt + jamWatt)) ).

@section jammer_files Files

| File | Role |
|------|------|
| @c src/jammer/jammer-spec.h | @ref mesh_sim::JammerSpec — one jammer's parameters (data, loaded from @c jammers.json). |
| @c src/jammer/jammer-model.h / .cc | @ref mesh_sim::JammerModel — computes total received jammer power (W) at a receiver, applying the gates below. |
| @c src/eval/link-evaluator.cc | Calls @ref mesh_sim::JammerModel::InterfPowerAtReceiver and folds it into SINR; active only when @c band==sub-6 and jammers exist. |
| @c src/config/config-loader.cc | Reads @c jammers.json into @c cfg.jammers. |
| @c src/config/config-validator.cc | Validates each jammer's fields. |
| @c src/setup/topology-builder.cc | Builds one ns-3 mobility model per jammer. |
| @c scripts/validation/make_jammers.py | Host-side generator: EW trials CSV → @c jammers.json. |

@section jammer_gates How interference is computed

Each tick, for each receiver, the model sums the received power of every jammer
that passes all gates, in this order:

-# **Time** — the current sim time is inside one of the jammer's
   @c intervals (empty = always on).
-# **Cycle** — for @c type=random jammers, an on/off draw (probability
   @c duty_cycle, seed-dependent, per second). @c type=constant is always on.
-# **Frequency** — the link carrier falls within the jammer's
   @c target_freq band (empty ⇒ no filtering).
-# **Range** — the receiver is within @c max_range_m (0 ⇒ unlimited).
-# **Directional** — the receiver is inside the 3-D beam cone defined by
   @c azimuth_deg, @c zenith_deg, and @c beamwidth_deg (omni when
   @c beamwidth_deg >= 360).

@endcode

@section jammer_schema jammers.json layout

| Field | Type | Meaning |
|-------|------|---------|
| @c id | string | Label (logs / validation). |
| @c enabled | bool | @c false drops the jammer at load. |
| @c type | string | @c constant or @c random. |
| @c target_freq | number[] | Target band (MHz). Empty ⇒ all frequencies. 2 values = @c [lo,hi] band; 1 value = spot ±2.5 MHz. |
| @c tx_power_dbm | number | Transmit power (dBm). |
| @c tx_array_gain_dbi | number | Antenna gain (dBi); added to EIRP. |
| @c duty_cycle | number | Fraction transmitting, [0,1]. |
| @c max_range_m | number | Range cutoff (m); 0 disables. |
| @c beamwidth_deg | number | Cone width; 360 = omni. |
| @c azimuth_deg | number | Horizontal pointing, deg from North (0=N, 90=E). |
| @c zenith_deg | number | Vertical pointing (0=up, 90=horizontal, 180=down). |
| @c position | {x,y,z} | Location (ENU metres, same frame as nodes). |
| @c velocity / @c waypoints / @c random_walk | — | Optional jammer motion. |
| @c intervals | {start,end}[] | Active windows in **sim seconds** (from scenario start). |

@warning A ground emitter must use @c zenith_deg = 90 (horizontal). A value of
@c 0 points the beam straight **up**, so no ground node is inside the cone and
the jammer affects nothing. @c make_jammers.py defaults to 90.

@section jammer_csv Mapping the EW trials CSV

| CSV column | Jammer field | Conversion |
|------------|--------------|------------|
| @c start / @c end (epoch) | @c intervals | epoch − scenario-start epoch, clipped to @c [0,duration] |
| @c "ew_strength (W)" | @c tx_power_dbm | @c 10*log10(W)+30 |
| @c ew_approx_lat / @c lon | @c position.x/y | ENU, fit from the trace's paired lat/lon ↔ east/north |
| @c "ew_approx_heading (deg from N)" | @c azimuth_deg | direct |
| @c ew_band_range | @c target_freq | @c "2210-2215" → @c [2210,2215] |
| @c ew_type | @c beamwidth_deg | @c directional → @c --beamwidth; else 360 |
| @c ew_type = none | — | no jammer (baseline trial) |

@section jammer_limits Limitations

- **Beamwidth is an assumption.** The CSV logs heading but not beamwidth;
  @c --beamwidth (default 60°) sets it. If the field shows nodes jammed well
  outside a narrow cone, widen it (try @c --beamwidth 360 to bound the case).
- **Hard-edged cone.** A receiver just inside the beam gets full power, just
  outside gets none — there is no gradual sidelobe rolloff.
- **Per-interval power/heading.** One @ref mesh_sim::JammerSpec has a single
  power and heading; a sweep is modeled as several specs (one per trial),
  which @c make_jammers.py produces when @c --trial is omitted.
- **SINR floor.** A strong jammer can drive SINR negative; that is physically
  correct (the field radios simply drop those links). Capacity/MCS are floored
  accordingly downstream.
