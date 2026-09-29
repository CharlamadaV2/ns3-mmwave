# jammer/

## Scope
Jammer interference power at a receiver, added to the SINR denominator in sub-6.
Uses the same ns-3 propagation model as mesh links. Traffic, routing, and mobility
are elsewhere; this only computes watts.

## Files
- **jammer-spec.h** -- POD `JammerSpec` and `Interval`; no ns-3 include. Held in `SimConfig::jammers`.
- **jammer-model.h/cc** -- `JammerModel` (`Configure`, `HasJammers`, `InterfPowerAtReceiver`) and private gates `ActiveAt`, `BurstOn`, `InBand`, `InBeam`.

## Behavior to preserve
- Gate order: interval, random burst, frequency, range, beam. Then power = EIRP
  (`tx_power_dbm + tx_array_gain_dbi`) through `CalcRxPower`; under 1 m it uses EIRP directly.
- `duty_cycle`: `random` = full-power on/off draw per whole second (hash of id, seed, second);
  `constant` = always on, power scaled by `duty_cycle`.
- `Configure` drops `enabled == false` specs; `ConfigLoader` defaults `enabled` to false.
  `jammers` and `mobModels` must be the same length and order (`NS_ASSERT_MSG`).
- Frequency gate is off when `carrierHz == 0` or `target_freq` is empty. Values are MHz.
- Beam is a hard cone: `zenith_deg` 0 = up, 90 = horizontal. Beamwidth >= 360 is omni.
- Returns linear watts, not dBm. No receiver antenna gain is applied.

## Changing a JammerSpec field
Update together: `jammer-spec.h`, `parseJammerSpec` in `config/config-loader.cc`,
the jammer block in `config/config-validator.cc`, `setup/topology-builder.cc`
(`CreateJammersAndMobility`), `io/run-logger.h` (documents the jammer log line), `README.md`, and
`scripts/validation/make_jammers.py`. No unit suite exists for `jammer/`
(needs ns-3), so it is unverified by standalone compile.

## Dependencies
- Depends on: `domain/` (`node-spec.h` for `Position`, `Velocity`, `Waypoint`, `RandomWalkParams`), ns-3 (propagation, mobility)
- Depended on by: `eval/` (`LinkEvaluator` owns a `JammerModel`); `jammer-spec.h` reaches `config/`, `setup/`, `io/`, `rl/` via `domain/sim-config.h`
