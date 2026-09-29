# eval/

## Scope
Per-link path loss, SINR, and capacity for all node pairs each tick. Wraps ns-3
propagation models; no cellular stack. Only `sinr-capacity.h` and `link-table.*`
are ns-3-free.

## Files
- **link-evaluator.h/cc** -- `LinkEvaluator` (`Configure`, `Evaluate`, `EvaluateAll`); owns a `JammerModel`.
- **sinr-capacity.h** -- header-only `SINR_MIN_DB`, `MCS_TABLE`, `SILVUS_MCS_TABLE`, `SinrToMcsIndex`, `SinrToSilvusMcsIndex`, `SinrToCapacity`.
- **link-table.h/cc** -- `LinkTable` symmetric N×N matrix: `Update`, `Get`, `MaxCapacity`, `ConnectedLinkCount`, `IsConnected`.

## Behavior to preserve
- `EvaluateAll` returns only i < j pairs in row-major order; `LinkTable::Update`
  throws `std::runtime_error` unless the count is N·(N−1)/2 and mirrors each entry.
- `Configure` throws `std::runtime_error` on null models; `Evaluate` before
  `Configure` is undefined (no checks). `txIdx`/`rxIdx` index per-node gain vectors.
- Path loss is floored at free-space loss (distance floored at 1 m).
- Jammer power counts only when `band == "sub-6"` and jammers exist, using the max
  of RX-end and TX-end power. Nonzero jammer power clamps SINR to >= 0 dB; otherwise SINR is plain SNR.
- `-999` sentinel comes only from a default `LinkResult` (e.g. `LinkTable` diagonal);
  the evaluator never writes it. Consumers filter on `sinr_db > -900`.
- The `-6.7` default in `LinkTable` and `SINR_MIN_DB` are separate literals; keep them equal.
- `"silvus"` MCS is chosen by `SinrToSilvusMcsIndex`; other models use `SinrToMcsIndex`.
  `SinrToCapacity` throws on an unknown model only when SINR >= `SINR_MIN_DB`.

## Changing the AMC model or link fields
Update together: `sinr-capacity.h`, `LinkEvaluator::Evaluate`, `amc_model` in
`domain/channel-config.h` (the validator does not check it; `SinrToCapacity`
throws), `domain/link-result.h`, and the consumers (`io/`, `routing/`, `rl/`). Unit tests: `make -C tests/unit/eval test`
(covers `sinr-capacity.h` and `link-table.cc`; `link-evaluator.cc` is untested standalone).

## Dependencies
- Depends on: `domain/`, `jammer/`, ns-3 (propagation, mobility, channel condition)
- Depended on by: `sim.cc`, `routing/` (link-table), `io/` (link-table, sinr-capacity), `rl/` (link-table)
