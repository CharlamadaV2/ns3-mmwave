# eval/

## Scope
Link evaluation -- wraps ns-3 propagation models to compute per-link path loss, SINR, and capacity.
This is the core physics engine replacing the full cellular stack.

## Files
- **link-evaluator.h/cc** -- `LinkEvaluator` calls `PropagationLossModel::CalcRxPower()`, adds beamforming gain, computes SINR and Shannon/AMC capacity. Returns `LinkResult`.
- **sinr-capacity.h** -- `SinrToCapacity()` pure-math function (Shannon + MCS table). Header-only, no ns-3 dependency, testable standalone.
- **link-table.h/cc** -- `LinkTable` stores the NxN link quality matrix. Updated each tick. Exposes per-link and aggregate queries.

## Dependencies
- Depends on: `domain/`, ns-3 modules (propagation, mobility, buildings)
- Depended on by: `routing/`, `io/`, `sim.cc`
