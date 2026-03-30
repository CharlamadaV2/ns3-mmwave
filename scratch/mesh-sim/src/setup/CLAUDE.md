# setup/

## Scope
The only layer that creates ns-3 objects -- nodes, mobility models, buildings, and propagation models.
No EPC, RRC, MAC, or protocol stack -- only the components needed for direct propagation model calls.

## Files
- **topology-builder.h/cc** -- Creates `ConstantPositionMobilityModel` per node, `Building` objects, configures `PropagationLossModel` and `ChannelConditionModel`. Returns handles needed by `LinkEvaluator`.

## Dependencies
- Depends on: `domain/`, ns-3 modules (mobility, propagation, buildings, mmwave)
- Depended on by: `sim.cc`

## Rules
- Buildings must be created before `BuildingsHelper::Install()`.
- Call `BuildingsHelper::MakeMobilityModelConsistent()` after any position change.
