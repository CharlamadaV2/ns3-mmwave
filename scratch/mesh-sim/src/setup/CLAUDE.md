# setup/

## Scope
The only layer that creates ns-3 objects: nodes, mobility models, buildings, and the
propagation and channel-condition models. No EPC, RRC, MAC, or protocol stack.

## Files
- **topology-builder.h/cc** -- `TopologyBuilder(cfg)`, `Build()`, `GetMobilityModels()`, `GetJammerMobilityModels()`, `GetPropagationModel()`, `GetConditionModel()`. Header tables map mobility strings and channel/scenario pairs to ns-3 classes.

## Behavior to preserve
- `Build()` order is nodes, jammers, buildings, propagation. Jammers get their own ns-3 nodes in
  the same container as mesh nodes.
- `GetMobilityModels()` follows `cfg.nodes` order and `GetJammerMobilityModels()` follows
  `cfg.jammers` order. `JammerModel::Configure` asserts equal lengths.
- In centralized RL mode, nodes in `cfg.rl.controlled_indices` get a `ConstantVelocityMobilityModel`
  at `ControlledStartPosition` with zero velocity, whatever their configured mobility.
- Buildings are created before `BuildingsHelper::Install()`. Install also runs (with no buildings)
  when `condition_model == "static_los"`.
- `BuildingsChannelConditionModel` is used when there are buildings or `static_los`; otherwise the
  propagation model's default statistical condition model.
- Jammer mobility precedence: waypoints, then non-zero velocity, then fixed. Jammer `random_walk` is ignored.
- Unknown building `type` / `ext_walls` strings silently fall back to `Residential` / `ConcreteWithWindows`.
- `MobilityBuildingInfo::IsIndoor()` re-evaluates lazily, so no explicit refresh is needed after mobility advances.

## Changing scenarios or models
A new scenario string needs a branch in `ConfigurePropagationModel` and an entry in the
`ValidateConfig` allowed list (`config/`). The two already differ: the validator accepts `InF`
for 3GPP, but the builder throws for it. There is no unit-test suite because it needs ns-3.

## Dependencies
- Depends on: `domain/`, `config/rl-control.h` (`ControlledStartPosition`), ns-3 modules (mobility, propagation, buildings, mmwave)
- Depended on by: `sim.cc`
