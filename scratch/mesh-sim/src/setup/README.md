@page src_setup src/setup

@brief Creates the ns-3 nodes, mobility models, buildings, and propagation models for a run.

`TopologyBuilder` is the only place in mesh-sim that creates ns-3 objects. It
reads a validated `SimConfig`, builds one node per mesh node and per jammer,
and hands the mobility models and the propagation and channel-condition
models to the rest of the simulator through getters. `sim.cc` builds one per
seed.

## Module Layout

| File | Role |
|------|------|
| @ref topology-builder.h "topology-builder.h" | `TopologyBuilder` class; header tables map config values to ns-3 classes. |
| @ref topology-builder.cc "topology-builder.cc" | Mobility install, building creation, propagation model setup. |
| `CLAUDE.md` | Scope and invariants for AI assistants. |

## Setup

This code needs a built ns-3 tree (mobility, propagation, buildings, mmwave
modules). It cannot be compiled standalone and has no unit-test suite. The user
handles builds; do not run `./ns3 build` from here.

## Run

There is no executable here. `sim.cc` uses the class in this order:

```cpp
TopologyBuilder tb(cfg);   // cfg must outlive tb
tb.Build();                // once, before any getter
auto mobility = tb.GetMobilityModels();       // cfg.nodes order
auto jammers  = tb.GetJammerMobilityModels(); // cfg.jammers order
auto loss     = tb.GetPropagationModel();
auto cond     = tb.GetConditionModel();
```

## Output

| Step | Where |
|------|-------|
| `Build()` | In-memory ns-3 objects only; no files written. |
| Getters | Return ns-3 smart pointers owned jointly with the builder. |

## Conventions

- Behavior depends only on the `SimConfig` passed in.
- Unknown mobility or propagation scenario strings throw `std::runtime_error`.
- Comments use Doxygen `/** */` blocks; full docs live in the header.

## Dependencies

- `src/domain/` (`SimConfig`, `NodeSpec`, `JammerSpec`, `BuildingSpec`).
- `src/config/rl-control.h` for `ControlledStartPosition`.
- ns-3 modules: core, mobility, propagation, buildings, and the NYU propagation models from ns3-mmwave.
