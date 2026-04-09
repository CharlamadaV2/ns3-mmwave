# rl/

## Scope
C++ side of the RL bridge. Handles stdin/stdout JSON IPC with a Python
Gymnasium environment. Each tick: writes observation+reward to stdout,
reads action from stdin, applies the action by setting the controlled
node's velocity toward the desired position via ConstantVelocityMobilityModel.

## Files
- **rl-bridge.h/cc** -- `RlBridge` class with `Step()` (IPC) and `ApplyAction()` (physics).
- **rl-agent.h** -- Legacy placeholder (no-op). Superseded by rl-bridge.

## Dependencies
- Depends on: `domain/`, `eval/link-table`, `routing/mesh-router`, ns-3 mobility
- Depended on by: `sim.cc`
