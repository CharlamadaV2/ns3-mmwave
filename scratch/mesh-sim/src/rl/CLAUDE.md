# rl/

## Scope
C++ side of the RL bridge for both control modes: legacy single-node
`Discrete(7)` and centralized `MultiDiscrete([5]*M)` over a resolved slot list.
Handles stdin/stdout JSON IPC with a Python Gymnasium environment: writes
observation+reward (every tick in legacy mode, every decision interval in
centralized mode), reads the action from stdin, and applies it by setting
controlled-node velocities via ConstantVelocityMobilityModel. Masks, speed
caps, per-tick bounds clamping, and action revalidation are owned here.
In centralized mode the bridge also exports raw per-decision facts (node,
link, and window sums); preset selection and normalization live in Python.

## Files
- **rl-bridge.h/cc** -- `RlBridge` class with `Step()` (IPC) and `ApplyAction()` (physics).
- **README.md** -- Mode/message/mask/action contract shared with the Python env.
- **policy-inputs.md** -- Observation presets, reward components, telemetry.
- **policy-input-tests.md / policy-lifecycle-tests.md / policy-comparison-tests.md** -- Test maps.
- **rl-agent.h** -- Legacy placeholder (no-op). Superseded by rl-bridge.

## Dependencies
- Depends on: `domain/`, `eval/link-table`, `routing/mesh-router`, ns-3 mobility
- Depended on by: `sim.cc`
