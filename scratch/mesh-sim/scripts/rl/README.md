# RL code map

Start with [Python setup](../../README.md#python-environment) and the
[training command](../../README.md#maskableppo-smoke-run) in the main README.
This training path controls one mesh node; the jammer is part of the scenario,
not an agent-controlled node.

One training episode follows this path:

1. [`train.py`](train.py) reads the CLI options, creates the output directory,
   starts training, and saves the model and `train_manifest.json`.
2. [`agents/mask_ppo.py`](agents/mask_ppo.py) connects MaskablePPO to the
   environment's valid-action mask.
3. [`env/mesh_env.py`](env/mesh_env.py) is the Gymnasium environment. Each
   reset starts a simulator process; each step exchanges one JSON message with
   the C++ bridge and records the episode. [`env/config.py`](env/config.py)
   reads the scenario seed and movement bounds.
4. [`src/rl/rl-bridge.cc`](../../src/rl/rl-bridge.cc) computes observations and
   rewards in the simulator, then applies the action it receives from Python.

[`bootstrap_venv.py`](bootstrap_venv.py) sets up Python dependencies; it does
not build the simulator. [`tests/fake_sim.py`](tests/fake_sim.py) exercises the
Python exchange without ns-3, while the simulator and regression checks cover
real runs. For saved files, see [Where output lands](../../README.md#where-output-lands).

[`agents/q_learning.py`](agents/q_learning.py) is an exploratory agent; the
training command above uses MaskablePPO, not Q-learning.
