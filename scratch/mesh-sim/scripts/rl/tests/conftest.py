"""Shared fake-simulator fixtures; older test modules keep their local definitions."""

import sys
from pathlib import Path

import pytest

FAKE_SIM = Path(__file__).resolve().parent / "fake_sim.py"

MULTI_RUN_INI = """[scenario]
name = fake-multi
seed = 1
duration_s = 1.0
tick_s = 0.1
nodes_file = nodes.json

[rl]
enabled = true
controlled_nodes = node-b, node-c
max_controlled_nodes = 3
action_profile = move_2d
decision_interval_s = 0.5
action_type = discrete
reward_type = all_links_los
step_size_m = 1.0
x_min = 0.0
x_max = 100.0
y_min = -50.0
y_max = 100.0
z_min = 0.0
z_max = 50.0
"""

NODES_JSON = """[
  {"id": "node-a", "role": "peer", "mobility": "fixed", "node_type": "drone",
   "position": {"x": 50.0, "y": 20.0, "z": 10.0}},
  {"id": "node-b", "role": "peer", "mobility": "waypoint", "node_type": "drone",
   "waypoints": [{"t": 0.0, "x": 100.0, "y": 0.0, "z": 10.0},
                 {"t": 1.0, "x": 100.0, "y": 40.0, "z": 10.0}]},
  {"id": "node-c", "role": "peer", "mobility": "constant_velocity", "node_type": "drone",
   "position": {"x": 97.0, "y": 50.0, "z": 10.0},
   "velocity": {"vx": 0.0, "vy": 0.0, "vz": 0.0}}
]
"""


@pytest.fixture
def sim_binary(tmp_path: Path) -> str:
    """Executable shim so the CLIs can spawn the fake sim like a real binary."""
    shim = tmp_path / "fake-sim"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_SIM}" "$@"\n')
    shim.chmod(0o755)
    return str(shim)


@pytest.fixture
def multi_run_config(tmp_path: Path) -> str:
    """Centralized multi-node scenario reusable across seeds."""
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    (scenario / "run.ini").write_text(MULTI_RUN_INI)
    (scenario / "nodes.json").write_text(NODES_JSON)
    return str(scenario / "run.ini")
