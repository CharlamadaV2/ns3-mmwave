"""Hand-written synthetic scenario and mapping fixtures, a stub planner, and fake binaries."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.baselines.solver import PlanRequest, PlanResult

# Axis-aligned area shared by the [rl] bounds and the rl_bounds geofence.
AREA = {"x_min": 0.0, "x_max": 400.0, "y_min": 0.0, "y_max": 400.0}

SCENARIO_SECTIONS = """[scenario]
name = baseline-synthetic
seed = 1
duration_s = 1.0
tick_s = 0.1
nodes_file = nodes.json

[rl]
enabled = true
controlled_nodes = uav-a, uav-b, uav-c, walker
max_controlled_nodes = 4
action_profile = move_2d
decision_interval_s = 0.5
action_type = discrete
reward_type = all_links_los
step_size_m = 1.0
x_min = 0.0
x_max = 400.0
y_min = 0.0
y_max = 400.0
z_min = 0.0
z_max = 100.0
"""

BASELINE_DEFAULTS = {
    "algorithm": "geometric",
    "objective": "coverage",
    "movable_nodes": "uav-a, uav-c",
    "seed": "7",
    "max_iterations": "60",
    "mapping_file": "mapping.json",
}

# uav-b has both `position` and `waypoints`; they differ on purpose. "gw" is an ordinary
# peer id; nothing is a gateway.
NODES = [
    {"id": "gw", "role": "peer", "mobility": "fixed", "node_type": "vehicle",
     "position": {"x": 200.0, "y": 200.0, "z": 2.0}},
    {"id": "uav-a", "role": "peer", "mobility": "fixed", "node_type": "drone",
     "position": {"x": 150.0, "y": 180.0, "z": 30.0}},
    {"id": "uav-b", "role": "peer", "mobility": "waypoint", "node_type": "drone",
     "position": {"x": 210.0, "y": 150.0, "z": 30.0},
     "waypoints": [{"t": 0.0, "x": 220.0, "y": 160.0, "z": 30.0},
                   {"t": 1.0, "x": 240.0, "y": 160.0, "z": 30.0}]},
    {"id": "uav-c", "role": "peer", "mobility": "random_walk", "node_type": "drone",
     "position": {"x": 170.0, "y": 140.0, "z": 30.0},
     "random_walk": {"bounds": {"x_min": 0.0, "x_max": 400.0, "y_min": 0.0,
                                "y_max": 400.0}, "speed_mps": 2.0}},
    {"id": "walker", "role": "peer", "mobility": "random_walk", "node_type": "pedestrian",
     "position": {"x": 180.0, "y": 220.0, "z": 1.5},
     "random_walk": {"bounds": {"x_min": 0.0, "x_max": 400.0, "y_min": 0.0,
                                "y_max": 400.0}, "speed_mps": 1.0}},
    {"id": "truck", "role": "peer", "mobility": "constant_velocity", "node_type": "vehicle",
     "position": {"x": 230.0, "y": 230.0, "z": 2.0},
     "velocity": {"vx": 0.5, "vy": 0.0, "vz": 0.0}},
]

MAPPING = {
    "baseline_mapping_version": 2,
    "geofence": {"source": "rl_bounds"},
    "platforms": {"nodes": {}},
}

CHANNEL_FACTS = {"contract": "mesh_channel_query_v1", "isolation": "fork_per_layout",
                 "sinr_threshold_db": -6.7, "probe_rx_gain_dbi": 12.0}
STUB_QUERY_STATS = {"requests": 1, "layouts": 2, "links": 30, "probe_links": 0,
                    "wall_s": 0.0}


def baseline_section(overrides: dict | None = None, drop: tuple = ()) -> str:
    """[baseline] text from BASELINE_DEFAULTS; a None override removes a key."""
    values = {**BASELINE_DEFAULTS, **(overrides or {})}
    lines = ["[baseline]"]
    lines += [f"{key} = {value}" for key, value in values.items()
              if value is not None and key not in drop]
    return "\n".join(lines) + "\n"


def scenario_ini(overrides: dict | None = None, drop: tuple = (), extra: str = "") -> str:
    """Full run.ini text: scenario, centralized [rl], [baseline], then `extra`."""
    return f"{SCENARIO_SECTIONS}\n{baseline_section(overrides, drop)}{extra}"


def write_scenario(root: str | Path, *, overrides: dict | None = None, drop: tuple = (),
                   ini_text: str | None = None, nodes: list | None = None,
                   mapping: dict | None = None) -> Path:
    """Write run.ini, nodes.json, and mapping.json into root; return the INI path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run_ini = root / "run.ini"
    run_ini.write_text(ini_text if ini_text is not None else scenario_ini(overrides, drop))
    (root / "nodes.json").write_text(json.dumps(NODES if nodes is None else nodes,
                                                indent=2) + "\n")
    (root / "mapping.json").write_text(json.dumps(MAPPING if mapping is None else mapping,
                                                  indent=2) + "\n")
    return run_ini


def stub_positions(request: PlanRequest) -> dict:
    """Spread selected nodes along y = 1/4 of the rectangle; everything else stays put."""
    rect = request.rectangle
    selected = [record for record in request.nodes if record.selected]
    positions = {record.id: (record.x, record.y, record.z) for record in request.nodes}
    width = rect["x_max"] - rect["x_min"]
    y = rect["y_min"] + 0.25 * (rect["y_max"] - rect["y_min"])
    for index, record in enumerate(selected):
        x = rect["x_min"] + width * (index + 1) / (len(selected) + 1)
        positions[record.id] = (x, y, record.z)
    return positions


def stub_solve(request: PlanRequest, log) -> PlanResult:
    """Deterministic stand-in for solver.solve; starts no channel-query worker."""
    log.write(f"stub planner: {request.method}/{request.objective}\n")
    return PlanResult(positions=stub_positions(request),
                      predictions={"stub": True, "method": request.method},
                      planner_wall_s=0.0, query_stats=dict(STUB_QUERY_STATS),
                      planner_settings={"stub": True},
                      channel={**CHANNEL_FACTS, "band": request.band or "mmwave"})


def install_stub_planner(monkeypatch) -> None:
    """Replace solver.solve so no planner or channel-query worker runs."""
    monkeypatch.setattr("scripts.baselines.solver.solve", stub_solve)


@pytest.fixture
def stub_planner(monkeypatch):
    install_stub_planner(monkeypatch)


class StubStrategy:
    """Planner strategy stand-in: scores start and stub layouts, returns the stub layout."""

    def __init__(self, positions=stub_positions):
        self.positions = positions
        self.scored = []

    def solve(self, request: PlanRequest, scorer, log):
        start = np.array([(r.x, r.y, r.z) for r in request.nodes])
        chosen = self.positions(request)
        layout = np.array([chosen[r.id] for r in request.nodes])
        self.scored = scorer.evaluate([start, layout])
        log.write("stub strategy scored 2 layouts\n")
        return layout, {"stub_strategy": True, "connected_pairs": int(
            self.scored[1].connected.sum() // 2)}, ["stub note"]

    def settings(self, request: PlanRequest) -> dict:
        return {"strategy": "stub", "method": request.method}


def install_stub_strategy(monkeypatch, strategy: StubStrategy | None = None) -> StubStrategy:
    """Keep solver.solve and the real channel scorer; replace only the search strategy."""
    strategy = strategy or StubStrategy()
    monkeypatch.setattr("scripts.baselines.solver._strategy", lambda method: strategy)
    return strategy


@pytest.fixture
def scenario(tmp_path: Path) -> Path:
    """Synthetic scenario with an active geometric [baseline]; returns run.ini."""
    return write_scenario(tmp_path / "scenario")


FAKE_CHILD = Path(__file__).with_name("fake_child.py")
FAKE_QUERY = Path(__file__).with_name("fake_query.py")


def _shim(directory: str | Path, script: Path) -> Path:
    path = Path(directory) / "fake-mesh-sim"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\nimport runpy\n"
                    f"runpy.run_path({str(script)!r}, run_name='__main__')\n")
    path.chmod(0o755)
    return path


def fake_child_binary(directory: str | Path) -> Path:
    """Executable shim that runs fake_child.py with this interpreter; never a real simulator."""
    return _shim(directory, FAKE_CHILD)


def fake_query_binary(directory: str | Path) -> Path:
    """Executable shim that serves fake_query.py's channel query; never a real simulator."""
    return _shim(directory, FAKE_QUERY)


@pytest.fixture
def fake_child(tmp_path: Path) -> Path:
    return fake_child_binary(tmp_path / "bin")


@pytest.fixture
def fake_query(tmp_path: Path) -> Path:
    return fake_query_binary(tmp_path / "query-bin")
