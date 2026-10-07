"""Real-binary rows: guard, runner, evaluation (R1-R6, R8, R12), channel query (R7, R9-R11)."""

# Set MESH_SIM_BIN to a built simulator; every run uses a synthetic scenario in tmp_path.

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.baselines import artifacts
from scripts.baselines.effective_inputs import start_position
from scripts.baselines.planners.channel import ChannelScorer
from scripts.baselines.tests.conftest import (NODES, SCENARIO_SECTIONS, scenario_ini,
                                              write_scenario)
from scripts.sim_support import simulator_env

MESH_SIM_BIN = os.environ.get("MESH_SIM_BIN")
pytestmark = pytest.mark.skipif(
    not MESH_SIM_BIN,
    reason="BLOCKED: MESH_SIM_BIN is not set; these rows need a human-built mesh-sim binary")

MESH_ROOT = Path(__file__).resolve().parents[3]
RL_ON = "[rl]\nenabled = true\n"
RL_OFF = "[rl]\nenabled = false\n"
SIM_SECTIONS = """
[channel]
band = mmwave
frequency_ghz = 28.0
tx_power_dbm = 30.0
scenario = RMa
channel_model = 3gpp
condition_model = static_los
blockage_enabled = false
bandwidth_mhz = 400.0
noise_figure_db = 5.0

[traffic]
model = constant
demand_mbps = 10.0
flow_topology = all_pairs

[routing]
algorithm = shortest_path
max_hops = 5
"""
GUARD_TEXT = "is active but this is a direct simulator run"
NOTICE_TEXT = "is not applied in RL mode; this run uses the scenario layout"
COMPARED_CSVS = ("positions.csv", "links.csv", "flows.csv", "routes.csv")
TIMING_FIELDS = ("wall_clock_start", "wall_clock_end", "wall_elapsed_s")
WRITER_TOL = 1e-6
SOURCE_START = [start_position(node) for node in NODES]
PLACEMENT_POLICIES = ("geometric", "optimization")
EVAL_POLICIES = ("hold", "random_valid", *PLACEMENT_POLICIES)
# Small grids and few iterations keep every planner row bounded.
BOUNDED_PLANNER = {"candidate_grid_cells": "16", "coverage_grid_cells": "36",
                   "max_iterations": "3"}
SOURCE_CONTROLLED = "controlled_nodes = uav-a, uav-b, uav-c, walker"
OWNERSHIP_FIX = "to the same ids, or `all`"


def _scenario(root: Path, *, algorithm: str | None = "geometric",
              rl_enabled: bool = True, overrides: dict | None = None,
              controlled: str | None = None) -> Path:
    if algorithm is None:
        text = f"{SCENARIO_SECTIONS}{SIM_SECTIONS}"
    else:
        text = scenario_ini({"algorithm": algorithm, **(overrides or {})},
                            extra=SIM_SECTIONS)
    if not rl_enabled:
        assert RL_ON in text
        text = text.replace(RL_ON, RL_OFF)
    if controlled is not None:
        assert SOURCE_CONTROLLED in text
        text = text.replace(SOURCE_CONTROLLED, f"controlled_nodes = {controlled}")
    return write_scenario(root, ini_text=text)


def _hashes(directory: Path) -> dict:
    return {p.name: artifacts.sha256_file(p) for p in sorted(directory.iterdir())
            if p.is_file()}


def _direct(run_config: Path, out: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [MESH_SIM_BIN, f"--run-config={run_config}", f"--output-dir={out}", *flags],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600,
        env=simulator_env(MESH_ROOT))


def _module(module: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", module, *args], cwd=MESH_ROOT,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          timeout=900)


def _runner(run_config: Path, out: Path, *args: str) -> subprocess.CompletedProcess:
    return _module("scripts.baselines.runner", "--sim-binary", MESH_SIM_BIN,
                   "--run-config", str(run_config), "--output-dir", str(out), *args)


def _first_tick_positions(seed_dir: Path) -> dict:
    """Roster index -> (x, y, z) for the first time step in positions.csv."""
    lines = [line for line in (seed_dir / "positions.csv").read_text().splitlines()
             if line and not line.startswith("#")]
    rows = [line.split(",") for line in lines[1:]]
    first = rows[0][0]
    return {int(row[1]): tuple(float(v) for v in row[2:5]) for row in rows if row[0] == first}


def _assert_xyz(actual, expected, label: str) -> None:
    assert all(abs(a - e) <= WRITER_TOL for a, e in zip(actual, expected)), (
        f"{label}: {actual} != {expected}")


def _summary_without_timing(seed_dir: Path) -> dict:
    summary = json.loads((seed_dir / "summary.json").read_text())
    return {k: v for k, v in summary.items() if k not in TIMING_FIELDS}


def _planned(plan: dict) -> list:
    nodes = sorted(plan["nodes"], key=lambda n: n["roster_index"])
    return [(n["planned"]["x"], n["planned"]["y"], n["planned"]["z"]) for n in nodes]


def _trace(seed_dir: Path, indices) -> list:
    """positions.csv rows of the given roster indices, in file order."""
    lines = [line for line in (seed_dir / "positions.csv").read_text().splitlines()
             if line and not line.startswith("#")]
    rows = [line.split(",") for line in lines[1:]]
    return [row for row in rows if int(row[1]) in set(indices)]


def _assert_v2_artifacts(manifest: dict, plan: dict, mode: str, movable: list,
                         controlled: list | None) -> None:
    assert manifest["baseline_manifest_version"] == 2
    assert plan["baseline_plan_version"] == 2
    assert manifest["ownership"] == {"mode": mode, "movable_resolved": movable,
                                     "controlled_resolved": controlled}
    scoring = manifest["channel_scoring"]
    assert (scoring["contract"], scoring["isolation"]) == ("mesh_channel_query_v1",
                                                           "fork_per_layout")
    assert scoring["mode_flag"] == ("--rl-mode" if mode == "evaluation" else None)
    assert scoring["queries"]["layouts"] >= 1
    assert manifest["fingerprint"] and manifest["sim_binary_sha256"]
    assert {n["id"] for n in plan["nodes"] if n["selected"]} == set(movable)


@pytest.mark.parametrize("rl_enabled", [False, True])
def test_r1_direct_run_refuses_active_method(tmp_path, rl_enabled):
    ini = _scenario(tmp_path / "s", rl_enabled=rl_enabled)
    out = tmp_path / "out"
    result = _direct(ini, out, "--seeds=1")
    assert result.returncode == 1, result.stderr[-2000:]
    assert GUARD_TEXT in result.stderr
    assert "python -m scripts.baselines.runner" in result.stderr
    assert not out.exists() or not any(out.iterdir())


def test_r2_none_and_absent_section_match(tmp_path):
    outputs = []
    for name, algorithm in (("none", "none"), ("absent", None)):
        ini = _scenario(tmp_path / name, algorithm=algorithm, rl_enabled=False)
        out = tmp_path / f"out-{name}"
        result = _direct(ini, out, "--seeds=1")
        assert result.returncode == 0, result.stderr[-2000:]
        outputs.append(out / "seed-1")
    first, second = outputs
    for name in sorted(p.name for p in first.glob("*.csv")):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    assert _summary_without_timing(first) == _summary_without_timing(second)


def test_r3_rl_mode_notice_and_original_layout(tmp_path):
    ini = _scenario(tmp_path / "s")
    out = tmp_path / "out"
    result = _direct(ini, out, "--rl-mode", "--seed=1")
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stderr.count(NOTICE_TEXT) == 1
    assert GUARD_TEXT not in result.stderr
    positions = _first_tick_positions(out / "seed-1")
    for index, expected in enumerate(SOURCE_START):
        _assert_xyz(positions[index], expected, NODES[index]["id"])


def test_r4_runner_none_matches_direct_run(tmp_path):
    ini = _scenario(tmp_path / "s", algorithm="none", rl_enabled=False)
    direct = tmp_path / "direct"
    result = _direct(ini, direct, "--seeds=1,2")
    assert result.returncode == 0, result.stderr[-2000:]
    run = tmp_path / "runner"
    launched = _runner(ini, run, "--seeds", "1,2")
    assert launched.returncode == 0, launched.stderr[-2000:]
    assert artifacts.read_json(run / artifacts.MANIFEST_NAME)["status"] == "complete"
    for seed in (1, 2):
        ours, theirs = run / f"seed-{seed}", direct / f"seed-{seed}"
        for name in COMPARED_CSVS:
            assert (ours / name).read_bytes() == (theirs / name).read_bytes(), (seed, name)
        assert _summary_without_timing(ours) == _summary_without_timing(theirs)


@pytest.mark.parametrize("method", PLACEMENT_POLICIES)
def test_r5_runner_places_selected_nodes(tmp_path, method):
    # Standalone has no ownership rule: [rl] lists four nodes, two are movable.
    ini = _scenario(tmp_path / "s", algorithm=method, overrides=BOUNDED_PLANNER)
    before = _hashes(ini.parent)
    run = tmp_path / "run"
    launched = _runner(ini, run, "--seeds", "1,2")
    manifest = artifacts.read_json(run / artifacts.MANIFEST_NAME)
    assert launched.returncode == 0, (manifest.get("error"), launched.stderr[-2000:])
    assert (manifest["status"], manifest["method"]) == ("complete", method)
    assert [s["status"] for s in manifest["seeds"]] == ["complete", "complete"]
    assert _hashes(ini.parent) == before

    plan = artifacts.read_json(run / manifest["plan"])
    _assert_v2_artifacts(manifest, plan, "standalone", ["uav-a", "uav-c"], None)
    planned = _planned(plan)
    selected = {n["roster_index"] for n in plan["nodes"] if n["selected"]}
    for seed in (1, 2):
        positions = _first_tick_positions(run / f"seed-{seed}")
        for index, source in enumerate(SOURCE_START):
            expected = planned[index] if index in selected else source
            _assert_xyz(positions[index], expected, f"seed {seed} {NODES[index]['id']}")
    assert (run / "inputs" / artifacts.PLAN_NAME).is_file()
    assert not list(run.rglob("rl_episode.json")) and not list(run.rglob("steps.jsonl"))


def _evaluate(ini: Path, out: Path, seeds: str, policies) -> subprocess.CompletedProcess:
    return _module("scripts.rl.evaluate", "--sim-binary", MESH_SIM_BIN,
                   "--run-config", str(ini), "--output-dir", str(out),
                   "--seeds", seeds, "--policies", ",".join(policies),
                   "--telemetry", "steps")


def _decision_zero(episode_dir: Path) -> tuple[list, list]:
    """(node ids, facts.nodes rows) of the reset record in steps.jsonl."""
    lines = (episode_dir / "steps.jsonl").read_text().splitlines()
    header, record = json.loads(lines[0]), json.loads(lines[1])
    assert record["decision"] == 0
    return list(header["contract"]["node_ids"]), record["facts"]["nodes"]


def test_r6_evaluation_suite_with_placement_policies(tmp_path):
    ini = _scenario(tmp_path / "s", overrides=BOUNDED_PLANNER, controlled="uav-c, uav-a")
    before = _hashes(ini.parent)
    out = tmp_path / "eval"
    result = _evaluate(ini, out, "1,2", EVAL_POLICIES)
    assert result.returncode == 0, result.stderr[-2000:]
    assert _hashes(ini.parent) == before

    manifest = json.loads((out / "eval_manifest.json").read_text())
    assert set(manifest["policies"]) == set(EVAL_POLICIES)
    assert all((out / name).is_dir() for name in EVAL_POLICIES)
    for name, block in manifest["policies"].items():
        assert ("baseline" in block) == (name in PLACEMENT_POLICIES), name
        assert [e["status"] for e in block["episodes"]] == ["completed", "completed"]

    expected_ids = [node["id"] for node in NODES]
    for name in EVAL_POLICIES:
        block = manifest["policies"][name]
        if name in PLACEMENT_POLICIES:
            baseline = block["baseline"]
            assert baseline["method"] == name and baseline["executor"] == "hold"
            prep = artifacts.read_json(out / baseline["manifest"])
            assert prep["status"] == "prepared"
            plan = artifacts.read_json(out / baseline["plan"])
            _assert_v2_artifacts(prep, plan, "evaluation", ["uav-a", "uav-c"],
                                 ["uav-c", "uav-a"])
            assert baseline["ownership"] == prep["ownership"]
            expected = _planned(plan)
        elif name == "hold":
            expected = SOURCE_START
        else:
            continue
        for episode in block["episodes"]:
            ids, rows = _decision_zero(Path(episode["episode_dir"]))
            assert ids == expected_ids
            for index, row in enumerate(rows):
                _assert_xyz(row[:3], expected[index], f"{name} seed {episode['seed']} "
                                                      f"{expected_ids[index]}")
            # Every RL slot is a placed node: no unplaced node is frozen by hold.
            slots = {expected_ids[index]: int(row[6]) for index, row in enumerate(rows)}
            assert {k: v for k, v in slots.items() if v >= 0} == {"uav-c": 0, "uav-a": 1}


def test_r6_evaluation_refuses_unequal_rosters(tmp_path):
    ini = _scenario(tmp_path / "s", overrides=BOUNDED_PLANNER)
    out = tmp_path / "eval"
    result = _evaluate(ini, out, "1", ("hold", "geometric"))
    assert result.returncode == 1
    assert ("movable but not controlled: -; controlled but not movable: uav-b, walker"
            in result.stderr)
    assert OWNERSHIP_FIX in result.stderr
    assert not (out / "eval_manifest.json").exists()
    assert not list(out.rglob("episode-*"))


# Channel-query rows use a CLI planning seed that differs from the INI seed. An ordinary
# run assigns each resolved seed to cfg.seed (sim.cc), so its JammerModel uses that seed too.
QUERY_INI_SEED = 5
QUERY_SEED = 3
QUERY_JAMMER_SEED = QUERY_SEED
QUERY_RUN_ID = 1
CAPACITY_TOL = 0.05 + 1e-6
QUERY_NODES = [
    {"id": "n0", "role": "peer", "mobility": "fixed", "node_type": "drone",
     "position": {"x": 0.0, "y": 0.0, "z": 10.0}, "tx_array_gain_dbi": 6.0},
    {"id": "n1", "role": "peer", "mobility": "fixed", "node_type": "vehicle",
     "position": {"x": 60.0, "y": 0.0, "z": 2.0}, "rx_array_gain_dbi": 3.0},
    {"id": "n2", "role": "peer", "mobility": "fixed", "node_type": "pedestrian",
     "position": {"x": 0.0, "y": 150.0, "z": 1.5},
     "tx_array_gain_dbi": 9.0, "rx_array_gain_dbi": 9.0},
    {"id": "n3", "role": "peer", "mobility": "fixed", "node_type": "drone",
     "position": {"x": 220.0, "y": 90.0, "z": 25.0}},
]
QUERY_PROBES = [[10.0, 10.0], [120.0, 40.0], [300.0, 0.0], [600.0, 600.0]]
WALL = [{"id": "wall", "bounds": {"x_min": 90.0, "x_max": 110.0, "y_min": -20.0,
                                  "y_max": 20.0, "z_min": 0.0, "z_max": 20.0},
         "type": "Office", "ext_walls": "ConcreteWithWindows", "n_floors": 3,
         "n_rooms_x": 2, "n_rooms_y": 4}]
WALL_NODES = [
    {"id": "w0", "role": "peer", "mobility": "fixed", "node_type": "drone",
     "position": {"x": 0.0, "y": 0.0, "z": 10.0}},
    {"id": "w1", "role": "peer", "mobility": "fixed", "node_type": "drone",
     "position": {"x": 200.0, "y": 0.0, "z": 10.0}},
    {"id": "w2", "role": "peer", "mobility": "fixed", "node_type": "drone",
     "position": {"x": 0.0, "y": 100.0, "z": 10.0}},
]
JAMMER = {"id": "jammer-0", "enabled": True, "type": "constant", "target_freq": [2400.0],
          "tx_power_dbm": 30.0, "tx_array_gain_dbi": 12.0, "duty_cycle": 1.0,
          "max_range_m": 0.0, "beamwidth_deg": 360.0, "azimuth_deg": 0.0,
          "zenith_deg": 90.0, "position": {"x": 50.0, "y": 0.0, "z": 10.0}}


def _query_scenario(root: Path, *, channel_model: str = "3gpp", scenario: str = "UMi",
                    condition_model: str = "auto", band: str = "mmwave",
                    frequency_ghz: float = 28.0, bandwidth_mhz: float = 400.0,
                    shadowing: bool = True, nodes: list | None = None,
                    buildings: list | None = None, jammers: list | None = None) -> Path:
    """Synthetic standalone scenario (no [baseline], RL off) for the channel-query rows."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "nodes.json").write_text(json.dumps(nodes or QUERY_NODES, indent=2) + "\n")
    assets = ""
    if buildings is not None:
        (root / "buildings.json").write_text(json.dumps(buildings, indent=2) + "\n")
        assets += "buildings_file = buildings.json\n"
    if jammers is not None:
        (root / "jammers.json").write_text(json.dumps(jammers, indent=2) + "\n")
        assets += "jammers_file = jammers.json\n"
    run_ini = root / "run.ini"
    run_ini.write_text(f"""[scenario]
name = channel-query-synthetic
seed = {QUERY_INI_SEED}
run_id = {QUERY_RUN_ID}
duration_s = 0.2
tick_s = 0.1
nodes_file = nodes.json
{assets}
[channel]
band = {band}
frequency_ghz = {frequency_ghz}
tx_power_dbm = 30.0
scenario = {scenario}
channel_model = {channel_model}
condition_model = {condition_model}
bandwidth_mhz = {bandwidth_mhz}
noise_figure_db = 5.0
tx_array_gain_dbi = 12.0
rx_array_gain_dbi = 10.0

[nyu_channel]
shadowing_enabled = {"true" if shadowing else "false"}

[traffic]
model = constant
demand_mbps = 10.0
flow_topology = all_pairs

[routing]
algorithm = shortest_path
max_hops = 5

[rl]
enabled = false

[output]
viz_tick_ms = 100
""")
    return run_ini


def _starts(nodes: list) -> np.ndarray:
    return np.array([start_position(node) for node in nodes])


def _query(run_config: Path, calls: list, *, nodes: list | None = None,
           seed: int = QUERY_SEED, band: str | None = None, probes: dict | None = None,
           mode: str = "standalone"):
    """One worker lifetime: evaluate each list of layouts in `calls`; return results, scorer."""
    nodes = nodes or QUERY_NODES
    log_path = run_config.parent / f"query-{seed}.log"
    with open(log_path, "a", encoding="utf-8") as log, ChannelScorer(
            MESH_SIM_BIN, run_config, seed, QUERY_RUN_ID, band, mode,
            [node["id"] for node in nodes], _starts(nodes), seed, log) as scorer:
        if probes is not None:
            scorer.set_probes(**probes)
        results = [scorer.evaluate(layouts) for layouts in calls]
    return results, scorer


def _first_tick_links(seed_dir: Path) -> dict:
    """(node_a, node_b) -> (sinr_db, capacity_mbps, is_los) at the first links.csv time."""
    lines = [line for line in (seed_dir / "links.csv").read_text().splitlines()
             if line and not line.startswith("#")]
    header = lines[0].split(",")
    rows = [dict(zip(header, line.split(","))) for line in lines[1:]]
    first = rows[0]["time_s"]
    assert float(first) == 0.0, f"first links.csv time is {first}"
    return {(int(r["node_a"]), int(r["node_b"])): (float(r["sinr_db"]),
                                                   float(r["capacity_mbps"]),
                                                   r["condition"] == "LOS")
            for r in rows if r["time_s"] == first}


def _assert_matches_links(result, links: dict, label: str) -> None:
    n = result.sinr_db.shape[0]
    assert len(links) == n * (n - 1) // 2, label
    for (a, b), (sinr, capacity, los) in links.items():
        assert abs(result.sinr_db[a, b] - sinr) <= WRITER_TOL, (label, a, b, "sinr")
        assert abs(result.capacity_mbps[a, b] - capacity) <= CAPACITY_TOL, (label, a, b)
        assert bool(result.is_los[a, b]) == los, (label, a, b, "is_los")


def _assert_same(first, second, label: str) -> None:
    for field in ("connected", "sinr_db", "capacity_mbps", "is_los"):
        assert np.array_equal(getattr(first, field), getattr(second, field)), (label, field)
    assert first.coverage == second.coverage, (label, "coverage")


def _direct_ok(run_config: Path, out: Path, *flags: str) -> Path:
    result = _direct(run_config, out, f"--seeds={QUERY_SEED}", *flags)
    assert result.returncode == 0, result.stderr[-2000:]
    return out / f"seed-{QUERY_SEED}"


def _parity(tmp_path: Path, channel_model: str, with_probes: bool) -> None:
    ini = _query_scenario(tmp_path / "s", channel_model=channel_model)
    links = _first_tick_links(_direct_ok(ini, tmp_path / "direct"))
    probes = {"points": QUERY_PROBES} if with_probes else None
    ((result,),), scorer = _query(ini, [[_starts(QUERY_NODES)]], probes=probes)
    init = scorer.init
    assert (init["seed"], init["run_id"], init["jammer_seed"]) == (
        QUERY_SEED, QUERY_RUN_ID, QUERY_JAMMER_SEED)
    assert init["channel"]["channel_model"] == channel_model
    assert (result.coverage is not None) == with_probes
    _assert_matches_links(result, links, f"{channel_model} probes={with_probes}")


def _repeatability(tmp_path: Path, channel_model: str) -> None:
    ini = _query_scenario(tmp_path / "s", channel_model=channel_model)
    before = _direct_ok(ini, tmp_path / "before")
    a = _starts(QUERY_NODES)
    b = a.copy()
    b[1, :2] += (25.0, -10.0)
    probes = {"points": QUERY_PROBES}
    (a1, b1, a2), _ = _query(ini, [[a], [b], [a]], probes=probes)
    (b2, a3, mixed), _ = _query(ini, [[b], [a], [b, a]], probes=probes)
    for label, result in (("A again", a2[0]), ("A second worker", a3[0]),
                          ("A in batch", mixed[1])):
        _assert_same(a1[0], result, label)
    for label, result in (("B second worker", b2[0]), ("B in batch", mixed[0])):
        _assert_same(b1[0], result, label)
    after = _direct_ok(ini, tmp_path / "after")
    for name in ("positions.csv", "links.csv"):
        assert (before / name).read_bytes() == (after / name).read_bytes(), name
    assert _summary_without_timing(before) == _summary_without_timing(after)


@pytest.mark.parametrize("with_probes", [False, True], ids=["no-probes", "probes"])
def test_r7_query_matches_direct_first_tick(tmp_path, with_probes):
    _parity(tmp_path, "3gpp", with_probes)


def test_r9_query_is_repeatable_and_leaves_runs_unchanged(tmp_path):
    _repeatability(tmp_path, "3gpp")


@pytest.mark.parametrize("with_probes", [False, True], ids=["no-probes", "probes"])
def test_r10_nyu_shadowing_parity(tmp_path, with_probes):
    _parity(tmp_path, "nyu", with_probes)


def test_r10_nyu_shadowing_repeatability(tmp_path):
    _repeatability(tmp_path, "nyu")


def test_r10_planning_seed_changes_nyu_draws(tmp_path):
    ini = _query_scenario(tmp_path / "s", channel_model="nyu")
    layout = [_starts(QUERY_NODES)]
    ((first,),), _ = _query(ini, [layout], seed=QUERY_SEED)
    ((second,),), _ = _query(ini, [layout], seed=QUERY_SEED + 1)
    assert not np.array_equal(first.sinr_db, second.sinr_db)


def test_r11_buildings_block_line_of_sight(tmp_path):
    common = {"channel_model": "nyu", "condition_model": "static_los", "shadowing": False,
              "nodes": WALL_NODES}
    walled = _query_scenario(tmp_path / "walled", buildings=WALL, **common)
    open_ini = _query_scenario(tmp_path / "open", **common)
    blocked = _starts(WALL_NODES)
    clear = blocked.copy()
    clear[1, 1] = 60.0
    (walled_results,), scorer = _query(walled, [[blocked, clear]], nodes=WALL_NODES)
    assert scorer.init["num_buildings"] == 1
    ((open_result,),), scorer = _query(open_ini, [[blocked]], nodes=WALL_NODES)
    assert scorer.init["num_buildings"] == 0
    behind, beside = walled_results
    assert not behind.is_los[0, 1] and beside.is_los[0, 1] and open_result.is_los[0, 1]
    assert behind.sinr_db[0, 1] < open_result.sinr_db[0, 1]
    assert behind.sinr_db[0, 1] < beside.sinr_db[0, 1]


def test_r11_random_jammer_matches_the_run_burst_seed(tmp_path):
    jammer = {**JAMMER, "type": "random", "duty_cycle": 0.5}
    ini = _query_scenario(tmp_path / "s", scenario="RMa", condition_model="static_los",
                          band="sub-6", frequency_ghz=2.4, bandwidth_mhz=20.0,
                          jammers=[jammer])
    links = _first_tick_links(_direct_ok(ini, tmp_path / "direct", "--band=sub-6"))
    ((result,),), scorer = _query(ini, [[_starts(QUERY_NODES)]], band="sub-6")
    init = scorer.init
    assert init["jammer_path_enabled"] is True and init["band_source"] == "cli"
    assert (init["seed"], init["jammer_seed"]) == (QUERY_SEED, QUERY_JAMMER_SEED)
    assert np.isfinite(result.sinr_db[~np.eye(len(QUERY_NODES), dtype=bool)]).all()
    _assert_matches_links(result, links, "random jammer")


def test_r11_constant_jammer_clamp_and_probe_thresholds(tmp_path):
    ini = _query_scenario(tmp_path / "s", scenario="RMa", condition_model="static_los",
                          band="sub-6", frequency_ghz=2.4, bandwidth_mhz=20.0,
                          jammers=[JAMMER])
    layout = [_starts(QUERY_NODES)]
    coverage = {}
    for threshold in (None, 10.0, 1000.0):
        probes = {"points": QUERY_PROBES}
        if threshold is not None:
            probes["sinr_db"] = threshold
        ((result,),), scorer = _query(ini, [layout], probes=probes)
        assert scorer.probe_rx_gain_dbi == 10.0
        coverage[threshold] = result.coverage
        off = ~np.eye(len(QUERY_NODES), dtype=bool)
        assert np.isfinite(result.sinr_db[off]).all()
        assert (result.sinr_db[off] >= 0.0).all() and result.connected[off].all()
    everything = list(range(len(QUERY_PROBES)))
    assert all(covered == everything for covered in coverage[None])
    assert all(set(c) <= set(d) for c, d in zip(coverage[10.0], coverage[None]))
    assert all(covered == [] for covered in coverage[1000.0])


# R8: evaluation-mode parity. The query runs with --rl-mode on an INI with centralized
# control and is compared with the facts.links of the RL reset message (decision 0).
QUERY_RL = """[rl]
enabled = true
controlled_nodes = n2, n0
max_controlled_nodes = 2
action_profile = move_2d
decision_interval_s = 0.1
action_type = discrete
reward_type = all_links_los
step_size_m = 1.0
x_min = -100.0
x_max = 400.0
y_min = -100.0
y_max = 400.0
z_min = 0.0
z_max = 100.0
"""


def _reset_links(run_config: Path, out: Path) -> tuple[dict, dict]:
    """(init, (i, j) -> (sinr_db, capacity_mbps, is_los)) from the RL reset message."""
    result = subprocess.run(
        [MESH_SIM_BIN, f"--run-config={run_config}", "--rl-mode", f"--seed={QUERY_SEED}",
         f"--output-dir={out}"], input="", capture_output=True, text=True, timeout=600,
        env=simulator_env(MESH_ROOT))
    assert result.returncode == 0, result.stderr[-2000:]
    messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    init = next(m for m in messages if m.get("type") == "init")
    reset = next(m for m in messages if m.get("type") == "step")
    assert (reset["decision"], reset["tick"]) == (0, 0)
    assert init["facts_columns"]["links"] == ["sinr_db", "capacity_mbps", "is_los"]
    n = len(init["node_ids"])
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rows = reset["facts"]["links"]
    assert len(rows) == len(pairs)
    return init, {pair: (row[0], row[1], bool(row[2])) for pair, row in zip(pairs, rows)}


@pytest.mark.parametrize("with_probes", [False, True], ids=["no-probes", "probes"])
def test_r8_evaluation_query_matches_rl_reset_links(tmp_path, with_probes):
    ini = _query_scenario(tmp_path / "s")
    text = ini.read_text()
    assert RL_OFF in text
    ini.write_text(text.replace(RL_OFF, QUERY_RL))
    init, links = _reset_links(ini, tmp_path / "direct")
    assert init["node_ids"] == [node["id"] for node in QUERY_NODES]
    probes = {"points": QUERY_PROBES} if with_probes else None
    ((result,),), scorer = _query(ini, [[_starts(QUERY_NODES)]], probes=probes,
                                  mode="evaluation")
    query_init = scorer.init
    assert query_init["rl_enabled"] is True and query_init["controlled_indices"] == [2, 0]
    assert (query_init["seed"], query_init["jammer_seed"]) == (QUERY_SEED, QUERY_JAMMER_SEED)
    assert (result.coverage is not None) == with_probes
    for (a, b), (sinr, capacity, los) in links.items():
        label = (f"probes={with_probes}", a, b)
        assert abs(result.sinr_db[a, b] - sinr) <= WRITER_TOL, (*label, "sinr")
        assert abs(result.capacity_mbps[a, b] - capacity) <= WRITER_TOL, (*label, "capacity")
        assert bool(result.is_los[a, b]) == los, (*label, "is_los")


# R12: gateway-free acceptance. Three all-movable nodes on a 300 m x 300 m area (AOI
# 90000 m2, below the 150000 m2 reference fixed cost) start connected and close together.
R12_AREA_M = 300.0
TRIO = [
    {"id": "a0", "role": "peer", "mobility": "fixed", "node_type": "drone",
     "position": {"x": 40.0, "y": 40.0, "z": 20.0}},
    {"id": "g1", "role": "peer", "mobility": "fixed", "node_type": "vehicle",
     "position": {"x": 60.0, "y": 40.0, "z": 2.0}},
    {"id": "g2", "role": "peer", "mobility": "fixed", "node_type": "pedestrian",
     "position": {"x": 40.0, "y": 60.0, "z": 1.5}},
]
# Partial selection: only m0 moves; the other three keep scenario mobility.
MIXED = [
    {"id": "m0", "role": "peer", "mobility": "fixed", "node_type": "drone",
     "position": {"x": 60.0, "y": 60.0, "z": 20.0}},
    {"id": "cv", "role": "peer", "mobility": "constant_velocity", "node_type": "vehicle",
     "position": {"x": 100.0, "y": 80.0, "z": 2.0},
     "velocity": {"vx": 2.0, "vy": 0.5, "vz": 0.0}},
    {"id": "wp", "role": "peer", "mobility": "waypoint", "node_type": "pedestrian",
     "position": {"x": 80.0, "y": 120.0, "z": 1.5},
     "waypoints": [{"t": 0.0, "x": 80.0, "y": 120.0, "z": 1.5},
                   {"t": 0.4, "x": 84.0, "y": 123.0, "z": 1.5}]},
    {"id": "rw", "role": "peer", "mobility": "random_walk", "node_type": "pedestrian",
     "position": {"x": 120.0, "y": 100.0, "z": 1.5},
     "random_walk": {"bounds": {"x_min": 0.0, "x_max": 300.0, "y_min": 0.0,
                                "y_max": 300.0}, "speed_mps": 1.5}},
]
ZERO_COST = {"aerial_fixed_cost_m2": "0", "aerial_cost_m2_per_m": "0",
             "ground_fixed_cost_m2": "0", "ground_cost_m2_per_m": "0"}


def _r12_scenario(root: Path, nodes: list, *, algorithm: str, movable: str,
                  controlled: str, rl_enabled: bool = True,
                  overrides: dict | None = None) -> Path:
    """Synthetic scenario without a mapping file (geofence = [rl] bounds)."""
    count = len(nodes) if controlled == "all" else len(controlled.split(","))
    baseline = {"algorithm": algorithm, "objective": "coverage", "movable_nodes": movable,
                "seed": "7", **BOUNDED_PLANNER, **(overrides or {})}
    text = f"""[scenario]
name = r12-synthetic
seed = 1
duration_s = 0.4
tick_s = 0.1
nodes_file = nodes.json

[rl]
enabled = {"true" if rl_enabled else "false"}
controlled_nodes = {controlled}
max_controlled_nodes = {count}
action_profile = move_2d
decision_interval_s = 0.1
action_type = discrete
reward_type = all_links_los
step_size_m = 1.0
x_min = 0.0
x_max = {R12_AREA_M}
y_min = 0.0
y_max = {R12_AREA_M}
z_min = 0.0
z_max = 100.0

[baseline]
""" + "".join(f"{key} = {value}\n" for key, value in baseline.items()) + SIM_SECTIONS
    root.mkdir(parents=True, exist_ok=True)
    (root / "nodes.json").write_text(json.dumps(nodes, indent=2) + "\n")
    run_ini = root / "run.ini"
    run_ini.write_text(text)
    return run_ini


def _runner_ok(ini: Path, run: Path) -> tuple[dict, dict]:
    launched = _runner(ini, run, "--seeds", "1")
    manifest = artifacts.read_json(run / artifacts.MANIFEST_NAME)
    assert launched.returncode == 0, (manifest.get("error"), launched.stderr[-2000:])
    assert manifest["status"] == "complete"
    return manifest, artifacts.read_json(run / manifest["plan"])


def _displacements(plan: dict) -> dict:
    return {n["id"]: n["displacement_m"] for n in plan["nodes"]}


@pytest.mark.parametrize("method", PLACEMENT_POLICIES)
def test_r12_all_movable_standalone(tmp_path, method):
    ids = [node["id"] for node in TRIO]
    ini = _r12_scenario(tmp_path / "s", TRIO, algorithm=method, movable="all",
                        controlled="all")
    manifest, plan = _runner_ok(ini, tmp_path / "run")
    _assert_v2_artifacts(manifest, plan, "standalone", ids, None)
    positions = _first_tick_positions(tmp_path / "run" / "seed-1")
    for index, expected in enumerate(_planned(plan)):
        _assert_xyz(positions[index], expected, f"{method} {ids[index]}")
    # Reference costs exceed the AOI, so no coverage gain pays for a move.
    assert all(d == 0.0 for d in _displacements(plan).values()), _displacements(plan)


def test_r12_zero_cost_moves_a_node_reference_cost_keeps(tmp_path):
    common = {"algorithm": "geometric", "movable": "all", "controlled": "all"}
    reference = _r12_scenario(tmp_path / "ref", TRIO, **common)
    zero = _r12_scenario(tmp_path / "zero", TRIO, overrides=ZERO_COST, **common)
    _, kept = _runner_ok(reference, tmp_path / "run-ref")
    _, moved = _runner_ok(zero, tmp_path / "run-zero")
    kept_d, moved_d = _displacements(kept), _displacements(moved)
    assert any(kept_d[node] == 0.0 and moved_d[node] > 0.0 for node in kept_d), (
        kept_d, moved_d)


def test_r12_all_movable_evaluation_suite(tmp_path):
    ids = [node["id"] for node in TRIO]
    ini = _r12_scenario(tmp_path / "s", TRIO, algorithm="geometric", movable="all",
                        controlled="all")
    out = tmp_path / "eval"
    result = _evaluate(ini, out, "1", EVAL_POLICIES)
    assert result.returncode == 0, result.stderr[-2000:]
    manifest = json.loads((out / "eval_manifest.json").read_text())
    assert set(manifest["policies"]) == set(EVAL_POLICIES)
    for name, block in manifest["policies"].items():
        assert [e["status"] for e in block["episodes"]] == ["completed"], name
        if name not in PLACEMENT_POLICIES:
            continue
        prep = artifacts.read_json(out / block["baseline"]["manifest"])
        plan = artifacts.read_json(out / block["baseline"]["plan"])
        _assert_v2_artifacts(prep, plan, "evaluation", ids, ids)
        episode_ids, rows = _decision_zero(Path(block["episodes"][0]["episode_dir"]))
        assert episode_ids == ids
        for index, row in enumerate(rows):
            _assert_xyz(row[:3], _planned(plan)[index], f"{name} {ids[index]}")
            assert int(row[6]) == index


MIXED_UNSELECTED = (1, 2, 3)


def test_r12_partial_selection_standalone_keeps_unselected_traces(tmp_path):
    ini = _r12_scenario(tmp_path / "s", MIXED, algorithm="geometric", movable="m0",
                        controlled="m0", overrides=ZERO_COST)
    manifest, plan = _runner_ok(ini, tmp_path / "run")
    _assert_v2_artifacts(manifest, plan, "standalone", ["m0"], None)
    reference = _r12_scenario(tmp_path / "ref", MIXED, algorithm="none", movable="m0",
                              controlled="m0", rl_enabled=False)
    direct = _direct(reference, tmp_path / "direct", "--seeds=1")
    assert direct.returncode == 0, direct.stderr[-2000:]
    ours, theirs = tmp_path / "run" / "seed-1", tmp_path / "direct" / "seed-1"
    trace = _trace(ours, MIXED_UNSELECTED)
    assert len({row[0] for row in trace}) > 1
    assert trace == _trace(theirs, MIXED_UNSELECTED)
    _assert_xyz(_first_tick_positions(ours)[0], _planned(plan)[0], "m0")


def test_r12_partial_selection_evaluation_keeps_unselected_traces(tmp_path):
    ini = _r12_scenario(tmp_path / "s", MIXED, algorithm="geometric", movable="m0",
                        controlled="m0", overrides=ZERO_COST)
    out = tmp_path / "eval"
    result = _evaluate(ini, out, "1", PLACEMENT_POLICIES)
    assert result.returncode == 0, result.stderr[-2000:]
    direct = _direct(ini, tmp_path / "direct", "--rl-mode", "--seed=1")
    assert direct.returncode == 0, direct.stderr[-2000:]
    theirs = _trace(tmp_path / "direct" / "seed-1", MIXED_UNSELECTED)
    assert len({row[0] for row in theirs}) > 1
    manifest = json.loads((out / "eval_manifest.json").read_text())
    for name in PLACEMENT_POLICIES:
        block = manifest["policies"][name]
        prep = artifacts.read_json(out / block["baseline"]["manifest"])
        assert prep["ownership"] == {"mode": "evaluation", "movable_resolved": ["m0"],
                                     "controlled_resolved": ["m0"]}
        episode, = block["episodes"]
        positions, = Path(episode["episode_dir"]).rglob("positions.csv")
        assert _trace(positions.parent, MIXED_UNSELECTED) == theirs, name
