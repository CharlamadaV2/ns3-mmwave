"""adapter.prepare() argument handling, seed/identity recording, ownership and region gates.

No real simulator runs: solver.solve is replaced by the `stub_planner` fixture (or a
local stand-in), and `fake_child` only provides an executable file to hash. Sources:
scripts/baselines/README.md ("What preparation does", "Hold executor contract",
"Channel scoring", "Penalty scale", the manifest field table) and the [baseline] table
("geometric accepts seed and max_iterations ... records them as null with status unused /
not_applicable").
"""

import copy
import json


from scripts.baselines import adapter, artifacts
from scripts.baselines.tests.conftest import MAPPING, NODES, scenario_ini, write_scenario
from scripts.sim_support import parse_seed_spec

CONTROLLED_LINE = "controlled_nodes = uav-a, uav-b, uav-c, walker"


def _eval_text(overrides=None, controlled="uav-a, uav-c", edit=None):
    text = scenario_ini(overrides).replace(CONTROLLED_LINE,
                                           f"controlled_nodes = {controlled}")
    return edit(text) if edit else text


def _eval_scenario(root, overrides=None, controlled="uav-a, uav-c", edit=None, **kwargs):
    return write_scenario(root, ini_text=_eval_text(overrides, controlled, edit), **kwargs)


def _prepare_eval(tmp_path, ini, binary, method="geometric", **kwargs):
    return adapter.prepare(ini, method, tmp_path / "eval" / method / "baseline",
                           mode="evaluation", sim_binary=binary, **kwargs)


def _manifest(prep):
    return artifacts.read_json(prep / artifacts.MANIFEST_NAME)


def _forbid_solver(monkeypatch):
    def forbidden(request, log):
        raise AssertionError("the planner must not run")
    monkeypatch.setattr("scripts.baselines.solver.solve", forbidden)


# --- arguments and the preparation directory --------------------------------------------


def test_existing_empty_prep_dir_is_used(tmp_path, stub_planner, fake_child):
    ini = write_scenario(tmp_path / "s")
    prep = tmp_path / "prep"
    prep.mkdir()
    prepared = adapter.prepare(ini, "geometric", prep, mode="standalone",
                               sim_binary=fake_child)
    assert _manifest(prep)["status"] == "prepared"
    assert prepared.effective_run_config.is_file()


# --- recorded optimizer settings --------------------------------------------------------

def test_method_none_records_seed_fields_as_not_applicable(tmp_path, monkeypatch):
    _forbid_solver(monkeypatch)
    ini = write_scenario(tmp_path / "s", overrides={"algorithm": "none"})
    prepared = adapter.prepare(ini, None, tmp_path / "run", mode="standalone")
    manifest = artifacts.read_json(prepared.manifest_path)
    assert (manifest["method"], manifest["objective"]) == ("none", None)
    assert (manifest["planner_seed"], manifest["planner_seed_status"]) == (
        None, "not_applicable")
    assert (manifest["max_iterations"], manifest["max_iterations_status"]) == (
        None, "not_applicable")


def test_geometric_records_ini_seed_and_iterations_as_null(tmp_path, stub_planner,
                                                           fake_child):
    ini = write_scenario(tmp_path / "s", overrides={"seed": "0", "max_iterations": "1"})
    prepared = adapter.prepare(ini, "geometric", tmp_path / "run", mode="standalone",
                               sim_binary=fake_child)
    manifest = artifacts.read_json(prepared.manifest_path)
    assert (manifest["planner_seed"], manifest["planner_seed_status"]) == (None, "unused")
    assert (manifest["max_iterations"], manifest["max_iterations_status"]) == (
        None, "not_applicable")


def test_optimization_records_seed_zero_and_one_iteration(tmp_path, stub_planner,
                                                          fake_child):
    ini = write_scenario(tmp_path / "s", overrides={"algorithm": "optimization",
                                                    "seed": "0", "max_iterations": "1"})
    prepared = adapter.prepare(ini, None, tmp_path / "run", mode="standalone",
                               sim_binary=fake_child)
    manifest = artifacts.read_json(prepared.manifest_path)
    assert (manifest["planner_seed"], manifest["planner_seed_status"]) == (0, "used")
    assert (manifest["max_iterations"], manifest["max_iterations_status"]) == (1, "used")


# --- planning seed and run_id -----------------------------------------------------------

def test_planning_seed_is_first_of_an_expanded_range(tmp_path, stub_planner, fake_child):
    text = scenario_ini().replace("seed = 1\n", "seed = 1\nrun_id = 3\n")
    ini = write_scenario(tmp_path / "s", ini_text=text)
    seeds = parse_seed_spec("5-7")
    prepared = adapter.prepare(ini, "geometric", tmp_path / "run", mode="standalone",
                               sim_binary=fake_child, simulation_seeds=seeds)
    manifest = artifacts.read_json(prepared.manifest_path)
    scoring = manifest["channel_scoring"]
    assert manifest["simulation_seeds"] == [5, 6, 7]
    assert (scoring["planning_seed"], scoring["jammer_seed"],
            scoring["planning_run_id"]) == (5, 5, 3)
    log = (prepared.prep_dir / "planner.log").read_text()
    assert "planning_seed=5 run_id=3" in log


def test_planning_seed_defaults_to_simulator_default(tmp_path, stub_planner, fake_child):
    ini = write_scenario(tmp_path / "s", ini_text=scenario_ini().replace("seed = 1\n", ""))
    prepared = adapter.prepare(ini, "geometric", tmp_path / "run", mode="standalone",
                               sim_binary=fake_child)
    scoring = artifacts.read_json(prepared.manifest_path)["channel_scoring"]
    assert (scoring["planning_seed"], scoring["planning_run_id"]) == (42, 1)
    assert artifacts.read_json(prepared.manifest_path)["simulation_seeds"] is None


# --- evaluation ownership and region ----------------------------------------------------

def test_evaluation_accepts_whitespace_and_empty_tokens(tmp_path, stub_planner, fake_child):
    ini = _eval_scenario(tmp_path / "s", overrides={"movable_nodes": " uav-a ,uav-c # set"},
                         controlled="  uav-c ,, uav-a , ")
    prepared = _prepare_eval(tmp_path, ini, fake_child)
    manifest = artifacts.read_json(prepared.manifest_path)
    assert manifest["ownership"] == {"mode": "evaluation",
                                     "movable_resolved": ["uav-a", "uav-c"],
                                     "controlled_resolved": ["uav-c", "uav-a"]}
    plan = json.loads(prepared.plan_path.read_text())
    slots = {n["id"]: n["slot"] for n in plan["nodes"] if n["selected"]}
    assert slots == {"uav-c": 0, "uav-a": 1}


def test_rectangle_geofence_with_partial_rl_bounds_uses_simulator_defaults(
        tmp_path, stub_planner, fake_child):
    # With a rectangle geofence the explicit-bounds rule does not apply; the region is the
    # rectangle intersected with the bounds the simulator enforces (absent x_max -> 2000).
    mapping = {**copy.deepcopy(MAPPING),
               "geofence": {"source": "rectangle_xy_m", "x_min": 0.0, "x_max": 500.0,
                            "y_min": 0.0, "y_max": 400.0}}
    ini = _eval_scenario(tmp_path / "s", mapping=mapping,
                         edit=lambda t: t.replace("x_max = 400.0\n", ""))
    prepared = _prepare_eval(tmp_path, ini, fake_child)
    log = (prepared.prep_dir / "planner.log").read_text()
    region = {"x_max": 500.0, "x_min": 0.0, "y_max": 400.0, "y_min": 0.0}
    assert f"evaluation_region={json.dumps(region, sort_keys=True)}" in log
    assert artifacts.read_json(prepared.manifest_path)["geofence"]["x_max"] == 500.0


# --- waypoint policy --------------------------------------------------------------------


def test_unselected_waypoint_node_is_fine_under_reject(tmp_path, stub_planner, fake_child):
    ini = _eval_scenario(tmp_path / "s", overrides={"waypoint_policy": "reject"})
    prepared = _prepare_eval(tmp_path, ini, fake_child)
    plan = json.loads(prepared.plan_path.read_text())
    uav_b = next(n for n in plan["nodes"] if n["id"] == "uav-b")
    assert not uav_b["selected"] and uav_b["displacement_m"] == 0.0


# --- penalty scale ----------------------------------------------------------------------


# --- datum through prepare --------------------------------------------------------------

def test_ground_level_nodes_prepare(tmp_path, stub_planner, fake_child):
    nodes = copy.deepcopy(NODES)
    for node in nodes:
        node.setdefault("position", {})["z"] = 0.0
        for waypoint in node.get("waypoints", []):
            waypoint["z"] = 0.0
    ini = _eval_scenario(tmp_path / "s", nodes=nodes)
    prepared = _prepare_eval(tmp_path, ini, fake_child)
    plan = json.loads(prepared.plan_path.read_text())
    assert {n["planned"]["z"] for n in plan["nodes"]} == {0.0}
