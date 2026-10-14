"""Plan fingerprint (v2), adapted-code and channel-runtime identity, end to end.

Contract: scripts/baselines/README.md baseline_manifest.json table -- `fingerprint` hashes
"method, objective, planner seed, max_iterations, waypoint policy, mapping hash,
penalty/grid/probe/threshold settings, planning seed and run_id, mode flag, band, the four
effective-input hashes, binary, channel-runtime and adapted-code hashes,
planner_settings, and ownership; never timing or paths"; "One plan is reused for every
simulation seed" (planning seed = first simulation seed). End-to-end cases run the real
geometric / optimization strategies against fake_query.py through a composite
executable that serves fake_child.py for the simulation itself.
"""

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

from scripts.baselines import artifacts, runner
from scripts.baselines.tests.conftest import FAKE_CHILD, FAKE_QUERY, write_scenario

FREE_MOVES = {"candidate_grid_cells": "16", "coverage_grid_cells": "16",
              "max_iterations": "15", "aerial_fixed_cost_m2": "0",
              "ground_fixed_cost_m2": "0", "aerial_cost_m2_per_m": "0",
              "ground_cost_m2_per_m": "0"}

BASE = {
    "method": "optimization", "objective": "balanced", "planner_seed": 7,
    "max_iterations": 60, "waypoint_policy": "reject", "mapping_sha256": None,
    "penalties": {"aerial": {"fixed_cost_m2": 150000.0, "cost_m2_per_m": 500.0,
                             "max_displacement_m": None},
                  "ground": {"fixed_cost_m2": 150000.0, "cost_m2_per_m": 100.0,
                             "max_displacement_m": None},
                  "aoi_m2": 160000.0,
                  "fixed_cost_over_aoi": {"aerial": 0.9375, "ground": 0.9375}},
    "channel_scoring": {
        "contract": "mesh_channel_query_v1", "isolation": "fork_per_layout",
        "planning_seed": 1, "planning_run_id": 1, "jammer_seed": 1, "mode_flag": None,
        "band": "mmwave", "band_source": "default", "sinr_threshold_db": -6.7,
        "coverage_sinr_db": -6.7,
        "probe": {"height_m": 1.5, "rx_gain_dbi": 12.0, "grid_cells": 400,
                  "min_resolution_m": 5.0, "cell_m": 20.0, "count": 400},
        "candidate_grid": {"cells": 400, "min_resolution_m": 5.0, "cell_m": 20.0,
                           "count": 400},
        "queries": {"requests": 3, "layouts": 70, "links": 1050, "probe_links": 1000,
                    "wall_s": 4.2}},
    "effective_scenario_identity": {"run_config": "effective-inputs/run.ini",
                                    "run_ini_sha256": "a" * 64, "nodes_json_sha256": "b" * 64,
                                    "buildings_json_sha256": None,
                                    "jammers_json_sha256": None},
    "source_scenario_identity": {"run_config": "source-inputs/run.ini",
                                 "run_ini_sha256": "9" * 64, "nodes_json_sha256": "b" * 64,
                                 "buildings_json_sha256": None,
                                 "jammers_json_sha256": None},
    "sim_binary_sha256": "c" * 64, "channel_runtime_sha256": "d" * 64,
    "planner_code_sha256": "e" * 64,
    "planner_settings": {"t0_area_frac": 0.02, "move_weights": {"nudge": 0.6, "jump": 0.4}},
    "ownership": {"mode": "standalone", "movable_resolved": ["a", "b"],
                  "controlled_resolved": None},
    "simulation_seeds": [1, 2], "seeds": [], "status": "prepared", "error": None,
    "planner_log": "planner.log", "sim_log": "sim.log", "source_inputs": "source-inputs",
    "effective_inputs": "effective-inputs", "planner_wall_s": 1.0,
}


def _set(manifest: dict, path: str, value) -> dict:
    changed = copy.deepcopy(manifest)
    *parents, leaf = path.split(".")
    target = changed
    for key in parents:
        target = target[key]
    target[leaf] = value
    return changed


@pytest.mark.parametrize("path,value", [
    ("method", "geometric"),
    ("objective", "coverage"),
    ("max_iterations", 61),
    ("waypoint_policy", "translate"),
    ("mapping_sha256", "1" * 64),
    ("penalties.aerial.fixed_cost_m2", 0.0),
    ("penalties.aerial.cost_m2_per_m", 1.0),
    ("penalties.ground.fixed_cost_m2", 1.0),
    ("penalties.ground.max_displacement_m", 10.0),
    ("channel_scoring.candidate_grid.min_resolution_m", 1.0),
    ("channel_scoring.candidate_grid.cell_m", 10.0),
    ("channel_scoring.candidate_grid.count", 1600),
    ("channel_scoring.probe.grid_cells", 100),
    ("channel_scoring.probe.min_resolution_m", 1.0),
    ("channel_scoring.probe.cell_m", 40.0),
    ("channel_scoring.probe.count", 100),
    ("channel_scoring.sinr_threshold_db", -5.0),
    ("channel_scoring.jammer_seed", 9),
    ("channel_scoring.band", "sub-6"),
    ("channel_scoring.mode_flag", "--rl-mode"),
    ("effective_scenario_identity.run_ini_sha256", "0" * 64),
    ("effective_scenario_identity.buildings_json_sha256", "0" * 64),
    ("effective_scenario_identity.jammers_json_sha256", "0" * 64),
    ("ownership.mode", "evaluation"),
    ("planner_settings.move_weights.jump", 0.5),
])
def test_fingerprint_covers_each_listed_field(path, value):
    assert artifacts.fingerprint(_set(BASE, path, value)) != artifacts.fingerprint(BASE)


@pytest.mark.parametrize("path,value", [
    ("simulation_seeds", [1, 3]),
    ("seeds", [{"seed": 1, "status": "complete", "summary": "seed-1/summary.json"}]),
    ("status", "complete"),
    ("error", "boom"),
    ("planner_log", "elsewhere/planner.log"),
    ("sim_log", None),
    ("source_inputs", "x"),
    ("effective_inputs", "y"),
    ("planner_wall_s", 123.0),
    ("source_scenario_identity.run_ini_sha256", "8" * 64),
    ("source_scenario_identity.run_config", "other/run.ini"),
    ("effective_scenario_identity.run_config", "moved/run.ini"),
    ("penalties.aoi_m2", 1.0),
    ("channel_scoring.band_source", "cli"),
    ("fingerprint", "f" * 64),
])
def test_fingerprint_ignores_outcome_timing_and_paths(path, value):
    assert artifacts.fingerprint(_set(BASE, path, value)) == artifacts.fingerprint(BASE)


# --- end to end through the runner ----------------------------------------------------------

def _composite(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "mesh-sim-composite"
    path.write_text(
        f"#!{sys.executable}\nimport runpy, sys\n"
        f"target = {str(FAKE_QUERY)!r} if '--channel-query' in sys.argv[1:] "
        f"else {str(FAKE_CHILD)!r}\nsys.argv[0] = target\n"
        "runpy.run_path(target, run_name='__main__')\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def composite(tmp_path) -> Path:
    return _composite(tmp_path / "bin")


@pytest.fixture
def query_record(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "query-record.jsonl"
    monkeypatch.setenv("FAKE_QUERY_RECORD", str(path))
    return path


def _run(binary, ini, out, *extra) -> dict:
    code = runner.main(["--sim-binary", str(binary), "--run-config", str(ini),
                        "--output-dir", str(out), *extra])
    manifest = artifacts.read_json(Path(out) / artifacts.MANIFEST_NAME)
    assert code == 0, manifest["error"]
    return manifest


def _query_argv(record: Path) -> list:
    lines = [json.loads(line) for line in record.read_text().splitlines()]
    return [line["argv"] for line in lines if "argv" in line]


def test_geometric_standalone_records_its_identity(tmp_path, composite, query_record):
    ini = write_scenario(tmp_path / "s", overrides=FREE_MOVES)
    out = tmp_path / "run"
    manifest = _run(composite, ini, out, "--seeds", "3,4")

    run = out.resolve()
    assert _query_argv(query_record) == [[f"--run-config={run}/effective-inputs.partial/run.ini",
                                          "--channel-query", "--seed=3"]]
    assert manifest["status"] == "complete"
    scoring = manifest["channel_scoring"]
    assert (scoring["contract"], scoring["isolation"]) == ("mesh_channel_query_v1",
                                                           "fork_per_layout")
    assert (scoring["mode_flag"], scoring["planning_seed"], scoring["jammer_seed"],
            scoring["planning_run_id"]) == (None, 3, 3, 1)
    assert (scoring["band"], scoring["band_source"]) == ("mmwave", "default")
    assert scoring["queries"]["requests"] >= 1 and scoring["queries"]["layouts"] >= 1
    assert manifest["sim_binary_sha256"] == artifacts.sha256_file(composite)
    runtime = artifacts.channel_runtime_identity(composite)
    assert manifest["channel_runtime_sha256"] == runtime["sha256"]
    assert manifest["channel_runtime_files"] == [str(composite.resolve())]
    assert manifest["planner_code_sha256"] == (
        artifacts.planner_code_identity()["aggregate_sha256"])
    assert manifest["planner_settings"]["strategy"] == "sequential_greedy"
    assert manifest["ownership"] == {"mode": "standalone",
                                     "movable_resolved": ["uav-a", "uav-c"],
                                     "controlled_resolved": None}
    assert manifest["fingerprint"] == artifacts.fingerprint(manifest)
    assert (manifest["planner_seed"], manifest["planner_seed_status"]) == (None, "unused")
    assert (manifest["max_iterations"], manifest["max_iterations_status"]) == (
        None, "not_applicable")

    plan = artifacts.read_json(out / "effective-inputs/baseline-plan.json")
    assert (plan["baseline_plan_version"], plan["method"], plan["objective"]) == (
        2, "geometric", "coverage")
    assert plan["initial_displacement_m_total"] == pytest.approx(
        sum(n["displacement_m"] for n in plan["nodes"]))
    assert manifest["initial_displacement_m_total"] == plan["initial_displacement_m_total"]
    assert plan["initial_displacement_m_total"] > 0
    nodes = {n["id"]: n for n in json.loads((out / "effective-inputs/nodes.json").read_text())}
    for entry in plan["nodes"]:
        assert entry["slot"] is None
        assert entry["planned"]["z"] == entry["original"]["z"]
        if not entry["selected"]:
            assert entry["planned"] == entry["original"]
        elif entry["displacement_m"] > 0:
            position = nodes[entry["id"]]["position"]
            assert (position["x"], position["y"], position["z"]) == (
                entry["planned"]["x"], entry["planned"]["y"], entry["planned"]["z"])
    log = (out / "planner.log").read_text()
    assert "channel query: " in log and "--channel-query --seed=3" in log
    assert "method=geometric objective=coverage" in log
    # The simulator archives every effective input, plan included.
    assert (out / "inputs").is_dir()


def test_fingerprint_ignores_paths_and_later_seeds_but_not_the_first_seed(tmp_path, composite):
    ini = write_scenario(tmp_path / "s", overrides=FREE_MOVES)
    elsewhere = tmp_path / "deeper" / "copy" / "of-scenario"
    shutil.copytree(ini.parent, elsewhere)
    other_bin = _composite(tmp_path / "other-bin")

    first = _run(composite, ini, tmp_path / "r1", "--seeds", "1,2")
    moved = _run(other_bin, elsewhere / "run.ini", tmp_path / "x" / "y" / "r2",
                 "--seeds", "1,3")
    swapped = _run(composite, ini, tmp_path / "r3", "--seeds", "2,1")

    assert moved["fingerprint"] == first["fingerprint"]
    assert (tmp_path / "x/y/r2/effective-inputs/baseline-plan.json").read_bytes() == (
        (tmp_path / "r1/effective-inputs/baseline-plan.json").read_bytes())
    assert moved["effective_scenario_identity"] == first["effective_scenario_identity"]
    assert swapped["channel_scoring"]["planning_seed"] == 2
    assert swapped["fingerprint"] != first["fingerprint"]


@pytest.mark.parametrize("change", ["band", "objective", "movable", "cap", "grid",
                                    "comment"])
def test_fingerprint_changes_with_each_planning_input(tmp_path, composite, change):
    base_ini = write_scenario(tmp_path / "base", overrides=FREE_MOVES)
    reference = _run(composite, base_ini, tmp_path / "r0")
    extra = ()
    overrides = dict(FREE_MOVES)
    if change == "band":
        extra = ("--band", "sub-6")
    elif change == "objective":
        overrides["objective"] = "resilience"
    elif change == "movable":
        overrides["movable_nodes"] = "uav-a"
    elif change == "cap":
        overrides["aerial_max_displacement_m"] = "500"
    elif change == "grid":
        overrides["coverage_grid_cells"] = "9"
    ini = write_scenario(tmp_path / "changed", overrides=overrides)
    if change == "comment":
        ini.write_text("# a comment changes the effective run.ini hash\n" + ini.read_text())
    changed = _run(composite, ini, tmp_path / "r1", *extra)
    assert changed["fingerprint"] != reference["fingerprint"]
    assert changed["fingerprint"] == artifacts.fingerprint(changed)


def test_optimization_is_deterministic_per_planner_seed(tmp_path, composite):
    overrides = {**FREE_MOVES, "algorithm": "optimization"}
    ini = write_scenario(tmp_path / "s", overrides=overrides)
    first = _run(composite, ini, tmp_path / "r1")
    again = _run(composite, ini, tmp_path / "r2")
    assert (first["planner_seed"], first["planner_seed_status"]) == (7, "used")
    assert (first["max_iterations"], first["max_iterations_status"]) == (15, "used")
    assert first["fingerprint"] == again["fingerprint"]
    assert (tmp_path / "r1/effective-inputs/baseline-plan.json").read_bytes() == (
        (tmp_path / "r2/effective-inputs/baseline-plan.json").read_bytes())
    reseeded = _run(composite, write_scenario(tmp_path / "s2",
                                              overrides={**overrides, "seed": "8"}),
                    tmp_path / "r3")
    longer = _run(composite, write_scenario(tmp_path / "s3",
                                            overrides={**overrides, "max_iterations": "16"}),
                  tmp_path / "r4")
    assert len({first["fingerprint"], reseeded["fingerprint"], longer["fingerprint"]}) == 3


def test_balanced_core_fraction_reaches_the_fingerprint(tmp_path, composite):
    reference = _run(composite, write_scenario(tmp_path / "a", overrides=FREE_MOVES),
                     tmp_path / "r0")
    ini = write_scenario(tmp_path / "b", overrides={**FREE_MOVES,
                                                    "balanced_core_fraction": "0.25"})
    manifest = _run(composite, ini, tmp_path / "r1")
    assert reference["planner_settings"]["balanced_core_fraction"] == 0.5
    assert manifest["planner_settings"]["balanced_core_fraction"] == 0.25
    assert manifest["fingerprint"] != reference["fingerprint"]


def test_requested_algorithm_does_not_change_the_plan_identity(tmp_path, composite):
    """The effective run.ini always says `algorithm = none`, so only the method counts."""
    from_ini = _run(composite, write_scenario(tmp_path / "a", overrides=FREE_MOVES),
                    tmp_path / "r0")
    overridden = _run(composite, write_scenario(
        tmp_path / "b", overrides={**FREE_MOVES, "algorithm": "optimization"}),
        tmp_path / "r1", "--algorithm", "geometric")
    assert (from_ini["requested_algorithm"], overridden["requested_algorithm"]) == (
        "geometric", "optimization")
    assert overridden["method"] == "geometric"
    assert overridden["effective_scenario_identity"]["run_ini_sha256"] == (
        from_ini["effective_scenario_identity"]["run_ini_sha256"])
    assert overridden["fingerprint"] == from_ini["fingerprint"]
