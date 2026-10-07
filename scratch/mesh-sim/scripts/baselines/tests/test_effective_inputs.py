"""Effective INI edits, staging, nodes.json rewrite, asset rebasing, and source snapshots."""

import copy
import json

import pytest

from scripts.baselines import adapter, artifacts, config, effective_inputs
from scripts.baselines.adapter import BaselinePreparationError
from scripts.baselines.effective_inputs import InputError
from scripts.baselines.solver import PlanResult
from scripts.baselines.tests.conftest import (CHANNEL_FACTS, NODES, scenario_ini,
                                              stub_positions, write_scenario)

COMMENTED_INI = """# synthetic scenario header
[scenario]
name = commented   # trailing note
nodes_file = nodes.json
buildings_file = ../shared/buildings.json

; policy knobs
[rl]
enabled = true
controlled_nodes = uav-a, uav-c
x_min = 0.0
x_max = 400.0
y_min = 0.0
y_max = 400.0

[baseline]
algorithm = geometric   # picked for this run
objective = coverage
movable_nodes = uav-a, uav-c
mapping_file = maps/mapping.json

[output]
dir = somewhere
"""


def _commented_scenario(tmp_path):
    ini = write_scenario(tmp_path / "scenario", ini_text=COMMENTED_INI)
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared/buildings.json").write_text('{"buildings": []}\n')
    (ini.parent / "maps").mkdir()
    (ini.parent / "mapping.json").rename(ini.parent / "maps/mapping.json")
    return ini


def _changed_lines(before: str, after: str) -> list:
    return [(a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b]


def test_edit_preserves_bytes_except_edited_keys():
    edits = {("baseline", "algorithm"): "none", ("rl", "enabled"): "false",
             ("scenario", "buildings_file"): "buildings.json"}
    after = effective_inputs.edit_ini_text(COMMENTED_INI, edits)
    assert len(after.splitlines()) == len(COMMENTED_INI.splitlines())
    assert _changed_lines(COMMENTED_INI, after) == [
        ("buildings_file = ../shared/buildings.json", "buildings_file = buildings.json"),
        ("enabled = true", "enabled = false"),
        ("algorithm = geometric   # picked for this run",
         "algorithm = none   # picked for this run"),
    ]


def test_edit_appends_missing_keys_and_sections():
    text = "[scenario]\nname = x\n\n[baseline]\nalgorithm = geometric\n"
    after = effective_inputs.edit_ini_text(
        text, {("scenario", "nodes_file"): "nodes.json", ("rl", "enabled"): "false"})
    assert after == ("[scenario]\nname = x\nnodes_file = nodes.json\n\n"
                     "[baseline]\nalgorithm = geometric\n\n[rl]\nenabled = false\n")


def test_edit_preserves_crlf_and_missing_final_newline():
    text = "[baseline]\r\nalgorithm = geometric\r\nobjective = coverage"
    after = effective_inputs.edit_ini_text(text, {("baseline", "algorithm"): "none",
                                                  ("baseline", "mapping_file"): "m.json"})
    assert after == ("[baseline]\r\nalgorithm = none\r\nobjective = coverage\r\n"
                     "mapping_file = m.json\r\n")


@pytest.mark.parametrize("mode,enabled", [("standalone", "false"), ("evaluation", "true")])
def test_effective_ini_per_mode(tmp_path, stub_planner, fake_child, mode, enabled):
    ini = _commented_scenario(tmp_path)
    prepared = adapter.prepare(ini, "geometric", tmp_path / "out" / mode / "baseline",
                               mode=mode, sim_binary=fake_child)
    text = prepared.effective_run_config.read_text()
    assert _changed_lines(COMMENTED_INI, text) == [
        ("buildings_file = ../shared/buildings.json", "buildings_file = buildings.json"),
        *([("enabled = true", "enabled = false")] if enabled == "false" else []),
        ("algorithm = geometric   # picked for this run",
         "algorithm = none   # picked for this run"),
        ("mapping_file = maps/mapping.json", "mapping_file = mapping.json"),
    ]
    parsed = config.read_ini(prepared.effective_run_config)
    assert config.ini_value(parsed, "rl", "enabled") == enabled
    assert config.load_baseline(prepared.effective_run_config).algorithm == "none"
    effective = prepared.effective_run_config.parent
    assert sorted(p.name for p in effective.iterdir()) == [
        "baseline-plan.json", "buildings.json", "mapping.json", "nodes.json", "run.ini"]
    assert (effective / "buildings.json").read_bytes() == (
        (tmp_path / "shared/buildings.json").read_bytes())
    identity = artifacts.read_json(prepared.manifest_path)["effective_scenario_identity"]
    assert identity["buildings_json_sha256"] == artifacts.sha256_file(
        tmp_path / "shared/buildings.json")
    assert identity["jammers_json_sha256"] is None


def test_source_snapshot_and_sources_unmodified(tmp_path, stub_planner, fake_child):
    ini = _commented_scenario(tmp_path)
    sources = [ini, ini.parent / "nodes.json", ini.parent / "maps/mapping.json",
               tmp_path / "shared/buildings.json"]
    before = {path: artifacts.sha256_file(path) for path in sources}
    prepared = adapter.prepare(ini, "geometric", tmp_path / "run", mode="standalone",
                               sim_binary=fake_child)
    assert {path: artifacts.sha256_file(path) for path in sources} == before
    snapshot = prepared.prep_dir / "source-inputs"
    assert sorted(p.name for p in snapshot.iterdir()) == [
        "buildings.json", "mapping.json", "nodes.json", "run.ini"]
    assert (snapshot / "run.ini").read_bytes() == ini.read_bytes()
    identity = artifacts.read_json(prepared.manifest_path)["source_scenario_identity"]
    assert identity == {
        "run_config": "source-inputs/run.ini",
        "run_ini_sha256": before[ini],
        "nodes_json_sha256": before[ini.parent / "nodes.json"],
        "buildings_json_sha256": before[tmp_path / "shared/buildings.json"],
        "jammers_json_sha256": None,
    }


def test_relocated_run_keeps_relative_references(tmp_path, stub_planner, fake_child):
    ini = write_scenario(tmp_path / "s")
    prepared = adapter.prepare(ini, "geometric", tmp_path / "run", mode="standalone",
                               sim_binary=fake_child)
    moved = tmp_path / "elsewhere"
    prepared.prep_dir.rename(moved)
    manifest = artifacts.read_json(moved / "baseline_manifest.json")
    for key in ("plan", "planner_log", "source_inputs", "effective_inputs"):
        assert (moved / manifest[key]).exists()
    identity = manifest["effective_scenario_identity"]
    assert artifacts.sha256_file(moved / identity["run_config"]) == identity["run_ini_sha256"]


def test_missing_asset_is_an_error(tmp_path, stub_planner, fake_child):
    ini = write_scenario(tmp_path / "s", ini_text=scenario_ini().replace(
        "nodes_file = nodes.json", "nodes_file = nodes.json\njammers_file = jammers.json"))
    with pytest.raises(BaselinePreparationError, match="jammers file not found"):
        adapter.prepare(ini, "geometric", tmp_path / "run", mode="standalone",
                        sim_binary=fake_child)
    assert not (tmp_path / "run/source-inputs").exists()


def test_same_basename_assets_are_an_error(tmp_path, stub_planner, fake_child):
    ini = write_scenario(tmp_path / "s", ini_text=scenario_ini().replace(
        "nodes_file = nodes.json",
        "nodes_file = nodes.json\nbuildings_file = a/data.json\njammers_file = b/data.json"))
    for sub in ("a", "b"):
        (ini.parent / sub).mkdir()
        (ini.parent / sub / "data.json").write_text("[]\n")
    with pytest.raises(BaselinePreparationError, match="same basename"):
        adapter.prepare(ini, "geometric", tmp_path / "run", mode="standalone",
                        sim_binary=fake_child)


def test_reserved_basename_is_an_error(tmp_path, stub_planner, fake_child):
    ini = write_scenario(tmp_path / "s", ini_text=scenario_ini().replace(
        "nodes_file = nodes.json", "nodes_file = nodes.json\nbuildings_file = b/nodes.json"))
    (ini.parent / "b").mkdir()
    (ini.parent / "b/nodes.json").write_text("[]\n")
    with pytest.raises(BaselinePreparationError, match="same basename"):
        adapter.prepare(ini, "geometric", tmp_path / "run", mode="standalone",
                        sim_binary=fake_child)


def test_duplicate_section_or_key_fails_before_writing(tmp_path, stub_planner, fake_child):
    text = scenario_ini() + "\n[rl]\nenabled = false\n"
    ini = write_scenario(tmp_path / "s", ini_text=text)
    with pytest.raises(BaselinePreparationError, match="already exists"):
        adapter.prepare(ini, "geometric", tmp_path / "run", mode="standalone",
                        sim_binary=fake_child)
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["baseline_manifest.json"]


def test_unselected_nodes_numerically_identical(tmp_path, stub_planner, fake_child):
    ini = write_scenario(tmp_path / "s")
    prepared = adapter.prepare(ini, "geometric", tmp_path / "run", mode="standalone",
                               sim_binary=fake_child)
    rewritten = json.loads((prepared.prep_dir / "effective-inputs/nodes.json").read_text())
    original = {n["id"]: n for n in NODES}
    for entry in rewritten:
        if entry["id"] in ("uav-a", "uav-c"):
            assert entry["position"]["z"] == original[entry["id"]]["position"]["z"]
            assert entry["position"] != original[entry["id"]]["position"]
            assert {k: v for k, v in entry.items() if k != "position"} == {
                k: v for k, v in original[entry["id"]].items() if k != "position"}
        else:
            assert entry == original[entry["id"]]
    plan = json.loads(prepared.plan_path.read_text())
    planned = {n["id"]: n["planned"] for n in plan["nodes"]}
    uav_a = next(e for e in rewritten if e["id"] == "uav-a")
    assert (uav_a["position"]["x"], uav_a["position"]["y"]) == (
        planned["uav-a"]["x"], planned["uav-a"]["y"])


def _waypoint_scenario(tmp_path, policy):
    ini = write_scenario(tmp_path / "s", overrides={"movable_nodes": "uav-a, uav-b",
                                                    "waypoint_policy": policy})
    ini.write_text(ini.read_text().replace("controlled_nodes = uav-a, uav-b, uav-c, walker",
                                           "controlled_nodes = uav-a, uav-b"))
    return ini


def test_controlled_waypoint_node_reject(tmp_path, stub_planner, fake_child):
    ini = _waypoint_scenario(tmp_path, "reject")
    with pytest.raises(BaselinePreparationError, match="node 'uav-b' uses waypoint mobility"):
        adapter.prepare(ini, "geometric", tmp_path / "out/geometric/baseline",
                        sim_binary=fake_child)
    prep = tmp_path / "out/geometric/baseline"
    assert not (prep / "effective-inputs").exists()
    assert not (prep / "effective-inputs.partial").exists()


def test_selected_stationary_waypoint_rejected_before_solver(tmp_path, stub_planner, fake_child,
                                                             monkeypatch):
    ini = _waypoint_scenario(tmp_path, "reject")
    called = False

    def stationary_solve(request, log):
        nonlocal called
        called = True
        positions = stub_positions(request)
        positions["uav-b"] = (220.0, 160.0, 30.0)
        return PlanResult(positions=positions, channel=dict(CHANNEL_FACTS))

    monkeypatch.setattr("scripts.baselines.solver.solve", stationary_solve)
    with pytest.raises(BaselinePreparationError, match="node 'uav-b' uses waypoint mobility"):
        adapter.prepare(ini, "geometric", tmp_path / "out/geometric/baseline",
                        sim_binary=fake_child)
    assert not called


def test_controlled_waypoint_node_translate(tmp_path, stub_planner, fake_child):
    ini = _waypoint_scenario(tmp_path, "translate")
    prepared = adapter.prepare(ini, "geometric", tmp_path / "out/geometric/baseline",
                               sim_binary=fake_child)
    plan = {n["id"]: n for n in json.loads(prepared.plan_path.read_text())["nodes"]}
    assert plan["uav-b"]["original"] == {"x": 220.0, "y": 160.0, "z": 30.0}
    target = plan["uav-b"]["planned"]
    dx, dy = target["x"] - 220.0, target["y"] - 160.0
    rewritten = {n["id"]: n for n in json.loads(
        (prepared.prep_dir / "effective-inputs/nodes.json").read_text())}
    source = next(n for n in NODES if n["id"] == "uav-b")
    for new, old in zip(rewritten["uav-b"]["waypoints"], source["waypoints"]):
        assert new["x"] == pytest.approx(old["x"] + dx, abs=1e-9)
        assert new["y"] == pytest.approx(old["y"] + dy, abs=1e-9)
        assert (new["t"], new["z"]) == (old["t"], old["z"])
    first = rewritten["uav-b"]["waypoints"][0]
    assert (first["x"], first["y"]) == pytest.approx((target["x"], target["y"]))
    assert (rewritten["uav-b"]["position"]["x"], rewritten["uav-b"]["position"]["y"]) == (
        target["x"], target["y"])
    assert rewritten["uav-b"]["position"]["z"] == source["position"]["z"]


def test_rewrite_nodes_rules():
    nodes = copy.deepcopy(NODES)
    out = effective_inputs.rewrite_nodes(nodes, {"truck": (1.0, 2.0)}, "reject")
    assert nodes == NODES
    truck = next(n for n in out if n["id"] == "truck")
    assert truck["position"] == {"x": 1.0, "y": 2.0, "z": 2.0}
    assert truck["velocity"] == {"vx": 0.5, "vy": 0.0, "vz": 0.0}
    bare = [{"id": "n", "mobility": "fixed"}]
    assert effective_inputs.rewrite_nodes(bare, {"n": (3.0, 4.0)}, "reject") == [
        {"id": "n", "mobility": "fixed", "position": {"z": 0.0, "x": 3.0, "y": 4.0}}]
    with pytest.raises(InputError, match="non-finite"):
        effective_inputs.rewrite_nodes(nodes, {"truck": (float("inf"), 0.0)}, "reject")
    with pytest.raises(InputError, match="no waypoints"):
        effective_inputs.start_position({"id": "w", "mobility": "waypoint"})


def test_method_none_copies_nodes_bytes(tmp_path):
    ini = write_scenario(tmp_path / "s", overrides={"algorithm": "none"})
    (ini.parent / "nodes.json").write_text(json.dumps(NODES))
    prepared = adapter.prepare(ini, None, tmp_path / "run", mode="standalone")
    assert (prepared.prep_dir / "effective-inputs/nodes.json").read_bytes() == (
        (ini.parent / "nodes.json").read_bytes())


def test_unmoved_plan_copies_nodes_bytes(tmp_path, stub_planner, fake_child, monkeypatch):
    def hold(request, log):
        return PlanResult(positions={r.id: (r.x, r.y, r.z) for r in request.nodes},
                          channel=dict(CHANNEL_FACTS))
    monkeypatch.setattr("scripts.baselines.solver.solve", hold)
    ini = write_scenario(tmp_path / "s")
    prepared = adapter.prepare(ini, "geometric", tmp_path / "run", mode="standalone",
                               sim_binary=fake_child)
    assert (prepared.prep_dir / "effective-inputs/nodes.json").read_bytes() == (
        (ini.parent / "nodes.json").read_bytes())
    assert json.loads(prepared.plan_path.read_text())["initial_displacement_m_total"] == 0.0


def _files(tmp_path):
    ini = _commented_scenario(tmp_path)
    cfg = config.load_baseline(ini)
    return effective_inputs.scenario_files(config.read_ini(ini), cfg)


def test_stage_then_finish(tmp_path):
    files = _files(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    staged = effective_inputs.stage_effective_inputs(files, run, "standalone")
    assert staged == run / "effective-inputs.partial/run.ini"
    assert sorted(p.name for p in staged.parent.iterdir()) == [
        "buildings.json", "mapping.json", "nodes.json", "run.ini"]
    assert (staged.parent / "nodes.json").read_bytes() == files.nodes.read_bytes()
    assert config.ini_value(config.read_ini(staged), "rl", "enabled") == "false"
    assert not (run / "effective-inputs").exists()
    with pytest.raises(InputError, match="written once"):
        effective_inputs.stage_effective_inputs(files, run, "standalone")

    moved = effective_inputs.rewrite_nodes(NODES, {"uav-a": (10.0, 20.0)}, "reject")
    identity = effective_inputs.finish_effective_inputs(files, run, moved, {"plan": 1})
    final = run / "effective-inputs"
    assert not (run / "effective-inputs.partial").exists()
    assert sorted(p.name for p in final.iterdir()) == [
        "baseline-plan.json", "buildings.json", "mapping.json", "nodes.json", "run.ini"]
    assert identity == {
        "run_config": "effective-inputs/run.ini",
        "run_ini_sha256": artifacts.sha256_file(final / "run.ini"),
        "nodes_json_sha256": artifacts.sha256_file(final / "nodes.json"),
        "buildings_json_sha256": artifacts.sha256_file(final / "buildings.json"),
        "jammers_json_sha256": None,
    }
    assert identity["nodes_json_sha256"] != artifacts.sha256_file(files.nodes)
    with pytest.raises(InputError, match="was not staged"):
        effective_inputs.finish_effective_inputs(files, run, None, {"plan": 1})


def test_finish_failure_removes_the_partial_dir(tmp_path):
    files = _files(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    effective_inputs.stage_effective_inputs(files, run, "evaluation")
    with pytest.raises(ValueError):
        effective_inputs.finish_effective_inputs(files, run, None, {"bad": float("nan")})
    assert not (run / "effective-inputs.partial").exists()
    assert not (run / "effective-inputs").exists()


def test_discard_and_stage_failure_leave_nothing(tmp_path):
    files = _files(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    effective_inputs.stage_effective_inputs(files, run, "standalone")
    effective_inputs.discard_staged_inputs(run)
    assert list(run.iterdir()) == []
    effective_inputs.discard_staged_inputs(run)
    files.buildings.unlink()
    with pytest.raises(FileNotFoundError):
        effective_inputs.stage_effective_inputs(files, run, "standalone")
    assert list(run.iterdir()) == []
