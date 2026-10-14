"""Source snapshots, effective-inputs staging, asset rebasing, and the nodes.json rewrite.

Contract: scripts/baselines/README.md "What preparation does" (steps 2-5) and "Outputs".
Method `none` needs no planner, so most cases call adapter.prepare(..., None, ...)
directly; the rewrite cases replace solver.solve with a deterministic stand-in.
"""

import copy
import json
from pathlib import Path

import pytest

from scripts.baselines import adapter, artifacts, config, effective_inputs
from scripts.baselines.effective_inputs import InputError
from scripts.baselines.solver import PlanResult
from scripts.baselines.tests.conftest import CHANNEL_FACTS, NODES, write_scenario

SMALL_NODES = [
    {"id": "a", "mobility": "fixed", "node_type": "drone",
     "position": {"x": 10.0, "y": 20.0, "z": 5.0}},
    {"id": "b", "mobility": "fixed", "node_type": "vehicle",
     "position": {"x": 30.0, "y": 40.0, "z": 1.0}},
]


def _cpp_view(text: str) -> dict:
    """(section, key) -> value as src/util/ini-parser.cc reads it."""
    view, section = {}, ""
    for raw in text.split("\n"):
        line = raw
        for marker in ("#", ";"):
            if marker in line:
                line = line[:line.find(marker)]
        line = line.strip(" \t\r\n")
        if not line:
            continue
        if line[0] == "[" and line[-1] == "]":
            section = line[1:-1].strip(" \t\r\n")
        elif "=" in line:
            key, value = line.split("=", 1)
            view[(section, key.strip(" \t\r\n"))] = value.strip(" \t\r\n")
    return view


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _asset_scenario(tmp_path: Path) -> dict:
    """INI with a non-standard name, nested nodes file, an absolute buildings path, etc."""
    root = tmp_path / "scenario"
    elsewhere = tmp_path / "shared-assets"
    paths = {
        "nodes": _write(root / "topo" / "layout.json", json.dumps(SMALL_NODES) + "\n"),
        "buildings": _write(elsewhere / "b.json", '{"buildings": []}\n'),
        "jammers": _write(root / "j" / "jam.json", '{"jammers": []}\n'),
        "mapping": _write(root / "maps" / "m.json", json.dumps(
            {"baseline_mapping_version": 2, "geofence": {"source": "rl_bounds"}}) + "\n"),
    }
    text = (f"# asset scenario\n[scenario]\nname = assets   ; note\nseed = 4\n"
            f"nodes_file = topo/layout.json\n"
            f"buildings_file = {paths['buildings']}   # absolute\n"
            f"jammers_file = ./j/jam.json\n\n"
            f"[rl]\nenabled = true\nx_min = 0.0\nx_max = 100.0\ny_min = 0.0\ny_max = 100.0\n\n"
            f"[baseline]\nalgorithm = none\nmapping_file = maps/m.json\n")
    paths["ini"] = _write(root / "my-scenario.ini", text)
    return paths


def test_snapshot_and_effective_inputs_rebase_every_reference(tmp_path):
    paths = _asset_scenario(tmp_path)
    before = {label: path.read_bytes() for label, path in paths.items()}
    prep = tmp_path / "run"
    prepared = adapter.prepare(paths["ini"], None, prep, mode="standalone")

    assert {label: path.read_bytes() for label, path in paths.items()} == before
    source = prep / "source-inputs"
    assert sorted(p.name for p in source.iterdir()) == [
        "b.json", "jam.json", "layout.json", "m.json", "my-scenario.ini"]
    for label, name in (("ini", "my-scenario.ini"), ("nodes", "layout.json"),
                        ("buildings", "b.json"), ("jammers", "jam.json"),
                        ("mapping", "m.json")):
        assert (source / name).read_bytes() == before[label], label

    effective = prep / "effective-inputs"
    assert prepared.effective_run_config == effective / "run.ini"
    assert sorted(p.name for p in effective.iterdir()) == [
        "b.json", "baseline-plan.json", "jam.json", "m.json", "nodes.json", "run.ini"]
    assert (effective / "nodes.json").read_bytes() == before["nodes"]
    for label, name in (("buildings", "b.json"), ("jammers", "jam.json"),
                        ("mapping", "m.json")):
        assert (effective / name).read_bytes() == before[label], label

    old = _cpp_view(before["ini"].decode())
    new = _cpp_view((effective / "run.ini").read_text())
    edits = {("scenario", "nodes_file"): "nodes.json",
             ("scenario", "buildings_file"): "b.json",
             ("scenario", "jammers_file"): "jam.json",
             ("baseline", "mapping_file"): "m.json",
             ("baseline", "algorithm"): "none", ("rl", "enabled"): "false"}
    assert {k: new[k] for k in edits} == edits
    assert {k: v for k, v in new.items() if k not in edits} == {
        k: v for k, v in old.items() if k not in edits}
    assert len((effective / "run.ini").read_text().splitlines()) == len(
        before["ini"].decode().splitlines())
    for key in [k for k in edits if k[1].endswith("_file")]:
        assert (effective / new[key]).is_file(), key

    manifest = artifacts.read_json(prepared.manifest_path)
    assert manifest["source_scenario_identity"] == {
        "run_config": "source-inputs/my-scenario.ini",
        "run_ini_sha256": artifacts.sha256_file(paths["ini"]),
        "nodes_json_sha256": artifacts.sha256_file(paths["nodes"]),
        "buildings_json_sha256": artifacts.sha256_file(paths["buildings"]),
        "jammers_json_sha256": artifacts.sha256_file(paths["jammers"])}
    identity = manifest["effective_scenario_identity"]
    assert identity["run_config"] == "effective-inputs/run.ini"
    assert identity["run_ini_sha256"] == artifacts.sha256_file(effective / "run.ini")
    assert identity["run_ini_sha256"] != manifest["source_scenario_identity"][
        "run_ini_sha256"]
    for key in ("nodes_json_sha256", "buildings_json_sha256", "jammers_json_sha256"):
        assert identity[key] == manifest["source_scenario_identity"][key], key
    assert manifest["mapping_sha256"] == artifacts.sha256_file(paths["mapping"])
    # Standalone preparation leaves the run open for the simulator.
    assert manifest["status"] == "prepared" and manifest["ended_at"] is None
    assert not (prep / "effective-inputs.partial").exists()


def test_method_none_plan_log_and_manifest(tmp_path):
    paths = _asset_scenario(tmp_path)
    prepared = adapter.prepare(paths["ini"], None, tmp_path / "run", mode="standalone")
    assert (prepared.prep_dir / "planner.log").read_text() == (
        "method none: no planner was run\n")
    plan = artifacts.read_json(prepared.plan_path)
    assert plan["baseline_plan_version"] == 2
    assert (plan["method"], plan["objective"], plan["planner_predictions"]) == (
        "none", None, None)
    assert plan["initial_displacement_m_total"] == 0.0
    assert [n["id"] for n in plan["nodes"]] == ["a", "b"]
    for index, node in enumerate(plan["nodes"]):
        assert set(node) == {"id", "roster_index", "slot", "role", "platform", "selected",
                             "original", "planned", "displacement_m"}
        assert node["roster_index"] == index
        assert (node["role"], node["selected"], node["slot"]) == ("fixed", False, None)
        assert node["planned"] == node["original"] == SMALL_NODES[index]["position"]
        assert node["displacement_m"] == 0.0
    manifest = artifacts.read_json(prepared.manifest_path)
    assert (manifest["method"], manifest["requested_algorithm"]) == ("none", "none")
    assert manifest["planner_log"] == "planner.log"
    assert manifest["plan"] == "effective-inputs/baseline-plan.json"
    assert manifest["fingerprint"] == artifacts.fingerprint(manifest)


def test_blank_asset_keys_mean_no_asset_and_keep_their_lines(tmp_path):
    root = tmp_path / "s"
    _write(root / "nodes.json", json.dumps(SMALL_NODES))
    text = ("[scenario]\nbuildings_file =\njammers_file =   # none\n"
            "nodes_file = nodes.json\n")
    ini = _write(root / "run.ini", text)
    prepared = adapter.prepare(ini, None, tmp_path / "run", mode="standalone")
    after = prepared.effective_run_config.read_text()
    assert after.splitlines()[:3] == text.splitlines()[:3]
    assert sorted(p.name for p in prepared.effective_run_config.parent.iterdir()) == [
        "baseline-plan.json", "nodes.json", "run.ini"]
    identity = artifacts.read_json(prepared.manifest_path)["effective_scenario_identity"]
    assert identity["buildings_json_sha256"] is None
    assert identity["jammers_json_sha256"] is None


def test_minimal_ini_gets_every_section_appended(tmp_path):
    root = tmp_path / "s"
    _write(root / "nodes.json", json.dumps(SMALL_NODES))
    ini = _write(root / "run.ini", "[scenario]\nseed = 5")
    prepared = adapter.prepare(ini, None, tmp_path / "run", mode="standalone")
    after = prepared.effective_run_config.read_text()
    assert after.startswith("[scenario]\nseed = 5\nnodes_file = nodes.json\n")
    view = _cpp_view(after)
    assert view == {("scenario", "seed"): "5", ("scenario", "nodes_file"): "nodes.json",
                    ("baseline", "algorithm"): "none", ("rl", "enabled"): "false"}
    config.read_ini(prepared.effective_run_config)


# --- nodes.json rewrite ----------------------------------------------------------------

def test_translate_shifts_every_waypoint_by_the_planned_offset():
    node = {"id": "w", "mobility": "waypoint", "node_type": "drone", "extra": {"k": 1},
            "waypoints": [{"t": 0.0, "x": 10.0, "y": -5.0, "z": 7.0, "speed": 3.0},
                          {"t": 5.0, "y": 20.0, "z": 7.0},
                          {"t": 9.0, "x": -3.5, "y": 0.25, "z": 9.0}]}
    nodes = [node, copy.deepcopy(SMALL_NODES[0])]
    source = copy.deepcopy(nodes)
    out = effective_inputs.rewrite_nodes(nodes, {"w": (13.0, -1.0)}, "translate")
    assert nodes == source
    moved = out[0]
    assert "position" not in moved
    assert [(p["x"], p["y"]) for p in moved["waypoints"]] == [
        (13.0, -1.0), (3.0, 24.0), (-0.5, 4.25)]
    assert [(p["t"], p["z"]) for p in moved["waypoints"]] == [(0.0, 7.0), (5.0, 7.0),
                                                              (9.0, 9.0)]
    assert moved["waypoints"][0]["speed"] == 3.0
    assert {k: v for k, v in moved.items() if k != "waypoints"} == {
        k: v for k, v in node.items() if k != "waypoints"}
    assert out[1] == SMALL_NODES[0] and out[1] is not nodes[1]


def test_translate_updates_a_present_position_but_not_its_z():
    node = {"id": "w", "mobility": "waypoint",
            "position": {"x": 0.0, "y": 0.0, "z": 50.0},
            "waypoints": [{"t": 0.0, "x": 1.0, "y": 2.0, "z": 3.0}]}
    out = effective_inputs.rewrite_nodes([node], {"w": (4.0, 6.0)}, "translate")
    assert out[0]["position"] == {"x": 4.0, "y": 6.0, "z": 50.0}
    assert out[0]["waypoints"] == [{"t": 0.0, "x": 4.0, "y": 6.0, "z": 3.0}]


def test_reject_refuses_a_moved_waypoint_node_but_not_other_mobilities():
    waypoint = {"id": "w", "mobility": "waypoint",
                "waypoints": [{"t": 0.0, "x": 1.0, "y": 2.0, "z": 3.0}]}
    with pytest.raises(InputError, match="waypoint_policy = translate"):
        effective_inputs.rewrite_nodes([waypoint], {"w": (4.0, 6.0)}, "reject")
    walker = {"id": "r", "mobility": "random_walk",
              "position": {"x": 1.0, "y": 1.0, "z": 1.5},
              "random_walk": {"bounds": {"x_min": 0.0, "x_max": 9.0}, "speed_mps": 1.0}}
    out = effective_inputs.rewrite_nodes([walker], {"r": (5.0, 6.0)}, "reject")
    assert out[0]["position"] == {"x": 5.0, "y": 6.0, "z": 1.5}
    assert out[0]["random_walk"] == walker["random_walk"]
    implicit = {"id": "f", "position": {"x": 1.0, "y": 2.0, "z": 3.0}}
    assert effective_inputs.rewrite_nodes([implicit], {"f": (7.0, 8.0)}, "reject")[0] == {
        "id": "f", "position": {"x": 7.0, "y": 8.0, "z": 3.0}}


def test_start_position_defaults_and_waypoint_start():
    assert effective_inputs.start_position({"id": "n"}) == (0.0, 0.0, 0.0)
    assert effective_inputs.start_position(
        {"id": "n", "position": {"x": 1}}) == (1.0, 0.0, 0.0)
    assert effective_inputs.start_position(
        {"id": "w", "mobility": "waypoint", "position": {"x": 9.0, "y": 9.0, "z": 9.0},
         "waypoints": [{"x": 1.0, "y": 2.0}]}) == (1.0, 2.0, 0.0)


def _offset_solve(offset_m: float):
    """solver.solve stand-in: move each selected node by `offset_m` along +x."""
    def solve(request, log):
        positions = {r.id: (r.x + (offset_m if r.selected else 0.0), r.y, r.z)
                     for r in request.nodes}
        return PlanResult(positions=positions, channel=dict(CHANNEL_FACTS))
    return solve


def test_translate_through_prepare_without_a_position_key(tmp_path, fake_child,
                                                          monkeypatch):
    monkeypatch.setattr("scripts.baselines.solver.solve", _offset_solve(12.5))
    nodes = copy.deepcopy(NODES)
    uav_b = next(n for n in nodes if n["id"] == "uav-b")
    del uav_b["position"]
    ini = write_scenario(tmp_path / "s", nodes=nodes,
                         overrides={"movable_nodes": "uav-b", "waypoint_policy": "translate"})
    prepared = adapter.prepare(ini, "geometric", tmp_path / "run", mode="standalone",
                               sim_binary=fake_child)
    rewritten = {n["id"]: n for n in json.loads(
        (prepared.prep_dir / "effective-inputs/nodes.json").read_text())}
    assert "position" not in rewritten["uav-b"]
    assert [(p["x"], p["y"], p["z"], p["t"]) for p in rewritten["uav-b"]["waypoints"]] == [
        (232.5, 160.0, 30.0, 0.0), (252.5, 160.0, 30.0, 1.0)]
    for node in nodes:
        if node["id"] != "uav-b":
            assert rewritten[node["id"]] == node
    plan = {n["id"]: n for n in artifacts.read_json(prepared.plan_path)["nodes"]}
    assert plan["uav-b"]["original"] == {"x": 220.0, "y": 160.0, "z": 30.0}
    assert plan["uav-b"]["planned"] == {"x": 232.5, "y": 160.0, "z": 30.0}
    assert plan["uav-b"]["displacement_m"] == pytest.approx(12.5)
