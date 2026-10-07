"""Node records, penalties, result validation, prepare() lifecycle, and the import boundary."""

import copy
import dataclasses
import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.baselines import adapter, artifacts, config
from scripts.baselines.adapter import BaselinePreparationError
from scripts.baselines.config import ConfigError
from scripts.baselines.planners.objective import MovementCost
from scripts.baselines.solver import PlanResult
from scripts.baselines.tests.conftest import (AREA, CHANNEL_FACTS, MAPPING, NODES,
                                              StubStrategy, install_stub_strategy,
                                              scenario_ini, stub_positions, write_scenario)

MESH_ROOT = Path(__file__).resolve().parents[3]
UNCAPPED = {"aerial": MovementCost(150000.0, 500.0), "ground": MovementCost(150000.0, 100.0)}
ROSTER = [node["id"] for node in NODES]
CONTROLLED_LINE = "controlled_nodes = uav-a, uav-b, uav-c, walker"
FIX_TEXT = "to the same ids, or `all`"


def _eval_ini(overrides=None, controlled="uav-a, uav-c") -> str:
    """Scenario text whose [rl] controlled_nodes equals the default movable_nodes."""
    text = scenario_ini(overrides)
    assert CONTROLLED_LINE in text
    return text.replace(CONTROLLED_LINE, f"controlled_nodes = {controlled}")


def _eval_scenario(root, overrides=None, controlled="uav-a, uav-c", **kwargs):
    return write_scenario(root, ini_text=_eval_ini(overrides, controlled), **kwargs)


def _records(tmp_path, overrides=None, mapping=None, nodes=None, controlled=None,
             no_mapping=False):
    drop = ("mapping_file",) if no_mapping else ()
    ini = write_scenario(tmp_path / "s", overrides=overrides, mapping=mapping, nodes=nodes,
                         drop=drop)
    cfg = config.load_baseline(ini)
    parsed = config.read_ini(ini)
    loaded_mapping = config.load_mapping(cfg.mapping_file, parsed)
    loaded_nodes = adapter.load_nodes(ini.parent / "nodes.json")
    return adapter.build_records(loaded_nodes, cfg, loaded_mapping, controlled), loaded_nodes


def _by_id(records):
    return {record.id: record for record in records}


def _movable(records):
    return [record.id for record in records if record.selected]


def test_role_and_platform_mapping(tmp_path):
    records, _ = _records(tmp_path, no_mapping=True)
    by_id = _by_id(records)
    assert [(r.id, r.role, r.platform, r.platform_source) for r in records] == [
        ("gw", "fixed", "ground", "node_type"),
        ("uav-a", "movable", "aerial", "node_type"),
        ("uav-b", "fixed", "aerial", "node_type"),
        ("uav-c", "movable", "aerial", "node_type"),
        ("walker", "fixed", "ground", "node_type"),
        ("truck", "fixed", "ground", "node_type"),
    ]
    assert {r.role for r in records} == {"movable", "fixed"}
    assert [r.roster_index for r in records] == list(range(6))
    assert by_id["uav-a"].selected and not by_id["gw"].selected
    assert all(r.slot is None for r in records)
    assert by_id["uav-c"].random_walk_bounds == AREA
    assert by_id["uav-a"].random_walk_bounds is None
    assert by_id["uav-b"].has_waypoints and not by_id["uav-a"].has_waypoints


@pytest.mark.parametrize("movable,expected", [
    ("walker", ["walker"]),
    ("truck, uav-a", ["uav-a", "truck"]),
    ("walker, gw, uav-c", ["gw", "uav-c", "walker"]),
    ("all", ["gw", "uav-a", "uav-b", "uav-c", "walker", "truck"]),
])
def test_single_mixed_reordered_and_all_selections(tmp_path, movable, expected):
    records, _ = _records(tmp_path, overrides={"movable_nodes": movable})
    assert _movable(records) == expected
    assert adapter.resolve_ids(config.load_baseline(
        tmp_path / "s/run.ini").movable_nodes, [r.id for r in records]) == expected


def test_mapping_overrides_platform(tmp_path):
    mapping = copy.deepcopy(MAPPING)
    mapping["platforms"] = {"nodes": {"walker": "aerial", "uav-a": "ground"}}
    by_id = _by_id(_records(tmp_path, mapping=mapping)[0])
    assert (by_id["walker"].platform, by_id["walker"].platform_source) == ("aerial", "mapping")
    assert (by_id["uav-a"].platform, by_id["uav-a"].platform_source) == ("ground", "mapping")
    assert by_id["truck"].platform_source == "node_type"


def test_slots_follow_controlled_order(tmp_path):
    by_id = _by_id(_records(tmp_path, controlled=("walker", "uav-a"))[0])
    assert (by_id["walker"].slot, by_id["uav-a"].slot, by_id["gw"].slot) == (0, 1, None)
    by_id = _by_id(_records(tmp_path / "all", controlled=("all",))[0])
    assert by_id["truck"].slot == 5


def test_start_position_uses_first_waypoint(tmp_path):
    uav_b = _by_id(_records(tmp_path)[0])["uav-b"]
    assert (uav_b.x, uav_b.y, uav_b.z) == (220.0, 160.0, 30.0)


def test_unknown_ids(tmp_path):
    with pytest.raises(ConfigError, match="movable_nodes names unknown.*ghost"):
        _records(tmp_path / "a", overrides={"movable_nodes": "uav-a, ghost"})
    mapping = copy.deepcopy(MAPPING)
    mapping["platforms"] = {"nodes": {"ghost": "aerial"}}
    with pytest.raises(ConfigError, match="platforms.nodes names unknown.*ghost"):
        _records(tmp_path / "b", mapping=mapping)


def test_duplicate_movable_ids(tmp_path):
    with pytest.raises(ConfigError, match="more than once"):
        _records(tmp_path, overrides={"movable_nodes": "uav-a, uav-c, uav-a"})


def test_unknown_node_type_needs_a_platform(tmp_path):
    nodes = copy.deepcopy(NODES)
    nodes[5]["node_type"] = "boat"
    with pytest.raises(ConfigError, match="node_type 'boat'.*platforms.nodes"):
        _records(tmp_path / "a", nodes=nodes)
    mapping = copy.deepcopy(MAPPING)
    mapping["platforms"] = {"nodes": {"truck": "ground"}}
    assert _by_id(_records(tmp_path / "b", nodes=nodes, mapping=mapping)[0])[
        "truck"].platform == "ground"


def test_duplicate_node_ids(tmp_path):
    nodes = copy.deepcopy(NODES) + [copy.deepcopy(NODES[0])]
    with pytest.raises(ConfigError, match="repeats node id"):
        _records(tmp_path, nodes=nodes)


def test_negative_z_rejected(tmp_path):
    nodes = copy.deepcopy(NODES)
    nodes[5]["position"]["z"] = -0.5
    records, _ = _records(tmp_path, nodes=nodes)
    with pytest.raises(ConfigError, match="node 'truck' has z=-0.5 below the ground datum"):
        adapter.check_datum(records)


def _movable_cfg(tmp_path, movable):
    return config.load_baseline(write_scenario(tmp_path, overrides={"movable_nodes": movable}))


@pytest.mark.parametrize("movable,controlled,slots", [
    ("walker", ("walker",), ["walker"]),
    ("truck, uav-a", ("uav-a", "truck"), ["uav-a", "truck"]),
    ("truck, uav-a", ("truck", "uav-a"), ["truck", "uav-a"]),
    ("walker, gw, uav-c", ("uav-c", "walker", "gw"), ["uav-c", "walker", "gw"]),
    ("all", ("all",), ROSTER),
    ("all", tuple(reversed(ROSTER)), list(reversed(ROSTER))),
    (", ".join(reversed(ROSTER)), ("all",), ROSTER),
])
def test_ownership_equality_accepts_equal_sets(tmp_path, movable, controlled, slots):
    cfg = _movable_cfg(tmp_path, movable)
    assert adapter.check_ownership(cfg, controlled, ROSTER) == slots


@pytest.mark.parametrize("movable,controlled,not_controlled,not_movable", [
    ("all", ("uav-a", "uav-b", "uav-c", "walker"), "gw, truck", "-"),
    ("uav-a, uav-c", ("uav-a", "uav-b", "uav-c", "walker"), "-", "uav-b, walker"),
    ("uav-a, uav-c", ("walker", "uav-a"), "uav-c", "walker"),
    ("uav-a, uav-c", ("all",), "-", "gw, uav-b, walker, truck"),
    ("truck", ("walker",), "truck", "walker"),
])
def test_ownership_mismatch_names_both_differences(tmp_path, movable, controlled,
                                                  not_controlled, not_movable):
    cfg = _movable_cfg(tmp_path, movable)
    with pytest.raises(ConfigError) as caught:
        adapter.check_ownership(cfg, controlled, ROSTER)
    message = str(caught.value)
    assert f"movable but not controlled: {not_controlled};" in message
    assert f"controlled but not movable: {not_movable}." in message
    assert FIX_TEXT in message


def test_ownership_rejects_legacy_unknown_and_duplicate_ids(tmp_path):
    cfg = _movable_cfg(tmp_path / "a", "uav-a, uav-c")
    with pytest.raises(ConfigError, match="legacy single-node control"):
        adapter.check_ownership(cfg, None, ROSTER)
    with pytest.raises(ConfigError,
                       match="controlled_nodes names unknown node id\\(s\\): ghost"):
        adapter.check_ownership(cfg, ("uav-a", "uav-c", "ghost"), ROSTER)
    with pytest.raises(ConfigError, match="controlled_nodes lists uav-a more than once"):
        adapter.check_ownership(cfg, ("uav-a", "uav-c", "uav-a"), ROSTER)
    ghost = dataclasses.replace(cfg, movable_nodes=("uav-a", "ghost"))
    with pytest.raises(ConfigError,
                       match="movable_nodes names unknown node id\\(s\\): ghost"):
        adapter.check_ownership(ghost, ("uav-a", "ghost"), ROSTER)


def test_evaluation_region_is_the_intersection():
    rl = {"x_min": 0.0, "x_max": 400.0, "y_min": -50.0, "y_max": 300.0}
    rect = {"x_min": 100.0, "x_max": 500.0, "y_min": 0.0, "y_max": 400.0}
    assert adapter.evaluation_region(rect, rl) == {"x_min": 100.0, "x_max": 400.0,
                                                   "y_min": 0.0, "y_max": 300.0}
    assert adapter.evaluation_region(dict(AREA), dict(AREA)) == AREA
    inner = {"x_min": 10.0, "x_max": 20.0, "y_min": 30.0, "y_max": 40.0}
    assert adapter.evaluation_region(inner, rl) == inner
    for disjoint in ({**rect, "x_min": 400.0}, {**rect, "y_min": 300.0, "y_max": 350.0},
                     {**rect, "x_min": 600.0, "x_max": 700.0}):
        with pytest.raises(ConfigError, match="do not overlap"):
            adapter.evaluation_region(disjoint, rl)


def test_penalties_resolution_and_scale(tmp_path):
    cfg = config.load_baseline(write_scenario(tmp_path / "d"))
    penalties = adapter.resolve_penalties(cfg)
    assert penalties["aerial"] == MovementCost(150000.0, 500.0, None)
    assert penalties["ground"] == MovementCost(150000.0, 100.0, None)
    custom = config.load_baseline(write_scenario(tmp_path / "c", overrides={
        "aerial_fixed_cost_m2": "0", "aerial_cost_m2_per_m": "0",
        "ground_fixed_cost_m2": "10", "ground_cost_m2_per_m": "2",
        "ground_max_displacement_m": "30"}))
    penalties = adapter.resolve_penalties(custom)
    assert penalties["aerial"] == MovementCost(0.0, 0.0, None)
    assert penalties["ground"] == MovementCost(10.0, 2.0, 30.0)
    block = adapter.penalties_block(penalties, 160000.0)
    assert block["ground"] == {"fixed_cost_m2": 10.0, "cost_m2_per_m": 2.0,
                               "max_displacement_m": 30.0}
    assert block["aoi_m2"] == 160000.0
    assert block["fixed_cost_over_aoi"] == {"ground": 10.0 / 160000.0, "aerial": 0.0}


def test_planning_identity(tmp_path):
    ini = config.read_ini(write_scenario(tmp_path / "a"))
    assert adapter.planning_identity(ini, None) == (1, 1)
    assert adapter.planning_identity(ini, [9, 3]) == (9, 1)
    text = scenario_ini().replace("seed = 1\n", "run_id = 4\n")
    ini = config.read_ini(write_scenario(tmp_path / "b", ini_text=text))
    assert adapter.planning_identity(ini, []) == (config.DEFAULT_SCENARIO_SEED, 4)
    bad = config.read_ini(write_scenario(tmp_path / "c", ini_text=scenario_ini().replace(
        "seed = 1", "seed = x")))
    with pytest.raises(ConfigError, match="\\[scenario\\] seed"):
        adapter.planning_identity(bad, None)


def test_resolve_band(tmp_path):
    plain = config.read_ini(write_scenario(tmp_path / "a"))
    assert adapter.resolve_band(plain, None) == ("mmwave", "default")
    assert adapter.resolve_band(plain, "sub-6") == ("sub-6", "cli")
    banded = config.read_ini(write_scenario(tmp_path / "b", ini_text=scenario_ini(
        extra="\n[channel]\nband = sub-6\n")))
    assert adapter.resolve_band(banded, None) == ("sub-6", "run.ini")
    assert adapter.resolve_band(banded, "mmwave") == ("mmwave", "cli")


class _Request:
    rectangle = dict(AREA)


def _validate(tmp_path, positions, rl_bounds=None, nodes=None, penalties=None):
    records, _ = _records(tmp_path, nodes=nodes)
    request = _Request()
    request.nodes = records
    base = stub_positions(request)
    base.update(positions)
    return adapter.validate_result(records, PlanResult(positions=base), dict(AREA),
                                   penalties or UNCAPPED, rl_bounds)


def test_valid_result_and_write_back(tmp_path):
    final = _validate(tmp_path, {"uav-c": (170.004, 140.003, 30.0)})
    assert final["uav-c"] == (170.0, 140.0, 30.0)
    assert final["uav-a"] == (400.0 / 3, 100.0, 30.0)
    assert final["gw"] == (200.0, 200.0, 2.0)


def test_inclusive_boundary_accepted(tmp_path):
    final = _validate(tmp_path, {"uav-a": (400.0, 0.0, 30.0), "uav-c": (0.0, 400.0, 30.0)},
                      rl_bounds=dict(AREA))
    assert final["uav-a"][:2] == (400.0, 0.0)


@pytest.mark.parametrize("positions,match", [
    ({"uav-a": (400.001, 100.0, 30.0)}, "outside the declared geofence rectangle"),
    ({"uav-a": (100.0, 100.0, 30.000002)}, "changed z of 'uav-a'"),
    ({"gw": (200.011, 200.0, 2.0)}, "moved fixed node 'gw'"),
    ({"truck": (230.0, 230.5, 2.0)}, "moved fixed node 'truck'"),
    ({"uav-a": (math.nan, 100.0, 30.0)}, "non-finite"),
    ({"uav-a": (100.0, 100.0)}, "non-finite"),
    ({"extra": (1.0, 1.0, 1.0)}, "unexpected: extra"),
])
def test_invalid_results(tmp_path, positions, match):
    with pytest.raises(ValueError, match=match):
        _validate(tmp_path, positions)


def test_tolerances_are_unchanged(tmp_path):
    final = _validate(tmp_path, {"gw": (200.009, 200.0, 2.0),
                                 "uav-a": (100.0, 100.0, 30.0000009)})
    assert final["gw"] == (200.0, 200.0, 2.0)
    assert final["uav-a"] == (100.0, 100.0, 30.0)


def test_missing_node_in_result(tmp_path):
    records, _ = _records(tmp_path)
    positions = {r.id: (r.x, r.y, r.z) for r in records if r.id != "truck"}
    with pytest.raises(ValueError, match="missing: truck"):
        adapter.validate_result(records, PlanResult(positions=positions), dict(AREA),
                                UNCAPPED)


def test_displacement_cap_per_platform(tmp_path):
    records, _ = _records(tmp_path)
    positions = {r.id: (r.x, r.y, r.z) for r in records}
    positions["uav-a"] = (150.0, 195.0, 30.0)
    capped = {**UNCAPPED, "aerial": MovementCost(150000.0, 500.0, 10.0)}
    with pytest.raises(ValueError, match="beyond \\[baseline\\] aerial_max_displacement_m 10"):
        adapter.validate_result(records, PlanResult(positions=positions), dict(AREA), capped)
    ground_only = {**UNCAPPED, "ground": MovementCost(0.0, 0.0, 10.0)}
    final = adapter.validate_result(records, PlanResult(positions=positions), dict(AREA),
                                    ground_only)
    assert final["uav-a"] == (150.0, 195.0, 30.0)


def test_rl_bounds_and_random_walk_bounds(tmp_path):
    with pytest.raises(ValueError, match="outside the \\[rl\\] x/y bounds"):
        _validate(tmp_path / "a", {}, rl_bounds={**AREA, "y_min": 150.0})
    nodes = copy.deepcopy(NODES)
    nodes[3]["random_walk"]["bounds"] = {"x_min": 0.0, "x_max": 200.0,
                                         "y_min": 0.0, "y_max": 400.0}
    with pytest.raises(ValueError, match="'uav-c'.*random_walk bounds"):
        _validate(tmp_path / "b", {}, nodes=nodes)


def test_random_walk_default_bounds_mirror_simulator(tmp_path):
    nodes = copy.deepcopy(NODES)
    del nodes[3]["random_walk"]
    with pytest.raises(ValueError, match="random_walk bounds"):
        _validate(tmp_path, {}, nodes=nodes)


RETIRED_MANIFEST_KEYS = ("rf", "planner_source", "origin", "ground_datum")


def _prepare_eval(tmp_path, ini, binary, method="geometric", **kwargs):
    out = tmp_path / "eval"
    return adapter.prepare(ini, method, out / method / "baseline", mode="evaluation",
                           sim_binary=binary, **kwargs)


def test_prepare_evaluation_scores_through_the_query(tmp_path, fake_query, monkeypatch):
    strategy = install_stub_strategy(monkeypatch)
    ini = _eval_scenario(tmp_path / "s", overrides={"algorithm": "none"},
                         controlled="uav-c, uav-a")
    prepared = _prepare_eval(tmp_path, ini, fake_query, band="sub-6",
                             simulation_seeds=[5, 2])
    manifest = artifacts.read_json(prepared.manifest_path)
    assert manifest["baseline_manifest_version"] == 2
    assert not set(RETIRED_MANIFEST_KEYS) & set(manifest)
    assert manifest["status"] == "prepared" and manifest["ended_at"] is not None
    assert (manifest["mode"], manifest["executor"]) == ("evaluation", "hold")
    assert (manifest["requested_algorithm"], manifest["method"]) == ("none", "geometric")
    assert (manifest["planner_seed"], manifest["planner_seed_status"]) == (None, "unused")
    assert manifest["simulation_seeds"] == [5, 2]
    assert manifest["geofence"] == {"source": "rl_bounds", **AREA}
    assert manifest["package_versions"].keys() == {"numpy"}
    scoring = manifest["channel_scoring"]
    assert {k: scoring[k] for k in ("contract", "isolation", "planning_seed",
                                    "planning_run_id", "jammer_seed", "mode_flag", "band",
                                    "band_source", "sinr_threshold_db",
                                    "coverage_sinr_db")} == {
        "contract": "mesh_channel_query_v1", "isolation": "fork_per_layout",
        "planning_seed": 5, "planning_run_id": 1, "jammer_seed": 5,
        "mode_flag": "--rl-mode", "band": "sub-6", "band_source": "cli",
        "sinr_threshold_db": -6.7, "coverage_sinr_db": -6.7}
    assert scoring["probe"] == {"height_m": 1.5, "rx_gain_dbi": 12.0, "grid_cells": 400,
                                "min_resolution_m": 5.0, "cell_m": 20.0, "count": 400}
    assert scoring["candidate_grid"] == {"cells": 400, "min_resolution_m": 5.0,
                                         "cell_m": 20.0, "count": 400}
    assert scoring["queries"]["requests"] == 1 and scoring["queries"]["layouts"] == 2
    assert scoring["queries"]["probe_links"] == 2 * 6 * 400
    assert manifest["penalties"]["aoi_m2"] == 160000.0
    assert manifest["penalties"]["fixed_cost_over_aoi"] == {"aerial": 0.9375,
                                                             "ground": 0.9375}
    assert manifest["ownership"] == {
        "mode": "evaluation", "movable_resolved": ["uav-a", "uav-c"],
        "controlled_resolved": ["uav-c", "uav-a"]}
    assert manifest["planner_settings"] == {"strategy": "stub", "method": "geometric"}
    assert manifest["sim_binary_sha256"] == artifacts.sha256_file(fake_query)
    assert manifest["channel_runtime_sha256"] == artifacts.channel_runtime_identity(
        fake_query)["sha256"]
    assert manifest["planner_code_sha256"] == artifacts.planner_code_identity()[
        "aggregate_sha256"]
    assert manifest["eval_manifest"] == "../../eval_manifest.json"
    assert manifest["sim_log"] is None
    assert manifest["fingerprint"] == artifacts.fingerprint(manifest)
    for key in ("plan", "planner_log", "source_inputs", "effective_inputs"):
        assert not Path(manifest[key]).is_absolute()
        assert (prepared.prep_dir / manifest[key]).exists()
    assert not (prepared.prep_dir / "effective-inputs.partial").exists()
    assert len(strategy.scored) == 2

    block = prepared.metadata
    assert block["baseline_manifest_version"] == 2
    assert block["planning_seed"] == 5
    assert block["ownership"] == manifest["ownership"]
    assert "rf_config_sha256" not in block and "planner_source_sha256" not in block
    assert block["manifest"] == "geometric/baseline/baseline_manifest.json"
    assert block["plan"] == "geometric/baseline/effective-inputs/baseline-plan.json"
    assert set(block["effective_scenario_identity"]) == set(artifacts.IDENTITY_HASH_KEYS)
    assert block["fingerprint"] == manifest["fingerprint"]

    plan = json.loads(prepared.plan_path.read_text())
    assert plan["baseline_plan_version"] == 2
    assert {n["role"] for n in plan["nodes"]} == {"movable", "fixed"}
    assert all("radios" not in n for n in plan["nodes"])
    selected = {n["id"]: n for n in plan["nodes"] if n["selected"]}
    assert set(selected) == {"uav-a", "uav-c"}
    assert selected["uav-c"]["slot"] == 0 and selected["uav-a"]["slot"] == 1
    assert all(n["slot"] is None for n in plan["nodes"] if not n["selected"])
    assert plan["planner_predictions"]["stub_strategy"] is True
    assert math.isclose(plan["initial_displacement_m_total"],
                        sum(n["displacement_m"] for n in plan["nodes"]))
    log = (prepared.prep_dir / "planner.log").read_text()
    staged = prepared.prep_dir.resolve() / "effective-inputs.partial/run.ini"
    assert (f"--run-config={staged} --channel-query --seed=5 --band=sub-6 --rl-mode"
            in log)
    assert "aoi_m2=160000.0" in log and "aerial fixed_cost_m2/aoi_m2=0.9375" in log
    assert f"evaluation_region={json.dumps(AREA, sort_keys=True)}" in log
    assert "WARNING" not in log
    assert "note: stub note" in log and "stub strategy scored 2 layouts" in log
    assert config.load_baseline(prepared.effective_run_config).algorithm == "none"


def test_prepare_evaluation_with_every_node_on_both_sides(tmp_path, fake_query, monkeypatch):
    strategy = install_stub_strategy(monkeypatch)
    ini = _eval_scenario(tmp_path / "s", overrides={"movable_nodes": "all",
                                                    "waypoint_policy": "translate"},
                         controlled="all")
    prepared = _prepare_eval(tmp_path, ini, fake_query)
    manifest = artifacts.read_json(prepared.manifest_path)
    assert manifest["ownership"] == {"mode": "evaluation", "movable_resolved": ROSTER,
                                     "controlled_resolved": ROSTER}
    plan = json.loads(prepared.plan_path.read_text())
    assert [(n["id"], n["slot"], n["selected"]) for n in plan["nodes"]] == [
        (node_id, index, True) for index, node_id in enumerate(ROSTER)]
    assert len(strategy.scored) == 2


@pytest.mark.parametrize("movable,controlled,match", [
    ("all", "uav-a, uav-b, uav-c, walker",
     "movable but not controlled: gw, truck; controlled but not movable: -"),
    ("uav-a, uav-c", "all",
     "movable but not controlled: -; controlled but not movable: gw, uav-b, walker, truck"),
])
def test_prepare_evaluation_refuses_unequal_rosters(tmp_path, fake_query, monkeypatch,
                                                    movable, controlled, match):
    strategy = install_stub_strategy(monkeypatch)
    ini = _eval_scenario(tmp_path / "s", overrides={"movable_nodes": movable,
                                                    "waypoint_policy": "translate"},
                         controlled=controlled)
    with pytest.raises(BaselinePreparationError, match=match):
        _prepare_eval(tmp_path, ini, fake_query)
    prep = tmp_path / "eval/geometric/baseline"
    _assert_no_inputs(prep)
    assert FIX_TEXT in artifacts.read_json(prep / "baseline_manifest.json")["error"]
    assert strategy.scored == []


def test_prepare_evaluation_region_and_standalone_rectangle(tmp_path, fake_query, monkeypatch):
    seen = []

    def positions(request):
        seen.append((request.mode, request.rectangle, request.rl_bounds))
        return stub_positions(request)

    install_stub_strategy(monkeypatch, StubStrategy(positions))
    rectangle = {"x_min": 100.0, "x_max": 500.0, "y_min": 0.0, "y_max": 400.0}
    mapping = {**copy.deepcopy(MAPPING),
               "geofence": {"source": "rectangle_xy_m", **rectangle}}
    ini = _eval_scenario(tmp_path / "s", mapping=mapping)
    region = {**AREA, "x_min": 100.0}
    prepared = _prepare_eval(tmp_path, ini, fake_query)
    assert seen[-1] == ("evaluation", rectangle, region)
    log = (prepared.prep_dir / "planner.log").read_text()
    assert f"evaluation_region={json.dumps(region, sort_keys=True)}" in log
    plan = json.loads(prepared.plan_path.read_text())
    for node in plan["nodes"]:
        if node["selected"]:
            assert adapter._inside(node["planned"]["x"], node["planned"]["y"], region)

    standalone = adapter.prepare(ini, "geometric", tmp_path / "run", mode="standalone",
                                 sim_binary=fake_query)
    assert seen[-1] == ("standalone", rectangle, None)
    assert "evaluation_region=" not in (standalone.prep_dir / "planner.log").read_text()


def test_validate_result_with_the_evaluation_region(tmp_path):
    region = adapter.evaluation_region(dict(AREA), {**AREA, "x_max": 300.0})
    final = _validate(tmp_path / "a", {"uav-a": (300.0, 100.0, 30.0)}, rl_bounds=region)
    assert final["uav-a"] == (300.0, 100.0, 30.0)
    with pytest.raises(ValueError, match="outside the \\[rl\\] x/y bounds"):
        _validate(tmp_path / "b", {"uav-a": (300.5, 100.0, 30.0)}, rl_bounds=region)


def test_query_runs_on_staged_source_nodes(tmp_path, fake_query, monkeypatch):
    seen = {}

    def positions(request):
        query_dir = Path(request.query_run_config).parent
        seen["nodes"] = (query_dir / "nodes.json").read_bytes()
        seen["files"] = sorted(p.name for p in query_dir.iterdir())
        return stub_positions(request)

    install_stub_strategy(monkeypatch, StubStrategy(positions))
    ini = write_scenario(tmp_path / "s")
    prepared = adapter.prepare(ini, "geometric", tmp_path / "run", mode="standalone",
                               sim_binary=fake_query)
    assert seen["nodes"] == (ini.parent / "nodes.json").read_bytes()
    assert seen["files"] == ["mapping.json", "nodes.json", "run.ini"]
    rewritten = (prepared.prep_dir / "effective-inputs/nodes.json").read_bytes()
    assert rewritten != seen["nodes"]


def test_prepare_standalone_resolves_method(tmp_path, stub_planner, fake_child):
    ini = write_scenario(tmp_path / "s", overrides={"algorithm": "optimization"})
    prepared = adapter.prepare(ini, None, tmp_path / "run", mode="standalone",
                               sim_binary=fake_child)
    manifest = artifacts.read_json(prepared.manifest_path)
    assert (manifest["status"], manifest["ended_at"]) == ("prepared", None)
    assert (manifest["requested_algorithm"], manifest["method"]) == (
        "optimization", "optimization")
    assert (manifest["planner_seed"], manifest["max_iterations"]) == (7, 60)
    assert manifest["planner_seed_status"] == "used"
    assert manifest["executor"] == "none"
    assert manifest["sim_log"] == "sim.log"
    assert manifest["eval_manifest"] is None
    assert manifest["sim_binary_sha256"] == artifacts.sha256_file(fake_child)
    assert manifest["channel_scoring"]["mode_flag"] is None
    assert manifest["channel_scoring"]["planning_seed"] == 1
    assert manifest["ownership"] == {"mode": "standalone",
                                     "movable_resolved": ["uav-a", "uav-c"],
                                     "controlled_resolved": None}
    assert manifest["channel_runtime_files"] == [str(fake_child.resolve())]
    assert prepared.metadata["manifest"] == "baseline_manifest.json"
    override = adapter.prepare(ini, "geometric", tmp_path / "run2", mode="standalone",
                               sim_binary=fake_child)
    assert artifacts.read_json(override.manifest_path)["method"] == "geometric"


def test_active_method_requires_the_binary(tmp_path, stub_planner):
    ini = write_scenario(tmp_path / "s")
    with pytest.raises(BaselinePreparationError, match="needs the simulator binary"):
        adapter.prepare(ini, "geometric", tmp_path / "run", mode="standalone")
    with pytest.raises(BaselinePreparationError, match="not an executable file"):
        adapter.prepare(ini, "geometric", tmp_path / "run2", mode="standalone",
                        sim_binary=tmp_path / "missing-bin")


def test_prepare_none_standalone(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("planner used for method none")
    monkeypatch.setattr("scripts.baselines.solver.solve", forbidden)
    monkeypatch.setattr(artifacts, "channel_runtime_identity", forbidden)
    ini = write_scenario(tmp_path / "s", overrides={"algorithm": "none"})
    prepared = adapter.prepare(ini, None, tmp_path / "run", mode="standalone")
    manifest = artifacts.read_json(prepared.manifest_path)
    assert manifest["method"] == "none"
    assert manifest["channel_scoring"] is None and manifest["penalties"] is None
    assert manifest["initial_displacement_m_total"] == 0.0
    plan = json.loads(prepared.plan_path.read_text())
    assert {n["role"] for n in plan["nodes"]} == {"fixed"}
    assert (tmp_path / "run/effective-inputs/nodes.json").read_bytes() == (
        (ini.parent / "nodes.json").read_bytes())


def test_prepare_none_imports_no_planner_module(tmp_path):
    ini = write_scenario(tmp_path / "s", overrides={"algorithm": "none"})
    script = textwrap.dedent(f"""
        import sys
        from scripts.baselines import adapter
        adapter.prepare({str(ini)!r}, None, {str(tmp_path / 'run')!r}, mode="standalone")
        print(sorted(m for m in sys.modules if m == 'scripts.baselines.solver'
                     or m.startswith('scripts.baselines.planners')))
    """)
    result = subprocess.run([sys.executable, "-c", script], cwd=MESH_ROOT, text=True,
                            capture_output=True, check=True)
    assert result.stdout.strip() == "[]"


def test_prepare_failure_records_failed(tmp_path, stub_planner, fake_child):
    ini = write_scenario(tmp_path / "s", overrides={"objective": "fastest"})
    with pytest.raises(BaselinePreparationError, match="baseline.objective"):
        _prepare_eval(tmp_path, ini, fake_child)
    prep = tmp_path / "eval/geometric/baseline"
    manifest = artifacts.read_json(prep / "baseline_manifest.json")
    assert manifest["status"] == "failed"
    assert "baseline.objective" in manifest["error"]
    assert manifest["ended_at"] is not None
    assert sorted(p.name for p in prep.iterdir()) == ["baseline_manifest.json"]


def _assert_no_inputs(prep: Path) -> None:
    assert not (prep / "effective-inputs").exists()
    assert not (prep / "effective-inputs.partial").exists()
    assert artifacts.read_json(prep / "baseline_manifest.json")["status"] == "failed"


def test_solver_failure_keeps_log_and_writes_no_inputs(tmp_path, fake_child, monkeypatch):
    def broken(request, log):
        log.write("about to fail\n")
        raise RuntimeError("solver exploded")
    monkeypatch.setattr("scripts.baselines.solver.solve", broken)
    with pytest.raises(BaselinePreparationError, match="solver exploded"):
        _prepare_eval(tmp_path, _eval_scenario(tmp_path / "s"), fake_child)
    prep = tmp_path / "eval/geometric/baseline"
    _assert_no_inputs(prep)
    log = (prep / "planner.log").read_text()
    assert "about to fail" in log and "FAILED: solver exploded" in log


def test_query_layout_error_aborts_preparation(tmp_path, fake_query, monkeypatch):
    install_stub_strategy(monkeypatch)
    monkeypatch.setenv("FAKE_QUERY_FAULT", "layout_error")
    with pytest.raises(BaselinePreparationError, match="injected layout error"):
        _prepare_eval(tmp_path, _eval_scenario(tmp_path / "s"), fake_query)
    _assert_no_inputs(tmp_path / "eval/geometric/baseline")


def test_invalid_plan_fails_preparation(tmp_path, fake_child, monkeypatch):
    def wrong_z(request, log):
        positions = stub_positions(request)
        x, y, _ = positions["uav-a"]
        positions["uav-a"] = (x, y, 1.5)
        return PlanResult(positions=positions, channel=dict(CHANNEL_FACTS))
    monkeypatch.setattr("scripts.baselines.solver.solve", wrong_z)
    with pytest.raises(BaselinePreparationError, match="changed z of 'uav-a'"):
        _prepare_eval(tmp_path, _eval_scenario(tmp_path / "s"), fake_child)
    _assert_no_inputs(tmp_path / "eval/geometric/baseline")


def test_band_mismatch_with_the_query_fails(tmp_path, fake_child, monkeypatch):
    def other_band(request, log):
        return PlanResult(positions=stub_positions(request),
                          channel={**CHANNEL_FACTS, "band": "sub-6"})
    monkeypatch.setattr("scripts.baselines.solver.solve", other_band)
    with pytest.raises(BaselinePreparationError, match="resolved band 'sub-6'"):
        _prepare_eval(tmp_path, _eval_scenario(tmp_path / "s"), fake_child)


@pytest.mark.parametrize("ini_edit,match", [
    (lambda text: text.replace("controlled_nodes = uav-a, uav-c\n", ""),
     "legacy single-node control"),
    (lambda text: text.replace("controlled_nodes = uav-a, uav-c",
                               "controlled_nodes = uav-a, walker"),
     "movable but not controlled: uav-c; controlled but not movable: walker"),
    (lambda text: text.replace("controlled_nodes = uav-a, uav-c", CONTROLLED_LINE),
     "movable but not controlled: -; controlled but not movable: uav-b, walker"),
    (lambda text: text.replace("x_max = 400.0\n", ""),
     "rl_bounds.*x_max.*loader defaults are not used"),
])
def test_evaluation_preparation_rules(tmp_path, stub_planner, fake_child, ini_edit, match):
    ini = write_scenario(tmp_path / "s", ini_text=ini_edit(_eval_ini()))
    with pytest.raises(BaselinePreparationError, match=match):
        _prepare_eval(tmp_path, ini, fake_child)


def test_negative_z_rejected_through_prepare(tmp_path, stub_planner, fake_child):
    nodes = copy.deepcopy(NODES)
    nodes[0]["position"]["z"] = -1.0
    ini = _eval_scenario(tmp_path / "s", nodes=nodes)
    with pytest.raises(BaselinePreparationError, match="below the ground datum"):
        _prepare_eval(tmp_path, ini, fake_child)


def test_scale_warning_is_logged_not_raised(tmp_path, stub_planner, fake_child):
    ini = _eval_scenario(tmp_path / "s", overrides={"ground_fixed_cost_m2": "160000"})
    prepared = _prepare_eval(tmp_path, ini, fake_child)
    log = (prepared.prep_dir / "planner.log").read_text()
    assert "ground fixed_cost_m2/aoi_m2=1.0000" in log
    assert "WARNING: ground_fixed_cost_m2 160000.0 >= aoi_m2 160000.0" in log
    assert "WARNING: aerial" not in log
    penalties = artifacts.read_json(prepared.manifest_path)["penalties"]
    assert penalties["fixed_cost_over_aoi"]["ground"] == 1.0


def test_prepare_argument_rules(tmp_path, stub_planner, fake_child):
    ini = write_scenario(tmp_path / "s")
    with pytest.raises(BaselinePreparationError, match="evaluation placement method"):
        _prepare_eval(tmp_path, ini, fake_child, method="none")
    with pytest.raises(TypeError):
        adapter.prepare(ini, "geometric", tmp_path / "b", planner_source=tmp_path)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "keep.txt").write_text("x")
    with pytest.raises(BaselinePreparationError, match="not empty"):
        adapter.prepare(ini, "geometric", busy, mode="standalone", sim_binary=fake_child)
    assert sorted(p.name for p in busy.iterdir()) == ["keep.txt"]


def test_fingerprint_stable_and_sensitive(tmp_path, stub_planner, fake_child):
    ini = _eval_scenario(tmp_path / "s")
    first = _prepare_eval(tmp_path / "1", ini, fake_child).metadata["fingerprint"]
    second = _prepare_eval(tmp_path / "2", ini, fake_child).metadata["fingerprint"]
    other = _eval_scenario(tmp_path / "o", overrides={"objective": "balanced"})
    third = _prepare_eval(tmp_path / "3", other, fake_child).metadata["fingerprint"]
    reseeded = _prepare_eval(tmp_path / "4", ini, fake_child,
                             simulation_seeds=[3]).metadata["fingerprint"]
    reordered = _prepare_eval(tmp_path / "5", _eval_scenario(
        tmp_path / "r", controlled="uav-c, uav-a"), fake_child).metadata["fingerprint"]
    assert first == second
    assert len({first, third, reseeded, reordered}) == 4


def test_manifest_status_helpers(tmp_path):
    path = tmp_path / "baseline_manifest.json"
    artifacts.write_json(path, artifacts.new_manifest("standalone", "none", "none", "none"))
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match="illegal"):
        artifacts.set_status(path, "running")
    with pytest.raises(ValueError, match="set_status"):
        artifacts.update_manifest(path, status="complete")
    artifacts.set_status(path, "prepared")
    artifacts.set_status(path, "running")
    records = [artifacts.seed_record(1, "complete", "seed-1/summary.json"),
               artifacts.seed_record(2, "missing", None)]
    artifacts.set_seed_records(path, records)
    manifest = artifacts.set_status(path, "failed", error="seed 2 has no summary.json")
    assert manifest["seeds"] == records
    assert manifest["ended_at"] is not None
    with pytest.raises(ValueError, match="illegal"):
        artifacts.set_status(path, "complete")
    with pytest.raises(ValueError, match="seed status"):
        artifacts.seed_record(3, "done", None)


def test_interrupted_transition(tmp_path):
    path = tmp_path / "m.json"
    artifacts.write_json(path, artifacts.new_manifest("standalone", "none", "none", "none"))
    artifacts.set_status(path, "prepared")
    artifacts.set_status(path, "running")
    manifest = artifacts.set_status(path, "interrupted", error="SIGINT")
    assert (manifest["status"], manifest["error"]) == ("interrupted", "SIGINT")


def test_baseline_modules_do_not_import_rl_or_ml_stacks():
    script = textwrap.dedent("""
        import sys
        import scripts.baselines.config, scripts.baselines.adapter
        import scripts.baselines.effective_inputs, scripts.baselines.artifacts
        import scripts.baselines.solver
        banned = ('gymnasium', 'torch', 'stable_baselines3', 'sb3_contrib')
        print(sorted(m for m in sys.modules
                     if m.split('.')[0] in banned or m.startswith('scripts.rl')))
        retired = ('pydantic', 'shapely', 'pyproj', 'yaml', 'models', 'core', 'planners')
        print(sorted(m for m in sys.modules if m.split('.')[0] in retired))
    """)
    result = subprocess.run([sys.executable, "-c", script], cwd=MESH_ROOT, text=True,
                            capture_output=True, check=True)
    assert result.stdout.splitlines() == ["[]", "[]"]
