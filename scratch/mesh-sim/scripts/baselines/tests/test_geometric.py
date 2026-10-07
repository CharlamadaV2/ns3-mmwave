"""Greedy peer-anchor planner against fake_query.py's distance channel."""

import json
import math

import numpy as np
import pytest

from scripts.baselines.planners import geometric
from scripts.baselines.planners.channel import PlannerError
from scripts.baselines.planners.objective import MovementCost
from scripts.baselines.tests.test_objective import (REFERENCE, ZERO, assert_plan_invariants,
                                                    entry, make_request, planning_scorer)


def _solve(tmp_path, monkeypatch, entries, movable, range_m=100.0, **kwargs):
    request = make_request(entries, movable, **kwargs)
    with planning_scorer(tmp_path, monkeypatch, entries, request, range_m) as (scorer, log):
        positions, diagnostics, notes = geometric.solve(request, scorer, log)
    json.dumps(diagnostics)
    assert_plan_invariants(request, positions)
    assert all(isinstance(note, str) for note in notes)
    return request, positions, diagnostics


def _moved(request, positions, node_id) -> float:
    k = [node.id for node in request.nodes].index(node_id)
    node = request.nodes[k]
    return math.hypot(positions[k, 0] - node.x, positions[k, 1] - node.y)


def test_needs_schedule_and_settings():
    assert geometric.needs("coverage", 3, 0.5) == [1, 1, 1]
    assert geometric.needs("resilience", 2, 0.5) == [2, 2]
    assert geometric.needs("balanced", 3, 0.5) == [2, 2, 1]
    assert geometric.needs("balanced", 4, 0.0) == [1, 1, 1, 1]
    request = make_request([entry("a", 1, 1, 1), entry("b", 2, 2, 1)], {"a", "b"},
                           objective="balanced")
    assert geometric.settings(request)["needs"] == [2, 1]
    json.dumps(geometric.settings(request))


def test_all_movable_bootstrap_and_first_stay_put_seeds_the_pool(tmp_path, monkeypatch):
    entries = [entry("A", 100.0, 100.0, 10.0), entry("B", 150.0, 100.0, 10.0),
               entry("C", 300.0, 300.0, 10.0)]
    request, positions, diagnostics = _solve(tmp_path, monkeypatch, entries, {"A", "B", "C"},
                                             penalties=REFERENCE)
    steps = diagnostics["geometric"]["steps"]
    assert steps[0]["pool"] == [] and steps[0]["eff_need"] == 0 and steps[0]["bootstrap"]
    assert not steps[0]["accepted"]
    assert steps[1]["pool"] == ["A"] and steps[1]["eff_need"] == 1
    assert steps[2]["pool"] == ["A", "B"]
    assert _moved(request, positions, "A") == 0.0 and _moved(request, positions, "B") == 0.0
    assert diagnostics["disconnected_selected"] == []
    assert diagnostics["score"]["total"] > diagnostics["geometric"]["start_total"]


def test_all_movable_zero_cost_never_loses_score(tmp_path, monkeypatch):
    entries = [entry("A", 100.0, 100.0, 10.0), entry("B", 150.0, 100.0, 2.0, "vehicle"),
               entry("C", 160.0, 160.0, 1.5, "pedestrian")]
    _, _, diagnostics = _solve(tmp_path, monkeypatch, entries, {"A", "B", "C"})
    assert diagnostics["score"]["total"] > diagnostics["geometric"]["start_total"]
    assert diagnostics["disconnected_selected"] == []


def test_core_and_pool_follow_choices_and_skip_disconnected_anchors(tmp_path, monkeypatch):
    entries = [entry("F1", 50.0, 50.0, 10.0), entry("F2", 130.0, 50.0, 10.0),
               entry("F3", 390.0, 390.0, 10.0), entry("M1", 300.0, 300.0, 10.0),
               entry("M2", 380.0, 380.0, 10.0)]
    request, positions, diagnostics = _solve(tmp_path, monkeypatch, entries, {"M1", "M2"})
    steps = diagnostics["geometric"]["steps"]
    # {F1,F2} and {F3,M2} tie in size; the core holds the smaller roster index.
    assert steps[0]["pool"] == ["F1", "F2"] and steps[0]["accepted"]
    assert steps[1]["pool"] == ["F1", "F2", "M1"] and steps[1]["accepted"]
    assert diagnostics["disconnected_selected"] == []
    assert "F3" not in diagnostics["core"]
    assert positions[2].tolist() == [390.0, 390.0, 10.0]


def test_two_anchor_links_do_not_prove_resilience(tmp_path, monkeypatch):
    entries = [entry("F1", 100.0, 200.0, 10.0), entry("F2", 180.0, 200.0, 10.0),
               entry("F3", 260.0, 200.0, 10.0), entry("M", 350.0, 350.0, 10.0)]
    _, _, diagnostics = _solve(tmp_path, monkeypatch, entries, {"M"}, objective="resilience")
    step = diagnostics["geometric"]["steps"][0]
    assert step["eff_need"] == 2 and step["accepted"] and step["anchor_links"] >= 2
    assert not step["degraded"]
    assert diagnostics["survives_single_node_loss"] is False  # F2 still cuts F3 off
    assert diagnostics["controlled_mesh_survives_single_loss"] is True


def test_two_anchor_need_degrades_to_one(tmp_path, monkeypatch):
    entries = [entry("F1", 55.0, 45.0, 2.0, "vehicle"), entry("F2", 145.0, 45.0, 2.0, "vehicle"),
               entry("M", 350.0, 350.0, 60.0)]
    _, _, diagnostics = _solve(tmp_path, monkeypatch, entries, {"M"}, objective="resilience")
    step = diagnostics["geometric"]["steps"][0]
    assert step["need"] == 2 and step["eff_need"] == 2 and step["degraded"]
    assert step["accepted"] and step["anchor_links"] == 1
    assert diagnostics["disconnected_selected"] == []


def test_small_pool_lowers_the_effective_need(tmp_path, monkeypatch):
    entries = [entry("F1", 50.0, 50.0, 10.0), entry("M1", 120.0, 50.0, 10.0),
               entry("M2", 350.0, 350.0, 10.0)]
    _, _, diagnostics = _solve(tmp_path, monkeypatch, entries, {"M1", "M2"},
                               objective="resilience")
    first = diagnostics["geometric"]["steps"][0]
    assert first["pool"] == ["F1"] and first["need"] == 2 and first["eff_need"] == 1
    assert not first["degraded"]


@pytest.mark.parametrize("penalties,moves", [(REFERENCE, False), (ZERO, True)])
def test_reference_penalties_stay_put_zero_cost_moves(tmp_path, monkeypatch, penalties,
                                                      moves):
    entries = [entry("F1", 200.0, 200.0, 10.0), entry("F2", 260.0, 200.0, 10.0),
               entry("M", 230.0, 230.0, 10.0)]
    request, positions, diagnostics = _solve(tmp_path, monkeypatch, entries, {"M"},
                                             penalties=penalties)
    assert (_moved(request, positions, "M") > 0) is moves
    assert (diagnostics["nodes"]["M"]["displacement_m"] > 0) is moves
    if not moves:
        assert positions.tolist() == [[n.x, n.y, n.z] for n in request.nodes]
        assert diagnostics["nodes"]["M"]["move_cost_m2"] == 0.0


@pytest.mark.parametrize("node_type,z", [("drone", 10.0), ("vehicle", 2.0),
                                         ("pedestrian", 1.5)])
@pytest.mark.parametrize("fixed,per_m,moves", [
    (0.0, 0.0, True), (1000.0, 0.0, True), (1e6, 0.0, False),
    (0.0, 10.0, True), (0.0, 1e6, False)])
def test_fixed_only_and_distance_only_costs(tmp_path, monkeypatch, node_type, z, fixed,
                                            per_m, moves):
    platform = "aerial" if node_type == "drone" else "ground"
    other = "ground" if platform == "aerial" else "aerial"
    penalties = {platform: MovementCost(fixed, per_m), other: MovementCost(0.0, 0.0)}
    entries = [entry("F", 200.0, 200.0, 2.0, "vehicle"), entry("M", 210.0, 200.0, z, node_type)]
    request, positions, diagnostics = _solve(tmp_path, monkeypatch, entries, {"M"},
                                             penalties=penalties)
    distance = _moved(request, positions, "M")
    assert (distance > 0) is moves
    report = diagnostics["nodes"]["M"]
    assert report["platform"] == platform
    assert report["displacement_m"] == pytest.approx(distance)
    expected = fixed + per_m * distance if moves else 0.0
    assert report["move_cost_m2"] == pytest.approx(expected)
    assert diagnostics["score"]["move_cost_m2"] == pytest.approx(expected)


def test_platform_costs_apply_per_node(tmp_path, monkeypatch):
    entries = [entry("F", 200.0, 200.0, 2.0, "vehicle"), entry("D", 210.0, 200.0, 10.0),
               entry("P", 190.0, 200.0, 1.5, "pedestrian")]
    penalties = {"aerial": MovementCost(1e6, 0.0), "ground": MovementCost(0.0, 0.0)}
    request, positions, _ = _solve(tmp_path, monkeypatch, entries, {"D", "P"},
                                   penalties=penalties)
    assert _moved(request, positions, "D") == 0.0
    assert _moved(request, positions, "P") > 0.0


def test_connectivity_terms_cannot_be_bypassed(tmp_path, monkeypatch):
    entries = [entry("F1", 50.0, 50.0, 10.0), entry("F2", 120.0, 50.0, 10.0),
               entry("M", 100.0, 100.0, 10.0)]
    _, positions, diagnostics = _solve(tmp_path, monkeypatch, entries, {"M"})
    step = diagnostics["geometric"]["steps"][0]
    assert 0 < step["feasible"] < step["candidates"]
    assert diagnostics["disconnected_selected"] == []
    assert math.dist(positions[2, :2], (50.0, 50.0)) <= 100.0 or \
        math.dist(positions[2, :2], (120.0, 50.0)) <= 100.0


def test_vulnerability_term_drives_the_choice(tmp_path, monkeypatch):
    entries = [entry("F1", 100.0, 100.0, 10.0), entry("M1", 170.0, 100.0, 10.0),
               entry("M2", 240.0, 100.0, 10.0)]
    _, _, diagnostics = _solve(tmp_path, monkeypatch, entries, {"M1", "M2"}, range_m=110.0,
                               objective="resilience")
    assert diagnostics["geometric"]["start_total"] < 0  # one vulnerability pair at start
    assert diagnostics["vulnerability_pairs"] == 0
    assert diagnostics["controlled_mesh_survives_single_loss"] is True
    assert diagnostics["score"]["total"] > diagnostics["geometric"]["start_total"]


def test_original_start_costs_and_random_walk_bounds(tmp_path, monkeypatch):
    walk = {"x_min": 0.0, "x_max": 160.0, "y_min": 0.0, "y_max": 160.0}
    entries = [entry("F", 120.0, 100.0, 2.0, "vehicle"),
               entry("W", 100.0, 100.0, 1.5, "pedestrian", mobility="random_walk", walk=walk)]
    penalties = {"aerial": MovementCost(0.0, 0.0), "ground": MovementCost(10.0, 1.0)}
    request, positions, diagnostics = _solve(tmp_path, monkeypatch, entries, {"W"},
                                             penalties=penalties)
    x, y = positions[1, :2]
    assert 0.0 <= x <= 160.0 and 0.0 <= y <= 160.0
    distance = math.hypot(x - 100.0, y - 100.0)
    assert distance > 0
    assert diagnostics["nodes"]["W"]["move_cost_m2"] == pytest.approx(10.0 + distance)


def test_evaluation_mode_keeps_moves_inside_rl_bounds(tmp_path, monkeypatch):
    rl = {"x_min": 0.0, "x_max": 200.0, "y_min": 0.0, "y_max": 400.0}
    entries = [entry("F", 150.0, 150.0, 10.0), entry("M", 160.0, 150.0, 10.0)]
    _, positions, _ = _solve(tmp_path, monkeypatch, entries, {"M"}, mode="evaluation",
                             rl_bounds=rl)
    assert positions[1, 0] <= 200.0 and not np.array_equal(positions[1], [160.0, 150.0, 10.0])


def test_waypoint_policy_reject_and_translate(tmp_path, monkeypatch):
    waypoints = [{"t": 0.0, "x": 220.0, "y": 200.0, "z": 30.0},
                 {"t": 5.0, "x": 260.0, "y": 200.0, "z": 30.0}]
    entries = [entry("F", 200.0, 200.0, 30.0),
               entry("W", 0.0, 0.0, 30.0, mobility="waypoint", waypoints=waypoints)]
    with pytest.raises(PlannerError, match="waypoint_policy"):
        _solve(tmp_path / "reject", monkeypatch, entries, {"W"})
    request, positions, _ = _solve(tmp_path / "translate", monkeypatch, entries, {"W"},
                                   waypoint_policy="translate")
    assert positions[1, 2] == 30.0 and _moved(request, positions, "W") > 0
