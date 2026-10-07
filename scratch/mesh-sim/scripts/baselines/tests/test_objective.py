"""Pure objective helpers plus request/fake-worker builders shared by the strategy tests."""

import io
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.baselines.planners.channel import ChannelScorer, LayoutResult, PlannerError
from scripts.baselines.planners.objective import (
    BIG, OBJECTIVES, GridSettings, LayoutCache, LayoutScore, MovementCost, ProbeSettings,
    ScoringContext, candidate_positions, check_layout, components, core, diagnose,
    prepare_scorer, rectangle_grid, score, vulnerability_pairs, vulnerability_weight)

FAKE_QUERY = Path(__file__).with_name("fake_query.py")
RECT = {"x_min": 0.0, "x_max": 400.0, "y_min": 0.0, "y_max": 400.0}
AOI = 160000.0
ZERO = {"aerial": MovementCost(0.0, 0.0), "ground": MovementCost(0.0, 0.0)}
REFERENCE = {"aerial": MovementCost(150000.0, 500.0), "ground": MovementCost(150000.0, 100.0)}
PLATFORM = {"drone": "aerial", "vehicle": "ground", "pedestrian": "ground"}


def entry(node_id, x, y, z, node_type="drone", mobility="fixed", walk=None,
          waypoints=None) -> dict:
    """One nodes.json entry; waypoint nodes start at waypoints[0]."""
    item = {"id": node_id, "role": "peer", "mobility": mobility, "node_type": node_type,
            "position": {"x": x, "y": y, "z": z}}
    if walk is not None:
        item["random_walk"] = {"bounds": walk, "speed_mps": 1.0}
    if waypoints is not None:
        item["waypoints"] = waypoints
    return item


def _start(item: dict) -> tuple:
    point = item["waypoints"][0] if item["mobility"] == "waypoint" else item["position"]
    return tuple(float(point[axis]) for axis in ("x", "y", "z"))


def make_request(entries, movable, *, objective="coverage", method="geometric",
                 penalties=None, mode="standalone", rl_bounds=None, rectangle=None,
                 candidate_cells=16, coverage_cells=64, min_resolution_m=5.0, probe=None,
                 planner_seed=1, max_iterations=None, balanced_core_fraction=0.5,
                 waypoint_policy="reject"):
    """Duck-typed stand-in for solver.PlanRequest."""
    nodes = []
    for index, item in enumerate(entries):
        x, y, z = _start(item)
        walk = item.get("random_walk", {}).get("bounds") \
            if item["mobility"] == "random_walk" else None
        nodes.append(SimpleNamespace(
            id=item["id"], roster_index=index, node_type=item["node_type"],
            mobility=item["mobility"], role="movable" if item["id"] in movable else "fixed",
            platform=PLATFORM[item["node_type"]], x=x, y=y, z=z,
            random_walk_bounds=dict(walk) if walk else None,
            has_waypoints=bool(item.get("waypoints"))))
    return SimpleNamespace(
        method=method, objective=objective, nodes=tuple(nodes),
        rectangle=dict(rectangle or RECT), rl_bounds=rl_bounds, mode=mode,
        penalties=dict(penalties or ZERO),
        grid=GridSettings(candidate_cells, coverage_cells, min_resolution_m),
        probe=probe or ProbeSettings(1.5, None, -6.7), planner_seed=planner_seed,
        max_iterations=max_iterations, balanced_core_fraction=balanced_core_fraction,
        waypoint_policy=waypoint_policy)


@contextmanager
def planning_scorer(tmp_path, monkeypatch, entries, request, range_m=100.0, seed=5):
    """ChannelScorer on fake_query.py (distance channel) with the request's probes set."""
    for name in ("FAKE_QUERY_FAULT", "FAKE_QUERY_FAULT_AT", "FAKE_QUERY_INIT_PATCH",
                 "FAKE_QUERY_PAD_BYTES", "FAKE_QUERY_RECORD", "FAKE_QUERY_PID_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FAKE_QUERY_RANGE_M", str(range_m))
    scenario = tmp_path / "scenario"
    scenario.mkdir(parents=True, exist_ok=True)
    (scenario / "run.ini").write_text(f"[scenario]\nname = planner-test\nseed = {seed}\n"
                                      "run_id = 1\nnodes_file = nodes.json\n\n"
                                      "[channel]\nrx_array_gain_dbi = 0.0\n")
    (scenario / "nodes.json").write_text(json.dumps(entries, indent=2) + "\n")
    binary = tmp_path / "bin" / "fake-mesh-sim"
    binary.parent.mkdir(exist_ok=True)
    binary.write_text(f"#!{sys.executable}\nimport runpy\n"
                      f"runpy.run_path({str(FAKE_QUERY)!r}, run_name='__main__')\n")
    binary.chmod(0o755)
    starts = [_start(item) for item in entries]
    with open(tmp_path / "planner.log", "w", encoding="utf-8") as log:
        with ChannelScorer(binary, scenario / "run.ini", seed, 1, None, request.mode,
                           [item["id"] for item in entries], starts, seed, log) as scorer:
            prepare_scorer(request, scorer)
            yield scorer, log


def assert_plan_invariants(request, positions) -> None:
    """Unselected nodes exactly at their start and every z unchanged."""
    positions = np.asarray(positions)
    for k, node in enumerate(request.nodes):
        assert positions[k, 2] == node.z
        if node.role != "movable":
            assert tuple(positions[k]) == (node.x, node.y, node.z)


class StubScorer:
    """Records set_probes/evaluate; evaluate returns canned results."""

    def __init__(self, results=None):
        self.probes = None
        self.calls = []
        self._results = results or {}

    def set_probes(self, points, *, height_m, rx_gain_dbi, sinr_db):
        self.calls.append(("set_probes", height_m, rx_gain_dbi, sinr_db))
        self.probes = {"points": [list(p) for p in np.asarray(points).tolist()],
                       "height_m": height_m, "rx_gain_dbi": rx_gain_dbi or 0.0,
                       "sinr_db": sinr_db}

    def evaluate(self, layouts):
        self.calls.append(("evaluate", len(layouts)))
        return [self._results.get(np.asarray(l).tobytes(), "result") for l in layouts]


def table(n, edges, coverage=None) -> LayoutResult:
    connected = np.zeros((n, n), dtype=bool)
    for i, j in edges:
        connected[i, j] = connected[j, i] = True
    sinr = np.where(connected, 0.0, -20.0)
    np.fill_diagonal(sinr, -np.inf)
    return LayoutResult(connected=connected, sinr_db=sinr, capacity_mbps=np.zeros((n, n)),
                        is_los=connected.copy(), coverage=coverage, wall_s=0.0)


def pure_context(entries, movable, **kwargs):
    kwargs.setdefault("coverage_cells", 16)
    request = make_request(entries, movable, **kwargs)
    stub = StubScorer()
    prepare_scorer(request, stub)
    return ScoringContext.build(request, stub), request


FOUR = [entry("a", 50.0, 50.0, 10.0), entry("b", 120.0, 50.0, 10.0),
        entry("c", 300.0, 300.0, 2.0, "vehicle"), entry("d", 350.0, 50.0, 1.5, "pedestrian")]


def test_movement_cost_formula_and_cap():
    cost = MovementCost(150000.0, 500.0)
    assert cost.cost(0.0) == 0.0 and cost.cost(1e-6) == 0.0
    assert cost.cost(2e-6) == pytest.approx(150000.0 + 500.0 * 2e-6)
    assert cost.cost(10.0) == 155000.0
    assert cost.within_cap(1e9)
    capped = MovementCost(0.0, 100.0, 50.0)
    assert capped.within_cap(50.0) and not capped.within_cap(50.0001)
    assert MovementCost(0.0, 0.0).cost(500.0) == 0.0


def test_vulnerability_weights_and_big():
    assert OBJECTIVES == ("coverage", "balanced", "resilience")
    assert vulnerability_weight("resilience", AOI) == AOI
    assert vulnerability_weight("balanced", AOI) == pytest.approx(0.02 * AOI)
    assert vulnerability_weight("coverage", AOI) == 0.0
    assert BIG(AOI) == 2 * AOI
    with pytest.raises(ValueError):
        vulnerability_weight("gateway", AOI)


def test_rectangle_grid_exact_division():
    points, cell, weights = rectangle_grid(RECT, 16, 5.0)
    assert cell == 100.0 and len(points) == 16
    assert np.allclose(weights, 10000.0)
    assert points[0].tolist() == [50.0, 50.0] and points[1].tolist() == [150.0, 50.0]


@pytest.mark.parametrize("rect,cells,min_res", [
    ({"x_min": 0.0, "x_max": 10.0, "y_min": 0.0, "y_max": 7.0}, 4, 1.0),
    ({"x_min": -13.0, "x_max": 401.7, "y_min": 5.5, "y_max": 233.0}, 400, 5.0),
    ({"x_min": 0.0, "x_max": 1000.0, "y_min": 0.0, "y_max": 3.0}, 50, 5.0),
    ({"x_min": 0.0, "x_max": 3.0, "y_min": 0.0, "y_max": 3.0}, 400, 5.0),
])
def test_rectangle_grid_clipped_weights_sum_to_aoi(rect, cells, min_res):
    points, cell, weights = rectangle_grid(rect, cells, min_res)
    aoi = (rect["x_max"] - rect["x_min"]) * (rect["y_max"] - rect["y_min"])
    assert cell == max(min_res, math.sqrt(aoi / cells))
    assert weights.sum() == pytest.approx(aoi, rel=1e-6)
    assert weights.sum() <= aoi * (1 + 1e-9)
    assert np.all(weights > 0) and np.all(weights <= cell * cell * (1 + 1e-9))
    assert np.all((points[:, 0] > rect["x_min"]) & (points[:, 0] < rect["x_max"]))
    assert np.all((points[:, 1] > rect["y_min"]) & (points[:, 1] < rect["y_max"]))


def test_rectangle_grid_single_cell_and_invalid_input():
    points, cell, weights = rectangle_grid({"x_min": 0.0, "x_max": 3.0, "y_min": 0.0,
                                            "y_max": 3.0}, 400, 5.0)
    assert cell == 5.0 and points.tolist() == [[1.5, 1.5]] and weights.tolist() == [9.0]
    with pytest.raises(ValueError):
        rectangle_grid({"x_min": 1.0, "x_max": 1.0, "y_min": 0.0, "y_max": 3.0}, 4, 1.0)
    with pytest.raises(ValueError):
        rectangle_grid(RECT, 0, 5.0)
    with pytest.raises(ValueError):
        rectangle_grid(RECT, 4, 0.0)


def test_components_and_core_tie_break():
    parts = components(table(5, [(2, 3), (0, 4)]).connected)
    assert parts == [frozenset({0, 4}), frozenset({1}), frozenset({2, 3})]
    assert core(parts) == frozenset({0, 4})
    assert core(components(table(5, [(1, 2), (3, 4)]).connected)) == frozenset({1, 2})
    assert core(components(table(5, [(3, 4), (2, 4)]).connected)) == frozenset({2, 3, 4})


def test_singleton_empty_and_all_disconnected_graphs():
    assert components(np.zeros((1, 1), dtype=bool)) == [frozenset({0})]
    assert core([frozenset({0})]) == frozenset({0})
    lonely = components(table(3, []).connected)
    assert lonely == [frozenset({0}), frozenset({1}), frozenset({2})]
    assert core(lonely) == frozenset({0})
    assert vulnerability_pairs(table(3, []).connected, {0}, [0]) == 0
    with pytest.raises(ValueError):
        core([])


def test_vulnerability_pairs_articulation_vs_two_connected():
    path = table(3, [(0, 1), (1, 2)]).connected
    assert vulnerability_pairs(path, {0, 1, 2}, [1]) == 1
    assert vulnerability_pairs(path, {0, 1, 2}, [0, 1, 2]) == 1
    cycle = table(4, [(0, 1), (1, 2), (2, 3), (3, 0)]).connected
    assert vulnerability_pairs(cycle, {0, 1, 2, 3}, [0, 1, 2, 3]) == 0
    star = table(4, [(0, 1), (0, 2), (0, 3)]).connected
    assert vulnerability_pairs(star, {0, 1, 2, 3}, [0]) == 3
    assert vulnerability_pairs(star, {0, 1, 2, 3}, [1, 2, 3]) == 0
    # Non-core node 3 never counts; an empty victim list is zero.
    tail = table(4, [(0, 1), (1, 2)]).connected
    assert vulnerability_pairs(tail, {0, 1, 2}, [1]) == 1
    assert vulnerability_pairs(tail, {0, 1, 2}, []) == 0


def test_prepare_scorer_sends_the_coverage_grid_once():
    request = make_request(FOUR, {"a"}, coverage_cells=16,
                           probe=ProbeSettings(2.5, None, -3.0))
    stub = StubScorer()
    prepare_scorer(request, stub)
    assert stub.calls == [("set_probes", 2.5, None, -3.0)]
    assert len(stub.probes["points"]) == 16
    ctx = ScoringContext.build(request, stub)
    assert ctx.aoi_m2 == AOI and ctx.cell_m == 100.0 and ctx.coverage_weights.sum() == AOI
    assert ctx.big == 2 * AOI and ctx.selected == (0,)
    assert len(ctx.candidate_points) == 16


def test_build_requires_matching_probes():
    request = make_request(FOUR, {"a"}, coverage_cells=16)
    with pytest.raises(ValueError, match="prepare_scorer"):
        ScoringContext.build(request, StubScorer())
    stub = StubScorer()
    prepare_scorer(make_request(FOUR, {"a"}, coverage_cells=64), stub)
    with pytest.raises(ValueError, match="coverage grid"):
        ScoringContext.build(request, stub)


def test_empty_selection_is_rejected():
    with pytest.raises(PlannerError, match="no movable node"):
        pure_context(FOUR, set())


def test_selected_waypoint_node_reject_and_translate():
    nodes = [entry("a", 0.0, 0.0, 10.0),
             entry("w", 0.0, 0.0, 10.0, mobility="waypoint",
                   waypoints=[{"t": 0.0, "x": 60.0, "y": 60.0, "z": 10.0},
                              {"t": 5.0, "x": 90.0, "y": 60.0, "z": 10.0}])]
    with pytest.raises(PlannerError, match="waypoint_policy"):
        pure_context(nodes, {"w"})
    ctx, _ = pure_context(nodes, {"w"}, waypoint_policy="translate")
    assert ctx.starts[1].tolist() == [60.0, 60.0, 10.0]
    pure_context(nodes, {"a"})  # an unselected waypoint node is fine under reject


def test_score_counts_coverage_of_the_core_only():
    ctx, _ = pure_context(FOUR, {"a", "c"})
    result = table(4, [(0, 1)], coverage=[[0, 1], [1, 2], [3], [5]])
    value = score(ctx, result, ctx.starts)
    assert value.coverage_m2 == 3 * 10000.0
    assert value.disconnected == 1  # selected c outside the core; unselected d is not charged
    assert value.vuln == 0 and value.move_cost_m2 == 0.0
    expected_sep = 0.4 * 100.0 ** 2 * np.mean([70.0 / math.hypot(400, 400),
                                               math.hypot(50, 250) / math.hypot(400, 400)])
    assert value.sep_m2 == pytest.approx(expected_sep)
    assert value.total == pytest.approx(value.coverage_m2 + value.sep_m2 - 2 * AOI)


def test_score_all_disconnected_charges_every_selected_node():
    ctx, _ = pure_context(FOUR, {"b", "c", "d"})
    value = score(ctx, table(4, [], coverage=[[0], [1], [2], [3]]), ctx.starts)
    assert value.coverage_m2 == 10000.0 and value.disconnected == 3


@pytest.mark.parametrize("objective,weight", [("coverage", 0.0), ("balanced", 0.02 * AOI),
                                              ("resilience", AOI)])
def test_score_movement_from_original_start_and_vulnerability(objective, weight):
    penalties = {"aerial": MovementCost(1000.0, 10.0), "ground": MovementCost(500.0, 1.0)}
    ctx, _ = pure_context(FOUR, {"a", "b", "c"}, objective=objective, penalties=penalties)
    layout = ctx.starts.copy()
    layout[1, :2] = [120.0, 80.0]   # b: 30 m
    layout[2, :2] = [300.0, 260.0]  # c: 40 m
    result = table(4, [(0, 1), (1, 2), (2, 3)], coverage=[[0], [1], [2], [3]])
    value = score(ctx, result, layout)
    assert value.move_cost_m2 == pytest.approx((1000 + 10 * 30) + (500 + 1 * 40))
    assert value.vuln == 2 + 2  # victims b and c each split the path
    assert value.total == pytest.approx(value.coverage_m2 + value.sep_m2 -
                                        value.move_cost_m2 - weight * value.vuln)


def test_diagnose_reports_resilience_movement_and_scale():
    small = {"x_min": 0.0, "x_max": 100.0, "y_min": 0.0, "y_max": 100.0}
    nodes = [entry("a", 10.0, 10.0, 10.0), entry("b", 50.0, 10.0, 10.0),
             entry("c", 90.0, 10.0, 2.0, "vehicle")]
    ctx, _ = pure_context(nodes, {"a", "c"}, rectangle=small, penalties=REFERENCE,
                          coverage_cells=4, candidate_cells=4)
    layout = ctx.starts.copy()
    layout[0, :2] = [10.0, 40.0]
    result = table(3, [(0, 1), (1, 2)], coverage=[[0], [1], [2, 3]])
    value = score(ctx, result, layout)
    log = io.StringIO()
    report = diagnose(ctx, result, layout, value, log=log)
    json.dumps(report)
    assert report["core"] == ["a", "b", "c"] and report["component_sizes"] == [3]
    assert report["survives_single_node_loss"] is False  # b is an articulation point
    assert report["controlled_mesh_survives_single_loss"] is True  # victims a, c only
    assert report["vulnerability_pairs"] == 0
    assert report["nodes"]["a"]["displacement_m"] == pytest.approx(30.0)
    assert report["nodes"]["a"]["move_cost_m2"] == pytest.approx(150000 + 500 * 30)
    assert report["nodes"]["b"]["move_cost_m2"] == 0.0
    assert report["coverage_fraction"] == pytest.approx(1.0)
    assert report["fixed_cost_over_aoi"] == {"aerial": 15.0, "ground": 15.0}
    assert len(report["scale_warnings"]) == 2 and "scale warning" in log.getvalue()
    assert report["score"]["total"] == value.total


def test_candidate_positions_respect_bounds_cap_and_keep_current_first():
    walk = {"x_min": 0.0, "x_max": 400.0, "y_min": 0.0, "y_max": 200.0}
    nodes = [entry("a", 380.0, 380.0, 10.0), entry("w", 10.0, 10.0, 1.5, "pedestrian",
                                                   mobility="random_walk", walk=walk)]
    rl = {"x_min": 0.0, "x_max": 300.0, "y_min": 0.0, "y_max": 400.0}
    ctx, _ = pure_context(nodes, {"a", "w"}, mode="evaluation", rl_bounds=rl,
                          penalties={"aerial": MovementCost(0, 0, 250.0),
                                     "ground": MovementCost(0, 0)})
    a = candidate_positions(ctx, 0)
    assert a[0].tolist() == [380.0, 380.0, 10.0]  # current position even outside [rl]
    assert np.all(a[:, 2] == 10.0)
    assert np.all(a[1:, 0] <= 300.0)
    assert np.all(np.hypot(a[1:, 0] - 380.0, a[1:, 1] - 380.0) <= 250.0)
    w = candidate_positions(ctx, 1)
    assert np.all(w[1:, 1] <= 200.0) and np.all(w[:, 2] == 1.5)
    standalone, _ = pure_context(nodes, {"a", "w"}, rl_bounds=rl)
    assert standalone.rl_bounds is None
    assert len(candidate_positions(standalone, 0)) > len(a)


def test_check_layout_and_cache():
    ctx, _ = pure_context(FOUR, {"a"})
    layout = ctx.starts.copy()
    layout[0, :2] = [1.0, 2.0]
    check_layout(ctx, layout)
    bad = layout.copy()
    bad[0, 2] += 1.0
    with pytest.raises(PlannerError, match="z"):
        check_layout(ctx, bad)
    bad = layout.copy()
    bad[1, 0] += 0.5
    with pytest.raises(PlannerError, match="unselected"):
        check_layout(ctx, bad)
    stub = StubScorer()
    cache = LayoutCache(stub)
    assert len(cache.evaluate([layout, ctx.starts, layout])) == 3
    assert len(cache.evaluate([ctx.starts])) == 1
    assert stub.calls == [("evaluate", 2)]
    assert isinstance(LayoutScore(0.0, 0.0, 0.0, 0, 0, 0.0).as_dict(), dict)
