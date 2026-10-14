"""Shared objective: exact arithmetic on hand-built link tables, grid geometry and boundaries.

Gaps over scripts/baselines/tests/test_objective.py: hand-computed totals with every term
non-zero, core ties beyond node 0, N = 1 / N = 2, all-disconnected with an unselected core,
clipped-cell weights and centres, the cap / stay-put / scale-warning boundaries, context
validation and LayoutCache key normalisation.
"""

import math

import numpy as np
import pytest

from scripts.baselines.planners.objective import (LayoutCache, MovementCost, candidate_positions, components, core, diagnose, rectangle_grid, score, vulnerability_pairs)
from scripts.baselines.tests.test_objective import AOI, ZERO, entry, pure_context, table

from ._planner_support import TableScorer

DIAG = math.hypot(400.0, 400.0)
CELL = 100.0  # 16 coverage cells over the 400 m x 400 m RECT


def _rect(x_min, x_max, y_min, y_max) -> dict:
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


# ---------------------------------------------------------------------------------------
# Grid geometry.

def test_grid_clipped_last_row_and_column_exact_values():
    points, cell, weights = rectangle_grid(_rect(0.0, 10.0, 0.0, 7.0), 4, 1.0)
    c = math.sqrt(70.0 / 4)
    assert cell == pytest.approx(c)
    xe = [0.0, c, 2 * c, 10.0]
    ye = [0.0, c, 7.0]
    expected_points = [[(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2]
                       for j in range(2) for i in range(3)]
    expected_weights = [(xe[i + 1] - xe[i]) * (ye[j + 1] - ye[j])
                        for j in range(2) for i in range(3)]
    assert np.allclose(points, expected_points)
    assert np.allclose(weights, expected_weights)
    assert weights.sum() == pytest.approx(70.0)


def test_grid_is_laid_from_the_lower_left_corner_of_an_offset_rectangle():
    points, cell, weights = rectangle_grid(_rect(-13.0, 387.0, 5.0, 205.0), 8, 1.0)
    assert cell == 100.0
    assert points[:, 0].tolist()[:4] == [37.0, 137.0, 237.0, 337.0]
    assert sorted(set(points[:, 1].tolist())) == [55.0, 155.0]
    assert np.allclose(weights, 10000.0)


def test_grid_min_resolution_dominates_and_keeps_edge_slivers():
    points, cell, weights = rectangle_grid(_rect(0.0, 10.5, 0.0, 10.0), 400, 5.0)
    assert cell == 5.0
    assert weights.tolist() == [25.0, 25.0, 2.5, 25.0, 25.0, 2.5]
    assert points[2].tolist() == [10.25, 2.5] and points[5].tolist() == [10.25, 7.5]


@pytest.mark.parametrize("rect,cells,min_res,count", [
    (_rect(0.0, 0.3, 0.0, 0.3), 9, 0.1, 9),
    (_rect(0.0, 0.7, 0.0, 0.7), 49, 0.01, 49),
    (_rect(1e6, 1e6 + 300.0, 2e6, 2e6 + 300.0), 9, 1.0, 9),
])
def test_grid_float_rounding_never_adds_a_sliver_row(rect, cells, min_res, count):
    points, cell, weights = rectangle_grid(rect, cells, min_res)
    assert len(points) == count
    assert weights.min() == pytest.approx(cell * cell)


# ---------------------------------------------------------------------------------------
# Components and core.

def test_core_prefers_size_then_smallest_roster_index():
    assert core(components(table(4, [(1, 2), (2, 3)]).connected)) == frozenset({1, 2, 3})
    tie = components(table(5, [(2, 3), (1, 4)]).connected)
    assert tie == [frozenset({0}), frozenset({1, 4}), frozenset({2, 3})]
    assert core(tie) == frozenset({1, 4})
    assert core(list(reversed(tie))) == frozenset({1, 4})  # independent of list order


def test_components_ignore_the_diagonal():
    connected = table(4, [(0, 1)]).connected.copy()
    np.fill_diagonal(connected, True)
    assert components(connected) == [frozenset({0, 1}), frozenset({2}), frozenset({3})]


# ---------------------------------------------------------------------------------------
# Vulnerability pairs.

def test_vulnerability_pairs_exact_counts():
    path = table(4, [(0, 1), (1, 2), (2, 3)]).connected
    assert vulnerability_pairs(path, {0, 1, 2, 3}, [1]) == 2       # 0 | 2-3
    assert vulnerability_pairs(path, {0, 1, 2, 3}, [1, 2]) == 4    # plus 0-1 | 3
    star = table(5, [(0, 1), (0, 2), (0, 3), (0, 4)]).connected
    assert vulnerability_pairs(star, {0, 1, 2, 3, 4}, [0]) == 6    # C(4, 2)
    two = table(2, [(0, 1)]).connected
    assert vulnerability_pairs(two, {0, 1}, [0, 1]) == 0          # one survivor, no pair


# ---------------------------------------------------------------------------------------
# score(): exact arithmetic.

def test_score_exact_total_with_every_term():
    nodes = [entry("a", 50.0, 50.0, 10.0), entry("b", 150.0, 50.0, 10.0),
             entry("c", 250.0, 50.0, 2.0, "vehicle"), entry("d", 350.0, 350.0, 1.5, "pedestrian")]
    penalties = {"aerial": MovementCost(1000.0, 10.0), "ground": MovementCost(500.0, 2.0)}
    ctx, _ = pure_context(nodes, {"a", "b", "d"}, objective="balanced", penalties=penalties)
    layout = ctx.starts.copy()
    layout[0, :2] = [50.0, 80.0]          # a: 30 m -> 1000 + 300
    layout[3, 0] += 5e-7                  # d: below the stay-put tolerance -> 0
    result = table(4, [(0, 1), (1, 2)], coverage=[[0, 1], [1, 2], [2, 15], [3, 4, 5]])
    value = score(ctx, result, layout)
    nn = [math.hypot(100.0, 30.0), 100.0, math.hypot(100.0 + 5e-7, 300.0)]
    sep = 0.4 * CELL ** 2 * np.mean([d / DIAG for d in nn])
    assert value.coverage_m2 == 4 * 10000.0          # probe 2 counted once; d excluded
    assert value.move_cost_m2 == pytest.approx(1300.0)
    assert value.disconnected == 1 and value.vuln == 1  # victim b splits a | c
    assert value.sep_m2 == pytest.approx(sep)
    assert value.total == pytest.approx(40000.0 + sep - 1300.0 - 2 * AOI - 0.02 * AOI)


@pytest.mark.parametrize("linked", [True, False])
def test_score_two_node_roster(linked):
    nodes = [entry("a", 100.0, 100.0, 10.0), entry("b", 300.0, 300.0, 10.0)]
    ctx, _ = pure_context(nodes, {"a", "b"}, objective="resilience")
    result = table(2, [(0, 1)] if linked else [], coverage=[[0], [15]])
    value = score(ctx, result, ctx.starts)
    sep = 0.4 * CELL ** 2 * math.hypot(200.0, 200.0) / DIAG
    assert value.vuln == 0                    # a two-node core has no splittable pair
    if linked:
        assert value.disconnected == 0 and value.coverage_m2 == 20000.0
        assert value.total == pytest.approx(20000.0 + sep)
    else:                                     # tie of singletons -> core {a}
        assert value.disconnected == 1 and value.coverage_m2 == 10000.0
        assert value.total == pytest.approx(10000.0 + sep - 2 * AOI)


def test_all_disconnected_core_is_the_unselected_first_node():
    nodes = [entry("f", 50.0, 50.0, 2.0, "vehicle"), entry("m1", 150.0, 150.0, 10.0),
             entry("m2", 250.0, 250.0, 10.0)]
    ctx, _ = pure_context(nodes, {"m1", "m2"}, objective="resilience")
    value = score(ctx, table(3, [], coverage=[[0], [5], [10]]), ctx.starts)
    assert value.coverage_m2 == 10000.0       # only f (the core) counts
    assert value.disconnected == 2 and value.vuln == 0
    report = diagnose(ctx, table(3, [], coverage=[[0], [5], [10]]), ctx.starts, value)
    assert report["core"] == ["f"] and report["component_sizes"] == [1, 1, 1]
    assert report["disconnected_selected"] == ["m1", "m2"]
    assert report["survives_single_node_loss"] is True
    assert report["score"]["connectivity_penalty_m2"] == 2 * 2 * AOI


def test_selected_node_outside_core_is_not_a_vulnerability_victim():
    nodes = [entry(f"n{k}", 20.0 + 50.0 * k, 20.0, 10.0) for k in range(7)]
    ctx, _ = pure_context(nodes, {"n1", "n5"}, objective="resilience")
    result = table(7, [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)], coverage=[[]] * 7)
    value = score(ctx, result, ctx.starts)
    assert value.vuln == 2           # only n1 (core); n5 would split n4 | n6 but is outside
    assert value.disconnected == 1


def test_coverage_uses_clipped_cell_areas_once_per_probe():
    small = _rect(0.0, 10.5, 0.0, 10.0)
    nodes = [entry("a", 2.0, 2.0, 10.0), entry("b", 9.0, 8.0, 10.0),
             entry("c", 10.0, 1.0, 10.0)]
    ctx, _ = pure_context(nodes, {"a"}, rectangle=small, coverage_cells=400,
                          candidate_cells=400)
    assert ctx.coverage_weights.tolist() == [25.0, 25.0, 2.5, 25.0, 25.0, 2.5]
    assert ctx.aoi_m2 == 105.0 and ctx.cell_m == 5.0
    result = table(3, [(0, 1), (1, 2)], coverage=[[2, 5], [5, 2], [0, 2]])
    assert score(ctx, result, ctx.starts).coverage_m2 == 25.0 + 2.5 + 2.5


# ---------------------------------------------------------------------------------------
# Movement, candidates and their 1e-6 / cap boundaries.


def test_candidate_positions_from_a_moved_current_still_cap_from_the_start():
    nodes = [entry("m", 50.0, 50.0, 10.0), entry("f", 350.0, 350.0, 2.0, "vehicle")]
    ctx, _ = pure_context(nodes, {"m"}, penalties={"aerial": MovementCost(0, 0, 150.0),
                                                   "ground": MovementCost(0, 0)})
    rows = candidate_positions(ctx, 0, current=[150.0, 150.0, 10.0])
    assert rows[0].tolist() == [150.0, 150.0, 10.0]
    assert [150.0, 150.0] not in rows[1:, :2].tolist()
    assert all(math.hypot(x - 50.0, y - 50.0) <= 150.0 for x, y in rows[1:, :2])
    assert [50.0, 50.0] in rows[1:, :2].tolist()  # the start is a move back from current


# ---------------------------------------------------------------------------------------
# diagnose() and context validation.


def test_diagnose_two_node_core_survives_single_loss():
    nodes = [entry("a", 50.0, 50.0, 10.0), entry("b", 150.0, 50.0, 10.0)]
    ctx, _ = pure_context(nodes, {"a", "b"}, objective="resilience")
    result = table(2, [(0, 1)], coverage=[[0], [1]])
    report = diagnose(ctx, result, ctx.starts, score(ctx, result, ctx.starts))
    assert report["survives_single_node_loss"] is True
    assert report["controlled_mesh_survives_single_loss"] is True
    assert report["coverage_fraction"] == pytest.approx(20000.0 / AOI)


def test_layout_cache_normalises_dtype_and_preserves_order():
    scorer = TableScorer(lambda array: float(array.sum()))
    cache = LayoutCache(scorer)
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = a + 1.0
    c = a + 2.0
    assert cache.evaluate([[[1, 2, 3], [4, 5, 6]]]) == [21.0]     # ints key like floats
    assert cache.evaluate([b, a, c, b]) == [27.0, 21.0, 33.0, 27.0]
    assert [len(call) for call in scorer.calls] == [1, 2]
    assert np.array_equal(scorer.calls[1][0], b) and np.array_equal(scorer.calls[1][1], c)
    assert cache.evaluate([np.asfortranarray(a)]) == [21.0]
    assert len(scorer.calls) == 2


def test_zero_penalties_never_charge_movement():
    ctx, _ = pure_context([entry("a", 50.0, 50.0, 10.0), entry("b", 60.0, 50.0, 10.0)],
                          {"a", "b"}, penalties=ZERO)
    layout = ctx.starts.copy()
    layout[:, 0] += 200.0
    assert score(ctx, table(2, [(0, 1)], coverage=[[], []]), layout).move_cost_m2 == 0.0
