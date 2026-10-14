"""Boundary-exact checks of adapter.validate_result on hand-built node records.

Source: scripts/baselines/README.md "What preparation does" step 4 -- validation fails,
and never clamps, on missing or extra ids, non-finite values, a selected node outside the
rectangle (the boundary counts as inside), a planned z that differs from the source z by
more than 1e-6, displacement beyond the platform cap, an unselected node moved by more
than 0.01 m, a position outside the [rl] x/y bounds (evaluation), and a random_walk node
outside its own random_walk.bounds. Coordinates are chosen so the distances are exactly
representable in binary floating point.
"""

import math

import numpy as np
import pytest

from scripts.baselines import adapter
from scripts.baselines.adapter import NodeRecord
from scripts.baselines.planners.objective import MovementCost
from scripts.baselines.solver import PlanResult

RECT = {"x_min": 0.0, "x_max": 64.0, "y_min": 0.0, "y_max": 64.0}
UNCAPPED = {"aerial": MovementCost(0.0, 0.0, None), "ground": MovementCost(0.0, 0.0, None)}


def _rec(node_id, x, y, z=0.0, selected=True, platform="aerial", index=0,
         mobility="fixed", rw=None):
    return NodeRecord(id=node_id, roster_index=index, node_type="drone", mobility=mobility,
                      role="movable" if selected else "fixed", platform=platform,
                      platform_source="node_type", x=x, y=y, z=z, random_walk_bounds=rw)


def _check(records, positions, penalties=None, rl_bounds=None, rect=None):
    return adapter.validate_result(tuple(records), PlanResult(positions=positions),
                                   dict(rect or RECT), penalties or UNCAPPED, rl_bounds)


# --- rectangle containment --------------------------------------------------------------

@pytest.mark.parametrize("planned", [(0.0, 32.0), (64.0, 32.0), (32.0, 0.0), (32.0, 64.0),
                                     (0.0, 0.0), (64.0, 64.0), (0.0, 64.0), (64.0, 0.0)])
def test_every_rectangle_edge_and_corner_counts_as_inside(planned):
    final = _check([_rec("a", 32.0, 32.0)], {"a": (*planned, 0.0)})
    assert final["a"] == (*planned, 0.0)


def test_unselected_node_outside_the_rectangle_is_left_alone():
    records = [_rec("a", 32.0, 32.0), _rec("far", 500.0, -500.0, selected=False, index=1)]
    final = _check(records, {"a": (10.0, 10.0, 0.0), "far": (500.0, -500.0, 0.0)},
                   rl_bounds=dict(RECT))
    assert final["far"] == (500.0, -500.0, 0.0)


# --- z tolerance ------------------------------------------------------------------------


# --- unselected nodes: 0.01 m round-trip tolerance --------------------------------------


# --- displacement caps ------------------------------------------------------------------

def _capped(cap, platform="aerial"):
    return {**UNCAPPED, platform: MovementCost(0.0, 0.0, cap)}


def test_displacement_exactly_at_the_cap_is_accepted():
    final = _check([_rec("a", 10.0, 10.0)], {"a": (13.0, 14.0, 0.0)}, _capped(5.0))
    assert final["a"] == (13.0, 14.0, 0.0)
    assert math.hypot(3.0, 4.0) == 5.0


def test_displacement_clearly_beyond_the_cap_fails_and_is_not_clamped():
    with pytest.raises(ValueError, match="'a' moved 5.020 m, beyond \\[baseline\\] "
                                         "aerial_max_displacement_m 5.0"):
        _check([_rec("a", 10.0, 10.0)], {"a": (15.02, 10.0, 0.0)}, _capped(5.0))


def test_cap_applies_only_to_the_node_platform():
    records = [_rec("air", 10.0, 10.0), _rec("car", 10.0, 20.0, platform="ground", index=1)]
    positions = {"air": (40.0, 10.0, 0.0), "car": (10.0, 21.0, 0.0)}
    final = _check(records, positions, _capped(1.0, "ground"))
    assert final["air"] == (40.0, 10.0, 0.0)
    with pytest.raises(ValueError, match="'air' moved 30.000 m"):
        _check(records, positions, _capped(1.0, "aerial"))


def test_cap_ignores_unselected_nodes():
    records = [_rec("a", 10.0, 10.0), _rec("f", 0.0, 0.0, selected=False, index=1)]
    final = _check(records, {"a": (10.0, 10.0, 0.0), "f": (0.0, 0.0, 0.0)}, _capped(1e-9))
    assert final["a"] == (10.0, 10.0, 0.0)


# --- [rl] bounds and random_walk bounds -------------------------------------------------

def test_rl_bounds_are_inclusive_and_strict_outside():
    rl = {"x_min": 8.0, "x_max": 16.0, "y_min": 8.0, "y_max": 16.0}
    assert _check([_rec("a", 12.0, 12.0)], {"a": (16.0, 8.0, 0.0)}, rl_bounds=rl)[
        "a"] == (16.0, 8.0, 0.0)
    with pytest.raises(ValueError, match="outside the \\[rl\\] x/y bounds"):
        _check([_rec("a", 12.0, 12.0)], {"a": (16.000000001, 8.0, 0.0)}, rl_bounds=rl)


def test_standalone_has_no_rl_bounds_check():
    final = _check([_rec("a", 32.0, 32.0)], {"a": (60.0, 60.0, 0.0)}, rl_bounds=None)
    assert final["a"] == (60.0, 60.0, 0.0)


def test_random_walk_bounds_are_inclusive_and_strict_outside():
    rw = {"x_min": 0.0, "x_max": 32.0, "y_min": 0.0, "y_max": 32.0}
    record = _rec("w", 16.0, 16.0, mobility="random_walk", rw=rw)
    assert _check([record], {"w": (32.0, 32.0, 0.0)})["w"] == (32.0, 32.0, 0.0)
    with pytest.raises(ValueError, match="'w'.*outside its random_walk bounds"):
        _check([record], {"w": (32.000000001, 32.0, 0.0)})


# --- shape and finiteness ---------------------------------------------------------------


def test_numpy_and_integer_values_are_accepted_as_floats():
    final = _check([_rec("a", 1.0, 1.0)], {"a": np.array([20, 30, 0])})
    assert final["a"] == (20.0, 30.0, 0.0)
    assert all(type(v) is float for v in final["a"])
