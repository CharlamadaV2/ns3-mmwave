"""Annealing planner: proposal rules (pure) and keep-best runs on fake_query.py."""

import json

import numpy as np
import pytest

from scripts.baselines.planners import optimization
from scripts.baselines.planners.channel import PlannerError
from scripts.baselines.planners.objective import MovementCost
from scripts.baselines.tests.test_objective import (REFERENCE, assert_plan_invariants, entry,
                                                    make_request, planning_scorer,
                                                    pure_context)

ITERATIONS = 25
MIXED = [entry("F", 200.0, 200.0, 2.0, "vehicle"), entry("A", 210.0, 200.0, 10.0),
         entry("P", 190.0, 200.0, 1.5, "pedestrian")]


def _solve(tmp_path, monkeypatch, entries, movable, **kwargs):
    kwargs.setdefault("max_iterations", ITERATIONS)
    request = make_request(entries, movable, method="optimization", **kwargs)
    with planning_scorer(tmp_path, monkeypatch, entries, request) as (scorer, log):
        positions, diagnostics, notes = optimization.solve(request, scorer, log)
        layouts = scorer.stats["layouts"]
    json.dumps(diagnostics)
    assert_plan_invariants(request, positions)
    assert any(note.startswith("sa iters=") for note in notes)
    return request, positions, diagnostics, layouts


def test_settings_drop_altitude_and_renormalize_weights():
    assert optimization.MOVES == ("nudge", "teleport", "swap")
    weights = optimization.MOVE_WEIGHTS
    assert weights == pytest.approx({"nudge": 0.55 / 0.85, "teleport": 0.20 / 0.85,
                                     "swap": 0.10 / 0.85})
    assert sum(weights.values()) == pytest.approx(1.0)
    settings = optimization.settings(make_request(MIXED, {"A"}, max_iterations=7))
    json.dumps(settings)
    assert "altitude" not in settings["move_weights"]
    assert settings["termination"] == {"max_iterations": 7}
    assert (settings["t0_area_frac"], settings["tf_area_frac"],
            settings["nudge_sigma_frac"]) == (0.02, 0.0002, 0.08)


def test_swap_exchanges_xy_only_and_keeps_each_z():
    ctx, _ = pure_context(MIXED, {"A", "P"})
    rng = np.random.default_rng(3)
    covered = np.zeros(len(ctx.coverage_weights), dtype=bool)
    proposal = optimization.propose(ctx, rng, ctx.starts, covered, "swap", 10.0)
    assert proposal[1].tolist() == [190.0, 200.0, 10.0]
    assert proposal[2].tolist() == [210.0, 200.0, 1.5]
    assert proposal[0].tolist() == ctx.starts[0].tolist()


def test_swap_participants_are_clamped_to_their_caps():
    far = [entry("A", 50.0, 50.0, 10.0), entry("B", 350.0, 350.0, 2.0, "vehicle")]
    penalties = {"aerial": MovementCost(0.0, 0.0, 40.0), "ground": MovementCost(0.0, 0.0, 60.0)}
    ctx, _ = pure_context(far, {"A", "B"}, penalties=penalties)
    covered = np.zeros(len(ctx.coverage_weights), dtype=bool)
    proposal = optimization.propose(ctx, np.random.default_rng(0), ctx.starts, covered,
                                    "swap", 1.0)
    assert ctx.displacement(0, proposal[0]) <= 40.0
    assert ctx.displacement(1, proposal[1]) <= 60.0
    assert proposal[:, 2].tolist() == [10.0, 2.0]


@pytest.mark.parametrize("move", ["nudge", "teleport", "swap"])
def test_proposals_stay_inside_rectangle_rl_and_walk_bounds(move):
    walk = {"x_min": 0.0, "x_max": 120.0, "y_min": 0.0, "y_max": 400.0}
    rl = {"x_min": 0.0, "x_max": 300.0, "y_min": 0.0, "y_max": 300.0}
    nodes = [entry("F", 200.0, 200.0, 2.0, "vehicle"), entry("A", 280.0, 280.0, 10.0),
             entry("W", 100.0, 100.0, 1.5, "pedestrian", mobility="random_walk", walk=walk)]
    ctx, _ = pure_context(nodes, {"A", "W"}, mode="evaluation", rl_bounds=rl,
                          penalties={"aerial": MovementCost(0.0, 0.0, 150.0),
                                     "ground": MovementCost(0.0, 0.0)})
    rng = np.random.default_rng(11)
    covered = np.zeros(len(ctx.coverage_weights), dtype=bool)
    rejected = 0
    for _ in range(300):
        proposal = optimization.propose(ctx, rng, ctx.starts, covered, move, 200.0)
        if proposal is None:
            rejected += 1
            continue
        assert proposal[:, 2].tolist() == ctx.starts[:, 2].tolist()
        assert proposal[0].tolist() == ctx.starts[0].tolist()
        for k in ctx.selected:
            assert ctx.allowed(k, *proposal[k, :2])
        assert ctx.displacement(1, proposal[1]) <= 150.0
    assert rejected > 0


def test_teleport_without_uncovered_probes_is_skipped():
    ctx, _ = pure_context(MIXED, {"A"})
    covered = np.ones(len(ctx.coverage_weights), dtype=bool)
    assert optimization.propose(ctx, np.random.default_rng(0), ctx.starts, covered,
                                "teleport", 5.0) is None


def test_determinism_per_seed_and_keep_best(tmp_path, monkeypatch):
    runs = [_solve(tmp_path / f"run{k}", monkeypatch, MIXED, {"A", "P"}, planner_seed=9)
            for k in range(2)]
    (_, first, report, layouts), (_, second, again, _) = runs
    assert np.array_equal(first, second)
    assert report["optimization"] == again["optimization"]
    sa = report["optimization"]
    assert sa["iterations"] == ITERATIONS
    assert sa["best_total"] >= sa["geometric_total"]
    assert sa["best_total"] >= sa["current_total"]
    assert report["score"]["total"] == sa["best_total"]
    # Two warm-set queries plus one per non-rejected proposal on top of the greedy run.
    assert layouts >= 2 + sa["queried_proposals"]


def test_mixed_z_run_never_changes_altitude(tmp_path, monkeypatch):
    _, positions, _, _ = _solve(tmp_path, monkeypatch, MIXED, {"A", "P"}, planner_seed=4,
                                objective="balanced")
    assert positions[:, 2].tolist() == [2.0, 10.0, 1.5]


def test_evaluation_run_respects_rl_and_walk_bounds(tmp_path, monkeypatch):
    walk = {"x_min": 0.0, "x_max": 230.0, "y_min": 0.0, "y_max": 230.0}
    rl = {"x_min": 100.0, "x_max": 300.0, "y_min": 100.0, "y_max": 300.0}
    nodes = [entry("F", 200.0, 200.0, 2.0, "vehicle"), entry("A", 210.0, 200.0, 10.0),
             entry("W", 190.0, 200.0, 1.5, "pedestrian", mobility="random_walk", walk=walk)]
    _, positions, _, _ = _solve(tmp_path, monkeypatch, nodes, {"A", "W"}, mode="evaluation",
                                rl_bounds=rl, planner_seed=2)
    for x, y in positions[1:, :2]:
        assert 100.0 <= x <= 300.0 and 100.0 <= y <= 300.0
    assert positions[2, 0] <= 230.0 and positions[2, 1] <= 230.0


def test_stay_put_is_reported_as_zero_displacement(tmp_path, monkeypatch):
    request, positions, report, _ = _solve(tmp_path, monkeypatch, MIXED, {"A", "P"},
                                           penalties=REFERENCE, planner_seed=1)
    assert positions.tolist() == [[n.x, n.y, n.z] for n in request.nodes]
    assert all(node["displacement_m"] == 0.0 for node in report["nodes"].values())
    assert report["score"]["move_cost_m2"] == 0.0


def test_max_iterations_is_required(tmp_path, monkeypatch):
    request = make_request(MIXED, {"A"}, method="optimization", max_iterations=None)
    with planning_scorer(tmp_path, monkeypatch, MIXED, request) as (scorer, log):
        with pytest.raises(PlannerError, match="max_iterations"):
            optimization.solve(request, scorer, log)
        assert scorer.stats["requests"] == 0
