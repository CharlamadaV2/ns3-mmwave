"""Annealing edges: schedule formulas, keep-best, seed choice, determinism, iteration bounds,
clamping and the move rules, with in-process and fake_query.py scorers.
"""

import json

import numpy as np
import pytest

from scripts.baselines.planners import geometric, optimization
from scripts.baselines.planners.objective import (MovementCost, ScoringContext, prepare_scorer, score)
from scripts.baselines.tests.test_objective import (AOI, assert_plan_invariants, entry, make_request, pure_context)

from ._planner_support import DistanceScorer

MIXED = [entry("F", 200.0, 200.0, 2.0, "vehicle"), entry("A", 210.0, 200.0, 10.0),
         entry("P", 190.0, 200.0, 1.5, "pedestrian")]
SPREAD = [entry("F", 200.0, 200.0, 2.0, "vehicle"), entry("A", 120.0, 150.0, 10.0),
          entry("B", 280.0, 260.0, 10.0), entry("P", 230.0, 120.0, 1.5, "pedestrian")]


class _Log:
    def write(self, text):
        pass

    def flush(self):
        pass


def _marking_geometric(monkeypatch, marker: dict):
    """Wrap geometric.solve to remember how many evaluate() calls the warm start used."""
    original = geometric.solve

    def wrapped(request, scorer, log):
        out = original(request, scorer, log)
        marker["calls"] = len(scorer.calls)
        marker["layouts"] = len(scorer.layouts)
        return out
    monkeypatch.setattr(geometric, "solve", wrapped)


def _run(entries, movable, scorer=None, **kwargs):
    kwargs.setdefault("max_iterations", 30)
    request = make_request(entries, movable, method="optimization", **kwargs)
    scorer = scorer or DistanceScorer()
    prepare_scorer(request, scorer)
    positions, diagnostics, notes = optimization.solve(request, scorer, _Log())
    json.dumps(diagnostics)
    assert_plan_invariants(request, positions)
    return request, scorer, positions, diagnostics, notes


# ---------------------------------------------------------------------------------------
# Seed choice and keep-best.


def test_better_start_layout_seeds_the_anneal(monkeypatch):
    def bad_greedy(request, scorer, log):
        layout = np.array([[n.x, n.y, n.z] for n in request.nodes], dtype=float)
        layout[1, :2] = [10.0, 390.0]   # A far from everyone: BIG penalty
        return layout, {"geometric": {"steps": [], "start_total": 0.0}}, ["bad"]
    monkeypatch.setattr(geometric, "solve", bad_greedy)
    _, _, positions, diagnostics, notes = _run(MIXED, {"A"}, max_iterations=5,
                                               planner_seed=2)
    sa = diagnostics["optimization"]
    assert sa["seed_layout"] == "current"
    assert sa["current_total"] > sa["geometric_total"] + AOI
    assert sa["best_total"] >= sa["current_total"]
    assert notes[0] == "bad" and "(seed: current)" in notes[-1]
    assert positions[1].tolist() != [10.0, 390.0, 10.0]


@pytest.mark.parametrize("objective", ["coverage", "resilience"])
def test_returns_the_best_layout_ever_queried(monkeypatch, objective):
    marker = {}
    _marking_geometric(monkeypatch, marker)
    caps = {"aerial": MovementCost(0.0, 0.0, 120.0), "ground": MovementCost(0.0, 0.0, 120.0)}
    request, scorer, positions, diagnostics, _ = _run(
        SPREAD, {"A", "B", "P"}, objective=objective, penalties=caps, max_iterations=40,
        planner_seed=5)
    ctx = ScoringContext.build(request, scorer)
    totals = [score(ctx, result, layout).total
              for call in scorer.calls[marker["calls"]:] for layout, result in call]
    sa = diagnostics["optimization"]
    # A proposal above the best is above the current layout too, so it is always accepted.
    assert sa["best_total"] == pytest.approx(max(totals))
    assert sa["best_total"] >= max(sa["geometric_total"], sa["current_total"])
    again = score(ctx, scorer.result(positions), positions)
    assert again.total == pytest.approx(sa["best_total"])
    assert diagnostics["score"]["total"] == sa["best_total"]
    for k in ctx.selected:
        assert ctx.displacement(k, positions[k]) <= 120.0


# ---------------------------------------------------------------------------------------
# Determinism.

def test_same_seed_replays_every_query_and_other_seed_differs(monkeypatch):
    runs = {}
    for label, seed in (("a", 11), ("b", 11), ("c", 12)):
        marker = {}
        _marking_geometric(monkeypatch, marker)
        _, scorer, positions, diagnostics, _ = _run(SPREAD, {"A", "B", "P"},
                                                    planner_seed=seed, max_iterations=25)
        runs[label] = (scorer.layouts, marker["layouts"], positions, diagnostics)
    (la, ma, pa, da), (lb, _, pb, db), (lc, mc, _, _) = runs["a"], runs["b"], runs["c"]
    assert len(la) == len(lb) and all(np.array_equal(x, y) for x, y in zip(la, lb))
    assert np.array_equal(pa, pb) and da == db
    assert ma == mc  # the greedy warm start does not depend on the planner seed
    sa_a, sa_c = la[ma:], lc[mc:]
    assert len(sa_a) != len(sa_c) or not all(np.array_equal(x, y)
                                             for x, y in zip(sa_a, sa_c))


# ---------------------------------------------------------------------------------------
# Clamp and move rules.

def _capped(cap=50.0, nodes=None):
    nodes = nodes or [entry("A", 100.0, 100.0, 10.0), entry("F", 300.0, 300.0, 2.0, "vehicle")]
    return pure_context(nodes, {"A"}, penalties={"aerial": MovementCost(0.0, 0.0, cap),
                                                 "ground": MovementCost(0.0, 0.0)})[0]


def test_teleport_targets_only_uncovered_probes():
    ctx, _ = pure_context(MIXED, {"A"})
    covered = np.ones(len(ctx.coverage_weights), dtype=bool)
    uncovered = [3, 9, 14]
    covered[uncovered] = False
    rng = np.random.default_rng(7)
    hits = set()
    for _ in range(60):
        proposal = optimization.propose(ctx, rng, ctx.starts, covered, "teleport", 1e-6)
        assert proposal is not None
        distance = np.hypot(*(ctx.coverage_points - proposal[1, :2]).T)
        nearest = int(np.argmin(distance))
        assert distance[nearest] < 1e-5 and nearest in uncovered
        hits.add(nearest)
        assert proposal[1, 2] == 10.0 and np.array_equal(proposal[[0, 2]], ctx.starts[[0, 2]])
    assert hits == set(uncovered)


def test_nudge_from_a_moved_layout_is_capped_from_the_original_start():
    ctx = _capped(50.0)
    layout = ctx.starts.copy()
    layout[0, :2] = [140.0, 100.0]       # already 40 m from the start
    covered = np.zeros(len(ctx.coverage_weights), dtype=bool)
    rng = np.random.default_rng(1)
    for _ in range(200):
        proposal = optimization.propose(ctx, rng, layout, covered, "nudge", 300.0)
        if proposal is not None:
            assert ctx.displacement(0, proposal[0]) <= 50.0
