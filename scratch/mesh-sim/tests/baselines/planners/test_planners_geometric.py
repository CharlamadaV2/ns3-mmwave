"""Greedy planner edges: acceptance threshold, anchor filter precedence, need fallback,
empty candidate sets, one batched query per node, roster order and determinism.
"""

import json
from contextlib import contextmanager

import numpy as np

from scripts.baselines.planners import geometric
from scripts.baselines.planners.channel import ChannelScorer
from scripts.baselines.planners.objective import LayoutScore, MovementCost, prepare_scorer
from scripts.baselines.tests.conftest import fake_query_binary
from scripts.baselines.tests.test_objective import (assert_plan_invariants, entry, make_request)

from ._planner_support import DistanceScorer, TableScorer, full_table


class _Log:
    def __init__(self):
        self.text = ""

    def write(self, text):
        self.text += text

    def flush(self):
        pass


def _run(entries, movable, scorer, **kwargs):
    request = make_request(entries, movable, **kwargs)
    prepare_scorer(request, scorer)
    log = _Log()
    positions, diagnostics, notes = geometric.solve(request, scorer, log)
    json.dumps(diagnostics)
    assert_plan_invariants(request, positions)
    return request, positions, diagnostics, notes, log


def _fake_score(totals: dict, default: float):
    """score() replacement: total by the moved node's (x, y); the start scores 0."""

    def fake(ctx, result, layout):
        layout = np.asarray(layout)
        if np.array_equal(layout, ctx.starts):
            total = 0.0
        else:
            total = totals.get(tuple(layout[1, :2].tolist()), default)
        return LayoutScore(0.0, 0.0, 0.0, 0, 0, float(total))
    return fake


PAIR = [entry("F", 200.0, 200.0, 2.0, "vehicle"), entry("M", 210.0, 200.0, 10.0)]


# ---------------------------------------------------------------------------------------
# Acceptance threshold (marginal must exceed SCORE_TIE_TOL_M2 = 1e-6 m^2).


def test_anchor_filter_takes_precedence_over_score(monkeypatch):
    totals = {(50.0, 50.0): 1e6, (350.0, 350.0): 10.0}
    monkeypatch.setattr(geometric, "score", _fake_score(totals, 1.0))

    def table(layout):
        cut = tuple(layout[1, :2].tolist()) == (50.0, 50.0)
        return full_table(2, missing=[(0, 1)] if cut else ())
    _, positions, diagnostics, _, _ = _run(PAIR, {"M"}, TableScorer(table))
    step = diagnostics["geometric"]["steps"][0]
    assert step["pool"] == ["F"] and step["eff_need"] == 1
    assert step["candidates"] == 16 and step["feasible"] == 15
    assert positions[1].tolist() == [350.0, 350.0, 10.0]
    assert step["marginal_m2"] == 10.0 and step["anchor_links"] == 1


# ---------------------------------------------------------------------------------------
# No feasible / no candidate cases: stay-put is always allowed.

def test_no_candidate_feasible_even_after_need_drops_to_one():
    entries = [entry("F1", 0.0, 0.0, 10.0), entry("F2", 5.0, 0.0, 10.0),
               entry("M", 2.0, 5.0, 10.0)]
    scorer = DistanceScorer(range_m=10.0)
    request, positions, diagnostics, notes, log = _run(entries, {"M"}, scorer,
                                                       objective="resilience")
    step = diagnostics["geometric"]["steps"][0]
    assert step["need"] == 2 and step["eff_need"] == 2 and step["degraded"]
    assert step["candidates"] == 16 and step["feasible"] == 0
    assert step["accepted"] is False and step["marginal_m2"] is None
    assert positions.tolist() == [[n.x, n.y, n.z] for n in request.nodes]
    assert "M: stay put" in notes and "no anchor-feasible candidate" in log.text
    assert [len(call) for call in scorer.calls] == [1, 16]


def test_no_candidate_inside_the_cap_queries_only_the_start():
    penalties = {"aerial": MovementCost(0.0, 0.0, 1.0), "ground": MovementCost(0.0, 0.0)}
    entries = [entry("F", 200.0, 200.0, 2.0, "vehicle"), entry("M", 210.0, 200.0, 10.0)]
    scorer = DistanceScorer()
    request, positions, diagnostics, notes, _ = _run(entries, {"M"}, scorer,
                                                     penalties=penalties)
    step = diagnostics["geometric"]["steps"][0]
    assert step["candidates"] == 0 and step["feasible"] == 0 and not step["accepted"]
    assert [len(call) for call in scorer.calls] == [1]
    assert positions.tolist() == [[n.x, n.y, n.z] for n in request.nodes]
    assert "M: stay put" in notes


# ---------------------------------------------------------------------------------------
# Query batching, order and bookkeeping.

THREE = [entry("F", 205.0, 205.0, 2.0, "vehicle"), entry("A", 120.0, 120.0, 10.0),
         entry("B", 280.0, 130.0, 10.0), entry("C", 210.0, 300.0, 1.5, "pedestrian")]


@contextmanager
def _fake_query_scorer(tmp_path, monkeypatch, entries, request, max_layouts):
    for name in ("FAKE_QUERY_FAULT", "FAKE_QUERY_FAULT_AT", "FAKE_QUERY_INIT_PATCH",
                 "FAKE_QUERY_PAD_BYTES", "FAKE_QUERY_PID_FILE", "FAKE_QUERY_RANGE_M"):
        monkeypatch.delenv(name, raising=False)
    record = tmp_path / "record.jsonl"
    monkeypatch.setenv("FAKE_QUERY_RECORD", str(record))
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    (scenario / "run.ini").write_text("[scenario]\nname = t\nseed = 3\nrun_id = 1\n"
                                      "nodes_file = nodes.json\n")
    (scenario / "nodes.json").write_text(json.dumps(entries))
    binary = fake_query_binary(tmp_path / "bin")
    starts = [[n.x, n.y, n.z] for n in request.nodes]
    with open(tmp_path / "planner.log", "w", encoding="utf-8") as log:
        with ChannelScorer(binary, scenario / "run.ini", 3, 1, None, "standalone",
                           [n.id for n in request.nodes], starts, 3, log,
                           max_layouts=max_layouts) as scorer:
            prepare_scorer(request, scorer)
            yield scorer, log, record


def test_node_candidates_split_into_worker_requests_by_max_layouts(tmp_path, monkeypatch):
    request = make_request(THREE, {"A", "B", "C"})
    with _fake_query_scorer(tmp_path, monkeypatch, THREE, request, 7) as (scorer, log, rec):
        positions, diagnostics, _ = geometric.solve(request, scorer, log)
        stats = scorer.stats
    steps = diagnostics["geometric"]["steps"]
    sizes = [json.loads(line)["layouts"] for line in rec.read_text().splitlines()
             if json.loads(line).get("type") == "evaluate"]
    expected = [1]
    for s in steps:
        expected += [7] * (s["candidates"] // 7) + ([s["candidates"] % 7]
                                                    if s["candidates"] % 7 else [])
    assert sizes == expected
    assert stats["layouts"] == 1 + sum(s["candidates"] for s in steps)
    assert_plan_invariants(request, positions)
    # Same plan as the in-process distance channel with fake_query's default 100 m range.
    _, same, _, _, _ = _run(THREE, {"A", "B", "C"}, DistanceScorer())
    assert np.allclose(positions, same)
