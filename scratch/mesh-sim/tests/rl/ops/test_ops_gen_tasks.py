"""Task table construction, tolerance table, prerequisites, and comparison outcomes.

Expectations come from scripts/rl/ops/README.md "Task unit", "Exit-code
mapping", and "cluster compare versus fetch".
"""

import copy
import json
from pathlib import Path

import pytest

from scripts.rl.ops import tasks

from ._ops_gen_helpers import leaf, make_plan


@pytest.fixture
def plan(tmp_path):
    return make_plan(tmp_path, n_tasks=3)


def _step(plan, step_id):
    return next(step for step in plan["steps"] if step["id"] == step_id)


def _write(step, text):
    out = Path(step["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / step["manifest"]).write_text(text)


def test_gen_indices_follow_train_order_whatever_the_evaluate_order(plan):
    shuffled = copy.deepcopy(plan)
    evaluations = [step for step in shuffled["steps"] if step["kind"] == "evaluate"]
    trains = [step for step in shuffled["steps"] if step["kind"] == "train"]
    compare = tasks.compare_step(shuffled)
    shuffled["steps"] = [compare, *reversed(evaluations), *trains]
    table = tasks.build_tasks(shuffled)
    assert [task["index"] for task in table] == [0, 1, 2]
    assert [task["id"] for task in table] == [leaf(0), leaf(1), leaf(2)]
    for task in table:
        assert task["evaluate"]["needs"] == [task["train"]["id"]]


def test_gen_plan_root_is_the_directory_of_the_plan_argument(plan, tmp_path):
    assert tasks.plan_root(plan) == tmp_path


def test_gen_compare_prerequisites_split_missing_and_partial(plan):
    _write(_step(plan, f"evaluate/{leaf(0)}"), json.dumps({"status": "completed"}))
    _write(_step(plan, f"evaluate/{leaf(1)}"), json.dumps({"status": "partial"}))
    _write(_step(plan, f"evaluate/{leaf(2)}"), "{broken")
    assert tasks.compare_prerequisites(plan) == {
        "ready": False, "missing": [f"evaluate/{leaf(2)}"],
        "partial": [f"evaluate/{leaf(1)}"]}


def test_gen_compare_prerequisites_are_ready_only_when_every_evaluation_is_done(plan):
    for index in range(3):
        _write(_step(plan, f"evaluate/{leaf(index)}"),
               json.dumps({"status": "completed"}))
    assert tasks.compare_prerequisites(plan) == {"ready": True, "missing": [],
                                                 "partial": []}


@pytest.mark.parametrize("text,expected", [
    (None, "absent"),
    ('{"status": "complete"}', "complete"),
    ('{"status": "incomplete"}', "incomplete"),
    ('{"status": "completed"}', "unreadable"),
    ('{"status": 1}', "unreadable"),
    ("{trunc", "unreadable"),
    ("", "unreadable"),
])
def test_gen_comparison_outcome_values(plan, text, expected):
    step = tasks.compare_step(plan)
    if text is not None:
        _write(step, text)
    outcome = tasks.comparison_outcome(plan)
    assert outcome == {"state": expected,
                       "path": str(Path(step["output_dir"]) / step["manifest"])}
