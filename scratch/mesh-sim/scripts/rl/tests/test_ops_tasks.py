"""Array-task mapping, exit-code tolerance, and the in-job runner with a stub executor."""

import copy
import json
from pathlib import Path

import pytest

from scripts.rl import experiment
from scripts.rl.cli_common import write_json
from scripts.rl.ops import run_task, tasks
from scripts.sim_support import find_mesh_root

TRACKED_MATRIX = find_mesh_root() / "inputs/experiments/bypass-smoke-matrix.json"
SIM_BINARY = "/nonexistent/mesh-sim-binary"
LEAVES = ("row-a/train-seed-1", "row-b/train-seed-1")


def _plan(root: Path) -> dict:
    """Minimal plan with the shape build_plan produces, without loading a matrix."""
    steps = []
    for leaf in LEAVES:
        steps.append({"id": f"train/{leaf}", "kind": "train", "module": "stub.train",
                      "args": [], "output_dir": str(root / "train" / leaf),
                      "manifest": "train_manifest.json", "needs": []})
        steps.append({"id": f"evaluate/{leaf}", "kind": "evaluate",
                      "module": "stub.evaluate", "args": [],
                      "output_dir": str(root / "eval" / leaf),
                      "manifest": "eval_manifest.json", "needs": [f"train/{leaf}"]})
    steps.append({"id": "compare", "kind": "compare", "module": "stub.compare",
                  "args": ["--plan", str(root / experiment.PLAN_NAME),
                           "--output-dir", str(root / "comparison")],
                  "output_dir": str(root / "comparison"),
                  "manifest": "comparison.json", "needs": []})
    return {"experiment_plan_version": 1, "steps": steps}


def _written_plan(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    write_json(root / experiment.PLAN_NAME, _plan(root))
    return root


def _write_manifest(plan: dict, step_id: str, payload) -> Path:
    step = next(step for step in plan["steps"] if step["id"] == step_id)
    out_dir = Path(step["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        (out_dir / step["manifest"]).write_text(json.dumps(payload))
    return out_dir


def _snapshot(path: Path) -> dict:
    return {str(item.relative_to(path)): item.read_bytes() if item.is_file() else None
            for item in sorted(path.rglob("*"))}


class _StubExecutor:
    """Records executed steps and fakes their manifests; no process is started."""

    def __init__(self, codes: dict[str, int] | None = None):
        self.codes = codes or {}
        self.executed: list[str] = []

    def __call__(self, step: dict) -> int:
        self.executed.append(step["id"])
        code = self.codes.get(step["id"], 0)
        if code == 1:
            return code
        out_dir = Path(step["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        if step["kind"] == "compare":
            status = "complete" if code == 0 else "incomplete"
        else:
            status = "completed" if code == 0 else "partial"
        (out_dir / step["manifest"]).write_text(json.dumps({"status": status}))
        return code


def test_tracked_matrix_maps_to_eight_paired_tasks(tmp_path):
    matrix = experiment.load_matrix(TRACKED_MATRIX)
    root = tmp_path / "root"
    plan = experiment.build_plan(matrix, root, SIM_BINARY)
    table = tasks.build_tasks(plan)

    assert [task["index"] for task in table] == list(range(8))
    assert [task["id"] for task in table] == [
        f"{row}/train-seed-{seed}"
        for row in ("local-delivery", "raw-delivery", "local-connectivity",
                    "local-legacy")
        for seed in (101, 102)]
    for task in table:
        assert task["train"]["id"] == f"train/{task['id']}"
        assert task["evaluate"]["needs"] == [task["train"]["id"]]
    assert tasks.plan_root(plan) == root.resolve()
    assert tasks.compare_step(plan)["id"] == "compare"


def _drop(plan: dict, step_id: str) -> dict:
    plan = copy.deepcopy(plan)
    plan["steps"] = [step for step in plan["steps"] if step["id"] != step_id]
    return plan


def _stray_evaluate(plan: dict) -> dict:
    plan = copy.deepcopy(plan)
    stray = copy.deepcopy(next(step for step in plan["steps"]
                               if step["id"] == "evaluate/row-a/train-seed-1"))
    stray["id"], stray["needs"] = "evaluate/row-c/train-seed-1", []
    plan["steps"].append(stray)
    return plan


def _bad_compare_args(plan: dict) -> dict:
    plan = copy.deepcopy(plan)
    next(step for step in plan["steps"] if step["kind"] == "compare")["args"] = []
    return plan


MALFORMED = [
    ("no compare step", lambda plan: _drop(plan, "compare"), "compare step"),
    ("missing evaluate", lambda plan: _drop(plan, "evaluate/row-a/train-seed-1"),
     "no evaluate step needs"),
    ("unpaired evaluate", _stray_evaluate, "without a train step"),
    ("no train steps", lambda plan: {"steps": [step for step in plan["steps"]
                                               if step["kind"] == "compare"]},
     "no train steps"),
]


@pytest.mark.parametrize("mutate,message", [case[1:] for case in MALFORMED],
                         ids=[case[0] for case in MALFORMED])
def test_malformed_plans_are_refused(tmp_path, mutate, message):
    with pytest.raises(ValueError, match=message):
        tasks.build_tasks(mutate(_plan(tmp_path)))


def test_plan_root_requires_a_plan_argument(tmp_path):
    with pytest.raises(ValueError, match="--plan"):
        tasks.plan_root(_bad_compare_args(_plan(tmp_path)))


TOLERANCE = [("train", 0, True), ("train", 1, False), ("train", 2, False),
             ("evaluate", 0, True), ("evaluate", 1, False), ("evaluate", 2, True),
             ("compare", 0, True), ("compare", 1, False), ("compare", 2, True)]


@pytest.mark.parametrize("kind,code,expected", TOLERANCE)
def test_tolerated_table(kind, code, expected):
    assert tasks.tolerated(kind, code) is expected


def test_tolerated_refuses_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown step kind"):
        tasks.tolerated("benchmark", 0)


def test_task_state_reports_both_steps_and_the_train_manifest_status(tmp_path):
    plan = _plan(tmp_path)
    _write_manifest(plan, "train/row-a/train-seed-1", {"status": "running"})
    task = tasks.build_tasks(plan)[0]

    state = tasks.task_fs_state(task)
    assert state["train"][0] == "blocked"
    assert state["train_manifest_status"] == "running"
    assert state["evaluate"] == ("pending", "")


def test_failed_training_skips_the_evaluation(tmp_path):
    root = _written_plan(tmp_path)
    executor = _StubExecutor(codes={"train/row-a/train-seed-1": 1})
    record = tmp_path / "records" / "task-0000.json"

    assert run_task.run_task(root, 0, record, executor) == 1
    assert executor.executed == ["train/row-a/train-seed-1"]
    assert not (root / "eval").exists()
    payload = json.loads(record.read_text())
    assert payload["record_version"] == 1 and payload["mode"] == "task"
    assert payload["task_index"] == 0 and payload["task_id"] == "row-a/train-seed-1"
    assert payload["exit_code"] == 1
    assert [step["id"] for step in payload["steps"]] == ["train/row-a/train-seed-1"]
    assert payload["steps"][0]["tolerated"] is False


def test_tolerated_evaluation_exit_is_recorded_but_not_a_failure(tmp_path):
    root = _written_plan(tmp_path)
    executor = _StubExecutor(codes={"evaluate/row-a/train-seed-1": 2})
    record = tmp_path / "task-0000.json"

    assert run_task.run_task(root, 0, record, executor) == 0
    evaluate = json.loads(record.read_text())["steps"][1]
    assert evaluate["action"] == "executed"
    assert evaluate["exit_code"] == 2 and evaluate["tolerated"] is True


def test_a_finished_training_is_skipped(tmp_path):
    root = _written_plan(tmp_path)
    plan = experiment.load_plan(root)
    _write_manifest(plan, "train/row-a/train-seed-1", {"status": "completed"})
    executor = _StubExecutor()
    record = tmp_path / "task-0000.json"

    assert run_task.run_task(root, 0, record, executor) == 0
    assert executor.executed == ["evaluate/row-a/train-seed-1"]
    steps = json.loads(record.read_text())["steps"]
    assert [step["action"] for step in steps] == ["skipped", "executed"]
    assert steps[0]["exit_code"] is None and steps[0]["tolerated"] is None


BLOCKED = [
    ("blocked train", "train/row-a/train-seed-1", None),
    ("partial evaluate", "evaluate/row-a/train-seed-1", {"status": "partial"}),
]


@pytest.mark.parametrize("step_id,payload", [case[1:] for case in BLOCKED],
                         ids=[case[0] for case in BLOCKED])
def test_a_dirty_directory_is_refused_and_left_untouched(tmp_path, step_id, payload):
    root = _written_plan(tmp_path)
    plan = experiment.load_plan(root)
    if step_id.startswith("evaluate/"):
        _write_manifest(plan, "train/row-a/train-seed-1", {"status": "completed"})
    out_dir = _write_manifest(plan, step_id, payload)
    before = _snapshot(out_dir)
    executor = _StubExecutor()

    assert run_task.run_task(root, 0, tmp_path / "record.json", executor) == 1
    assert step_id not in executor.executed
    assert _snapshot(out_dir) == before


def test_an_out_of_range_task_index_is_refused(tmp_path):
    root = _written_plan(tmp_path)
    executor = _StubExecutor()

    assert run_task.run_task(root, 2, None, executor) == 1
    assert executor.executed == []


def _finish_evaluations(root: Path, partial: str | None = None) -> dict:
    plan = experiment.load_plan(root)
    for leaf in LEAVES:
        step_id = f"evaluate/{leaf}"
        status = "partial" if step_id == partial else "completed"
        _write_manifest(plan, step_id, {"status": status})
    return plan


@pytest.mark.parametrize("partial", [None, "evaluate/row-a/train-seed-1"],
                         ids=["missing evaluation", "partial evaluation"])
def test_compare_refuses_unfinished_evaluations_by_default(tmp_path, partial, capsys):
    root = _written_plan(tmp_path)
    if partial is not None:
        _finish_evaluations(root, partial)
    record = tmp_path / "record.json"

    assert run_task.main(["--output-root", str(root), "--compare",
                          "--record", str(record)]) == 1
    assert not (root / "comparison").exists()
    assert not record.exists()
    assert "evaluate/row-a/train-seed-1" in capsys.readouterr().err


def test_compare_runs_unfinished_evaluations_only_when_allowed(tmp_path):
    root = _written_plan(tmp_path)
    _finish_evaluations(root, "evaluate/row-a/train-seed-1")
    executor = _StubExecutor()
    record = tmp_path / "compare.json"

    assert run_task.run_compare(root, True, record, executor) == 0
    assert executor.executed == ["compare"]
    payload = json.loads(record.read_text())
    assert payload["mode"] == "compare" and payload["task_index"] is None
    assert payload["steps"][0]["exit_code"] == 0


def test_a_tolerated_compare_exit_is_not_a_failure(tmp_path):
    root = _written_plan(tmp_path)
    _finish_evaluations(root)
    executor = _StubExecutor(codes={"compare": 2})
    record = tmp_path / "compare.json"

    assert run_task.run_compare(root, False, record, executor) == 0
    assert json.loads(record.read_text())["steps"][0] == {
        "id": "compare", "kind": "compare", "action": "executed",
        "exit_code": 2, "tolerated": True}


def test_compare_mode_requires_exactly_one_selector(tmp_path):
    root = _written_plan(tmp_path)

    assert run_task.main(["--output-root", str(root)]) == 1
    assert run_task.main(["--output-root", str(root), "--compare",
                          "--task-index", "0"]) == 1
    assert run_task.main(["--output-root", str(root), "--task-index", "0",
                          "--allow-incomplete"]) == 1


OUTCOMES = [
    ("absent", None, "absent"),
    ("complete", {"status": "complete"}, "complete"),
    ("incomplete", {"status": "incomplete"}, "incomplete"),
    ("unreadable", "{not json", "unreadable"),
]


@pytest.mark.parametrize("payload,expected", [case[1:] for case in OUTCOMES],
                         ids=[case[0] for case in OUTCOMES])
def test_comparison_outcome_reads_the_comparison_status(tmp_path, payload, expected):
    plan = _plan(tmp_path)
    step = tasks.compare_step(plan)
    if payload is not None:
        out_dir = Path(step["output_dir"])
        out_dir.mkdir(parents=True)
        (out_dir / step["manifest"]).write_text(
            payload if isinstance(payload, str) else json.dumps(payload))

    outcome = tasks.comparison_outcome(plan)
    assert outcome["state"] == expected
    assert outcome["path"] == str(Path(step["output_dir"]) / step["manifest"])
