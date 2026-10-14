"""In-job runner: the exit-code mapping, record schema, refusals, and a real subprocess.

Expectations come from scripts/rl/ops/README.md "Runner: run_task.py",
"Exit-code mapping", and "Record schema (--record)".
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from scripts.rl import experiment
from scripts.rl.cli_common import write_json
from scripts.rl.ops import run_task

from ._ops_gen_helpers import leaf, make_plan, tree, write_manifest, written_plan

RECORD_KEYS = {"record_version", "mode", "task_index", "task_id", "host",
               "started_at", "ended_at", "steps", "exit_code"}
STEP_KEYS = {"id", "kind", "action", "exit_code", "tolerated"}


class _Executor:
    """Returns scripted exit codes; writes a completed manifest only for exit 0/2."""

    def __init__(self, codes=None):
        self.codes = codes or {}
        self.executed = []

    def __call__(self, step):
        self.executed.append(step["id"])
        code = self.codes.get(step["kind"], 0)
        if code in (0, 2):
            out = Path(step["output_dir"])
            out.mkdir(parents=True, exist_ok=True)
            status = "complete" if step["kind"] == "compare" else "completed"
            (out / step["manifest"]).write_text(json.dumps({"status": status}))
        return code


@pytest.fixture
def root(tmp_path):
    return written_plan(tmp_path, n_tasks=2)


def _record(path: Path) -> dict:
    record = json.loads(path.read_text())
    assert set(record) == RECORD_KEYS
    assert record["record_version"] == 1 and record["host"]
    assert datetime.fromisoformat(record["started_at"]) \
        <= datetime.fromisoformat(record["ended_at"])
    for step in record["steps"]:
        assert set(step) == STEP_KEYS
    return record


MAPPING = [
    ("both succeed", 0, 0, 0, ["train", "evaluate"]),
    ("evaluation warns with 2", 0, 2, 0, ["train", "evaluate"]),
    ("evaluation fails", 0, 1, 1, ["train", "evaluate"]),
    ("evaluation exits 3", 0, 3, 1, ["train", "evaluate"]),
    ("evaluation killed by a signal", 0, -15, 1, ["train", "evaluate"]),
    ("training exits 2", 2, 0, 1, ["train"]),
    ("training fails", 1, 0, 1, ["train"]),
    ("training killed by a signal", -9, 0, 1, ["train"]),
]


@pytest.mark.parametrize("train,evaluate,expected,executed",
                         [case[1:] for case in MAPPING],
                         ids=[case[0] for case in MAPPING])
def test_gen_task_exit_code_mapping(root, tmp_path, train, evaluate, expected,
                                    executed):
    executor = _Executor({"train": train, "evaluate": evaluate})
    record_path = tmp_path / "records" / "nested" / "task-0001.json"
    assert run_task.run_task(root, 1, record_path, executor) == expected
    assert [step_id.split("/", 1)[0] for step_id in executor.executed] == executed

    record = _record(record_path)
    assert record["mode"] == "task" and record["exit_code"] == expected
    assert record["task_index"] == 1 and record["task_id"] == leaf(1)
    codes = {"train": train, "evaluate": evaluate}
    tolerated = {"train": (0,), "evaluate": (0, 2)}
    assert record["steps"] == [
        {"id": f"{kind}/{leaf(1)}", "kind": kind, "action": "executed",
         "exit_code": codes[kind], "tolerated": codes[kind] in tolerated[kind]}
        for kind in executed]


def test_gen_a_finished_task_executes_nothing_and_records_two_skips(root, tmp_path,
                                                                    capsys):
    write_manifest(root, f"train/{leaf(0)}", {"status": "completed"})
    write_manifest(root, f"evaluate/{leaf(0)}", {"status": "completed"})
    executor = _Executor()
    record_path = tmp_path / "r.json"
    assert run_task.run_task(root, 0, record_path, executor) == 0
    assert executor.executed == []
    steps = _record(record_path)["steps"]
    assert [(step["action"], step["exit_code"], step["tolerated"]) for step in steps] \
        == [("skipped", None, None)] * 2
    assert capsys.readouterr().out.count("already done") == 2


def test_gen_a_partial_evaluation_is_refused_without_touching_it(root, tmp_path,
                                                                 capsys):
    write_manifest(root, f"train/{leaf(0)}", {"status": "completed"})
    out = write_manifest(root, f"evaluate/{leaf(0)}", {"status": "partial"})
    before = tree(out)
    executor = _Executor()
    assert run_task.run_task(root, 0, tmp_path / "r.json", executor) == 1
    assert executor.executed == []
    assert "some episodes did not complete" in capsys.readouterr().err
    assert tree(out) == before
    assert _record(tmp_path / "r.json")["exit_code"] == 1


def test_gen_a_blocked_training_stops_before_the_evaluation_exists(root, tmp_path,
                                                                   capsys):
    out = write_manifest(root, f"train/{leaf(0)}", {"status": "running"})
    executor = _Executor()
    assert run_task.run_task(root, 0, tmp_path / "r.json", executor) == 1
    assert executor.executed == []
    assert f"move or delete {out} to retry" in capsys.readouterr().err
    assert not (root / "eval").exists()
    steps = _record(tmp_path / "r.json")["steps"]
    assert [step["kind"] for step in steps] == ["train"]


def test_gen_a_blocked_evaluation_is_refused_after_a_pending_training(root, tmp_path):
    # README order: pending train is executed (1.), then a blocked evaluate exits 1 (3.).
    write_manifest(root, f"evaluate/{leaf(0)}", None)
    executor = _Executor()
    assert run_task.run_task(root, 0, tmp_path / "r.json", executor) == 1
    assert executor.executed == [f"train/{leaf(0)}"]
    steps = _record(tmp_path / "r.json")["steps"]
    assert [(step["kind"], step["action"]) for step in steps] \
        == [("train", "executed"), ("evaluate", "skipped")]


# --- compare mode ------------------------------------------------------------


def test_gen_an_incomplete_compare_writes_nothing_at_all(root, tmp_path, capsys):
    write_manifest(root, f"train/{leaf(0)}", {"status": "completed"})
    write_manifest(root, f"evaluate/{leaf(0)}", {"status": "partial"})
    record_path = tmp_path / "records" / "compare.json"
    before = tree(root)
    executor = _Executor()
    assert run_task.run_compare(root, False, record_path, executor) == 1
    err = capsys.readouterr().err
    assert f"partial evaluation: evaluate/{leaf(0)}" in err
    assert f"missing evaluation: evaluate/{leaf(1)}" in err
    assert "--allow-incomplete" in err
    assert executor.executed == [] and tree(root) == before
    assert not record_path.exists() and not record_path.parent.exists()


@pytest.mark.parametrize("code,expected", [(0, 0), (2, 0), (1, 1), (3, 1)])
def test_gen_compare_exit_code_mapping_and_record(root, tmp_path, code, expected,
                                                  capsys):
    for index in range(2):
        write_manifest(root, f"train/{leaf(index)}", {"status": "completed"})
        write_manifest(root, f"evaluate/{leaf(index)}", {"status": "completed"})
    record_path = tmp_path / "compare.json"
    assert run_task.run_compare(root, False, record_path,
                                _Executor({"compare": code})) == expected
    record = _record(record_path)
    assert record["mode"] == "compare"
    assert record["task_index"] is None and record["task_id"] is None
    assert record["steps"] == [{"id": "compare", "kind": "compare",
                                "action": "executed", "exit_code": code,
                                "tolerated": code in (0, 2)}]
    state = "complete" if code in (0, 2) else "absent"
    assert f"compare  exit {code}  {state}" in capsys.readouterr().out


def test_gen_allow_incomplete_runs_and_reports_the_outcome_state(root, tmp_path,
                                                                 capsys):
    executor = _Executor()
    assert run_task.run_compare(root, True, tmp_path / "c.json", executor) == 0
    assert executor.executed == ["compare"]
    assert "compare  exit 0  complete" in capsys.readouterr().out


# --- the default executor runs a real process from the mesh root ------------


def test_gen_the_default_executor_runs_modules_from_the_mesh_root(tmp_path,
                                                                  monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    plan = make_plan(root, n_tasks=1)
    train, evaluate = plan["steps"][0], plan["steps"][1]
    # Importable only when the child's cwd is the mesh root.
    train.update(module="scripts.sim_support", args=[])
    # argparse rejects the flag with exit 2: tolerated for an evaluation.
    evaluate.update(module="json.tool", args=["--no-such-flag"])
    write_json(root / experiment.PLAN_NAME, plan)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    record_path = tmp_path / "r.json"
    assert run_task.main(["--output-root", str(root), "--task-index", "0",
                          "--record", str(record_path)]) == 0
    steps = _record(record_path)["steps"]
    assert [(step["exit_code"], step["tolerated"]) for step in steps] \
        == [(0, True), (2, True)]
