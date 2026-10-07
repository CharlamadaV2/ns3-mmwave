"""Cluster config, sbatch argv, scheduler parsers, reconciliation, and the CLI."""

import json
import os
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.rl import experiment
from scripts.rl.cli_common import write_json
from scripts.rl.ops import cluster, reconcile, receipts, slurm, tasks
from scripts.rl.tests import fake_slurm

ROWS = ("local-delivery", "raw-delivery", "local-connectivity", "local-legacy")
SEEDS = (101, 102)
EXAMPLE_CONFIG = Path(cluster.__file__).resolve().parent / "cluster-config.example.json"


def _task_id(index: int) -> str:
    return f"{ROWS[index // 2]}/train-seed-{SEEDS[index % 2]}"


def _plan(root: Path, binary: Path, run_config: Path) -> dict:
    steps = []
    for index in range(len(ROWS) * len(SEEDS)):
        leaf = _task_id(index)
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
    return {"experiment_plan_version": 1, "sim_binary": str(binary),
            "rows": [{"name": row, "run_config": str(run_config)} for row in ROWS],
            "steps": steps}


def _config_body(venv: Path) -> dict:
    return {"cluster_config_version": 1, "name": "fake-site", "venv": str(venv),
            "setup_lines": ["module load python/3.11"],
            "task": {"partition": None, "account": None, "qos": None,
                     "constraint": None, "time": "04:00:00", "mem": "4G",
                     "cpus_per_task": 2},
            "compare": {"time": "00:30:00", "mem": "2G", "cpus_per_task": 1},
            "max_array_size": 1000, "max_concurrent_tasks": None}


@pytest.fixture
def venv(tmp_path: Path) -> Path:
    path = tmp_path / "venv"
    (path / "bin").mkdir(parents=True)
    (path / "bin" / "python").write_text("#!/bin/sh\nexit 0\n")
    return path


@pytest.fixture
def env(tmp_path: Path, venv: Path, monkeypatch) -> SimpleNamespace:
    """Plan, cluster config, and the fake scheduler on PATH."""
    fake = fake_slurm.install(tmp_path)
    monkeypatch.setenv("PATH", f"{fake.bin}{os.pathsep}{os.environ['PATH']}")
    root = tmp_path / "run"
    root.mkdir()
    binary = tmp_path / "mesh-sim"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    run_config = tmp_path / "run.ini"
    run_config.write_text("[scenario]\nname = fake\n")
    write_json(root / experiment.PLAN_NAME, _plan(root, binary, run_config))
    config = tmp_path / "cluster.json"
    write_json(config, _config_body(venv))
    return SimpleNamespace(fake=fake, root=root, config=config, binary=binary,
                           tmp=tmp_path)


def _run(env: SimpleNamespace, command: str, *extra: str) -> int:
    argv = [command, "--output-root", str(env.root)]
    if command in ("plan", "submit", "resume"):
        argv += ["--cluster-config", str(env.config)]
    return cluster.main([*argv, *extra])


def _write_manifest(root: Path, step_id: str, payload) -> Path:
    plan = experiment.load_plan(root)
    step = next(item for item in plan["steps"] if item["id"] == step_id)
    out_dir = Path(step["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        (out_dir / step["manifest"]).write_text(json.dumps(payload))
    return out_dir


def _complete(root: Path, index: int, evaluate: str = "completed") -> None:
    _write_manifest(root, f"train/{_task_id(index)}", {"status": "completed"})
    _write_manifest(root, f"evaluate/{_task_id(index)}", {"status": evaluate})


def _states(env: SimpleNamespace) -> dict[int, str]:
    plan = experiment.load_plan(env.root)
    table = tasks.build_tasks(plan)
    entries = receipts.load(env.root)
    snapshot = slurm.snapshot(receipts.job_ids(entries), receipts.user())
    return {row["index"]: row["state"]
            for row in reconcile.task_rows(table, entries, snapshot)}


def _covering(env: SimpleNamespace, index: int) -> list[list[str]]:
    """Submitted sbatch argvs whose array spec contains this task index."""
    found = []
    for argv in env.fake.submissions():
        spec = next((token.split("=", 1)[1] for token in argv
                     if token.startswith("--array=")), None)
        if spec is None:
            continue
        for token in spec.split("%")[0].split(","):
            low, separator, high = token.partition("-")
            if index in (range(int(low), int(high) + 1) if separator else [int(low)]):
                found.append(argv)
    return found


def _tree(path: Path) -> dict:
    return {str(item.relative_to(path)): item.read_bytes() if item.is_file() else None
            for item in sorted(path.rglob("*"))}


def test_the_example_config_is_refused_unedited():
    with pytest.raises(ValueError, match="placeholder"):
        slurm.load_cluster_config(EXAMPLE_CONFIG)


CONFIG_REFUSALS = [
    ("missing top-level key", lambda body: body.pop("max_array_size"),
     "missing required keys"),
    ("missing task key", lambda body: body["task"].pop("qos"),
     "missing required keys"),
    ("unknown key", lambda body: body.update(sbatch_args=["--exclusive"]),
     "unknown keys"),
    ("free-form task option", lambda body: body["task"].update(extra_args="-x"),
     "no free-form sbatch option"),
    ("illegal null", lambda body: body["task"].update(time=None),
     "task.time must be a non-empty string"),
    ("illegal null compare mem", lambda body: body["compare"].update(mem=None),
     "compare.mem must be a non-empty string"),
    ("bad cpu type", lambda body: body["task"].update(cpus_per_task="2"),
     "task.cpus_per_task must be an integer"),
    ("bad time", lambda body: body["task"].update(time="4h"), "HH:MM:SS"),
    ("bad memory", lambda body: body["task"].update(mem="lots"), "4G or 4000M"),
    ("placeholder", lambda body: body.update(name="<REQUIRED label>"), "placeholder"),
    ("placeholder in setup", lambda body: body.update(setup_lines=["<module load>"]),
     "placeholder"),
    ("bad version", lambda body: body.update(cluster_config_version=2),
     "cluster_config_version"),
    ("zero concurrency", lambda body: body.update(max_concurrent_tasks=0),
     "max_concurrent_tasks must be an integer"),
    ("relative venv", lambda body: body.update(venv="venv"), "absolute path"),
]


@pytest.mark.parametrize("mutate,message", [case[1:] for case in CONFIG_REFUSALS],
                         ids=[case[0] for case in CONFIG_REFUSALS])
def test_cluster_config_refusals(tmp_path, venv, mutate, message):
    body = _config_body(venv)
    mutate(body)
    path = tmp_path / "bad.json"
    write_json(path, body)
    with pytest.raises(ValueError, match=message):
        slurm.load_cluster_config(path)


def test_a_venv_without_bin_python_is_refused(tmp_path):
    body = _config_body(tmp_path / "empty-venv")
    (tmp_path / "empty-venv").mkdir()
    path = tmp_path / "config.json"
    write_json(path, body)
    with pytest.raises(ValueError, match="bin/python does not exist"):
        slurm.load_cluster_config(path)


def test_a_valid_config_keeps_every_value(tmp_path, venv):
    path = tmp_path / "config.json"
    write_json(path, _config_body(venv))
    config = slurm.load_cluster_config(path)
    assert config["task"]["cpus_per_task"] == 2
    assert config["max_concurrent_tasks"] is None
    assert slurm.resources(config, "compare") == {
        "partition": None, "account": None, "qos": None, "constraint": None,
        "time": "00:30:00", "mem": "2G", "cpus_per_task": 1}


ARRAY_SPECS = [([0], None, "0"), ([0, 1, 2, 3, 4, 5, 6, 7], None, "0-7"),
               ([1, 3], None, "1,3"), ([0, 1, 2, 5, 6], None, "0-2,5-6"),
               ([3, 1, 2], None, "1-3"), ([0, 1, 2, 3], 4, "0-3%4")]


@pytest.mark.parametrize("indices,concurrent,expected", ARRAY_SPECS)
def test_array_spec_compression(indices, concurrent, expected):
    assert slurm.array_spec(indices, concurrent) == expected


def test_sbatch_argv_omits_flags_whose_config_value_is_null(tmp_path, venv):
    path = tmp_path / "config.json"
    write_json(path, _config_body(venv))
    config = slurm.load_cluster_config(path)
    argv = slurm.sbatch_argv(config, "tasks", "meshops-x-tasks", "/logs/slurm-%A_%a.out",
                             "/scripts/0001-tasks.sh", array="0-7")
    assert argv[:4] == ["sbatch", "--parsable", "--no-requeue",
                        "--job-name=meshops-x-tasks"]
    assert "--array=0-7" in argv and "--output=/logs/slurm-%A_%a.out" in argv
    assert not [flag for flag in argv if flag.startswith(("--partition", "--account",
                                                          "--qos", "--constraint"))]
    assert argv[-4:] == ["--time=04:00:00", "--mem=4G", "--cpus-per-task=2",
                         "/scripts/0001-tasks.sh"]


def test_sbatch_argv_carries_placement_and_dependency(tmp_path, venv):
    body = _config_body(venv)
    body["task"].update(partition="short", account="proj", qos="normal",
                        constraint="avx2")
    path = tmp_path / "config.json"
    write_json(path, body)
    config = slurm.load_cluster_config(path)
    argv = slurm.sbatch_argv(config, "compare", "meshops-y-compare",
                             "/logs/compare-%j.out", "/scripts/0001-compare.sh",
                             dependency="afterany:1000:1001")
    assert "--dependency=afterany:1000:1001" in argv
    assert "--partition=short" in argv and "--constraint=avx2" in argv
    assert "--cpus-per-task=1" in argv
    assert not [flag for flag in argv if flag.startswith("--array")]


def test_the_job_script_quotes_paths_and_exports_threads(tmp_path, venv):
    path = tmp_path / "config.json"
    write_json(path, _config_body(venv))
    config = slurm.load_cluster_config(path)
    root = tmp_path / "out put"
    script = slurm.render_task_script(config, tmp_path / "mesh root", root,
                                      root / "cluster/records/0001")
    assert script.startswith("#!/bin/bash\nset -euo pipefail\nmodule load python/3.11\n")
    assert "export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=" in script
    assert shlex.quote(str(root)) in script
    assert shlex.quote(str(tmp_path / "mesh root")) in script
    assert f"{venv}/bin/python scripts/rl/bootstrap_venv.py --venv {venv} --check" \
        in script.replace("'", "")
    assert '--task-index "$SLURM_ARRAY_TASK_ID"' in script
    assert "task-$(printf '%04d' \"$SLURM_ARRAY_TASK_ID\").json" in script

    compare = slurm.render_compare_script(config, tmp_path / "mesh", root,
                                          root / "cluster/records/0001")
    assert "--compare" in compare and "SLURM_ARRAY_TASK_ID" not in compare
    assert "--allow-incomplete" not in compare
    assert "compare.json" in compare


def test_parse_squeue_expands_collapsed_array_elements():
    text = ("123_[2-5]|PENDING|Priority|2026-01-01T09:00:00\n"
            "123_7|RUNNING|None|N/A\n"
            "456|PENDING|Dependency|N/A\n"
            "123_[9-10%2]|PENDING|JobArrayTaskLimit|N/A\n")
    jobs = slurm.parse_squeue(text)
    assert sorted(jobs) == ["123_10", "123_2", "123_3", "123_4", "123_5", "123_7",
                            "123_9", "456"]
    assert jobs["123_3"] == {"state": "PENDING", "reason": "Priority",
                             "start": "2026-01-01T09:00:00"}
    assert jobs["123_7"]["state"] == "RUNNING" and jobs["123_7"]["start"] == "N/A"


def test_parse_sacct_drops_step_rows_and_normalizes_cancellation():
    text = ("1000|COMPLETED|0:0\n"
            "1000_0|COMPLETED|0:0\n"
            "1000_0.batch|COMPLETED|0:0\n"
            "1000_0.extern|COMPLETED|0:0\n"
            "1000_1|CANCELLED by 1024|0:15\n"
            "1000_2|TIMEOUT|0:1\n")
    jobs = slurm.parse_sacct(text)
    assert sorted(jobs) == ["1000", "1000_0", "1000_1", "1000_2"]
    assert jobs["1000_1"]["state"] == "CANCELLED"
    assert jobs["1000_2"] == {"state": "TIMEOUT", "exit": "0:1"}


JOB_IDS = [("1234\n", "1234"), ("1234;cluster-a\n", "1234"), ("", None),
           ("sbatch: error: invalid partition\n", None)]


@pytest.mark.parametrize("text,expected", JOB_IDS)
def test_parse_job_id(text, expected):
    assert slurm.parse_job_id(text) == expected


def test_a_missing_scheduler_binary_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    snapshot = slurm.snapshot(["1000"], "someone")
    assert snapshot["queue"]["ok"] is False and snapshot["accounting"]["ok"] is False
    assert snapshot["queue"]["jobs"] == {}


def _snapshot(queue=None, accounting=None, queue_ok=True, accounting_ok=True) -> dict:
    return {"queue": {"ok": queue_ok, "jobs": queue or {},
                      "error": "" if queue_ok else "squeue is unavailable"},
            "accounting": {"ok": accounting_ok, "jobs": accounting or {},
                           "error": "" if accounting_ok else "accounting is disabled"}}


def _receipt(**extra) -> dict:
    receipt = {"receipt_version": 1, "submission": "0001", "state": "submitted",
               "job_name": "meshops-token-tasks", "indices": [0], "job_id": "1000",
               "compare": None, "cancel_requests": [], "human_assertions": []}
    receipt.update(extra)
    return receipt


def _intent(**extra) -> dict:
    return _receipt(**{"state": "submitting", "job_id": None, **extra})


RECONCILE_CASES = [
    ("pending", {}, [_receipt()],
     _snapshot(queue={"1000_0": {"state": "PENDING", "reason": "Priority",
                                 "start": "2026-01-01T09:00:00"}}), "pending"),
    ("running", {}, [_receipt()],
     _snapshot(queue={"1000_0": {"state": "RUNNING", "reason": "None",
                                 "start": "N/A"}}), "running"),
    ("completed", {"train": {"status": "completed"},
                   "evaluate": {"status": "completed"}}, [], _snapshot(), "completed"),
    ("partial", {"train": {"status": "completed"},
                 "evaluate": {"status": "partial"}}, [], _snapshot(), "partial"),
    ("completed beats a terminal accounting row",
     {"train": {"status": "completed"}, "evaluate": {"status": "completed"}},
     [_receipt()], _snapshot(accounting={"1000_0": {"state": "TIMEOUT", "exit": "0:1"}}),
     "completed"),
    ("unknown when the queue query failed", {}, [_receipt()],
     _snapshot(queue_ok=False), "unknown"),
    ("unknown while an intent has no job id", {}, [_intent()], _snapshot(), "unknown"),
    ("unknown when a submitted job left no trace", {}, [_receipt()],
     _snapshot(), "unknown"),
    ("unknown when a parent row is the only accounting record", {}, [_receipt()],
     _snapshot(accounting={"1000": {"state": "COMPLETED", "exit": "0:0"}}), "unknown"),
    ("unknown when accounting claims success but the run is incomplete",
     {"train": {"status": "completed"}}, [_receipt()],
     _snapshot(accounting={"1000_0": {"state": "COMPLETED", "exit": "0:0"}}), "unknown"),
    ("unknown for an unowned running train manifest", {"train": {"status": "running"}},
     [], _snapshot(), "unknown"),
    ("canceled from accounting", {}, [_receipt()],
     _snapshot(accounting={"1000_0": {"state": "CANCELLED", "exit": "0:15"}}),
     "canceled"),
    ("canceled from a recorded request when accounting is unavailable", {},
     [_receipt(cancel_requests=[{"job_ids": ["1000_0"], "indices": [0]}])],
     _snapshot(accounting_ok=False), "canceled"),
    ("failed from accounting", {}, [_receipt()],
     _snapshot(accounting={"1000_0": {"state": "TIMEOUT", "exit": "0:1"}}), "failed"),
    ("failed for an unowned blocked directory", {"train": None}, [], _snapshot(),
     "failed"),
    ("failed once a human asserted the job inactive", {}, [
        _receipt(human_assertions=[{"kind": "inactive_job", "job_id": "1000"}])],
     _snapshot(), "failed"),
    ("completed from the filesystem when accounting is unavailable",
     {"train": {"status": "completed"}, "evaluate": {"status": "completed"}},
     [_receipt()], _snapshot(accounting_ok=False), "completed"),
    ("unknown when accounting is unavailable and nothing was recorded", {},
     [_receipt()], _snapshot(accounting_ok=False), "unknown"),
    ("unsubmitted", {}, [], _snapshot(), "unsubmitted"),
    ("unsubmitted after the intent was abandoned", {},
     [_intent(state="abandoned")], _snapshot(), "unsubmitted"),
]


@pytest.mark.parametrize("filesystem,entries,snapshot,expected",
                         [case[1:] for case in RECONCILE_CASES],
                         ids=[case[0] for case in RECONCILE_CASES])
def test_reconcile_state_table(env, filesystem, entries, snapshot, expected):
    for kind, payload in filesystem.items():
        _write_manifest(env.root, f"{kind}/{_task_id(0)}", payload)
    table = tasks.build_tasks(experiment.load_plan(env.root))
    row = reconcile.task_rows(table, entries, snapshot)[0]
    assert row["state"] == expected
    if expected in ("unknown", "canceled", "failed", "pending", "running"):
        assert row["detail"]


def test_a_pending_row_reports_the_schedulers_own_start_estimate_only(env):
    table = tasks.build_tasks(experiment.load_plan(env.root))
    queued = _snapshot(queue={"1000_0": {"state": "PENDING", "reason": "Priority",
                                         "start": "2026-01-01T09:00:00"},
                              "1000_1": {"state": "PENDING", "reason": "Priority",
                                         "start": "N/A"}})
    rows = reconcile.task_rows(table, [_receipt(indices=[0, 1])], queued)
    assert rows[0]["start"] == "2026-01-01T09:00:00"
    assert rows[1]["start"] is None


def test_an_active_element_is_never_resume_eligible(env):
    table = tasks.build_tasks(experiment.load_plan(env.root))
    entries = [_receipt(cancel_requests=[{"job_ids": ["1000_0"], "indices": [0]}])]
    snapshot = _snapshot(queue={"1000_0": {"state": "RUNNING", "reason": "None",
                                           "start": "N/A"}}, accounting_ok=False)
    rows = reconcile.task_rows(table, entries, snapshot)
    targets = reconcile.resume_targets(table, rows, entries, snapshot)
    assert 0 not in targets["eligible"]


COMPARE_ROWS = [("absent with no compare job", None, "unsubmitted"),
                ("complete", {"status": "complete"}, "completed"),
                ("incomplete", {"status": "incomplete"}, "partial")]


@pytest.mark.parametrize("payload,expected", [case[1:] for case in COMPARE_ROWS],
                         ids=[case[0] for case in COMPARE_ROWS])
def test_the_compare_row_follows_comparison_json(env, payload, expected):
    plan = experiment.load_plan(env.root)
    if payload is not None:
        _write_manifest(env.root, "compare", payload)
    assert reconcile.compare_row(plan, [], _snapshot())["state"] == expected


def test_the_compare_row_is_pending_while_its_job_is_queued(env):
    plan = experiment.load_plan(env.root)
    entries = [_receipt(compare={"job_name": "meshops-token-compare", "job_id": "1009",
                                 "state": "submitted", "argv": [], "depends_on": []})]
    snapshot = _snapshot(queue={"1009": {"state": "PENDING", "reason": "Dependency",
                                         "start": "N/A"}})
    assert reconcile.compare_row(plan, entries, snapshot)["state"] == "pending"


def test_plan_writes_the_task_table_and_submits_nothing(env, capsys):
    assert _run(env, "plan") == 0
    table = json.loads((env.root / "cluster" / "tasks.json").read_text())
    assert table["tasks_version"] == 1 and len(table["tasks"]) == 8
    assert table["tasks"][0] == {"index": 0, "id": _task_id(0),
                                 "train_id": f"train/{_task_id(0)}",
                                 "evaluate_id": f"evaluate/{_task_id(0)}"}
    out = capsys.readouterr().out
    assert "sbatch --parsable --no-requeue" in out and "NNNN" in out
    assert env.fake.submissions() == []
    assert not (env.root / "cluster" / "receipts").exists()


def test_a_differing_task_table_is_refused(env):
    assert _run(env, "plan") == 0
    path = env.root / "cluster" / "tasks.json"
    payload = json.loads(path.read_text())
    payload["plan_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    assert _run(env, "plan") == 1


def test_the_submit_resume_sequence(env, capsys):
    assert _run(env, "submit", "--tasks", "0") == 0
    first = env.fake.last_job_id()
    receipt = json.loads((env.root / "cluster/receipts/0001.json").read_text())
    assert receipt["state"] == "submitted" and receipt["job_id"] == first
    assert receipt["array_spec"] == "0" and receipt["indices"] == [0]
    assert receipt["compare"] is None
    assert receipt["job_name"].startswith("meshops-") \
        and receipt["job_name"].endswith("-tasks")
    assert receipt["cluster_config"]["name"] == "fake-site"
    assert len(receipt["cluster_config_sha256"]) == 64
    assert "no compare job was submitted" in capsys.readouterr().out
    assert (env.root / "cluster/scripts/0001-tasks.sh").is_file()

    states = _states(env)
    assert states[0] == "pending" and states[1] == "unsubmitted"
    env.fake.set_job(first, "RUNNING")
    assert _states(env)[0] == "running"

    _complete(env.root, 0)
    env.fake.set_job(first, "COMPLETED")
    assert _states(env)[0] == "completed"

    assert _run(env, "resume") == 0
    second = env.fake.job_ids()[1]
    receipt = json.loads((env.root / "cluster/receipts/0002.json").read_text())
    assert receipt["array_spec"] == "1-7" and receipt["indices"] == list(range(1, 8))
    assert receipt["compare"]["state"] == "submitted"
    assert receipt["compare"]["depends_on"] == [second]
    compare_argv = env.fake.submissions()[-1]
    assert f"--dependency=afterany:{second}" in compare_argv
    assert "--no-requeue" in compare_argv and "--parsable" in compare_argv

    before = len(env.fake.submissions())
    assert _run(env, "resume") == 0
    assert len(env.fake.submissions()) == before
    assert "nothing to resume" in capsys.readouterr().out


def test_a_blocked_directory_is_reported_and_never_submitted(env, capsys):
    assert _run(env, "submit") == 0
    array = env.fake.job_ids()[0]
    env.fake.set_job(array, "COMPLETED")
    env.fake.set_element(array, 3, "TIMEOUT", exit_code="0:1")
    _write_manifest(env.root, f"train/{_task_id(3)}", None)
    for index in (0, 1, 2, 4, 5, 6, 7):
        _complete(env.root, index)
    assert _states(env)[3] == "failed"

    blocked = Path(experiment.load_plan(env.root)["steps"][6]["output_dir"])
    before = _tree(blocked)
    submissions = len(env.fake.submissions())
    assert _run(env, "resume") == 0
    out = capsys.readouterr().out
    assert "move or delete" in out and str(blocked) in out
    assert len(env.fake.submissions()) == submissions
    assert _tree(blocked) == before


def test_sbatch_dying_after_queuing_leaves_a_recoverable_intent(env, capsys):
    env.fake.die_after_queue()
    assert _run(env, "submit", "--tasks", "0") == 1
    receipt = json.loads((env.root / "cluster/receipts/0001.json").read_text())
    assert receipt["state"] == "submit_uncertain" and receipt["job_id"] is None
    assert "connection lost" in receipt["error"]
    assert _states(env)[0] == "unknown"

    env.fake.die_after_queue(False)
    assert _run(env, "resume") == 0
    assert "recovered job id" in capsys.readouterr().out
    assert len(_covering(env, 0)) == 1
    receipt = json.loads((env.root / "cluster/receipts/0001.json").read_text())
    assert receipt["state"] == "submitted" and receipt["job_id"] == env.fake.job_ids()[0]


def test_an_unmatched_intent_stays_unknown_until_it_is_abandoned(env, capsys):
    env.fake.fail("sbatch")
    assert _run(env, "submit", "--tasks", "0") == 1
    assert env.fake.submissions() == []
    assert _states(env)[0] == "unknown"

    env.fake.fail("sbatch", False)
    assert _run(env, "resume") == 0
    assert _covering(env, 0) == []
    assert _states(env)[0] == "unknown"

    assert _run(env, "resume", "--abandon-intent", "0001:tasks") == 0
    receipt = json.loads((env.root / "cluster/receipts/0001.json").read_text())
    assert receipt["state"] == "abandoned"
    assertion = receipt["human_assertions"][0]
    assert assertion["kind"] == "abandon_intent" and assertion["role"] == "tasks"
    assert assertion["job_name"] == receipt["job_name"] and assertion["user"]
    assert assertion["asserted_at"]
    assert len(_covering(env, 0)) == 1


def test_abandoning_an_intent_that_reached_the_scheduler_is_refused(env, capsys):
    env.fake.die_after_queue()
    assert _run(env, "submit", "--tasks", "0") == 1
    env.fake.die_after_queue(False)
    env.fake.clone_job(env.fake.last_job_id())
    assert _run(env, "resume", "--abandon-intent", "0001:tasks") == 1
    assert "did reach the scheduler" in capsys.readouterr().err
    receipt = json.loads((env.root / "cluster/receipts/0001.json").read_text())
    assert receipt["state"] == "submit_uncertain"
    assert receipt["human_assertions"] == []


def test_one_queue_match_recovers_an_intent_when_accounting_fails(env, capsys):
    env.fake.die_after_queue()
    assert _run(env, "submit", "--tasks", "0") == 1
    env.fake.die_after_queue(False)
    env.fake.fail("sacct")

    assert _run(env, "resume") == 0
    assert "recovered job id" in capsys.readouterr().out
    receipt = json.loads((env.root / "cluster/receipts/0001.json").read_text())
    assert receipt["state"] == "submitted" and receipt["job_id"] == env.fake.job_ids()[0]
    assert len(_covering(env, 0)) == 1


def test_a_known_job_without_any_record_needs_a_human_assertion(env, capsys):
    assert _run(env, "submit", "--tasks", "0") == 0
    job_id = env.fake.last_job_id()
    env.fake.drop_element(job_id, 0)
    assert _states(env)[0] == "unknown"

    assert _run(env, "resume") == 0
    assert len(_covering(env, 0)) == 1

    assert _run(env, "resume", f"--inactive-job={job_id}") == 0
    receipt = json.loads((env.root / "cluster/receipts/0001.json").read_text())
    assert receipt["human_assertions"][0]["kind"] == "inactive_job"
    assert receipt["human_assertions"][0]["job_id"] == job_id
    assert len(_covering(env, 0)) == 2


def test_an_assertion_cannot_override_an_active_job(env, capsys):
    assert _run(env, "submit", "--tasks", "0") == 0
    job_id = env.fake.last_job_id()
    assert _run(env, "resume", f"--inactive-job={job_id}") == 1
    assert "still shows this job as active" in capsys.readouterr().err
    receipt = json.loads((env.root / "cluster/receipts/0001.json").read_text())
    assert receipt["human_assertions"] == []


def test_an_assertion_cannot_override_a_populated_step_directory(env, capsys):
    assert _run(env, "submit", "--tasks", "0") == 0
    job_id = env.fake.last_job_id()
    env.fake.drop_element(job_id, 0)
    _write_manifest(env.root, f"train/{_task_id(0)}", None)

    assert _run(env, "resume", f"--inactive-job={job_id}") == 0
    assert "move or delete" in capsys.readouterr().out
    assert len(_covering(env, 0)) == 1


DUPLICATE_CASES = ["active element", "unresolved intent"]


@pytest.mark.parametrize("case", DUPLICATE_CASES)
def test_no_second_job_is_created_for_a_covered_task(env, case, capsys):
    if case == "unresolved intent":
        env.fake.fail("sbatch")
        assert _run(env, "submit", "--tasks", "0") == 1
        env.fake.fail("sbatch", False)
    else:
        assert _run(env, "submit", "--tasks", "0") == 0
    covering = len(_covering(env, 0))

    assert _run(env, "submit", "--tasks", "0") == 1
    assert _run(env, "submit") == 0
    assert _run(env, "resume") == 0
    assert len(_covering(env, 0)) == covering


def test_submit_refuses_a_task_that_is_not_unsubmitted(env, capsys):
    assert _run(env, "submit", "--tasks", "0") == 0
    assert _run(env, "submit", "--tasks", "0") == 1
    assert "state unsubmitted" in capsys.readouterr().err


def test_a_present_lock_refuses_submission(env, capsys):
    (env.root / "cluster").mkdir(parents=True, exist_ok=True)
    (env.root / "cluster" / "submit.lock").write_text('{"pid": 4242, "host": "login1"}')
    assert _run(env, "submit") == 1
    assert "4242" in capsys.readouterr().err
    assert env.fake.submissions() == []
    assert (env.root / "cluster" / "submit.lock").is_file()


def test_the_lock_is_released_after_a_submission(env):
    assert _run(env, "submit", "--tasks", "0") == 0
    assert not (env.root / "cluster" / "submit.lock").exists()


def test_cancel_passes_exact_element_ids(env, capsys):
    assert _run(env, "submit") == 0
    array = env.fake.job_ids()[0]
    assert _run(env, "cancel", "--tasks", "2,5") == 0
    assert env.fake.scancel_calls()[-1] == [f"{array}_2", f"{array}_5"]
    assert array not in env.fake.scancel_calls()[-1]
    receipt = json.loads((env.root / "cluster/receipts/0001.json").read_text())
    assert receipt["cancel_requests"][-1]["indices"] == [2, 5]
    assert _states(env)[2] == "canceled"
    assert _states(env)[1] == "pending"


def test_cancel_by_submission_targets_the_array_and_its_compare_job(env):
    assert _run(env, "submit") == 0
    array, compare = env.fake.job_ids()[0], env.fake.job_ids()[1]
    assert _run(env, "cancel", "--submission", "0001") == 0
    called = env.fake.scancel_calls()[-1]
    assert compare in called and len(called) == 9
    assert all(job != array for job in called)


def test_cancel_requires_exactly_one_selector(env, capsys):
    assert _run(env, "submit") == 0
    assert _run(env, "cancel") == 1
    assert _run(env, "cancel", "--submission", "0001", "--tasks", "1") == 1
    assert "exactly one" in capsys.readouterr().err


def test_a_failed_scancel_is_not_recorded_as_a_cancellation(env, capsys):
    assert _run(env, "submit") == 0
    env.fake.fail("scancel")
    assert _run(env, "cancel", "--tasks", "2") == 1
    assert "no cancellation was recorded" in capsys.readouterr().err
    receipt = json.loads((env.root / "cluster/receipts/0001.json").read_text())
    assert receipt["cancel_requests"] == []


def test_cancel_refuses_while_the_queue_query_fails(env, capsys):
    assert _run(env, "submit") == 0
    env.fake.fail("squeue")
    assert _run(env, "cancel", "--tasks", "2") == 1
    assert "queue is unavailable" in capsys.readouterr().err
    assert env.fake.scancel_calls() == []
    receipt = json.loads((env.root / "cluster/receipts/0001.json").read_text())
    assert receipt["cancel_requests"] == []


def test_cancel_never_selects_a_job_outside_the_receipts(env):
    assert _run(env, "cancel", "--tasks", "0,1") == 0
    assert env.fake.scancel_calls() == []


DRY_RUNS = [("submit", (), False), ("resume", (), False),
            ("cancel", ("--tasks", "0"), True)]


@pytest.mark.parametrize("command,extra,prepare", DRY_RUNS,
                         ids=[case[0] for case in DRY_RUNS])
def test_dry_runs_change_nothing(env, command, extra, prepare, capsys):
    if prepare:
        assert _run(env, "submit") == 0
    before_tree, before_state = _tree(env.root), env.fake.read()
    capsys.readouterr()
    assert _run(env, command, *extra, "--dry-run") == 0
    assert "dry run" in capsys.readouterr().out
    assert _tree(env.root) == before_tree
    assert env.fake.read() == before_state


def test_a_submit_dry_run_previews_the_argv_and_script(env, capsys):
    assert _run(env, "submit", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "--array=0-7" in out
    assert "scripts.rl.ops.run_task" in out and "bootstrap_venv.py" in out
    assert "afterany:<every active array job id>" in out


def test_a_plan_from_another_root_is_refused(env, tmp_path, capsys):
    elsewhere = tmp_path / "copied"
    elsewhere.mkdir()
    (elsewhere / experiment.PLAN_NAME).write_text(
        (env.root / experiment.PLAN_NAME).read_text())
    assert cluster.main(["submit", "--output-root", str(elsewhere),
                         "--cluster-config", str(env.config)]) == 1
    assert "was written for" in capsys.readouterr().err
    assert env.fake.submissions() == []


def test_a_non_executable_binary_is_refused(env, capsys):
    env.binary.chmod(0o644)
    assert _run(env, "plan") == 1
    assert "not executable" in capsys.readouterr().err


def test_status_renders_and_exits_zero_without_a_scheduler(env, monkeypatch, capsys):
    assert _run(env, "submit", "--tasks", "0") == 0
    monkeypatch.setenv("PATH", str(env.tmp / "empty-bin"))
    assert _run(env, "status") == 0
    out = capsys.readouterr().out
    assert "scheduler queue unavailable" in out and "unknown" in out


def test_status_json_lists_every_task_and_the_comparison(env, capsys):
    assert _run(env, "submit", "--tasks", "0") == 0
    capsys.readouterr()
    assert _run(env, "status", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status_version"] == 1
    assert payload["scheduler"] == {"queue_ok": True, "accounting_ok": True,
                                    "queue_error": "", "accounting_error": ""}
    assert [row["index"] for row in payload["tasks"]] == list(range(8))
    assert payload["tasks"][0]["state"] == "pending"
    assert payload["tasks"][0]["steps"]["train"]["state"] == "pending"
    assert payload["compare"]["state"] == "unsubmitted"


def test_status_never_writes(env, capsys):
    env.fake.die_after_queue()
    assert _run(env, "submit", "--tasks", "0") == 1
    env.fake.die_after_queue(False)
    before = _tree(env.root)
    assert _run(env, "status") == 0
    assert _tree(env.root) == before


def test_ops_never_writes_the_plan_or_a_step_directory(env):
    before = (env.root / experiment.PLAN_NAME).read_bytes()
    assert _run(env, "plan") == 0
    assert _run(env, "submit") == 0
    assert _run(env, "status") == 0
    assert _run(env, "cancel", "--tasks", "0") == 0
    assert _run(env, "resume") == 0
    assert (env.root / experiment.PLAN_NAME).read_bytes() == before
    assert sorted(item.name for item in env.root.iterdir()) == [
        "cluster", experiment.PLAN_NAME]


class _CompareStub:
    """Writes comparison.json like compare.py would and reports a raw exit code."""

    def __init__(self, status: str, code: int):
        self.status, self.code, self.calls = status, code, 0

    def __call__(self, collected: list[int]):
        def execute(step: dict) -> int:
            self.calls += 1
            out_dir = Path(step["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / step["manifest"]).write_text(
                json.dumps({"status": self.status}))
            collected.append(self.code)
            return self.code
        return execute


OUTCOMES = [("complete", 0, 0, "complete"),
            ("complete", 2, 0, "complete with health counters or seed overlap"),
            ("incomplete", 1, 1, "incomplete")]


@pytest.mark.parametrize("status,raw,exit_code,line", OUTCOMES,
                         ids=[case[0] + f"-{case[1]}" for case in OUTCOMES])
def test_manual_compare_reports_each_outcome(env, monkeypatch, capsys, status, raw,
                                             exit_code, line):
    for index in range(8):
        _complete(env.root, index)
    stub = _CompareStub(status, raw)
    monkeypatch.setattr(cluster, "_compare_executor", stub)
    assert _run(env, "compare") == exit_code
    out = capsys.readouterr().out
    assert line in out and f"raw exit code: {raw}" in out


def test_manual_compare_refuses_unfinished_evaluations(env, monkeypatch, capsys):
    _complete(env.root, 0)
    stub = _CompareStub("complete", 0)
    monkeypatch.setattr(cluster, "_compare_executor", stub)
    assert _run(env, "compare") == 1
    assert stub.calls == 0
    assert not (env.root / "comparison").exists()
    assert "missing evaluation" in capsys.readouterr().err


def test_manual_compare_refuses_allow_incomplete_while_a_task_is_active(env, monkeypatch,
                                                                       capsys):
    assert _run(env, "submit", "--tasks", "0") == 0
    stub = _CompareStub("incomplete", 1)
    monkeypatch.setattr(cluster, "_compare_executor", stub)
    assert _run(env, "compare", "--allow-incomplete") == 1
    assert stub.calls == 0
    assert "needs every task to be settled" in capsys.readouterr().err


def test_manual_compare_refuses_while_a_scheduled_compare_is_active(env, monkeypatch,
                                                                    capsys):
    assert _run(env, "submit") == 0
    for index in range(8):
        _complete(env.root, index)
    stub = _CompareStub("complete", 0)
    monkeypatch.setattr(cluster, "_compare_executor", stub)
    assert _run(env, "compare") == 1
    assert stub.calls == 0
    assert "compare job" in capsys.readouterr().err


def test_manual_compare_refuses_while_a_compare_intent_is_unresolved(env, monkeypatch,
                                                                     capsys):
    assert _run(env, "submit") == 0
    path = env.root / "cluster/receipts/0001.json"
    receipt = json.loads(path.read_text())
    receipt["compare"].update(state="submit_uncertain", job_id=None)
    path.write_text(json.dumps(receipt))
    for index in range(8):
        _complete(env.root, index)
    stub = _CompareStub("complete", 0)
    monkeypatch.setattr(cluster, "_compare_executor", stub)
    assert _run(env, "compare") == 1
    assert stub.calls == 0
    assert "no job id" in capsys.readouterr().err


def test_manual_compare_refuses_while_a_compare_job_has_no_terminal_record(
        env, monkeypatch, capsys):
    assert _run(env, "submit") == 0
    env.fake.drop_job(env.fake.job_ids()[1])
    for index in range(8):
        _complete(env.root, index)
    stub = _CompareStub("complete", 0)
    monkeypatch.setattr(cluster, "_compare_executor", stub)
    assert _run(env, "compare") == 1
    assert stub.calls == 0
    assert "no record of having finished" in capsys.readouterr().err


def test_a_compare_job_without_a_terminal_record_blocks_a_new_compare_submission(
        env, capsys):
    assert _run(env, "submit") == 0
    array, compare_job = env.fake.job_ids()[0], env.fake.job_ids()[1]
    env.fake.set_job(array, "FAILED", exit_code="1:0")
    env.fake.drop_job(compare_job)
    submissions = len(env.fake.submissions())

    assert _run(env, "resume") == 1
    assert "no record of having finished" in capsys.readouterr().err
    assert env.fake.scancel_calls() == []
    assert len(env.fake.submissions()) == submissions + 1  # the array only


COMPARE_EVIDENCE = ["human assertion", "terminal accounting record"]


@pytest.mark.parametrize("evidence", COMPARE_EVIDENCE)
def test_evidence_that_a_compare_job_ended_unblocks_manual_compare(env, monkeypatch,
                                                                   evidence):
    assert _run(env, "submit") == 0
    array, compare_job = env.fake.job_ids()[0], env.fake.job_ids()[1]
    for index in range(8):
        _complete(env.root, index)
    env.fake.set_job(array, "COMPLETED")
    if evidence == "human assertion":
        env.fake.drop_job(compare_job)
        assert _run(env, "resume", f"--inactive-job={compare_job}") == 0
    else:
        env.fake.set_job(compare_job, "COMPLETED")

    stub = _CompareStub("complete", 0)
    monkeypatch.setattr(cluster, "_compare_executor", stub)
    assert _run(env, "compare") == 0
    assert stub.calls == 1


def test_a_new_compare_job_cancels_and_verifies_the_earlier_one(env, capsys):
    assert _run(env, "submit") == 0
    first_compare = env.fake.job_ids()[1]
    array = env.fake.job_ids()[0]
    env.fake.set_job(array, "FAILED", exit_code="1:0")
    assert _run(env, "resume") == 0
    assert [first_compare] in env.fake.scancel_calls()
    assert env.fake.jobs()[first_compare]["elements"][first_compare]["state"] \
        == "CANCELLED"
    assert "cancelled earlier compare job" in capsys.readouterr().out


def test_a_failed_cancellation_blocks_a_second_compare_job(env, capsys):
    assert _run(env, "submit") == 0
    array = env.fake.job_ids()[0]
    env.fake.set_job(array, "FAILED", exit_code="1:0")
    env.fake.fail("scancel")
    submissions = len(env.fake.submissions())
    assert _run(env, "resume") == 1
    assert "no new compare job was submitted" in capsys.readouterr().err
    assert len(env.fake.submissions()) == submissions + 1  # the array only
