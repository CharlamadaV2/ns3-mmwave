"""Cluster CLI flows against fake_slurm: submission protocol, compare, cancel, refusals.

Expectations come from scripts/rl/ops/README.md "Cluster runs" and its
subsections. The scheduler is always scripts/rl/tests/fake_slurm.py.
"""

import json
import os
import textwrap

import pytest

from scripts.rl.ops import receipts, slurm

from ._ops_gen_helpers import (array_of, cluster_env, complete, flag, live_states, read_receipt, run_cluster)


@pytest.fixture
def env(tmp_path, monkeypatch):
    return cluster_env(tmp_path, monkeypatch, n_tasks=4)


def _lock(env):
    return env.root / "cluster" / "submit.lock"


def _receipt_files(env):
    directory = env.root / "cluster" / "receipts"
    return sorted(path.name for path in directory.glob("*.json")) \
        if directory.is_dir() else []


# --- submission protocol -----------------------------------------------------


def test_gen_the_intent_is_on_disk_and_the_lock_held_while_sbatch_runs(env,
                                                                       monkeypatch):
    seen = []
    original = slurm.submit

    def spy(argv):
        name = flag(argv, "job-name")
        stored = [json.loads(path.read_text())
                  for path in (env.root / "cluster" / "receipts").glob("*.json")]
        role = "compare" if name.endswith("-compare") else "tasks"
        holder = next(entry if role == "tasks" else entry["compare"]
                      for entry in stored
                      if (entry if role == "tasks" else entry.get("compare") or {})
                      .get("job_name") == name)
        seen.append({"role": role, "state": holder["state"],
                     "job_id": holder["job_id"], "locked": _lock(env).is_file(),
                     "script": os.path.isfile(argv[-1]),
                     "argv_recorded": holder["argv"] == argv})
        return original(argv)

    monkeypatch.setattr(slurm, "submit", spy)
    assert run_cluster(env, "submit") == 0
    assert [entry["role"] for entry in seen] == ["tasks", "compare"]
    for entry in seen:
        assert entry == {**entry, "state": "submitting", "job_id": None, "locked": True,
                         "script": True, "argv_recorded": True}
    assert not _lock(env).exists()


def test_gen_an_interrupted_sbatch_leaves_an_uncertain_intent_and_no_lock(env,
                                                                          monkeypatch):
    def interrupted(argv):
        raise KeyboardInterrupt

    monkeypatch.setattr(slurm, "submit", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_cluster(env, "submit", "--tasks", "0")
    stored = read_receipt(env, "0001")
    assert stored["state"] == "submit_uncertain" and stored["job_id"] is None
    assert "KeyboardInterrupt" in stored["error"]
    assert not _lock(env).exists()
    assert live_states(env)[0] == "unknown"


def test_gen_dry_runs_and_plan_never_query_the_scheduler(env, monkeypatch, capsys):
    assert run_cluster(env, "submit", "--tasks", "0,1") == 0
    capsys.readouterr()

    def forbidden(argv):
        raise AssertionError(f"scheduler queried: {argv}")

    monkeypatch.setattr(slurm, "_run", forbidden)
    assert run_cluster(env, "submit", "--dry-run") == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "row-0" in line]
    assert "unknown" in lines[0] and "unknown" in lines[1]
    assert "unsubmitted" in lines[2]
    assert "--array=2-3" in out
    assert run_cluster(env, "resume", "--dry-run") == 0
    assert run_cluster(env, "plan") == 0


# --- compare job -------------------------------------------------------------


def test_gen_the_compare_job_waits_on_every_active_array(env):
    assert run_cluster(env, "submit", "--tasks", "0") == 0
    first = env.fake.last_job_id()
    assert run_cluster(env, "submit") == 0
    second = env.fake.job_ids()[1]
    compare_argv = env.fake.submissions()[-1]
    assert flag(compare_argv, "dependency") == f"afterany:{first}:{second}"
    assert array_of(compare_argv) is None
    stored = read_receipt(env, "0002")
    assert stored["compare"]["depends_on"] == [first, second]
    assert stored["compare"]["state"] == "submitted"


def test_gen_a_lost_compare_submission_is_not_requeued_by_a_later_resume(env,
                                                                         monkeypatch,
                                                                         capsys):
    original = slurm.submit

    def compare_fails(argv):
        if flag(argv, "job-name").endswith("-compare"):
            return {"ok": False, "stdout": "", "stderr": "socket timed out",
                    "returncode": 1, "argv": argv, "job_id": None}
        return original(argv)

    monkeypatch.setattr(slurm, "submit", compare_fails)
    assert run_cluster(env, "submit") == 1
    stored = read_receipt(env, "0001")
    assert stored["state"] == "submitted"
    assert stored["compare"]["state"] == "submit_uncertain"
    monkeypatch.setattr(slurm, "submit", original)

    before = len(env.fake.submissions())
    assert run_cluster(env, "resume") == 0
    assert "nothing to resume" in capsys.readouterr().out
    assert len(env.fake.submissions()) == before
    assert _receipt_files(env) == ["0001.json"]


def test_gen_accounting_alone_still_covers_a_canary_when_the_queue_is_down(env):
    # A PENDING accounting row is a positive active observation, so the canary is
    # covered and the compare job waits on both arrays.
    assert run_cluster(env, "submit", "--tasks", "0") == 0
    first = env.fake.last_job_id()
    env.fake.fail("squeue")
    assert run_cluster(env, "resume") == 0
    second = read_receipt(env, "0002")
    assert second["indices"] == [1, 2, 3]
    assert second["compare"]["depends_on"] == [first, second["job_id"]]


def test_gen_a_canary_with_no_scheduler_view_never_queues_compare(env, capsys):
    assert run_cluster(env, "submit", "--tasks", "0") == 0
    env.fake.fail("squeue")
    env.fake.fail("sacct")
    capsys.readouterr()

    assert run_cluster(env, "resume") == 0
    out = capsys.readouterr().out
    second = read_receipt(env, "0002")
    assert second["indices"] == [1, 2, 3] and second["compare"] is None
    assert "no compare job was submitted" in out
    assert len(env.fake.submissions()) == 2


# --- reconciled states through the fake --------------------------------------


def test_gen_a_requeued_element_is_pending_and_never_resubmitted(env):
    assert run_cluster(env, "submit", "--no-compare") == 0
    array = env.fake.last_job_id()
    env.fake.set_element(array, 1, "REQUEUED")
    assert live_states(env)[1] == "pending"
    assert run_cluster(env, "resume") == 0
    assert len(env.fake.submissions()) == 1


@pytest.mark.parametrize("state", ["NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED",
                                   "TIMEOUT"])
def test_gen_a_terminal_failure_is_resubmitted_alone(env, state):
    assert run_cluster(env, "submit", "--no-compare") == 0
    array = env.fake.last_job_id()
    env.fake.set_job(array, "COMPLETED")
    for index in (0, 2, 3):
        complete(env.root, index)
    env.fake.set_element(array, 1, state, exit_code="0:9")
    assert live_states(env)[1] == "failed"

    assert run_cluster(env, "resume") == 0
    assert array_of(env.fake.submissions()[1]) == "1"
    assert flag(env.fake.submissions()[2], "dependency") \
        == f"afterany:{env.fake.job_ids()[1]}"


def test_gen_a_vanished_element_without_accounting_is_never_resubmitted(env):
    assert run_cluster(env, "submit", "--no-compare") == 0
    array = env.fake.last_job_id()
    env.fake.fail("sacct")
    env.fake.drop_element(array, 1)
    assert live_states(env)[1] == "unknown"
    assert run_cluster(env, "resume") == 0
    assert len(env.fake.submissions()) == 1


def test_gen_status_shows_the_scheduler_start_estimate_for_queued_rows(env, capsys):
    assert run_cluster(env, "submit", "--no-compare") == 0
    array = env.fake.last_job_id()
    env.fake.set_element(array, 0, "PENDING", reason="Resources",
                         start="2026-05-01T12:00:00")
    env.fake.set_element(array, 1, "RUNNING", start="2026-04-30T08:00:00")
    capsys.readouterr()
    assert run_cluster(env, "status") == 0
    out = capsys.readouterr().out
    assert out.count("scheduler-estimated start:") == 1
    assert "scheduler-estimated start: 2026-05-01T12:00:00 (may change)" in out


def test_gen_status_json_has_a_compare_row(env, capsys):
    assert run_cluster(env, "submit") == 0
    capsys.readouterr()
    assert run_cluster(env, "status", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    compare = payload["compare"]
    assert compare["state"] == "pending" and compare["job_id"] == env.fake.job_ids()[1]
    assert compare["outcome"] == "absent" and compare["receipt"] == "0001"


# --- human assertions --------------------------------------------------------


# --- cancellation ------------------------------------------------------------


def test_gen_cancel_never_confuses_task_1_with_task_11(tmp_path, monkeypatch):
    env = cluster_env(tmp_path, monkeypatch, n_tasks=12)
    assert run_cluster(env, "submit", "--no-compare") == 0
    array = env.fake.last_job_id()
    assert run_cluster(env, "cancel", "--tasks", "1") == 0
    assert env.fake.scancel_calls() == [[f"{array}_1"]]
    states = live_states(env)
    assert states[1] == "canceled" and states[11] == "pending" and states[10] == "pending"


def test_gen_cancel_picks_the_active_element_from_the_right_receipt(env):
    assert run_cluster(env, "submit") == 0
    array, _ = env.fake.job_ids()
    env.fake.set_element(array, 2, "TIMEOUT", exit_code="0:1")
    assert run_cluster(env, "resume") == 0
    resubmitted = read_receipt(env, "0002")["job_id"]

    assert run_cluster(env, "cancel", "--tasks", "2") == 0
    assert env.fake.scancel_calls()[-1] == [f"{resubmitted}_2"]
    request = read_receipt(env, "0002")["cancel_requests"][-1]
    assert request["job_ids"] == [f"{resubmitted}_2"] and request["indices"] == [2]
    assert request["selector"] == "--tasks 2"
    assert request["user"] and request["requested_at"]
    assert all(f"{array}_2" not in entry["job_ids"]
               for entry in read_receipt(env, "0001")["cancel_requests"])


def test_gen_the_fake_scheduler_is_the_only_one_reachable(env):
    # Guard for this module: every scheduler command resolves to the fake shims.
    for command in ("sbatch", "squeue", "sacct", "scancel"):
        path = os.popen(f"command -v {command}").read().strip()
        assert path == str(env.fake.bin / command), textwrap.shorten(path, 80)
