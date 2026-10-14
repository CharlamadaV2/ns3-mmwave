"""Exhaustive reconcile coverage: receipt x scheduler x filesystem -> reported state.

Expectations come from the "Reported states", "Resume rules", and "Only one
writer of comparison.json" sections of scripts/rl/ops/README.md.
"""

import pytest

from scripts.rl import experiment
from scripts.rl.ops import reconcile

from ._ops_gen_helpers import (a, compare_element, intent, leaf, q, receipt, snap,
                               table, write_manifest, written_plan)

QUEUED = ("PENDING", "CONFIGURING", "REQUEUED")
RUNNING = ("RUNNING", "COMPLETING", "SUSPENDED", "RESIZING", "SIGNALING", "STAGE_OUT")
FAILURES = ("FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED",
            "BOOT_FAIL", "DEADLINE")


@pytest.fixture
def root(tmp_path):
    return written_plan(tmp_path, n_tasks=3)


def _row(root, entries, snapshot, index=0):
    return reconcile.task_rows(table(root), entries, snapshot)[index]


@pytest.mark.parametrize("source", ["queue", "accounting"])
@pytest.mark.parametrize("state", QUEUED + RUNNING)
def test_gen_every_active_state_maps_to_pending_or_running(root, source, state):
    rows = {source: {"1000_0": q(state) if source == "queue" else a(state)}}
    row = _row(root, [receipt()], snap(**rows))
    assert row["state"] == ("pending" if state in QUEUED else "running")
    assert row["receipt"] == "0001" and row["element"] == "1000_0"
    assert row["detail"].startswith(f"job 1000_0 {state}")


@pytest.mark.parametrize("state", FAILURES)
def test_gen_every_explicit_terminal_failure_maps_to_failed(root, state):
    row = _row(root, [receipt()], snap(accounting={"1000_0": a(state, "0:9")}))
    assert row["state"] == "failed"
    assert row["detail"] == f"job 1000_0 {state} (exit 0:9)"


def test_gen_accounting_cancelled_maps_to_canceled(root):
    row = _row(root, [receipt()], snap(accounting={"1000_0": a("CANCELLED")}))
    assert row["state"] == "canceled" and "CANCELLED" in row["detail"]


def test_gen_accounting_completed_with_an_incomplete_filesystem_is_unknown(root):
    row = _row(root, [receipt()], snap(accounting={"1000_0": a("COMPLETED")}))
    assert row["state"] == "unknown"
    assert "COMPLETED" in row["detail"] and "pending" in row["detail"]


def test_gen_a_terminal_state_seen_only_in_the_queue_is_not_a_failure(root):
    # squeue --states=all may list a finished element, but only accounting proves
    # a terminal failure; without it the task stays unknown, never resubmittable.
    row = _row(root, [receipt()], snap(queue={"1000_0": q("FAILED")}))
    assert row["state"] == "unknown"


DISAGREEMENTS = [
    ("queue running, accounting failed", {"1000_0": q("RUNNING")},
     {"1000_0": a("FAILED")}, "running"),
    ("queue pending, accounting completed", {"1000_0": q("PENDING")},
     {"1000_0": a("COMPLETED")}, "pending"),
    ("queue absent, accounting running", {}, {"1000_0": a("RUNNING")}, "running"),
    ("queue absent, accounting requeued", {}, {"1000_0": a("REQUEUED")}, "pending"),
    ("queue suspended, accounting cancelled", {"1000_0": q("SUSPENDED")},
     {"1000_0": a("CANCELLED")}, "running"),
]


@pytest.mark.parametrize("queue,accounting,expected",
                         [case[1:] for case in DISAGREEMENTS],
                         ids=[case[0] for case in DISAGREEMENTS])
def test_gen_any_active_observation_wins_a_queue_accounting_disagreement(
        root, queue, accounting, expected):
    assert _row(root, [receipt()], snap(queue, accounting))["state"] == expected


QUERY_FAILURES = [
    ("queue failed, accounting terminal", dict(queue_ok=False,
                                               accounting={"1000_0": a("TIMEOUT")}),
     [], "unknown"),
    ("queue failed, accounting cancelled", dict(queue_ok=False,
                                                accounting={"1000_0": a("CANCELLED")}),
     [], "unknown"),
    ("queue failed, accounting running", dict(queue_ok=False,
                                              accounting={"1000_0": a("RUNNING")}),
     [], "running"),
    ("queue failed, accounting pending", dict(queue_ok=False,
                                              accounting={"1000_0": a("PENDING")}),
     [], "pending"),
    ("both failed", dict(queue_ok=False, accounting_ok=False), [], "unknown"),
    ("both failed with a recorded cancel", dict(queue_ok=False, accounting_ok=False),
     [{"job_ids": ["1000_0"], "indices": [0]}], "unknown"),
    ("accounting failed, element left the queue", dict(accounting_ok=False), [],
     "unknown"),
    ("accounting failed, cancel recorded for this index", dict(accounting_ok=False),
     [{"job_ids": ["1000_0"], "indices": [0]}], "canceled"),
    ("accounting failed, cancel recorded for a sibling only", dict(accounting_ok=False),
     [{"job_ids": ["1000_1"], "indices": [1]}], "unknown"),
    ("accounting failed, cancelled element still running",
     dict(accounting_ok=False, queue={"1000_0": q("RUNNING")}),
     [{"job_ids": ["1000_0"], "indices": [0]}], "running"),
]


@pytest.mark.parametrize("snapshot_args,cancels,expected",
                         [case[1:] for case in QUERY_FAILURES],
                         ids=[case[0] for case in QUERY_FAILURES])
def test_gen_scheduler_query_failures_degrade_to_unknown(root, snapshot_args, cancels,
                                                         expected):
    entries = [receipt(indices=[0, 1], cancel_requests=cancels)]
    assert _row(root, entries, snap(**snapshot_args))["state"] == expected


def test_gen_a_queue_failure_names_the_error_and_blocks_resubmission(root):
    entries, snapshot = [receipt()], snap(queue_ok=False)
    rows = reconcile.task_rows(table(root), entries, snapshot)
    row = rows[0]
    assert row["state"] == "unknown"
    assert "queue query failed" in row["detail"]
    assert "squeue is unavailable" in row["detail"]
    assert "resubmission is blocked" in row["detail"]
    # Uncovered siblings stay unsubmitted and may still be submitted.
    targets = reconcile.resume_targets(table(root), rows, entries, snapshot)
    assert targets == {"eligible": [1, 2], "blocked": []}


@pytest.mark.parametrize("state", ["submitting", "submit_uncertain"])
def test_gen_an_unresolved_intent_is_unknown_and_names_its_job_name(root, state):
    row = _row(root, [intent(state=state)], snap())
    assert row["state"] == "unknown"
    assert "never returned a job id" in row["detail"]
    assert "meshops-0001-tasks" in row["detail"]
    assert row["element"] is None


def test_gen_partial_array_elements_are_reconciled_one_by_one(root):
    entries = [receipt(indices=[0, 1, 2])]
    snapshot = snap(queue={"1000_1": q("RUNNING")},
                    accounting={"1000_0": a("TIMEOUT", "0:1"), "1000": a("COMPLETED")})
    states = [row["state"] for row in reconcile.task_rows(table(root), entries,
                                                          snapshot)]
    # 1000_2 has neither a queue row nor its own accounting row; the parent
    # "1000" record is not evidence for it.
    assert states == ["failed", "running", "unknown"]


def test_gen_a_receipt_that_does_not_cover_the_task_is_ignored(root):
    entries = [receipt(indices=[1])]
    rows = reconcile.task_rows(table(root), entries, snap())
    assert rows[0]["state"] == "unsubmitted" and rows[0]["receipt"] is None


MULTI = [
    ("older failed, newer queued", {"1001_0": q("PENDING")}, {"1000_0": a("TIMEOUT")},
     None, "pending", "0002"),
    ("older still running, newer failed", {"1000_0": q("RUNNING")},
     {"1001_0": a("FAILED")}, None, "running", "0001"),
    ("older cancelled, newer failed", {},
     {"1000_0": a("CANCELLED"), "1001_0": a("OUT_OF_MEMORY")}, None, "failed", "0002"),
    ("older failed, newer cancelled", {},
     {"1000_0": a("TIMEOUT"), "1001_0": a("CANCELLED")}, None, "canceled", "0002"),
    ("older failed, newer unresolved intent", {}, {"1000_0": a("TIMEOUT")},
     "submit_uncertain", "unknown", "0002"),
    ("older failed, newer abandoned intent", {}, {"1000_0": a("TIMEOUT")},
     "abandoned", "failed", "0001"),
]


@pytest.mark.parametrize("queue,accounting,newer_intent,expected,latest",
                         [case[1:] for case in MULTI], ids=[case[0] for case in MULTI])
def test_gen_several_covering_receipts(root, queue, accounting, newer_intent, expected,
                                       latest):
    newer = (intent("0002", state=newer_intent) if newer_intent
             else receipt("0002", "1001"))
    row = _row(root, [receipt("0001", "1000"), newer], snap(queue, accounting))
    assert row["state"] == expected
    assert row["receipt"] == latest


# --- filesystem rules --------------------------------------------------------


def test_gen_a_completed_filesystem_beats_a_failed_queue_query(root):
    write_manifest(root, f"train/{leaf(0)}", {"status": "completed"})
    write_manifest(root, f"evaluate/{leaf(0)}", {"status": "completed"})
    assert _row(root, [receipt()], snap(queue_ok=False))["state"] == "completed"


def test_gen_a_partial_evaluation_beats_an_unresolved_intent(root):
    write_manifest(root, f"train/{leaf(0)}", {"status": "completed"})
    write_manifest(root, f"evaluate/{leaf(0)}", {"status": "partial"})
    row = _row(root, [intent()], snap())
    assert row["state"] == "partial" and row["detail"]


def test_gen_an_active_job_beats_a_completed_filesystem(root):
    # pending/running are the first rules of the table.
    write_manifest(root, f"train/{leaf(0)}", {"status": "completed"})
    write_manifest(root, f"evaluate/{leaf(0)}", {"status": "completed"})
    assert _row(root, [receipt()], snap({"1000_0": q("COMPLETING")}))["state"] \
        == "running"


def test_gen_a_failed_task_with_finished_training_is_resume_eligible(root):
    # README "Resume rules": train done plus evaluate pending qualifies.
    write_manifest(root, f"train/{leaf(0)}", {"status": "completed"})
    entries, snapshot = [receipt()], snap(accounting={"1000_0": a("TIMEOUT")})
    rows = reconcile.task_rows(table(root), entries, snapshot)
    assert rows[0]["state"] == "failed"
    targets = reconcile.resume_targets(table(root), rows, entries, snapshot)
    assert 0 in targets["eligible"]


@pytest.mark.parametrize("dirty", ["train", "evaluate"])
def test_gen_a_failed_task_with_a_dirty_directory_is_blocked_with_the_retry_text(
        root, dirty):
    if dirty == "train":
        out = write_manifest(root, f"train/{leaf(0)}", {"status": "running"})
    else:
        write_manifest(root, f"train/{leaf(0)}", {"status": "completed"})
        out = write_manifest(root, f"evaluate/{leaf(0)}", None)
    entries, snapshot = [receipt()], snap(accounting={"1000_0": a("TIMEOUT")})
    rows = reconcile.task_rows(table(root), entries, snapshot)
    assert rows[0]["state"] == "failed"
    targets = reconcile.resume_targets(table(root), rows, entries, snapshot)
    assert targets["eligible"] == [1, 2]
    assert targets["blocked"] == [{"index": 0, "id": leaf(0),
                                   "detail": f"move or delete {out} to retry"}]


def test_gen_an_unowned_running_train_manifest_is_unknown_not_failed(root):
    # unknown precedes failed: the step dir is blocked, but its manifest says running.
    write_manifest(root, f"train/{leaf(0)}", {"status": "running"})
    row = _row(root, [], snap())
    assert row["state"] == "unknown" and "running" in row["detail"]


def test_gen_an_unowned_blocked_directory_is_failed_with_the_retry_text(root):
    write_manifest(root, f"train/{leaf(0)}", {"status": "failed"})
    out = write_manifest(root, f"evaluate/{leaf(0)}", None)
    row = _row(root, [], snap())
    # train is blocked too (status failed); the train directory is reported first.
    train_dir = out.parents[2] / "train" / leaf(0)
    assert row["state"] == "failed"
    assert row["detail"] == f"move or delete {train_dir} to retry"


# --- human assertions --------------------------------------------------------


def _asserted(job_id="1000"):
    return [{"kind": "inactive_job", "job_id": job_id, "user": "me",
             "asserted_at": "2026-01-01T00:00:00+00:00"}]


def test_gen_an_inactive_assertion_turns_a_traceless_job_into_a_resumable_failure(root):
    entries = [receipt(human_assertions=_asserted())]
    rows = reconcile.task_rows(table(root), entries, snap())
    assert rows[0]["state"] == "failed" and rows[0]["asserted_inactive"] is True
    assert "1000" in rows[0]["detail"]
    assert reconcile.resume_targets(table(root), rows, entries, snap())["eligible"] \
        == [0, 1, 2]


def test_gen_an_inactive_assertion_never_overrides_an_active_observation(root):
    entries = [receipt(human_assertions=_asserted())]
    snapshot = snap(accounting={"1000_0": a("RUNNING")})
    rows = reconcile.task_rows(table(root), entries, snapshot)
    assert rows[0]["state"] == "running" and rows[0]["asserted_inactive"] is False
    assert 0 not in reconcile.resume_targets(table(root), rows, entries,
                                             snapshot)["eligible"]


def test_gen_an_inactive_assertion_does_not_clear_an_unresolved_intent(root):
    entries = [receipt("0001", human_assertions=_asserted()),
               intent("0002", state="submit_uncertain")]
    row = _row(root, entries, snap())
    assert row["state"] == "unknown" and row["asserted_inactive"] is False


def test_gen_an_assertion_for_another_job_does_not_count(root):
    entries = [receipt(human_assertions=_asserted("999"))]
    assert _row(root, entries, snap())["state"] == "unknown"


# --- helpers used by submit, resume, and cancel ------------------------------


def test_gen_active_elements_lists_exact_element_ids_only():
    entry = receipt(indices=[0, 1, 2, 3])
    snapshot = snap(queue={"1000_1": q("RUNNING"), "1000": q("RUNNING")},
                    accounting={"1000_2": a("PENDING"), "1000_0": a("TIMEOUT"),
                                "1000_3": a("COMPLETED")})
    assert reconcile.active_elements(entry, snapshot) == ["1000_1", "1000_2"]
    assert reconcile.active_elements(intent(), snapshot) == []
    assert reconcile.active_job_ids([entry, receipt("0002", "1001")], snapshot) \
        == ["1000"]


def test_gen_a_parent_array_row_is_not_an_active_element():
    snapshot = snap(queue={"1000": q("RUNNING")}, accounting={"1000": a("RUNNING")})
    assert reconcile.active_elements(receipt(indices=[0, 1]), snapshot) == []


def test_gen_a_sibling_with_a_longer_index_never_blocks_resume(tmp_path):
    root = written_plan(tmp_path, n_tasks=12)
    entries = [receipt("0001", "1000", indices=[1]),
               receipt("0002", "1001", indices=[11])]
    snapshot = snap(queue={"1001_11": q("RUNNING")},
                    accounting={"1000_1": a("FAILED")})
    rows = reconcile.task_rows(table(root), entries, snapshot)
    assert rows[1]["state"] == "failed" and rows[11]["state"] == "running"
    targets = reconcile.resume_targets(table(root), rows, entries, snapshot)
    assert 1 in targets["eligible"] and 11 not in targets["eligible"]
    assert all(entry["index"] != 1 for entry in targets["blocked"])


def test_gen_uncovered_for_compare_counts_finished_and_active_tasks_as_covered():
    states = ["completed", "partial", "pending", "running", "unknown", "failed",
              "canceled", "unsubmitted"]
    rows = [{"index": index, "state": state} for index, state in enumerate(states)]
    assert reconcile.uncovered_for_compare(rows, [6]) == [4, 5, 7]
    assert reconcile.uncovered_for_compare(rows, [4, 5, 6, 7]) == []


def test_gen_unresolved_intents_lists_both_roles_and_skips_settled_ones():
    entries = [intent("0001", state="submit_uncertain"),
               receipt("0002", "1001",
                       compare=compare_element(None, "submitting", "meshops-x-compare")),
               intent("0003", state="abandoned"),
               receipt("0004", "1004", compare=compare_element("1005")),
               receipt("0005", "1006", state="submitting")]
    found = reconcile.unresolved_intents(entries)
    assert found == [
        {"submission": "0001", "role": "tasks", "job_name": "meshops-0001-tasks",
         "state": "submit_uncertain"},
        {"submission": "0002", "role": "compare", "job_name": "meshops-x-compare",
         "state": "submitting"}]


# --- compare row -------------------------------------------------------------


def _plan(root):
    return experiment.load_plan(root)


def _comparison(root, payload):
    write_manifest(root, "compare", payload)


COMPARE_ROWS = [
    ("running in the queue", {"queue": {"2000": q("RUNNING")}}, None, "running"),
    ("running in accounting only", {"accounting": {"2000": a("RUNNING")}}, None,
     "running"),
    ("queue failed", {"queue_ok": False}, None, "unknown"),
    ("queue failed even with a complete comparison", {"queue_ok": False},
     {"status": "complete"}, "unknown"),
    ("finished, complete", {"accounting": {"2000": a("COMPLETED")}},
     {"status": "complete"}, "completed"),
    ("finished, incomplete", {"accounting": {"2000": a("FAILED")}},
     {"status": "incomplete"}, "partial"),
    ("unreadable comparison", {}, {"status": "weird"}, "unknown"),
    ("cancelled without output", {"accounting": {"2000": a("CANCELLED")}}, None,
     "canceled"),
    ("failed without output", {"accounting": {"2000": a("FAILED", "1:0")}}, None,
     "failed"),
    ("timed out without output", {"accounting": {"2000": a("TIMEOUT")}}, None,
     "failed"),
    ("completed without output", {"accounting": {"2000": a("COMPLETED")}}, None,
     "failed"),
    ("no record at all", {}, None, "unknown"),
]


@pytest.mark.parametrize("snapshot_args,payload,expected",
                         [case[1:] for case in COMPARE_ROWS],
                         ids=[case[0] for case in COMPARE_ROWS])
def test_gen_compare_row_state_table(root, snapshot_args, payload, expected):
    if payload is not None:
        _comparison(root, payload)
    entries = [receipt(compare=compare_element("2000"))]
    row = reconcile.compare_row(_plan(root), entries, snap(**snapshot_args))
    assert row["state"] == expected
    assert row["receipt"] == "0001" and row["job_id"] == "2000"


def test_gen_compare_row_reports_an_unresolved_compare_intent(root):
    entries = [receipt(compare=compare_element(None, "submit_uncertain",
                                               "meshops-q-compare"))]
    row = reconcile.compare_row(_plan(root), entries, snap())
    assert row["state"] == "unknown" and "meshops-q-compare" in row["detail"]


def test_gen_compare_row_uses_the_latest_compare_job(root):
    entries = [receipt("0001", "1000", compare=compare_element("2000")),
               receipt("0002", "1001"),
               receipt("0003", "1002", compare=compare_element("2001"))]
    snapshot = snap(queue={"2001": q("PENDING", "Dependency")},
                    accounting={"2000": a("CANCELLED")})
    row = reconcile.compare_row(_plan(root), entries, snapshot)
    assert row["state"] == "pending" and row["receipt"] == "0003"
    assert row["job_id"] == "2001"


def test_gen_compare_row_with_an_abandoned_intent_reads_unsubmitted(root):
    entries = [receipt(compare=compare_element(None, "abandoned"))]
    assert reconcile.compare_row(_plan(root), entries, snap())["state"] == "unsubmitted"


# --- compare blockers --------------------------------------------------------

BLOCKERS = [
    ("no compare job", [receipt()], {}, False),
    ("active compare job", [receipt(compare=compare_element("2000"))],
     {"queue": {"2000": q("PENDING")}}, True),
    ("no trace of the compare job", [receipt(compare=compare_element("2000"))], {},
     True),
    ("accounting proves it finished", [receipt(compare=compare_element("2000"))],
     {"accounting": {"2000": a("COMPLETED")}}, False),
    ("queue shows it finished", [receipt(compare=compare_element("2000"))],
     {"queue": {"2000": q("CANCELLED")}}, False),
    ("cancel recorded for the compare job",
     [receipt(compare=compare_element("2000"),
              cancel_requests=[{"job_ids": ["2000"], "indices": []}])], {}, False),
    ("cancel recorded only for task elements",
     [receipt(compare=compare_element("2000"),
              cancel_requests=[{"job_ids": ["1000_0"], "indices": [0]}])], {}, True),
    ("cancel recorded for a job id with the same prefix",
     [receipt(compare=compare_element("2000"),
              cancel_requests=[{"job_ids": ["20001"], "indices": []}])], {}, True),
    ("compare job asserted inactive on another receipt",
     [receipt("0001", compare=compare_element("2000")),
      receipt("0002", "1001", human_assertions=[{"kind": "inactive_job",
                                                 "job_id": "2000"}])], {}, False),
    ("assertion names another job",
     [receipt(compare=compare_element("2000"),
              human_assertions=[{"kind": "inactive_job", "job_id": "1000"}])], {},
     True),
    ("abandoned compare intent", [receipt(compare=compare_element(None, "abandoned"))],
     {}, False),
    ("unresolved compare intent",
     [receipt(compare=compare_element(None, "submit_uncertain"))], {}, True),
    ("queue failed with a compare element",
     [receipt(compare=compare_element("2000"))],
     {"queue_ok": False, "accounting": {"2000": a("COMPLETED")}}, True),
    ("queue failed with only an abandoned compare element",
     [receipt(compare=compare_element(None, "abandoned"))], {"queue_ok": False}, True),
    ("queue failed without any compare element", [receipt()], {"queue_ok": False},
     False),
    ("accounting down, compare job left the queue",
     [receipt(compare=compare_element("2000"))], {"accounting_ok": False}, True),
]


@pytest.mark.parametrize("entries,snapshot_args,blocked",
                         [case[1:] for case in BLOCKERS],
                         ids=[case[0] for case in BLOCKERS])
def test_gen_compare_blockers(entries, snapshot_args, blocked):
    reasons = reconcile.compare_blockers(entries, snap(**snapshot_args))
    assert bool(reasons) is blocked, reasons
