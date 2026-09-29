"""Derive task states from filesystem state, receipts, and a scheduler snapshot."""

from scripts.rl.ops import tasks

QUEUED_STATES = frozenset({"PENDING", "CONFIGURING", "REQUEUED"})
ACTIVE_STATES = QUEUED_STATES | frozenset({"RUNNING", "COMPLETING", "SUSPENDED",
                                           "RESIZING", "SIGNALING", "STAGE_OUT"})
UNRESOLVED_STATES = ("submitting", "submit_uncertain")
RESUMABLE_STATES = ("unsubmitted", "failed", "canceled")
_FINISHED = frozenset({"COMPLETED", "CANCELLED"})


def is_active(state: str) -> bool:
    """Whether a scheduler state means the job may still run."""
    return state in ACTIVE_STATES


def is_queued(state: str) -> bool:
    """Whether a scheduler state means the job is waiting to start."""
    return state in QUEUED_STATES


def element_of(receipt: dict, role: str = "tasks") -> dict | None:
    """The tasks or compare part of a receipt, or None when there is no compare job."""
    if role == "compare":
        element = receipt.get("compare")
        return element if isinstance(element, dict) else None
    return receipt


def covering(receipts: list[dict], index: int) -> list[dict]:
    """Receipts that submitted this task index and were not abandoned."""
    return [receipt for receipt in receipts
            if index in receipt.get("indices", [])
            and receipt.get("state") != "abandoned"]


def _observe(snapshot: dict, element_id: str | None) -> dict:
    """Queue and accounting rows for one element id; None for a failed query or no id."""
    queue, accounting = snapshot["queue"], snapshot["accounting"]
    return {"queue": queue["jobs"].get(element_id) if queue["ok"] and element_id
                     else None,
            "accounting": accounting["jobs"].get(element_id)
                          if accounting["ok"] and element_id else None}


def _element_id(receipt: dict, index: int) -> str | None:
    job_id = receipt.get("job_id")
    return f"{job_id}_{index}" if job_id else None


def _cancel_requested(receipt: dict, index: int) -> bool:
    return any(index in request.get("indices", [])
               for request in receipt.get("cancel_requests", []))


def _asserted_inactive(receipts: list[dict]) -> set[str]:
    return {str(entry.get("job_id")) for receipt in receipts
            for entry in receipt.get("human_assertions", [])
            if entry.get("kind") == "inactive_job"}


def active_elements(receipt: dict, snapshot: dict) -> list[str]:
    """Exact array-element ids of this receipt that the scheduler still shows active."""
    job_id = receipt.get("job_id")
    if not job_id:
        return []
    found = []
    for index in receipt.get("indices", []):
        element_id = f"{job_id}_{index}"
        observed = _observe(snapshot, element_id)
        states = [entry["state"] for entry in observed.values() if entry]
        if any(is_active(state) for state in states):
            found.append(element_id)
    return found


def active_compare(receipt: dict, snapshot: dict) -> str | None:
    """Job id of this receipt's compare job when it is still active."""
    element = element_of(receipt, "compare")
    if not element or not element.get("job_id"):
        return None
    observed = _observe(snapshot, str(element["job_id"]))
    states = [entry["state"] for entry in observed.values() if entry]
    return str(element["job_id"]) if any(is_active(state) for state in states) else None


def active_job_ids(receipts: list[dict], snapshot: dict) -> list[str]:
    """Array job ids that still have at least one active element."""
    return sorted({str(receipt["job_id"]) for receipt in receipts
                   if receipt.get("job_id") and active_elements(receipt, snapshot)})


def unresolved_intents(receipts: list[dict]) -> list[dict]:
    """Submissions whose sbatch response was never parsed into a job id."""
    found = []
    for receipt in receipts:
        for role in ("tasks", "compare"):
            element = element_of(receipt, role)
            if element and element.get("state") in UNRESOLVED_STATES \
                    and not element.get("job_id"):
                found.append({"submission": receipt["submission"], "role": role,
                              "job_name": element["job_name"],
                              "state": element["state"]})
    return found


def _active_row(index: int, receipts: list[dict], snapshot: dict, queued: bool):
    for receipt in receipts:
        element_id = _element_id(receipt, index)
        observed = _observe(snapshot, element_id)
        for entry in observed.values():
            if not entry:
                continue
            state = entry["state"]
            if not is_active(state) or is_queued(state) is not queued:
                continue
            detail = f"job {element_id} {state}"
            reason = entry.get("reason", "")
            if reason and reason not in ("None", "N/A"):
                detail = f"{detail} ({reason})"
            start = entry.get("start", "")
            return (detail, start if start and start != "N/A" else None,
                    receipt, element_id)
    return None


def _latest(receipts: list[dict]) -> dict | None:
    """Receipt with the highest submission number, or None."""
    return max(receipts, key=lambda receipt: receipt["submission"]) if receipts else None


def _unknown_reason(index: int, fs: dict, covers: list[dict], snapshot: dict) -> str | None:
    if covers and not snapshot["queue"]["ok"]:
        error = snapshot["queue"].get("error") or "no output"
        return f"the queue query failed ({error}); resubmission is blocked"
    for receipt in covers:
        if receipt.get("state") in UNRESOLVED_STATES:
            return (f"submission {receipt['submission']} never returned a job id; "
                    f"searched job name {receipt['job_name']}")
    for receipt in covers:
        element_id = _element_id(receipt, index)
        if not element_id:
            continue
        observed = _observe(snapshot, element_id)
        accounting = observed["accounting"]
        if observed["queue"] is None and accounting is None:
            if not snapshot["accounting"]["ok"] and _cancel_requested(receipt, index):
                return None
            return (f"job {element_id} is not in the queue and accounting holds no "
                    "terminal record for it")
        if accounting and accounting["state"] == "COMPLETED":
            return (f"accounting reports {element_id} COMPLETED but train is "
                    f"{fs['train'][0]} and evaluate is {fs['evaluate'][0]}")
    if not covers and fs["train_manifest_status"] == "running":
        return "no receipt covers this task and its train manifest says running"
    return None


def _terminal_row(index: int, covers: list[dict], snapshot: dict) -> tuple[str, str] | None:
    receipt = _latest(covers)
    if receipt is None:
        return None
    element_id = _element_id(receipt, index)
    observed = _observe(snapshot, element_id)
    accounting = observed["accounting"]
    if accounting and accounting["state"] == "CANCELLED":
        return "canceled", f"job {element_id} CANCELLED"
    if not snapshot["accounting"]["ok"] and observed["queue"] is None \
            and _cancel_requested(receipt, index):
        return "canceled", (f"accounting is unavailable; submission "
                            f"{receipt['submission']} recorded a cancellation of "
                            f"{element_id}")
    if accounting and not is_active(accounting["state"]) \
            and accounting["state"] not in _FINISHED:
        return "failed", (f"job {element_id} {accounting['state']} "
                          f"(exit {accounting.get('exit', '?')})")
    return None


def task_row(task: dict, receipts: list[dict], snapshot: dict) -> dict:
    """One task's reported state, following the first-match-wins state table."""
    index = task["index"]
    fs = tasks.task_fs_state(task)
    covers = covering(receipts, index)
    row = {"index": index, "id": task["id"], "state": "unknown", "detail": "",
           "start": None, "receipt": None, "element": None,
           "steps": {kind: {"state": fs[kind][0], "detail": fs[kind][1]}
                     for kind in ("train", "evaluate")},
           "train_manifest_status": fs["train_manifest_status"],
           "asserted_inactive": False}

    for queued, name in ((True, "pending"), (False, "running")):
        found = _active_row(index, covers, snapshot, queued)
        if found:
            detail, start, receipt, element_id = found
            row.update({"state": name, "detail": detail, "start": start,
                        "receipt": receipt["submission"], "element": element_id})
            return row

    train, evaluate = fs["train"][0], fs["evaluate"][0]
    if train == "done" and evaluate in ("done", "partial"):
        row["state"] = "completed" if evaluate == "done" else "partial"
        row["detail"] = "" if evaluate == "done" else fs["evaluate"][1]
        return row

    latest = _latest(covers)
    row["receipt"] = latest["submission"] if latest else None
    row["element"] = _element_id(latest, index) if latest else None

    unknown = _unknown_reason(index, fs, covers, snapshot)
    if unknown:
        row.update({"state": "unknown", "detail": unknown})
        known = [str(receipt["job_id"]) for receipt in covers if receipt.get("job_id")]
        intents = [receipt for receipt in covers
                   if receipt.get("state") in UNRESOLVED_STATES]
        asserted = _asserted_inactive(covers)
        if known and not intents and all(job in asserted for job in known):
            row.update({"state": "failed", "asserted_inactive": True,
                        "detail": f"human asserted job(s) {','.join(known)} inactive"})
        return row

    terminal = _terminal_row(index, covers, snapshot)
    if terminal:
        row["state"], row["detail"] = terminal
        return row

    if not covers:
        if "blocked" in (train, evaluate):
            blocked = "train" if train == "blocked" else "evaluate"
            row.update({"state": "failed", "detail": fs[blocked][1]})
            return row
        if train == "pending" and evaluate == "pending":
            row.update({"state": "unsubmitted", "detail": ""})
            return row

    row.update({"state": "unknown",
                "detail": "no rule matched this receipt and filesystem combination"})
    return row


def task_rows(table: list[dict], receipts: list[dict], snapshot: dict) -> list[dict]:
    """Reported state of every task, in task-index order."""
    return [task_row(task, receipts, snapshot) for task in table]


def compare_row(plan: dict, receipts: list[dict], snapshot: dict) -> dict:
    """State of the comparison, from the latest compare job and comparison.json."""
    outcome = tasks.comparison_outcome(plan)
    submissions = [receipt for receipt in receipts if element_of(receipt, "compare")]
    latest = _latest(submissions)
    row = {"id": "compare", "state": "unsubmitted", "detail": "",
           "outcome": outcome["state"], "receipt": None, "job_id": None,
           "start": None}
    element = element_of(latest, "compare") if latest else None
    if element:
        row["receipt"] = latest["submission"]
        row["job_id"] = element.get("job_id")
        job_id = str(element["job_id"]) if element.get("job_id") else None
        observed = _observe(snapshot, job_id)
        for entry in observed.values():
            if entry and is_active(entry["state"]):
                row["state"] = "pending" if is_queued(entry["state"]) else "running"
                row["detail"] = f"job {job_id} {entry['state']}"
                start = entry.get("start", "")
                row["start"] = start if start and start != "N/A" else None
                return row
        if element.get("state") in UNRESOLVED_STATES:
            row.update({"state": "unknown",
                        "detail": (f"compare submission {latest['submission']} never "
                                   f"returned a job id; searched job name "
                                   f"{element['job_name']}")})
            return row
        if job_id and not snapshot["queue"]["ok"]:
            row.update({"state": "unknown", "detail": "the queue query failed"})
            return row

    if outcome["state"] == "complete":
        row["state"] = "completed"
        return row
    if outcome["state"] == "incomplete":
        row.update({"state": "partial", "detail": "comparison.json is incomplete"})
        return row
    if outcome["state"] == "unreadable":
        row.update({"state": "unknown", "detail": f"unreadable {outcome['path']}"})
        return row

    if element and element.get("job_id"):
        observed = _observe(snapshot, str(element["job_id"]))
        accounting = observed["accounting"]
        if accounting and accounting["state"] == "CANCELLED":
            row.update({"state": "canceled",
                        "detail": f"job {element['job_id']} CANCELLED"})
        elif accounting and accounting["state"] not in _FINISHED:
            row.update({"state": "failed",
                        "detail": f"job {element['job_id']} {accounting['state']}"})
        elif accounting and accounting["state"] == "COMPLETED":
            row.update({"state": "failed",
                        "detail": "the compare job finished but wrote no comparison"})
        else:
            row.update({"state": "unknown",
                        "detail": f"job {element['job_id']} has no terminal record"})
    return row


def _cancel_recorded(receipt: dict, job_id: str) -> bool:
    return any(job_id in [str(entry) for entry in request.get("job_ids", [])]
               for request in receipt.get("cancel_requests", []))


def _terminated(receipt: dict, job_id: str, snapshot: dict, asserted: set[str]) -> bool:
    """Whether anything observed proves this job can no longer write its output."""
    observed = _observe(snapshot, job_id)
    if any(entry and not is_active(entry["state"]) for entry in observed.values()):
        return True
    return _cancel_recorded(receipt, job_id) or job_id in asserted


def compare_blockers(receipts: list[dict], snapshot: dict) -> list[str]:
    """Reasons another writer of comparison.json may already exist."""
    reasons = []
    asserted = _asserted_inactive(receipts)
    for receipt in receipts:
        element = element_of(receipt, "compare")
        if not element or element.get("state") == "abandoned" \
                or not element.get("job_id"):
            continue
        job_id = str(element["job_id"])
        if active_compare(receipt, snapshot):
            reasons.append(f"submission {receipt['submission']} compare job {job_id} "
                           "is still active")
        elif not _terminated(receipt, job_id, snapshot, asserted):
            reasons.append(f"submission {receipt['submission']} compare job {job_id} "
                           "has no record of having finished; assert it with "
                           f"resume --inactive-job {job_id} once you confirm it is gone")
    for intent in unresolved_intents(receipts):
        if intent["role"] == "compare":
            reasons.append(f"submission {intent['submission']} compare job is "
                           f"{intent['state']} with no job id "
                           f"(job name {intent['job_name']})")
    if not snapshot["queue"]["ok"] and any(element_of(receipt, "compare")
                                           for receipt in receipts):
        reasons.append("the queue query failed, so an earlier compare job cannot be "
                       "ruled out")
    return reasons


def runner_steps_clean(task: dict) -> tuple[bool, str]:
    """Whether every step the in-job runner would execute is pending on disk."""
    fs = tasks.task_fs_state(task)
    train, evaluate = fs["train"][0], fs["evaluate"][0]
    if train not in ("pending", "done"):
        return False, f"move or delete {task['train']['output_dir']} to retry"
    if evaluate != "pending":
        return False, f"move or delete {task['evaluate']['output_dir']} to retry"
    return True, ""


def resume_targets(table: list[dict], rows: list[dict], receipts: list[dict],
                   snapshot: dict) -> dict:
    """Split resumable tasks into the safe ones and those a dirty directory blocks."""
    eligible, blocked = [], []
    by_index = {row["index"]: row for row in rows}
    for task in table:
        row = by_index[task["index"]]
        if row["state"] not in RESUMABLE_STATES:
            continue
        active = [element for receipt in receipts
                  for element in active_elements(receipt, snapshot)
                  if element.endswith(f"_{task['index']}")
                  and task["index"] in receipt.get("indices", [])]
        if active:
            blocked.append({"index": task["index"], "id": task["id"],
                            "detail": f"{active[0]} is still active"})
            continue
        clean, detail = runner_steps_clean(task)
        if clean:
            eligible.append(task["index"])
        else:
            blocked.append({"index": task["index"], "id": task["id"], "detail": detail})
    return {"eligible": eligible, "blocked": blocked}


def uncovered_for_compare(rows: list[dict], submitted: list[int]) -> list[int]:
    """Task indices a comparison would not wait for, so no compare job may be queued."""
    covered = set(submitted)
    for row in rows:
        if row["state"] in ("completed", "partial", "pending", "running"):
            covered.add(row["index"])
    return sorted(row["index"] for row in rows if row["index"] not in covered)


