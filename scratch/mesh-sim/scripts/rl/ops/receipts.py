"""Cluster bookkeeping under <output-root>/cluster/: task table, submit lock, receipts."""

import json
import os
import socket
import getpass
from pathlib import Path

from scripts.rl.cli_common import now_iso, write_json

RECEIPT_VERSION = 1
TASKS_TABLE_VERSION = 1
LOCK_NAME = "submit.lock"
TABLE_NAME = "tasks.json"
MAX_SUBMISSIONS = 9999

_ROLES = ("tasks", "compare")
_TABLE_DIFFERS = ("cluster/tasks.json describes a different plan; use a new "
                  "--output-root or remove the stale file by hand")


def layout(output_root) -> dict:
    """Absolute paths of every cluster bookkeeping location under the output root."""
    cluster = Path(output_root).resolve() / "cluster"
    return {"cluster": cluster, "receipts": cluster / "receipts",
            "scripts": cluster / "scripts", "logs": cluster / "logs",
            "records": cluster / "records", "lock": cluster / LOCK_NAME,
            "table": cluster / TABLE_NAME}


def user() -> str:
    """Account name recorded in receipts and used for scheduler queries."""
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return str(os.getuid())


def task_table(plan_sha256: str, table: list[dict]) -> dict:
    """Deterministic index -> step mapping stored beside the plan."""
    return {"tasks_version": TASKS_TABLE_VERSION, "plan_sha256": plan_sha256,
            "tasks": [{"index": task["index"], "id": task["id"],
                       "train_id": task["train"]["id"],
                       "evaluate_id": task["evaluate"]["id"]} for task in table]}


def write_task_table(output_root, payload: dict) -> Path:
    """Write cluster/tasks.json; an existing file describing another plan is refused."""
    path = layout(output_root)["table"]
    if path.is_file():
        if json.loads(path.read_text()) != json.loads(json.dumps(payload)):
            raise ValueError(_TABLE_DIFFERS)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return path


def load(output_root) -> list[dict]:
    """Every readable receipt, ordered by submission number."""
    directory = layout(output_root)["receipts"]
    if not directory.is_dir():
        return []
    found = []
    for path in sorted(directory.glob("*.json")):
        try:
            receipt = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"unreadable receipt {path}: {exc}") from exc
        found.append(receipt)
    return sorted(found, key=lambda receipt: receipt["submission"])


def path_for(output_root, submission: str) -> Path:
    """Path of one receipt file."""
    return layout(output_root)["receipts"] / f"{submission}.json"


def allocate(output_root) -> tuple[str, Path]:
    """Reserve the next submission number by exclusive creation of its receipt file."""
    directory = layout(output_root)["receipts"]
    directory.mkdir(parents=True, exist_ok=True)
    for number in range(1, MAX_SUBMISSIONS + 1):
        submission = f"{number:04d}"
        path = directory / f"{submission}.json"
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
        except FileExistsError:
            continue
        return submission, path
    raise ValueError(f"more than {MAX_SUBMISSIONS} submissions in {directory}")


def intent(submission: str, name: str, indices: list[int], array_spec: str,
           plan_sha256: str, config: dict, config_sha256: str, script,
           argv: list[str]) -> dict:
    """Receipt written before sbatch runs, so an accepted job is never nameless."""
    return {"receipt_version": RECEIPT_VERSION, "submission": submission,
            "state": "submitting", "created_at": now_iso(),
            "host": socket.gethostname(), "user": user(), "job_name": name,
            "indices": list(indices), "array_spec": array_spec,
            "plan_sha256": plan_sha256, "cluster_config": config,
            "cluster_config_sha256": config_sha256, "script": str(script),
            "argv": list(argv), "job_id": None, "compare": None,
            "cancel_requests": [], "human_assertions": []}


def compare_intent(name: str, argv: list[str], script, depends_on: list[str]) -> dict:
    """Compare-job intent stored under a receipt's `compare` key."""
    return {"job_name": name, "job_id": None, "state": "submitting",
            "created_at": now_iso(), "script": str(script), "argv": list(argv),
            "depends_on": list(depends_on)}


def _element(receipt: dict, role: str) -> dict:
    if role not in _ROLES:
        raise ValueError(f"unknown job role {role!r}; valid: {list(_ROLES)}")
    if role == "compare":
        if not isinstance(receipt.get("compare"), dict):
            raise ValueError(f"receipt {receipt['submission']} holds no compare job")
        return receipt["compare"]
    return receipt


def save(path, receipt: dict) -> dict:
    """Persist a receipt atomically."""
    write_json(path, receipt)
    return receipt


def finalize(path, receipt: dict, role: str, job_id: str) -> dict:
    """Record the parsed job id; only this marks a submission as accepted."""
    element = _element(receipt, role)
    element["job_id"] = job_id
    element["state"] = "submitted"
    element["submitted_at"] = now_iso()
    return save(path, receipt)


def mark_uncertain(path, receipt: dict, role: str, detail: str) -> dict:
    """Leave a no-ID intent after sbatch failed, was interrupted, or was unparseable."""
    element = _element(receipt, role)
    element["state"] = "submit_uncertain"
    element["job_id"] = None
    element["error"] = detail[-2000:]
    element["uncertain_at"] = now_iso()
    return save(path, receipt)


def mark_recovered(path, receipt: dict, role: str, job_id: str) -> dict:
    """Write back a job id found by exact job-name lookup."""
    element = _element(receipt, role)
    element["job_id"] = job_id
    element["state"] = "submitted"
    element["recovered_at"] = now_iso()
    return save(path, receipt)


def mark_abandoned(path, receipt: dict, role: str) -> dict:
    """Mark an intent the human asserted never reached the scheduler."""
    element = _element(receipt, role)
    element["state"] = "abandoned"
    element["abandoned_at"] = now_iso()
    return save(path, receipt)


def add_assertion(path, receipt: dict, assertion: dict) -> dict:
    """Append a recorded human assertion with its time and account name."""
    entry = {**assertion, "asserted_at": now_iso(), "user": user()}
    receipt.setdefault("human_assertions", []).append(entry)
    save(path, receipt)
    return entry


def add_cancel_request(path, receipt: dict, request: dict) -> dict:
    """Record a cancellation that scancel reported as successful."""
    entry = {**request, "requested_at": now_iso(), "user": user()}
    receipt.setdefault("cancel_requests", []).append(entry)
    save(path, receipt)
    return entry


def job_ids(receipts: list[dict]) -> list[str]:
    """Every known job id in the receipts, tasks and compare alike."""
    ids = []
    for receipt in receipts:
        for role in _ROLES:
            element = receipt if role == "tasks" else receipt.get("compare")
            if isinstance(element, dict) and element.get("job_id"):
                ids.append(str(element["job_id"]))
    return sorted(set(ids))


def acquire_lock(output_root) -> Path:
    """Create cluster/submit.lock exclusively; refuses while another submit holds it."""
    paths = layout(output_root)
    paths["cluster"].mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(paths["lock"], os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            held = paths["lock"].read_text().strip()
        except OSError:
            held = "unreadable"
        raise ValueError(f"{paths['lock']} exists: {held}. Remove it by hand after "
                         "confirming no submit is running") from None
    with os.fdopen(handle, "w") as fh:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(),
                   "created_at": now_iso()}, fh)
        fh.write("\n")
    return paths["lock"]


def release_lock(path) -> None:
    """Remove the submit lock this process created."""
    Path(path).unlink(missing_ok=True)
