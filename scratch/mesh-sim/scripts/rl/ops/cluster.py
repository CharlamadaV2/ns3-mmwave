#!/usr/bin/env python3
"""SLURM front end for an experiment plan: plan, submit, status, resume, cancel, compare."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.rl.cli_common import sha256_file
from scripts.rl.ops import reconcile, receipts, run_task, slurm, tasks
from scripts.sim_support import find_mesh_root

STATUS_VERSION = 1
PROVISIONAL = "NNNN"

_OUTCOME_LINES = {("complete", 0): "complete",
                  ("complete", 2): "complete with health counters or seed overlap",
                  ("incomplete", 1): "incomplete"}


class Refused(ValueError):
    """A checked refusal reported on stderr with exit code 1."""


def _plan_and_tasks(output_root):
    plan = tasks.load_plan(output_root)
    return plan, tasks.build_tasks(plan)


def _validate(plan: dict, table: list[dict], output_root, config: dict,
              need_sbatch: bool) -> None:
    root = Path(output_root).resolve()
    errors = []
    if tasks.plan_root(plan) != root:
        errors.append(f"the plan was written for {tasks.plan_root(plan)}, not {root}; "
                      "generate the plan on this filesystem instead of copying it")
    binary = Path(plan["sim_binary"])
    if not binary.is_absolute():
        errors.append(f"sim_binary {binary} is not an absolute path")
    elif not binary.is_file():
        errors.append(f"sim_binary {binary} does not exist")
    elif not os.access(binary, os.X_OK):
        errors.append(f"sim_binary {binary} is not executable")
    for row in plan.get("rows", []):
        if not Path(row["run_config"]).is_file():
            errors.append(f"row {row['name']}: run_config {row['run_config']} is missing")
    if len(table) > config["max_array_size"]:
        errors.append(f"{len(table)} tasks exceed max_array_size "
                      f"{config['max_array_size']}; chunking is not implemented")
    if need_sbatch and not slurm.available("sbatch"):
        errors.append("sbatch is not on PATH")
    if errors:
        raise Refused("\n".join(errors))


def _snapshot(entries: list[dict]) -> dict:
    return slurm.snapshot(receipts.job_ids(entries), receipts.user())


def _offline_snapshot(reason: str) -> dict:
    return {"queue": {"ok": False, "jobs": {}, "error": reason},
            "accounting": {"ok": False, "jobs": {}, "error": reason}}


def _print_rows(rows: list[dict], compare: dict | None = None) -> None:
    for row in rows:
        line = f"{row['index']:>4}  {row['id']:<40}  {row['state']}"
        print(f"{line}  {row['detail']}".rstrip())
        if row["state"] == "pending" and row["start"]:
            print(f"        scheduler-estimated start: {row['start']} (may change)")
    if compare is not None:
        print(f"{'':>4}  {'compare':<40}  {compare['state']}  "
              f"{compare['detail']}".rstrip())


def _paths(output_root, submission: str) -> dict:
    layout = receipts.layout(output_root)
    return {"script_tasks": layout["scripts"] / f"{submission}-tasks.sh",
            "script_compare": layout["scripts"] / f"{submission}-compare.sh",
            "logs": layout["logs"] / submission,
            "records": layout["records"] / submission}


def _write_task_table(output_root, table: list[dict]) -> str:
    plan_sha = sha256_file(Path(output_root).resolve() / tasks.PLAN_NAME)
    receipts.write_task_table(output_root, receipts.task_table(plan_sha, table))
    return plan_sha


def _preview(config: dict, output_root, indices: list[int], with_compare: bool) -> None:
    """Print the tasks sbatch command with provisional names; submits nothing."""
    paths = _paths(output_root, PROVISIONAL)
    spec = slurm.array_spec(indices, config["max_concurrent_tasks"])
    argv = slurm.sbatch_argv(config, "tasks", "meshops-<random>-tasks",
                             paths["logs"] / "slurm-%A_%a.out",
                             paths["script_tasks"], array=spec)
    print(f"provisional receipt {PROVISIONAL} and job names; the real values are "
          "generated when a receipt is allocated. Nothing was submitted")
    print(" ".join(argv))
    print(slurm.render_task_script(config, find_mesh_root(), Path(output_root).resolve(),
                                   paths["records"]))
    if with_compare:
        compare_argv = slurm.sbatch_argv(
            config, "compare", "meshops-<random>-compare",
            paths["logs"] / "compare-%j.out", paths["script_compare"],
            dependency="afterany:<every active array job id>")
        print(" ".join(compare_argv))
        print(slurm.render_compare_script(config, find_mesh_root(),
                                          Path(output_root).resolve(), paths["records"]))


def _recover_intents(output_root, entries: list[dict]) -> list[dict]:
    """Look no-ID intents up by exact job name; write an ID back only when unambiguous."""
    for intent in reconcile.unresolved_intents(entries):
        receipt = next(entry for entry in entries
                       if entry["submission"] == intent["submission"])
        path = receipts.path_for(output_root, intent["submission"])
        found = slurm.find_jobs_by_name(intent["job_name"], receipts.user())
        label = f"submission {intent['submission']} {intent['role']}"
        # One exact match is a positive observation even when the other query failed.
        if len(found["job_ids"]) == 1:
            receipts.mark_recovered(path, receipt, intent["role"], found["job_ids"][0])
            print(f"{label}: recovered job id {found['job_ids'][0]} by job name")
        elif found["job_ids"]:
            print(f"{label}: job name matches {found['job_ids']}; too ambiguous to "
                  "record an id", file=sys.stderr)
        elif not found["ok"]:
            print(f"{label}: the job-name query failed ({found['error']}); the intent "
                  "stays unresolved and blocks resubmission", file=sys.stderr)
        else:
            print(f"{label}: no job carries job name {intent['job_name']}; an empty "
                  "query is not proof that sbatch failed, so the intent stays "
                  "unresolved", file=sys.stderr)
    return receipts.load(output_root)


def _assert_inactive_job(output_root, entries: list[dict], snapshot: dict,
                         job_id: str) -> None:
    holders = [entry for entry in entries
               if str(entry.get("job_id")) == job_id
               or str((entry.get("compare") or {}).get("job_id")) == job_id]
    if not holders:
        raise Refused(f"--inactive-job {job_id}: no receipt holds that job id")
    for receipt in holders:
        if reconcile.active_elements(receipt, snapshot) or \
                reconcile.active_compare(receipt, snapshot):
            raise Refused(f"--inactive-job {job_id}: the scheduler still shows this "
                          "job as active")
        entry = receipts.add_assertion(
            receipts.path_for(output_root, receipt["submission"]), receipt,
            {"kind": "inactive_job", "job_id": job_id})
        print(f"recorded: {receipt['submission']} job {job_id} asserted inactive by "
              f"{entry['user']} at {entry['asserted_at']}")


def _abandon_intent(output_root, entries: list[dict], token: str) -> None:
    submission, _, role = token.partition(":")
    if role not in ("tasks", "compare"):
        raise Refused(f"--abandon-intent {token}: expected NNNN:tasks or NNNN:compare")
    receipt = next((entry for entry in entries if entry["submission"] == submission),
                   None)
    if receipt is None:
        raise Refused(f"--abandon-intent {token}: no receipt {submission}")
    element = reconcile.element_of(receipt, role)
    if element is None or element.get("state") not in reconcile.UNRESOLVED_STATES \
            or element.get("job_id"):
        raise Refused(f"--abandon-intent {token}: that submission is not an "
                      "unresolved no-ID intent")
    found = slurm.find_jobs_by_name(element["job_name"], receipts.user())
    if found["job_ids"]:
        raise Refused(f"--abandon-intent {token}: job name {element['job_name']} "
                      f"matches {found['job_ids']}; it did reach the scheduler")
    path = receipts.path_for(output_root, submission)
    entry = receipts.add_assertion(path, receipt,
                                   {"kind": "abandon_intent", "role": role,
                                    "job_name": element["job_name"],
                                    "queried": found["ok"]})
    receipts.mark_abandoned(path, receipt, role)
    print(f"recorded: {submission} {role} intent abandoned by {entry['user']} at "
          f"{entry['asserted_at']} (searched job name {element['job_name']})")


def _select_submit(rows: list[dict], requested: list[int] | None) -> list[int]:
    ready = [row["index"] for row in rows if row["state"] == "unsubmitted"]
    if requested is None:
        return ready
    known = {row["index"]: row for row in rows}
    unknown = [index for index in requested if index not in known]
    if unknown:
        raise Refused(f"--tasks names indices outside 0..{len(rows) - 1}: {unknown}")
    wrong = [f"{index} is {known[index]['state']}" for index in requested
             if index not in ready]
    if wrong:
        raise Refused("submit only starts tasks in state unsubmitted: "
                      + "; ".join(wrong) + "\nuse resume for failed or canceled tasks")
    return sorted(set(requested))


def _submit_job(output_root, path, receipt: dict, role: str, argv: list[str]) -> str:
    try:
        result = slurm.submit(argv)
    except BaseException as exc:
        receipts.mark_uncertain(path, receipt, role, f"{type(exc).__name__}: {exc}")
        raise
    if not result["ok"] or not result["job_id"]:
        detail = (result["stderr"] or result["stdout"] or "no output").strip()
        receipts.mark_uncertain(path, receipt, role, detail)
        raise Refused(f"sbatch for {role} did not return a job id "
                      f"(exit {result['returncode']}): {detail[:400]}\n"
                      f"submission {receipt['submission']} is left as a no-ID intent; "
                      "it is never retried automatically. Run resume to search for it "
                      f"by job name, or abandon it with --abandon-intent "
                      f"{receipt['submission']}:{role} once you confirm SLURM never "
                      "accepted it")
    receipts.finalize(path, receipt, role, result["job_id"])
    return result["job_id"]


def _submit_tasks(output_root, config: dict, plan_sha: str, config_sha: str,
                  indices: list[int]) -> tuple[dict, Path, str]:
    submission, path = receipts.allocate(output_root)
    paths = _paths(output_root, submission)
    spec = slurm.array_spec(indices, config["max_concurrent_tasks"])
    name = slurm.job_name("tasks")
    argv = slurm.sbatch_argv(config, "tasks", name, paths["logs"] / "slurm-%A_%a.out",
                             paths["script_tasks"], array=spec)
    receipt = receipts.intent(submission, name, indices, spec, plan_sha, config,
                              config_sha, paths["script_tasks"], argv)
    receipts.save(path, receipt)

    root = Path(output_root).resolve()
    paths["script_tasks"].parent.mkdir(parents=True, exist_ok=True)
    paths["script_tasks"].write_text(
        slurm.render_task_script(config, find_mesh_root(), root, paths["records"]))
    paths["script_tasks"].chmod(0o755)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    paths["records"].mkdir(parents=True, exist_ok=True)

    job_id = _submit_job(output_root, path, receipt, "tasks", argv)
    print(f"submission {submission}: job {job_id} array {spec} "
          f"({len(indices)} task{'s' if len(indices) != 1 else ''})")
    return receipt, path, job_id


def _clear_earlier_compare(output_root, entries: list[dict], snapshot: dict) -> None:
    """Cancel any still-active compare job from earlier receipts before queuing a new one."""
    for receipt in entries:
        job_id = reconcile.active_compare(receipt, snapshot)
        if not job_id:
            continue
        result = slurm.cancel([job_id])
        if not result["ok"]:
            raise Refused(f"scancel {job_id} failed ({result['stderr'].strip()}); no "
                          "new compare job was submitted and no cancellation was "
                          "recorded")
        receipts.add_cancel_request(
            receipts.path_for(output_root, receipt["submission"]), receipt,
            {"job_ids": [job_id], "indices": [], "role": "compare",
             "reason": "superseded by a new compare job"})
        print(f"cancelled earlier compare job {job_id}")
    fresh = _snapshot(entries)
    blockers = reconcile.compare_blockers(entries, fresh)
    if blockers:
        raise Refused("another writer of comparison.json may still exist:\n"
                      + "\n".join(f"  {reason}" for reason in blockers))


def _submit_compare(output_root, config: dict, entries: list[dict], snapshot: dict,
                    rows: list[dict], submitted: list[int], receipt: dict,
                    path) -> None:
    uncovered = reconcile.uncovered_for_compare(rows, submitted)
    if uncovered:
        print(f"no compare job was submitted: tasks {uncovered} are neither finished "
              "nor covered by an active job")
        return
    _clear_earlier_compare(output_root, entries, snapshot)

    fresh = _snapshot(entries)
    depends = sorted(set(reconcile.active_job_ids(entries, fresh))
                     | {str(receipt["job_id"])})
    paths = _paths(output_root, receipt["submission"])
    name = slurm.job_name("compare")
    argv = slurm.sbatch_argv(config, "compare", name,
                             paths["logs"] / "compare-%j.out", paths["script_compare"],
                             dependency="afterany:" + ":".join(depends))
    receipt["compare"] = receipts.compare_intent(name, argv, paths["script_compare"],
                                                 depends)
    receipts.save(path, receipt)
    paths["script_compare"].write_text(
        slurm.render_compare_script(config, find_mesh_root(),
                                    Path(output_root).resolve(), paths["records"]))
    paths["script_compare"].chmod(0o755)
    job_id = _submit_job(output_root, path, receipt, "compare", argv)
    print(f"submission {receipt['submission']}: compare job {job_id} "
          f"afterany:{':'.join(depends)}")


def _submit_flow(args, mode: str) -> int:
    config = slurm.load_cluster_config(args.cluster_config)
    plan, table = _plan_and_tasks(args.output_root)
    _validate(plan, table, args.output_root, config, need_sbatch=not args.dry_run)
    root = Path(args.output_root).resolve()
    requested = _parse_indices(getattr(args, "tasks", None))

    if args.dry_run:
        entries = receipts.load(root)
        offline = _offline_snapshot("dry run: the scheduler was not queried")
        rows = reconcile.task_rows(table, entries, offline)
        indices = (_select_submit(rows, requested) if mode == "submit"
                   else reconcile.resume_targets(table, rows, entries, offline)["eligible"])
        print("dry run: no lock, receipt, script, tasks.json, or scheduler call")
        _print_rows(rows)
        if not indices:
            print(f"{mode} would submit nothing")
            return 0
        _preview(config, root, indices,
                 with_compare=not getattr(args, "no_compare", False)
                 and not reconcile.uncovered_for_compare(rows, indices))
        return 0

    plan_sha = _write_task_table(root, table)
    config_sha = sha256_file(args.cluster_config)
    lock = receipts.acquire_lock(root)
    try:
        entries = _recover_intents(root, receipts.load(root))
        snapshot = _snapshot(entries)
        for job_id in getattr(args, "inactive_job", None) or []:
            _assert_inactive_job(root, entries, snapshot, job_id)
        for token in getattr(args, "abandon_intent", None) or []:
            _abandon_intent(root, entries, token)
        entries = receipts.load(root)
        snapshot = _snapshot(entries)
        rows = reconcile.task_rows(table, entries, snapshot)

        if mode == "submit":
            indices = _select_submit(rows, requested)
        else:
            targets = reconcile.resume_targets(table, rows, entries, snapshot)
            for blocked in targets["blocked"]:
                print(f"{blocked['index']:>4}  {blocked['id']}  not submitted: "
                      f"{blocked['detail']}")
            indices = targets["eligible"]
        if not indices:
            _print_rows(rows)
            print(f"nothing to {mode}")
            return 0

        receipt, path, _ = _submit_tasks(root, config, plan_sha, config_sha, indices)
        if getattr(args, "no_compare", False):
            print("--no-compare: no comparison job was submitted")
            return 0
        _submit_compare(root, config, entries, snapshot, rows, indices, receipt, path)
        return 0
    finally:
        receipts.release_lock(lock)


def _parse_indices(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    try:
        indices = [int(token) for token in raw.split(",") if token.strip()]
    except ValueError as exc:
        raise Refused(f"--tasks must be a comma-separated index list: {exc}") from exc
    if not indices:
        raise Refused("--tasks must name at least one task index")
    return sorted(set(indices))


def _cmd_plan(args) -> int:
    config = slurm.load_cluster_config(args.cluster_config)
    plan, table = _plan_and_tasks(args.output_root)
    _validate(plan, table, args.output_root, config, need_sbatch=False)
    root = Path(args.output_root).resolve()
    _write_task_table(root, table)
    for task in table:
        print(f"{task['index']:>4}  {task['id']:<40}  {task['train']['id']} -> "
              f"{task['evaluate']['id']}")
    _preview(config, root, [task["index"] for task in table], with_compare=True)
    return 0


def _cmd_status(args) -> int:
    plan, table = _plan_and_tasks(args.output_root)
    entries = receipts.load(args.output_root)
    snapshot = _snapshot(entries)
    rows = reconcile.task_rows(table, entries, snapshot)
    compare = reconcile.compare_row(plan, entries, snapshot)
    scheduler = {"queue_ok": snapshot["queue"]["ok"],
                 "accounting_ok": snapshot["accounting"]["ok"],
                 "queue_error": snapshot["queue"].get("error", ""),
                 "accounting_error": snapshot["accounting"].get("error", "")}
    if args.json:
        json.dump({"status_version": STATUS_VERSION,
                   "output_root": str(Path(args.output_root).resolve()),
                   "scheduler": scheduler, "tasks": rows, "compare": compare},
                  sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    _print_rows(rows, compare)
    for name, key in (("queue", "queue"), ("accounting", "accounting")):
        if not snapshot[key]["ok"]:
            print(f"scheduler {name} unavailable: "
                  f"{snapshot[key].get('error') or 'query failed'}; states that depend "
                  "on it are reported as unknown")
    return 0


def _cmd_cancel(args) -> int:
    table = _plan_and_tasks(args.output_root)[1]
    if (args.submission is None) == (args.tasks is None):
        raise Refused("pass exactly one of --submission or --tasks")
    entries = receipts.load(args.output_root)
    snapshot = _snapshot(entries)
    if not snapshot["queue"]["ok"]:
        raise Refused("the scheduler queue is unavailable "
                      f"({snapshot['queue'].get('error') or 'query failed'}), so active "
                      "job elements cannot be resolved; nothing was cancelled and "
                      "nothing was recorded")
    targets: dict[str, list[str]] = {}
    indices: dict[str, list[int]] = {}

    if args.submission is not None:
        receipt = next((entry for entry in entries
                        if entry["submission"] == args.submission), None)
        if receipt is None:
            raise Refused(f"no receipt {args.submission}")
        elements = reconcile.active_elements(receipt, snapshot)
        compare_job = reconcile.active_compare(receipt, snapshot)
        targets[receipt["submission"]] = elements + ([compare_job] if compare_job else [])
        indices[receipt["submission"]] = [int(element.rsplit("_", 1)[1])
                                          for element in elements]
    else:
        wanted = _parse_indices(args.tasks)
        known = {task["index"] for task in table}
        unknown = [index for index in wanted if index not in known]
        if unknown:
            raise Refused(f"--tasks names indices outside the plan: {unknown}")
        for receipt in entries:
            elements = [element for element in reconcile.active_elements(receipt, snapshot)
                        if int(element.rsplit("_", 1)[1]) in wanted]
            if elements:
                targets[receipt["submission"]] = elements
                indices[receipt["submission"]] = [int(element.rsplit("_", 1)[1])
                                                  for element in elements]

    flat = [element for elements in targets.values() for element in elements]
    if not flat:
        print("no active job from these receipts matches the selection")
        return 0
    if args.dry_run:
        print("dry run: " + " ".join(["scancel", *flat]))
        return 0
    result = slurm.cancel(flat)
    if not result["ok"]:
        print(f"scancel failed (exit {result['returncode']}): "
              f"{result['stderr'].strip()}\nno cancellation was recorded",
              file=sys.stderr)
        return 1
    for submission, elements in targets.items():
        receipt = next(entry for entry in entries
                       if entry["submission"] == submission)
        receipts.add_cancel_request(
            receipts.path_for(args.output_root, submission), receipt,
            {"job_ids": elements, "indices": indices.get(submission, []),
             "selector": f"--submission {args.submission}" if args.submission
                         else f"--tasks {args.tasks}"})
        print(f"cancelled {' '.join(elements)}")
    return 0


def _compare_executor(collected: list[int]):
    def execute(step: dict) -> int:
        command = [sys.executable, "-m", step["module"], *step["args"]]
        code = subprocess.run(command, cwd=str(find_mesh_root())).returncode
        collected.append(code)
        return code
    return execute


def _cmd_compare(args) -> int:
    plan, table = _plan_and_tasks(args.output_root)
    entries = receipts.load(args.output_root)
    snapshot = _snapshot(entries)
    blockers = reconcile.compare_blockers(entries, snapshot)
    if blockers:
        print("refusing to run compare here; a scheduled comparison may write the "
              "same files:", file=sys.stderr)
        for reason in blockers:
            print(f"  {reason}", file=sys.stderr)
        return 1
    if args.allow_incomplete:
        busy = [row for row in reconcile.task_rows(table, entries, snapshot)
                if row["state"] in ("pending", "running", "unknown")]
        if busy:
            print("--allow-incomplete needs every task to be settled first:",
                  file=sys.stderr)
            for row in busy:
                print(f"  {row['index']} {row['id']} {row['state']}", file=sys.stderr)
            return 1

    raw: list[int] = []
    code = run_task.run_compare(args.output_root, args.allow_incomplete, None,
                               _compare_executor(raw))
    if not raw:
        return code
    outcome = tasks.comparison_outcome(plan)["state"]
    line = _OUTCOME_LINES.get((outcome, raw[0]), f"{outcome} (exit {raw[0]})")
    print(f"{line}; raw exit code: {raw[0]}")
    return code


def _cmd_submit(args) -> int:
    return _submit_flow(args, "submit")


def _cmd_resume(args) -> int:
    return _submit_flow(args, "resume")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run an experiment plan as SLURM array tasks plus one compare job")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, help_text, config=True):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--output-root", required=True,
                         help="Directory holding experiment_plan.json")
        if config:
            cmd.add_argument("--cluster-config", required=True,
                             help="Cluster JSON; see cluster-config.example.json")
        return cmd

    add("plan", "Validate, write cluster/tasks.json, and print the sbatch shape")
    submit = add("submit", "Submit tasks that were never submitted")
    submit.add_argument("--tasks", default=None,
                        help="Comma-separated task indices; omitted -> every "
                             "unsubmitted task")
    submit.add_argument("--no-compare", action="store_true",
                        help="Do not queue the dependent comparison job")
    submit.add_argument("--dry-run", action="store_true",
                        help="Print the argv and scripts; touch nothing")

    status = add("status", "Print every task's state", config=False)
    status.add_argument("--json", action="store_true", help="Machine-readable output")

    resume = add("resume", "Submit unsubmitted, failed, or canceled tasks with clean "
                           "step directories")
    resume.add_argument("--inactive-job", action="append", default=[], metavar="ID",
                        help="Assert that this known job id is no longer active")
    resume.add_argument("--abandon-intent", action="append", default=[],
                        metavar="NNNN:tasks|NNNN:compare",
                        help="Assert that this no-ID submission never reached SLURM")
    resume.add_argument("--dry-run", action="store_true",
                        help="Print the argv and scripts; touch nothing")

    cancel = add("cancel", "Cancel active jobs recorded in the receipts", config=False)
    cancel.add_argument("--submission", default=None,
                        help="Receipt number: cancels its array and compare job")
    cancel.add_argument("--tasks", default=None,
                        help="Comma-separated task indices: cancels exact array elements")
    cancel.add_argument("--dry-run", action="store_true",
                        help="Print the scancel argv; cancel nothing")

    compare = add("compare", "Run the compare step on this host", config=False)
    compare.add_argument("--allow-incomplete", action="store_true",
                         help="Compare even when an evaluation is missing or partial")
    return p


_COMMANDS = {"plan": _cmd_plan, "submit": _cmd_submit, "status": _cmd_status,
             "resume": _cmd_resume, "cancel": _cmd_cancel, "compare": _cmd_compare}


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except Refused as exc:
        print(exc, file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
