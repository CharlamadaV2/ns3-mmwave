#!/usr/bin/env python3
"""Copy selected experiment results to another machine and record what arrived."""

import argparse
import copy
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.rl.cli_common import now_iso, sha256_file, write_json
from scripts.rl.ops import tasks
from scripts.sim_support import find_mesh_root

FETCH_MANIFEST_VERSION = 1
FETCH_MANIFEST_NAME = "fetch_manifest.json"

ALWAYS_INCLUDE = ("experiment_plan.json", "cluster/tasks.json",
                  "cluster/receipts/*.json")

CATEGORY_INCLUDES = {
    "comparison": ("comparison/comparison.json", "comparison/episodes.csv"),
    "manifests": ("train/**/train_manifest.json", "eval/**/eval_manifest.json",
                  "train/**/episode-*/rl_episode.json",
                  "eval/**/episode-*/rl_episode.json",
                  "cluster/records/**", "benchmark/**"),
    "models": ("train/**/maskable_ppo_mesh.zip", "train/**/best_model.zip"),
    "selection-logs": ("train/**/evaluations.npz",),
    "inputs": ("train/**/episode-*/inputs/**", "eval/**/episode-*/inputs/**"),
    "telemetry": ("train/**/episode-*/steps.jsonl",
                  "eval/**/episode-*/steps.jsonl"),
    "episode-data": ("train/**/episode-*/**", "eval/**/episode-*/**"),
    "logs": ("cluster/logs/**",),
}

_HOME_PARENTS = (Path("/Users"), Path("/home"))
_MANIFEST_NAMES = re.compile(r"fetch_manifest(\.\d+)?\.json")
_NO_RSYNC = "rsync was not found on PATH; install it or adjust PATH"


def parse_categories(select: str | None) -> list[str]:
    """Split the --select comma list, keeping order and refusing unknown names."""
    chosen: list[str] = []
    for token in (select or "").split(","):
        name = token.strip()
        if not name or name in chosen:
            continue
        if name not in CATEGORY_INCLUDES:
            raise ValueError(f"unknown --select category {name!r}; valid: "
                             f"{', '.join(sorted(CATEGORY_INCLUDES))}")
        chosen.append(name)
    return chosen


def build_argv(remote: str, dest: Path, categories: list[str]) -> list[str]:
    """rsync argv; --ignore-existing always present so nothing local is overwritten."""
    rules = list(ALWAYS_INCLUDE)
    for category in categories:
        rules.extend(rule for rule in CATEGORY_INCLUDES[category]
                     if rule not in rules)
    return ["rsync", "-a", "--prune-empty-dirs", "--ignore-existing",
            *(f"--include={rule}" for rule in rules),
            "--include=*/", "--exclude=*",
            f"{remote.rstrip('/')}/", f"{dest}/"]


def _local_source(remote: str) -> Path | None:
    """Resolve --remote as a local path, or None when it names a remote host."""
    if remote.startswith("rsync://"):
        return None
    head = remote.split("/", 1)[0]
    if ":" in head:
        return None
    return Path(remote).expanduser().resolve()


def _destination(dest: str) -> tuple[Path, list[Path]]:
    """The destination as written plus its symlink target, so neither is bypassed."""
    path = Path(dest).expanduser()
    literal = path.parent.resolve() / path.name if path.name else path.resolve()
    targets = [literal]
    if path.is_symlink():
        targets.append(path.resolve())
    return literal, targets


def _is_home(path: Path) -> bool:
    return (path == Path.home().resolve() or path == Path("/root")
            or path.parent in _HOME_PARENTS)


def _refusal(literal: Path, targets: list[Path], source: Path | None,
             update: bool) -> str | None:
    mesh_root = find_mesh_root()
    for target in targets:
        if target == Path(target.anchor):
            return f"refusing to fetch into the filesystem root {target}"
        if _is_home(target):
            return f"refusing to fetch into the home directory {target}"
        if target == mesh_root:
            return f"refusing to fetch into the mesh-sim checkout root {target}"
        if source is not None and (source == target or target in source.parents):
            return f"local source {source} is inside the destination {target}"
    if literal.exists() and not literal.is_dir():
        return f"{literal} exists and is not a directory"
    if literal.is_dir() and any(literal.iterdir()) and not update:
        return f"{literal} is not empty; pass --update to add only missing files"
    return None


def _rebase(plan: dict, remote_root: Path, dest: Path) -> dict:
    """Copy the plan in memory with each output_dir moved under the destination."""
    plan = copy.deepcopy(plan)
    for step in plan.get("steps") or []:
        try:
            relative = Path(step["output_dir"]).relative_to(remote_root)
        except (KeyError, TypeError, ValueError):
            continue
        step["output_dir"] = str(dest / relative)
    return plan


def _fetched_plan(dest: Path) -> dict | None:
    """Read the fetched plan and rebase its absolute remote paths onto the dest."""
    try:
        plan = json.loads((dest / tasks.PLAN_NAME).read_text())
        return _rebase(plan, tasks.plan_root(plan), dest)
    except (OSError, ValueError, TypeError):
        return None


def _task_state(state: dict) -> str:
    train, evaluate = state["train"][0], state["evaluate"][0]
    if evaluate == "done":
        return "completed"
    if evaluate == "partial":
        return "partial"
    if train == "pending" and evaluate == "pending":
        return "missing"
    return "failed"


def _task_rows(plan: dict | None, manifests: bool) -> tuple[list[dict], bool | None]:
    """Task states from fetched files only; not_fetched unless manifests came along."""
    try:
        table = tasks.build_tasks(plan) if plan else []
    except ValueError:
        return [], None
    if not manifests:
        return [{"index": task["index"], "id": task["id"], "state": "not_fetched"}
                for task in table], None
    rows = [{"index": task["index"], "id": task["id"],
             "state": _task_state(tasks.task_fs_state(task))} for task in table]
    if not rows:
        return rows, None
    return rows, any(row["state"] != "completed" for row in rows)


def _comparison_state(plan: dict | None, dest: Path, selected: bool) -> str:
    if not selected:
        return "not_fetched"
    if plan is None:
        plan = {"steps": [{"id": "compare", "kind": "compare",
                           "output_dir": str(dest / "comparison"),
                           "manifest": "comparison.json"}]}
    try:
        return tasks.comparison_outcome(plan)["state"]
    except ValueError:
        return "absent"


def _files(dest: Path) -> list[dict]:
    entries = []
    for path in sorted(dest.rglob("*")):
        relative = path.relative_to(dest)
        if not path.is_file():
            continue
        if relative.parent == Path(".") and _MANIFEST_NAMES.fullmatch(relative.name):
            continue
        entries.append({"path": relative.as_posix(), "bytes": path.stat().st_size,
                        "sha256": sha256_file(path)})
    return entries


def _rotate_manifest(dest: Path) -> None:
    """Keep an earlier --update manifest as fetch_manifest.<n>.json."""
    current = dest / FETCH_MANIFEST_NAME
    if not current.is_file():
        return
    index = 1
    while (dest / f"fetch_manifest.{index}.json").exists():
        index += 1
    current.rename(dest / f"fetch_manifest.{index}.json")


def write_manifest(dest: Path, remote: str, categories: list[str],
                   argv: list[str]) -> dict:
    """Record what was fetched, distinguishing not fetched from absent remotely."""
    plan = _fetched_plan(dest)
    rows, incomplete = _task_rows(plan, "manifests" in categories)
    payload = {"fetch_manifest_version": FETCH_MANIFEST_VERSION, "remote": remote,
               "selection": list(categories), "argv": list(argv),
               "fetched_at": now_iso(), "files": _files(dest), "tasks": rows,
               "comparison": _comparison_state(plan, dest,
                                               "comparison" in categories),
               "snapshot_of_incomplete_run": incomplete}
    _rotate_manifest(dest)
    write_json(dest / FETCH_MANIFEST_NAME, payload)
    return payload


def fetch(remote: str, dest: str, select: str | None = None, update: bool = False,
          dry_run: bool = False) -> int:
    """Validate the destination, run rsync, then write the provenance manifest."""
    try:
        categories = parse_categories(select)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    literal, targets = _destination(dest)
    refusal = _refusal(literal, targets, _local_source(remote), update)
    if refusal:
        print(refusal, file=sys.stderr)
        return 1

    argv = build_argv(remote, literal, categories)
    if dry_run:
        print(shlex.join(argv))
        return 0
    if shutil.which(argv[0]) is None:
        print(_NO_RSYNC, file=sys.stderr)
        return 1

    literal.parent.mkdir(parents=True, exist_ok=True)
    code = subprocess.run(argv).returncode
    if code != 0:
        print(f"rsync exited {code}; no fetch manifest was written", file=sys.stderr)
        return 1

    literal.mkdir(parents=True, exist_ok=True)
    payload = write_manifest(literal, remote, categories, argv)
    print(f"fetched {len(payload['files'])} files to {literal}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Copy selected experiment results from a run to this machine")
    p.add_argument("--remote", required=True,
                   help="rsync source: [user@host:]/abs/output-root")
    p.add_argument("--dest", required=True,
                   help="Local destination directory; nothing existing is overwritten")
    p.add_argument("--select", default=None,
                   help="Comma list of categories: "
                        f"{', '.join(sorted(CATEGORY_INCLUDES))}")
    p.add_argument("--update", action="store_true",
                   help="Add only files absent from a non-empty destination")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the rsync argv and exit without transferring")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    return fetch(args.remote, args.dest, args.select, args.update, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
