"""Map an experiment plan onto array tasks and read their filesystem state."""

import json
from pathlib import Path

from scripts.rl.experiment import PLAN_NAME, load_plan, step_state

TASKS_VERSION = 1

_TOLERATED = {"train": (0,), "evaluate": (0, 2), "compare": (0, 2)}
_OUTCOMES = ("complete", "incomplete")

__all__ = ["TASKS_VERSION", "PLAN_NAME", "load_plan", "step_state", "plan_root",
           "build_tasks", "compare_step", "tolerated", "task_fs_state",
           "compare_prerequisites", "comparison_outcome"]


def _steps(plan: dict, kind: str) -> list[dict]:
    steps = plan.get("steps")
    if not isinstance(steps, list):
        raise ValueError("plan has no step list")
    return [step for step in steps if step.get("kind") == kind]


def compare_step(plan: dict) -> dict:
    """Return the plan's single compare step."""
    steps = _steps(plan, "compare")
    if len(steps) != 1:
        raise ValueError(f"plan must hold exactly one compare step, found {len(steps)}")
    return steps[0]


def plan_root(plan: dict) -> Path:
    """Output root the plan was written for, taken from the compare step's --plan."""
    args = compare_step(plan).get("args") or []
    if "--plan" not in args or args.index("--plan") + 1 >= len(args):
        raise ValueError("compare step has no --plan argument")
    path = Path(args[args.index("--plan") + 1])
    if path.name != PLAN_NAME:
        raise ValueError(f"compare step --plan is not a {PLAN_NAME} path: {path}")
    return path.parent


def build_tasks(plan: dict) -> list[dict]:
    """Pair every train step with the evaluate step that needs it, in plan order."""
    compare_step(plan)
    evaluations = _steps(plan, "evaluate")
    by_need: dict[str, dict] = {}
    for step in evaluations:
        for need in step.get("needs") or []:
            by_need.setdefault(need, step)

    tasks, paired = [], set()
    for index, train in enumerate(_steps(plan, "train")):
        evaluate = by_need.get(train["id"])
        if evaluate is None:
            raise ValueError(f"no evaluate step needs {train['id']}")
        paired.add(evaluate["id"])
        tasks.append({"index": index, "id": train["id"].removeprefix("train/"),
                      "train": train, "evaluate": evaluate})
    unpaired = [step["id"] for step in evaluations if step["id"] not in paired]
    if unpaired:
        raise ValueError(f"evaluate steps without a train step: {unpaired}")
    if not tasks:
        raise ValueError("plan has no train steps")
    return tasks


def tolerated(kind: str, code: int) -> bool:
    """Whether this exit code counts as success for that kind of step."""
    if kind not in _TOLERATED:
        raise ValueError(f"unknown step kind {kind!r}; valid: {sorted(_TOLERATED)}")
    return code in _TOLERATED[kind]


def _manifest_status(step: dict) -> str | None:
    path = Path(step["output_dir"]) / step["manifest"]
    try:
        status = json.loads(path.read_text()).get("status")
    except (OSError, ValueError):
        return None
    return status if isinstance(status, str) else None


def task_fs_state(task: dict) -> dict:
    """State of both of a task's steps plus the train manifest's own status field."""
    return {"train": step_state(task["train"]),
            "evaluate": step_state(task["evaluate"]),
            "train_manifest_status": _manifest_status(task["train"])}


def compare_prerequisites(plan: dict) -> dict:
    """Which evaluations are not finished, so a comparison would be incomplete."""
    missing, partial = [], []
    for step in _steps(plan, "evaluate"):
        state = step_state(step)[0]
        if state == "partial":
            partial.append(step["id"])
        elif state != "done":
            missing.append(step["id"])
    return {"ready": not missing and not partial, "missing": missing,
            "partial": partial}


def comparison_outcome(plan: dict) -> dict:
    """Read comparison.json's status, which step_state alone cannot distinguish."""
    step = compare_step(plan)
    path = Path(step["output_dir"]) / step["manifest"]
    if not path.is_file():
        return {"state": "absent", "path": str(path)}
    try:
        status = json.loads(path.read_text()).get("status")
    except (OSError, ValueError):
        return {"state": "unreadable", "path": str(path)}
    state = status if status in _OUTCOMES else "unreadable"
    return {"state": state, "path": str(path)}
