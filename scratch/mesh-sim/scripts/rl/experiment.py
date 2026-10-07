#!/usr/bin/env python3
"""Expand an explicit experiment matrix into train, evaluate, and compare steps."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from scripts.rl.cli_common import (MANIFEST_NAME, automatic_output_root,
                                   sha256_file, write_json)
from scripts.rl.env.config import read_action_profile
from scripts.rl.env.selection import resolve_selection
from scripts.rl.evaluate import POLICY_NAMES
from scripts.rl.policy.evaluate import EVAL_MANIFEST_NAME
from scripts.sim_support import find_mesh_root, parse_seed_spec

MATRIX_VERSION = 1
PLAN_VERSION = 1
PLAN_NAME = "experiment_plan.json"
COMPARISON_NAME = "comparison.json"

_TOP_KEYS = ("matrix_version", "name", "description", "run_config", "band",
             "seeds", "training", "evaluation", "rows")
_SEED_KEYS = ("training", "model_selection", "held_out")
_TRAINING_KEYS = ("total_timesteps", "n_steps", "gamma", "ent_coef",
                  "eval_every_steps", "checkpoint_every_steps", "keep_checkpoints")
_EVALUATION_KEYS = ("model", "policies")
_ROW_KEYS = ("name", "observation_preset", "action_profile", "reward_components",
             "reward_weights", "run_config", "band")
_BANDS = ("mmwave", "sub-6")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_EXPLICIT = "rows are explicit; no product expansion"
_PLAN_DIFFERS = "matrix, binary, or --rows changed; use a new --output-root"


def _check_keys(mapping, allowed: tuple[str, ...], where: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{where} must be a JSON object")
    unknown = sorted(key for key in mapping if key not in allowed)
    if unknown:
        raise ValueError(f"{where} has unknown keys {unknown}; "
                         f"valid keys: {list(allowed)}")


def _scalar(value, where: str):
    if isinstance(value, list):
        raise ValueError(f"{where} must be a single value: {_EXPLICIT}")
    return value


def _name(value, where: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.match(value):
        raise ValueError(f"{where} must match [a-z0-9][a-z0-9-]*, got {value!r}")
    return value


def _int(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer, got {value!r}")
    return value


def _training(raw) -> dict:
    _check_keys(raw, _TRAINING_KEYS, "training")
    settings = {}
    for key, value in raw.items():
        value = _scalar(value, f"training.{key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"training.{key} must be a number, got {value!r}")
        settings[key] = value
    if _int(settings.get("eval_every_steps", 0), "training.eval_every_steps") < 0:
        raise ValueError("training.eval_every_steps must be >= 0")
    return settings


def _evaluation(raw, training: dict) -> dict:
    _check_keys(raw, _EVALUATION_KEYS, "evaluation")
    model = _scalar(raw.get("model", "final"), "evaluation.model")
    if not isinstance(model, str) or not model:
        raise ValueError(f"evaluation.model must be a string, got {model!r}")
    if model == "best" and training.get("eval_every_steps", 0) <= 0:
        raise ValueError("evaluation.model 'best' requires training.eval_every_steps > 0")
    policies = raw.get("policies")
    if not isinstance(policies, list) or not all(isinstance(p, str) for p in policies):
        raise ValueError("evaluation.policies must be an array of policy names")
    unknown = [name for name in policies if name not in POLICY_NAMES]
    if unknown:
        raise ValueError(f"evaluation.policies has unknown names {unknown}; "
                         f"valid choices: {list(POLICY_NAMES)}")
    if len(set(policies)) != len(policies):
        raise ValueError(f"evaluation.policies must be distinct, got {policies}")
    if "model" not in policies:
        raise ValueError("evaluation.policies must contain 'model'")
    if len(policies) < 2:
        raise ValueError("evaluation.policies must contain at least one baseline "
                         "besides 'model'")
    return {"model": model, "policies": list(policies)}


def _seed_list(value, where: str) -> list[int]:
    if isinstance(value, str):
        return parse_seed_spec(value)
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a seed spec string or an array of integers")
    seeds = [_int(seed, where) for seed in value]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{where} must be distinct, got {seeds}")
    return seeds


def _seeds(raw, training: dict) -> dict:
    _check_keys(raw, _SEED_KEYS, "seeds")
    train_seeds = _seed_list(raw.get("training"), "seeds.training")
    if not train_seeds:
        raise ValueError("seeds.training must list at least one seed")
    held_out = _seed_list(raw.get("held_out"), "seeds.held_out")
    if not held_out:
        raise ValueError("seeds.held_out must list at least one seed")

    cadence = training.get("eval_every_steps", 0)
    selection = _scalar(raw.get("model_selection"), "seeds.model_selection")
    if cadence > 0 and selection is None:
        raise ValueError("seeds.model_selection is required when "
                         "training.eval_every_steps > 0")
    if cadence <= 0 and selection is not None:
        raise ValueError("seeds.model_selection requires training.eval_every_steps > 0")
    if selection is not None:
        selection = _int(selection, "seeds.model_selection")

    roles = {"seeds.training": set(train_seeds), "seeds.held_out": set(held_out)}
    if selection is not None:
        roles["seeds.model_selection"] = {selection}
    names = sorted(roles)
    for first, second in ((a, b) for i, a in enumerate(names) for b in names[i + 1:]):
        shared = sorted(roles[first] & roles[second])
        if shared:
            raise ValueError(f"{first} and {second} share seeds {shared}; "
                             "seed roles must be disjoint")
    return {"training": train_seeds, "model_selection": selection,
            "held_out": held_out}


def _string_array(value, where: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty JSON array")
    if not all(isinstance(entry, str) for entry in value):
        raise ValueError(f"{where} must contain strings")
    return list(value)


def _weight_array(value, where: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty JSON array")
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) for w in value):
        raise ValueError(f"{where} must contain numbers")
    return [float(w) for w in value]


def _band(value, where: str) -> str | None:
    value = _scalar(value, where)
    if value is None:
        return None
    if value not in _BANDS:
        raise ValueError(f"{where} must be one of {list(_BANDS)}, got {value!r}")
    return value


def _run_config(value, where: str) -> str:
    value = _scalar(value, where)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a path to a run.ini")
    path = Path(value)
    if not path.is_absolute():
        path = find_mesh_root() / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{where}: run config not found: {path}")
    return str(path)


def _row(raw, index: int, defaults: dict) -> dict:
    where = f"rows[{index}]"
    _check_keys(raw, _ROW_KEYS, where)
    name = _name(_scalar(raw.get("name"), f"{where}.name"), f"{where}.name")
    preset = _scalar(raw.get("observation_preset"), f"{name}.observation_preset")
    if not isinstance(preset, str) or not preset:
        raise ValueError(f"{name}.observation_preset must be a string")
    profile = _scalar(raw.get("action_profile"), f"{name}.action_profile")
    components = _string_array(raw.get("reward_components"), f"{name}.reward_components")
    weights = _weight_array(raw.get("reward_weights"), f"{name}.reward_weights")

    run_config = _run_config(raw.get("run_config", defaults["run_config"]),
                             f"{name}.run_config")
    band = _band(raw.get("band", defaults["band"]), f"{name}.band")

    configured = read_action_profile(run_config)
    if profile != configured:
        raise ValueError(f"{name}.action_profile is {profile!r} but {run_config} "
                         f"configures {configured!r}")
    try:
        resolve_selection(run_config, observation_preset=preset,
                          reward_components=",".join(components),
                          reward_weights=",".join(str(w) for w in weights))
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc

    return {"name": name, "observation_preset": preset, "action_profile": profile,
            "reward_components": components, "reward_weights": weights,
            "run_config": run_config, "band": band}


def _rows(raw, defaults: dict) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("rows must be a non-empty JSON array")
    rows = [_row(entry, index, defaults) for index, entry in enumerate(raw)]
    names = [row["name"] for row in rows]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"row names must be unique, repeated: {duplicates}")
    return rows


def load_matrix(path) -> dict:
    """Load and fully validate a matrix file; no simulator and no output is touched."""
    matrix_path = Path(path).resolve()
    data = json.loads(matrix_path.read_text())
    _check_keys(data, _TOP_KEYS, "matrix")
    if data.get("matrix_version") != MATRIX_VERSION:
        raise ValueError(f"matrix_version must be {MATRIX_VERSION}, got "
                         f"{data.get('matrix_version')!r}")
    name = _name(_scalar(data.get("name"), "name"), "name")
    description = _scalar(data.get("description", ""), "description")
    if not isinstance(description, str):
        raise ValueError("description must be a string")

    training = _training(data.get("training", {}))
    evaluation = _evaluation(data.get("evaluation", {}), training)
    seeds = _seeds(data.get("seeds", {}), training)
    defaults = {"run_config": data.get("run_config"), "band": data.get("band")}
    rows = _rows(data.get("rows"), defaults)
    return {"path": str(matrix_path), "sha256": sha256_file(matrix_path),
            "name": name, "description": description, "seeds": seeds,
            "training": training, "evaluation": evaluation, "rows": rows}


def _train_args(matrix: dict, row: dict, sim_binary: str, out_dir: str,
                seed: int) -> list[str]:
    args = ["--sim-binary", sim_binary, "--run-config", row["run_config"]]
    if row["band"] is not None:
        args += ["--band", row["band"]]
    args += ["--output-dir", out_dir, "--verbose", "0",
             "--observation-preset", row["observation_preset"],
             "--reward-components", ",".join(row["reward_components"]),
             "--reward-weights", ",".join(str(w) for w in row["reward_weights"]),
             "m-ppo"]
    for key in _TRAINING_KEYS:
        if key in matrix["training"]:
            args += [f"--{key.replace('_', '-')}", str(matrix["training"][key])]
    args += ["--seed", str(seed), "--eval-episodes", "1"]
    selection = matrix["seeds"]["model_selection"]
    if selection is not None:
        args += ["--eval-seed", str(selection)]
    return args


def _evaluate_args(matrix: dict, row: dict, sim_binary: str, run_dir: str,
                   out_dir: str) -> list[str]:
    return ["--sim-binary", sim_binary, "--run-dir", run_dir,
            "--model", matrix["evaluation"]["model"], "--output-dir", out_dir,
            "--seeds", ",".join(str(seed) for seed in matrix["seeds"]["held_out"]),
            "--policies", ",".join(matrix["evaluation"]["policies"]),
            "--label", row["name"]]


def build_plan(matrix: dict, output_root, sim_binary: str,
               rows: list[str] | None = None) -> dict:
    """Expand the matrix into an ordered, deterministic step list."""
    selected = matrix["rows"]
    if rows is not None:
        known = {row["name"] for row in matrix["rows"]}
        unknown = [name for name in rows if name not in known]
        if unknown:
            raise ValueError(f"--rows names not in the matrix: {unknown}; "
                             f"available: {sorted(known)}")
        selected = [row for row in matrix["rows"] if row["name"] in set(rows)]

    root = Path(output_root).resolve()
    binary = str(Path(sim_binary).expanduser())
    steps, evaluations = [], []
    for row in selected:
        for seed in matrix["seeds"]["training"]:
            leaf = f"{row['name']}/train-seed-{seed}"
            train_dir = str(root / "train" / leaf)
            eval_dir = str(root / "eval" / leaf)
            steps.append({
                "id": f"train/{leaf}", "kind": "train", "module": "scripts.rl.train",
                "args": _train_args(matrix, row, binary, train_dir, seed),
                "output_dir": train_dir, "manifest": MANIFEST_NAME, "needs": []})
            steps.append({
                "id": f"evaluate/{leaf}", "kind": "evaluate",
                "module": "scripts.rl.evaluate",
                "args": _evaluate_args(matrix, row, binary, train_dir, eval_dir),
                "output_dir": eval_dir, "manifest": EVAL_MANIFEST_NAME,
                "needs": [f"train/{leaf}"]})
            evaluations.append({"label": row["name"], "training_seed": seed,
                                "eval_dir": eval_dir})

    comparison_dir = str(root / "comparison")
    steps.append({
        "id": "compare", "kind": "compare", "module": "scripts.rl.compare",
        "args": ["--plan", str(root / PLAN_NAME), "--output-dir", comparison_dir],
        "output_dir": comparison_dir, "manifest": COMPARISON_NAME, "needs": []})

    return {
        "experiment_plan_version": PLAN_VERSION,
        "matrix": {"path": matrix["path"], "sha256": matrix["sha256"],
                   "name": matrix["name"]},
        "sim_binary": binary,
        "rows_filter": list(rows) if rows is not None else None,
        "seeds": matrix["seeds"],
        "rows": selected,
        "steps": steps,
        "evaluations": evaluations,
    }


def step_state(step: dict) -> tuple[str, str]:
    """Derive one step's state from its output directory and manifest only."""
    out_dir = Path(step["output_dir"])
    retry = f"move or delete {out_dir} to retry"
    if not out_dir.exists():
        return "pending", ""
    manifest = out_dir / step["manifest"]
    if not manifest.is_file():
        return "blocked", retry
    try:
        status = json.loads(manifest.read_text()).get("status")
    except (OSError, ValueError):
        return "blocked", retry
    if step["kind"] == "compare":
        return "done", ""
    if status == "completed":
        return "done", ""
    if status == "partial" and step["kind"] == "evaluate":
        return "partial", "some episodes did not complete"
    return "blocked", retry


def _subprocess_execute(step: dict) -> int:
    """Run one step in a fresh process from the mesh root with inherited stdio."""
    command = [sys.executable, "-m", step["module"], *step["args"]]
    return subprocess.run(command, cwd=str(find_mesh_root())).returncode


def _print_states(states: dict[str, tuple[str, str]]) -> None:
    for step_id, (state, detail) in states.items():
        print(f"{step_id}  {state}  {detail}".rstrip())


def run_plan(plan: dict, execute=None) -> int:
    """Run pending steps sequentially; compare always runs last."""
    execute = execute or _subprocess_execute
    states: dict[str, tuple[str, str]] = {}
    work = [step for step in plan["steps"] if step["kind"] != "compare"]
    for step in work:
        state, detail = step_state(step)
        if state != "pending":
            states[step["id"]] = (state, detail)
            continue
        unmet = [need for need in step["needs"]
                 if states.get(need, ("pending", ""))[0] != "done"]
        if unmet:
            states[step["id"]] = ("skipped", f"{unmet[0]} did not complete")
            continue
        code = execute(step)
        tolerated = code == 0 or (step["kind"] == "evaluate" and code == 2)
        states[step["id"]] = ("done", "") if tolerated else ("failed", f"exit {code}")

    compare = next(step for step in plan["steps"] if step["kind"] == "compare")
    compare_code = execute(compare)
    states[compare["id"]] = (("done", "") if compare_code in (0, 2)
                             else ("failed", f"exit {compare_code}"))
    _print_states(states)

    if any(states[step["id"]][0] != "done" for step in work):
        return 1
    return compare_code


def _write_plan(plan: dict, output_root) -> None:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / PLAN_NAME
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text())
        if existing != json.loads(json.dumps(plan)):
            raise ValueError(_PLAN_DIFFERS)
        return
    write_json(plan_path, plan)


def load_plan(output_root) -> dict:
    """Read the plan written by `plan` or `run`."""
    plan_path = Path(output_root).resolve() / PLAN_NAME
    if not plan_path.is_file():
        raise ValueError(f"no {PLAN_NAME} in {plan_path.parent}; run plan first")
    return json.loads(plan_path.read_text())


def _parse_rows(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    names = [token.strip() for token in raw.split(",") if token.strip()]
    if not names:
        raise ValueError("--rows must name at least one row")
    if len(set(names)) != len(names):
        raise ValueError(f"--rows must be distinct, got {names}")
    return names


def automatic_run_label(matrix: dict, rows: list[str] | None) -> str:
    """Make a one-row output identify both its matrix and physical scene."""
    if rows is None or len(rows) != 1:
        return matrix["name"]
    row = next((entry for entry in matrix["rows"] if entry["name"] == rows[0]), None)
    if row is None:
        raise ValueError(f"--rows name not in matrix: {rows[0]}")
    scene = re.sub(r"[^a-z0-9-]+", "-", Path(row["run_config"]).parent.name.lower()).strip("-")
    return f"{matrix['name']}-{scene}-{row['name']}"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Expand and run an RL experiment matrix")
    sub = p.add_subparsers(dest="command", required=True)
    for name, help_text in (("plan", "Write experiment_plan.json"),
                            ("run", "Write the plan, then run its pending steps")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--matrix", required=True, help="Matrix JSON file")
        cmd.add_argument("--output-root", default=None,
                         help="Override the automatic timestamped output root")
        cmd.add_argument("--sim-binary", required=True, help="Path to mesh-sim executable")
        cmd.add_argument("--rows", default=None,
                         help="Comma-separated row names; omitted -> every row")
    status = sub.add_parser("status", help="Print the state of every planned step")
    status.add_argument("--output-root", required=True)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "status":
            plan = load_plan(args.output_root)
        else:
            matrix = load_matrix(args.matrix)
            selected_rows = _parse_rows(args.rows)
            root = args.output_root or str(automatic_output_root(
                automatic_run_label(matrix, selected_rows)))
            plan = build_plan(matrix, root, args.sim_binary, selected_rows)
            _write_plan(plan, root)
            if args.output_root is None:
                print(f"Experiment output: {Path(root).resolve()}", flush=True)
    except (ValueError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.command == "run":
        return run_plan(plan)
    _print_states({step["id"]: step_state(step) for step in plan["steps"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
