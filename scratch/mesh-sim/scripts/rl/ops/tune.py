#!/usr/bin/env python3
"""Search the already-wired PPO knobs with Optuna over one matrix row and seed."""

import argparse
import copy
import json
import math
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

from scripts.rl.cli_common import (MANIFEST_NAME as TRAIN_MANIFEST_NAME,
                                   automatic_output_root, now_iso,
                                   package_versions, sha256_file, write_json)
from scripts.rl.experiment import build_plan, load_matrix
from scripts.sim_support import find_mesh_root

STUDY_VERSION = 1
STUDY_MANIFEST_VERSION = 1
TRIAL_VERSION = 1
STUDY_MANIFEST_NAME = "study_manifest.json"
TRIAL_NAME = "trial.json"
PIN_NAME = "requirements-tuning.txt"
N_STARTUP_TRIALS = 10
MAX_TRIALS = 50

_TOP_KEYS = ("study_version", "name", "description", "matrix", "row",
             "training_seed", "sampler", "n_trials", "search_space")
_SAMPLER_KEYS = ("type", "seed")
_ENTRY_KEYS = {"categorical": ("type", "choices"),
               "int": ("type", "low", "high", "log"),
               "float": ("type", "low", "high", "log")}
# Only these reach MaskablePPO today (mask_ppo.py); widening needs the whole
# constructor/CLI/manifest chain, tracked by TODO-RL-TUNE-1.
_SEARCHABLE = {"n_steps": ("int", "categorical"), "gamma": ("float",),
               "ent_coef": ("float",)}
# Seed-carrying flags of the train step built by experiment._train_args.
_SEED_FLAGS = ("--seed", "--eval-seed")
_INSTALL_HINT = ("install with: .venv/bin/python -m pip install -r "
                 f"{PIN_NAME}")


def _check_keys(mapping, allowed: tuple[str, ...], where: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{where} must be a JSON object")
    unknown = sorted(key for key in mapping if key not in allowed)
    if unknown:
        raise ValueError(f"{where} has unknown keys {unknown}; "
                         f"valid keys: {list(allowed)}")


def _int(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an integer, got {value!r}")
    return value


def _number(value, where: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    return value


def _bounds(key: str, entry: dict, where: str) -> None:
    low, high, log = entry["low"], entry["high"], entry["log"]
    if key == "n_steps":
        _int(low, f"{where}.low")
        _int(high, f"{where}.high")
        if low < 2:
            raise ValueError(f"{where}.low must be >= 2, got {low}")
    if key == "gamma" and not (0 < low and high <= 1):
        raise ValueError(f"{where} must stay inside (0, 1], got low={low}, high={high}")
    if key == "ent_coef" and low < 0:
        raise ValueError(f"{where}.low must be >= 0, got {low}")
    if log and low <= 0:
        raise ValueError(f"{where}.log requires low > 0, got {low}")


def _entry(key: str, raw) -> dict:
    where = f"search_space.{key}"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a JSON object")
    kind = raw.get("type")
    if kind not in _SEARCHABLE[key]:
        raise ValueError(f"{where}.type must be one of {list(_SEARCHABLE[key])}, "
                         f"got {kind!r}")
    _check_keys(raw, _ENTRY_KEYS[kind], where)
    if kind == "categorical":
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{where}.choices must be a non-empty array")
        values = [_int(choice, f"{where}.choices") for choice in choices]
        if len(set(values)) != len(values):
            raise ValueError(f"{where}.choices must be distinct, got {values}")
        if any(value < 2 for value in values):
            raise ValueError(f"{where}.choices must all be >= 2, got {values}")
        return {"type": kind, "choices": values}

    entry = {"type": kind, "low": _number(raw.get("low"), f"{where}.low"),
             "high": _number(raw.get("high"), f"{where}.high"),
             "log": raw.get("log", False)}
    if not isinstance(entry["log"], bool):
        raise ValueError(f"{where}.log must be true or false, got {entry['log']!r}")
    if not entry["low"] < entry["high"]:
        raise ValueError(f"{where} must have low < high, got low={entry['low']}, "
                         f"high={entry['high']}")
    _bounds(key, entry, where)
    return entry


def _search_space(raw) -> dict:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("search_space must be a non-empty JSON object")
    for key in raw:
        if key == "total_timesteps":
            raise ValueError("search_space.total_timesteps is not searchable: the "
                             "training budget is fixed per study and comes from the "
                             "matrix")
        if key not in _SEARCHABLE:
            raise ValueError(f"search_space.{key} is not wired to MaskablePPO; "
                             f"searchable knobs: {sorted(_SEARCHABLE)}; "
                             "see the open TODO-RL-TUNE-1")
    return {key: _entry(key, raw[key]) for key in raw}


def _sampler(raw) -> dict:
    _check_keys(raw, _SAMPLER_KEYS, "sampler")
    if raw.get("type") != "tpe":
        raise ValueError(f"sampler.type must be 'tpe', got {raw.get('type')!r}")
    return {"type": "tpe", "seed": _int(raw.get("seed"), "sampler.seed")}


def _matrix(raw, study_path: Path) -> dict:
    if not isinstance(raw, str) or not raw:
        raise ValueError("matrix must be a path to a matrix JSON file")
    path = Path(raw)
    if not path.is_absolute():
        path = find_mesh_root() / path
    matrix = load_matrix(path)
    training = matrix["training"]
    if matrix["seeds"]["model_selection"] is None:
        raise ValueError(f"{path}: the objective needs seeds.model_selection")
    budget = training.get("total_timesteps")
    if budget is None:
        raise ValueError(f"{path}: training.total_timesteps is required")
    cadence = training.get("eval_every_steps", 0)
    if not 0 < cadence <= budget:
        raise ValueError(f"{path}: training.eval_every_steps must satisfy "
                         f"0 < {cadence} <= total_timesteps {budget}, or no model "
                         "selection runs and every objective is null")
    return matrix


def load_study(path) -> dict:
    """Load and fully validate a study spec; nothing is launched or created."""
    study_path = Path(path).resolve()
    body = json.loads(study_path.read_text())
    _check_keys(body, _TOP_KEYS, "study")
    if body.get("study_version") != STUDY_VERSION:
        raise ValueError(f"study_version must be {STUDY_VERSION}, got "
                         f"{body.get('study_version')!r}")
    name = body.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"name must be a non-empty string, got {name!r}")
    description = body.get("description", "")
    if not isinstance(description, str):
        raise ValueError("description must be a string")

    matrix = _matrix(body.get("matrix"), study_path)
    row = body.get("row")
    known = [entry["name"] for entry in matrix["rows"]]
    if row not in known:
        raise ValueError(f"row {row!r} is not in {matrix['path']}; available: {known}")
    seed = _int(body.get("training_seed"), "training_seed")
    if seed not in matrix["seeds"]["training"]:
        raise ValueError(f"training_seed {seed} is not in seeds.training "
                         f"{matrix['seeds']['training']}")
    n_trials = _int(body.get("n_trials"), "n_trials")
    if not 1 <= n_trials <= MAX_TRIALS:
        raise ValueError(f"n_trials must be between 1 and {MAX_TRIALS}, got {n_trials}")

    return {"path": str(study_path), "sha256": sha256_file(study_path), "body": body,
            "name": name, "description": description, "matrix": matrix, "row": row,
            "training_seed": seed, "sampler": _sampler(body.get("sampler", {})),
            "n_trials": n_trials,
            "search_space": _search_space(body.get("search_space"))}


def read_tuning_pin(path) -> str:
    """Read the single optuna pin; read_pins is not reused because it enforces DIRECT_DEPS."""
    for line in Path(path).read_text().splitlines():
        name, separator, version = line.split("#", 1)[0].strip().partition("==")
        if name == "optuna" and separator and version:
            return version
    raise ValueError(f"{path} must contain one exact pin: optuna==<version>")


def load_optuna(pin_file=None):
    """Import Optuna lazily and refuse any version other than the pinned one."""
    pin_file = Path(pin_file) if pin_file else find_mesh_root() / PIN_NAME
    pin = read_tuning_pin(pin_file)
    try:
        import optuna
    except ImportError as exc:
        raise ValueError(f"optuna is not installed ({exc}); {_INSTALL_HINT}") from exc
    if optuna.__version__ != pin:
        raise ValueError(f"optuna {optuna.__version__} is installed but {pin_file} "
                         f"pins optuna=={pin}; {_INSTALL_HINT}")
    return optuna


class _OptunaStudy:
    """ask/tell over an in-memory Optuna study; the only Optuna-aware object here."""

    def __init__(self, space: dict, seed: int, pin_file=None):
        self._optuna = load_optuna(pin_file)
        self.version = self._optuna.__version__
        self._optuna.logging.set_verbosity(self._optuna.logging.WARNING)
        self._distributions = _distributions(self._optuna, space)
        self._study = self._optuna.create_study(
            direction="maximize",
            sampler=self._optuna.samplers.TPESampler(
                seed=seed, n_startup_trials=N_STARTUP_TRIALS))

    def ask(self) -> tuple:
        trial = self._study.ask(self._distributions)
        return trial, dict(trial.params)

    def tell(self, handle, objective: float | None) -> None:
        if objective is None:
            self._study.tell(handle, state=self._optuna.trial.TrialState.FAIL)
        else:
            self._study.tell(handle, objective)


class _CallableStudy:
    """Adapter so a plain `sampler(space, number) -> params` callable can drive trials."""

    def __init__(self, sampler, space: dict):
        self.version = None
        self._sampler = sampler
        self._space = space
        self._number = 0

    def ask(self) -> tuple:
        params = self._sampler(self._space, self._number)
        self._number += 1
        return None, dict(params)

    def tell(self, handle, objective: float | None) -> None:
        return None


def _distributions(optuna, space: dict) -> dict:
    built = {}
    for key, entry in space.items():
        if entry["type"] == "categorical":
            built[key] = optuna.distributions.CategoricalDistribution(entry["choices"])
        elif entry["type"] == "int":
            built[key] = optuna.distributions.IntDistribution(
                low=entry["low"], high=entry["high"], log=entry["log"])
        else:
            built[key] = optuna.distributions.FloatDistribution(
                low=entry["low"], high=entry["high"], log=entry["log"])
    return built


def _seed_values(args: list[str]) -> set[str]:
    """Values that follow a seed-bearing flag; a sampled number is not a seed."""
    return {args[position + 1] for position, token in enumerate(args)
            if token in _SEED_FLAGS and position + 1 < len(args)}


def build_trial(spec: dict, params: dict, trial_dir, sim_binary: str) -> dict:
    """Build one trial's training command through build_plan; no plan file is written."""
    matrix = copy.deepcopy(spec["matrix"])
    matrix["training"].update(params)
    matrix["seeds"]["training"] = [spec["training_seed"]]
    plan = build_plan(matrix, trial_dir, sim_binary, rows=[spec["row"]])
    step = next(step for step in plan["steps"] if step["kind"] == "train")
    forbidden = {str(seed) for seed in spec["matrix"]["seeds"]["held_out"]}
    if forbidden & _seed_values(step["args"]):
        raise ValueError("refusing a trial command that names a held-out seed")
    return {"module": step["module"], "args": step["args"],
            "train_dir": step["output_dir"], "training": matrix["training"]}


def _command(trial: dict) -> list[str]:
    return [sys.executable, "-m", trial["module"], *trial["args"]]


def _subprocess_execute(trial: dict) -> int:
    """Run one trial's training in a fresh process from the mesh root."""
    return subprocess.run(_command(trial), cwd=str(find_mesh_root())).returncode


def _objective(train_dir) -> tuple[float | None, str | None]:
    manifest = Path(train_dir) / TRAIN_MANIFEST_NAME
    try:
        value = json.loads(manifest.read_text()).get("best_mean_reward")
    except (OSError, ValueError) as exc:
        return None, f"unreadable {manifest}: {type(exc).__name__}: {exc}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, (f"best_mean_reward is {value!r}: no model-selection evaluation "
                      "produced a best model")
    if not math.isfinite(value):
        return None, f"best_mean_reward is {value!r}"
    return float(value), None


def _fixed_training(spec: dict) -> dict:
    return {key: value for key, value in spec["matrix"]["training"].items()
            if key not in spec["search_space"]}


def _manifest(spec: dict, study, sim_binary: str, started_at: str) -> dict:
    matrix = spec["matrix"]
    return {
        "study_manifest_version": STUDY_MANIFEST_VERSION,
        "status": "running",
        "spec": {"path": spec["path"], "sha256": spec["sha256"], "body": spec["body"]},
        "matrix": {"path": matrix["path"], "sha256": matrix["sha256"],
                   "name": matrix["name"]},
        "row": spec["row"],
        "seed_roles": {"training": spec["training_seed"],
                       "model_selection": matrix["seeds"]["model_selection"],
                       "held_out_used": False},
        "objective": {"name": "best_mean_reward", "direction": "maximize",
                      "source": TRAIN_MANIFEST_NAME, "seed_role": "model_selection",
                      "episodes_per_evaluation": 1},
        "sampler": {**spec["sampler"], "n_startup_trials": N_STARTUP_TRIALS},
        "fixed_training": _fixed_training(spec),
        "optuna_version": study.version,
        "package_versions": package_versions(),
        "python_version": platform.python_version(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "sim_binary": sim_binary,
        "trials": [],
        "best": None,
        "started_at": started_at,
        "ended_at": None,
    }


def _best(records: list[dict], trainings: dict) -> dict | None:
    complete = [record for record in records if record["state"] == "complete"]
    if not complete:
        return None
    best = max(complete, key=lambda record: record["objective"])
    return {"number": best["number"], "objective": best["objective"],
            "training": trainings[best["number"]]}


def _params_text(params: dict) -> str:
    return " ".join(f"{key}={value!r}" for key, value in params.items())


def _run_trials(spec: dict, study, output_root, sim_binary: str, execute) -> int:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(spec, study, sim_binary, now_iso())
    write_json(root / STUDY_MANIFEST_NAME, manifest)

    summaries: list[dict] = []
    trainings: dict[int, dict] = {}
    for number in range(spec["n_trials"]):
        handle, params = study.ask()
        trial_dir = root / "trials" / f"trial-{number:04d}"
        trial = build_trial(spec, params, trial_dir, sim_binary)
        started_at = now_iso()
        trial_dir.mkdir(parents=True, exist_ok=True)
        code = execute(trial)
        if code == 0:
            objective, failure = _objective(trial["train_dir"])
        else:
            objective, failure = None, f"training exited {code}"
        study.tell(handle, objective)

        trainings[number] = trial["training"]
        write_json(trial_dir / TRIAL_NAME, {
            "trial_version": TRIAL_VERSION, "number": number, "params": params,
            "training": trial["training"], "module": trial["module"],
            "args": trial["args"], "train_dir": trial["train_dir"],
            "state": "complete" if failure is None else "failed",
            "objective": objective, "failure": failure, "exit_code": code,
            "started_at": started_at, "ended_at": now_iso()})
        summaries.append({"number": number,
                          "state": "complete" if failure is None else "failed",
                          "objective": objective, "params": params})
        print(f"trial {number:04d} {summaries[-1]['state']} "
              f"objective={objective} {_params_text(params)}"
              + (f" ({failure})" if failure else ""))

        manifest["trials"] = summaries
        manifest["best"] = _best(summaries, trainings)
        write_json(root / STUDY_MANIFEST_NAME, manifest)

    failed = [record["number"] for record in summaries if record["state"] == "failed"]
    manifest["status"] = "failed" if failed else "completed"
    manifest["ended_at"] = now_iso()
    write_json(root / STUDY_MANIFEST_NAME, manifest)
    if failed:
        print(f"{len(failed)} of {spec['n_trials']} trials produced no objective: "
              f"{failed}", file=sys.stderr)
    return 1 if failed else 0


def _dry_run(spec: dict, study, output_root, sim_binary: str) -> int:
    previewed = min(spec["n_trials"], N_STARTUP_TRIALS)
    root = Path(output_root).resolve()
    for number in range(previewed):
        _, params = study.ask()
        trial = build_trial(spec, params, root / "trials" / f"trial-{number:04d}",
                            sim_binary)
        print(f"trial {number:04d} params: {_params_text(params)}")
        print(f"trial {number:04d} command: "
              + " ".join(shlex.quote(part) for part in _command(trial)))
    if spec["n_trials"] > previewed:
        print(f"the remaining {spec['n_trials'] - previewed} trials depend on earlier "
              "objectives and cannot be previewed")
    print("dry run: no output directory, record, or process was created")
    return 0


def _check_output_root(output_root, dry_run: bool) -> None:
    if dry_run:
        return
    root = Path(output_root).resolve()
    for name in (STUDY_MANIFEST_NAME, "trials"):
        if (root / name).exists():
            raise ValueError(f"{root / name} already exists; studies are not resumed, "
                             "choose a new --output-root")


def _check_binary(sim_binary: str, dry_run: bool) -> str:
    binary = Path(sim_binary).expanduser()
    if not dry_run and not (binary.is_file() and os.access(binary, os.X_OK)):
        raise ValueError(f"--sim-binary {binary} is not an executable file")
    return str(binary)


def _open_study(spec: dict, sampler):
    if sampler is not None:
        return _CallableStudy(sampler, spec["search_space"])
    return _OptunaStudy(spec["search_space"], spec["sampler"]["seed"])


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Optuna smoke over the wired PPO knobs for one matrix row and seed")
    p.add_argument("--study", required=True, help="Study spec JSON file")
    p.add_argument("--output-root", default=None,
                   help="Override the automatic timestamped study output root")
    p.add_argument("--sim-binary", required=True, help="Path to mesh-sim executable")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the startup parameter sets and commands; create nothing")
    return p


def main(argv=None, sampler=None, execute=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        spec = load_study(args.study)
        if args.output_root is None:
            args.output_root = str(automatic_output_root(spec["name"]))
            if not args.dry_run:
                print(f"Tuning output: {Path(args.output_root).resolve()}", flush=True)
        _check_output_root(args.output_root, args.dry_run)
        binary = _check_binary(args.sim_binary, args.dry_run)
        study = _open_study(spec, sampler)
        if args.dry_run:
            return _dry_run(spec, study, args.output_root, binary)
        return _run_trials(spec, study, args.output_root, binary,
                           execute or _subprocess_execute)
    except (ValueError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
