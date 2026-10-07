"""Load and validate evaluation manifests and flatten their episode records."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

EVAL_MANIFEST_NAME = "eval_manifest.json"
REQUIRED_MANIFEST_VERSION = 2
METRIC_SOURCE = {"kind": "telemetry_window", "warmup_excluded": False}

# metric -> (higher_is_better, comparable_across_reward_definitions), lexicographic
METRICS = {"connectivity": (True, True), "delivery_ratio": (True, True),
           "los_fraction": (True, True), "return": (True, False),
           "unroutable_fraction": (False, True)}
GROUP_METRICS = tuple(name for name, (_, shared) in METRICS.items() if shared)
CSV_METRICS = ("delivery_ratio", "connectivity", "los_fraction",
               "unroutable_fraction", "first_all_los_decision",
               "travel_m_total", "displacement_m_final")
CSV_COLUMNS = ("label", "training_seed", "model_sha256", "eval_dir", "policy", "seed",
               "status", "decisions", "return", "delivery_ratio", "connectivity",
               "los_fraction", "unroutable_fraction", "first_all_los_decision",
               "travel_m_total", "displacement_m_final",
               "mask_violations", "revalidated_slots_total", "actions_sha256",
               "held_out", "metric_source", "episode_dir", "summary_json", "error")


class ComparisonError(Exception):
    """A manifest set that cannot be compared; the caller writes nothing."""


@dataclass(frozen=True)
class Evaluation:
    """One loaded eval_manifest.json with its episode records keyed by policy and seed."""

    eval_dir: str
    manifest: dict
    sha256: str
    seeds: tuple
    records: dict

    label: str | None
    training_seed: int | None
    model_sha256: str | None
    held_out: bool | None


def _metric_value(record, metric):
    if metric == "return":
        return record.get("return")
    return (record.get("metrics") or {}).get(metric)


def _policy_order(names) -> list:
    """`model` first, every other policy lexicographically."""
    return (["model"] if "model" in names else []) + sorted(n for n in names if n != "model")


def _records(manifest: dict, path: Path) -> tuple:
    seeds = tuple(int(s) for s in manifest.get("seeds") or ())
    records: dict = {}
    for name, block in (manifest.get("policies") or {}).items():
        by_seed: dict = {}
        for episode in block.get("episodes") or ():
            seed = int(episode["seed"])
            if seed not in seeds:
                raise ComparisonError(f"{path}: policy {name} has a record for seed "
                                      f"{seed}, which is not in {list(seeds)}")
            if seed in by_seed:
                raise ComparisonError(
                    f"{path}: policy {name} has more than one record for seed {seed}")
            for metric in METRICS:
                value = _metric_value(episode, metric)
                if isinstance(value, float) and not math.isfinite(value):
                    raise ComparisonError(f"{path}: {name} seed {seed} {metric} is not "
                                          f"a finite number: {value!r}")
            by_seed[seed] = episode
        records[name] = by_seed
    return seeds, records


def load_evaluation(eval_dir) -> Evaluation | None:
    """Read one evaluation directory; None when it holds no readable manifest."""
    path = Path(eval_dir) / EVAL_MANIFEST_NAME
    if not path.is_file():
        return None
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(manifest, dict):
        return None
    version = manifest.get("eval_manifest_version")
    if version != REQUIRED_MANIFEST_VERSION:
        raise ComparisonError(
            f"{path} has eval_manifest_version {version!r}, not "
            f"{REQUIRED_MANIFEST_VERSION}; re-evaluate with current tooling")
    seeds, records = _records(manifest, path)
    return Evaluation(str(Path(eval_dir).resolve()), manifest,
                      hashlib.sha256(raw).hexdigest(), seeds, records,
                      manifest.get("label"),
                      (manifest.get("training") or {}).get("seed"),
                      (manifest.get("bundle") or {}).get("model_sha256"),
                      (manifest.get("seed_roles") or {}).get("held_out"))


def load_evaluations(entries) -> tuple:
    """Load every requested evaluation; unreadable ones become missing entries."""
    evaluations, missing, seen = [], [], set()
    for entry in entries:
        resolved = str(Path(entry["eval_dir"]).resolve())
        if resolved in seen:
            raise ComparisonError(f"evaluation directory {resolved} was given twice")
        seen.add(resolved)
        evaluation = load_evaluation(entry["eval_dir"])
        if evaluation is None:
            missing.append({"label": entry.get("label"),
                            "training_seed": entry.get("training_seed"),
                            "eval_dir": resolved, "reason": "no eval_manifest.json"})
        else:
            evaluations.append(evaluation)
    return evaluations, missing


def _row(evaluation: Evaluation, policy: str, seed: int, record: dict) -> dict:
    metrics = record.get("metrics") or {}
    row = {"label": evaluation.label, "training_seed": evaluation.training_seed,
           "model_sha256": evaluation.model_sha256, "eval_dir": evaluation.eval_dir,
           "policy": policy, "seed": seed, "status": record.get("status"),
           "decisions": record.get("decisions"), "return": record.get("return")}
    row.update({metric: metrics.get(metric) for metric in CSV_METRICS})
    row.update({"mask_violations": record.get("mask_violations"),
                "revalidated_slots_total": record.get("revalidated_slots_total"),
                "actions_sha256": record.get("actions_sha256"),
                "held_out": evaluation.held_out, "metric_source": METRIC_SOURCE["kind"],
                "episode_dir": record.get("episode_dir"),
                "summary_json": record.get("summary_json"), "error": record.get("error")})
    return row


def episode_rows(evaluations) -> list:
    """One flat row per (eval_dir, policy, seed) in canonical order."""
    return [_row(evaluation, policy, seed, evaluation.records[policy][seed])
            for evaluation in evaluations
            for policy in _policy_order(evaluation.records)
            for seed in sorted(evaluation.records[policy])]
