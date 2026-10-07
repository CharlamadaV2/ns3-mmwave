"""Compute paired model-minus-baseline statistics and the comparison payload."""

import math

from scripts.rl.policy.compare_inputs import (CSV_COLUMNS, CSV_METRICS, EVAL_MANIFEST_NAME,
                                              GROUP_METRICS, METRIC_SOURCE, METRICS,
                                              REQUIRED_MANIFEST_VERSION, ComparisonError,
                                              Evaluation, _metric_value, _policy_order,
                                              episode_rows, load_evaluation,
                                              load_evaluations)
from scripts.stats import t_critical_95

__all__ = ["ACROSS_RUNS_KIND", "COMPARISON_VERSION", "CSV_COLUMNS", "CSV_METRICS",
           "EVAL_MANIFEST_NAME", "GROUP_METRICS", "METRICS", "METRIC_SOURCE",
           "PAIRED_KIND", "PRIMARY_METRIC", "REQUIRED_MANIFEST_VERSION",
           "ComparisonError", "Evaluation", "build_comparison", "episode_rows",
           "exit_code", "load_evaluation", "load_evaluations"]

COMPARISON_VERSION = 1
PRIMARY_METRIC = "delivery_ratio"
PAIRED_KIND = "paired_t_across_evaluation_seeds"
ACROSS_RUNS_KIND = "t_across_training_runs"

_SHA_FIELDS = ("run_ini_sha256", "nodes_json_sha256", "buildings_json_sha256",
               "jammers_json_sha256")
_TRAINING_SETTINGS = ("total_timesteps", "n_steps", "gamma", "ent_coef",
                      "eval_every_steps", "eval_episodes")


def _sample_std(values) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _interval(values, kind: str, source: str, fixed_model: bool, omitted: str) -> dict:
    if len(values) < 2:
        return {"interval": None, "interval_omitted": omitted}
    n = len(values)
    mean = sum(values) / n
    t = t_critical_95(n)
    half = t * _sample_std(values) / math.sqrt(n)
    return {"interval": {"kind": kind, "variability_source": source,
                         "fixed_model": fixed_model, "level": 0.95, "n": n,
                         "df": n - 1, "t": t, "half_width": half,
                         "low": mean - half, "high": mean + half}}


def _exclusion(policy: str, record, metric: str):
    if record is None:
        return f"{policy}:missing"
    status = record.get("status")
    if status != "completed":
        return f"{policy}:{status}"
    if _metric_value(record, metric) is None:
        return f"{policy}:metric_null"
    return None


def _pairs(evaluation: Evaluation, baseline: str, metric: str) -> tuple:
    model = evaluation.records.get("model") or {}
    other = evaluation.records.get(baseline) or {}
    pairs, excluded = [], []
    for seed in sorted(evaluation.seeds):
        reasons = [reason for reason in (_exclusion("model", model.get(seed), metric),
                                         _exclusion(baseline, other.get(seed), metric))
                   if reason is not None]
        if reasons:
            excluded.append({"seed": seed, "reasons": reasons})
            continue
        left = _metric_value(model[seed], metric)
        right = _metric_value(other[seed], metric)
        pairs.append({"seed": seed, "model": left, "baseline": right,
                      "difference": left - right})
    return pairs, excluded


def _comparison(evaluation: Evaluation, baseline: str, metric: str) -> dict:
    pairs, excluded = _pairs(evaluation, baseline, metric)
    diffs = [pair["difference"] for pair in pairs]
    n = len(diffs)
    std = _sample_std(diffs) if n > 1 else None
    result = {"baseline": baseline, "metric": metric,
              "n_expected": len(evaluation.seeds), "n_used": n,
              "model_mean": (sum(p["model"] for p in pairs) / n) if n else None,
              "baseline_mean": (sum(p["baseline"] for p in pairs) / n) if n else None,
              "mean_difference": (sum(diffs) / n) if n else None,
              "std_difference": std,
              "zero_variance": bool(n > 1 and std == 0.0)}
    result.update(_interval(diffs, PAIRED_KIND, "evaluation_seeds", True,
                            "fewer than 2 usable pairs"))
    result["pairs"] = pairs
    result["excluded"] = excluded
    return result


def _health(evaluation: Evaluation) -> dict:
    health = {}
    for policy in _policy_order(evaluation.records):
        records = evaluation.records[policy].values()
        health[policy] = {
            "mask_violations": sum(int(r.get("mask_violations") or 0) for r in records),
            "revalidated_slots": sum(int(r.get("revalidated_slots_total") or 0)
                                     for r in records)}
    return health


def _group_key(evaluation: Evaluation) -> dict:
    manifest = evaluation.manifest
    selection = manifest.get("selection") or {}
    training = manifest.get("training") or {}
    hyper = training.get("hyperparameters") or {}
    key = {"observation_preset": selection.get("observation_preset"),
           "reward_components": list(selection.get("reward_components") or ()),
           "reward_weights": list(selection.get("reward_weights") or ()),
           "observation_schema_sha256": (manifest.get("observation_schema") or {}).get("sha256"),
           "reward_schema_sha256": (manifest.get("reward_schema") or {}).get("sha256"),
           "band": manifest.get("band"),
           "model_selection": (manifest.get("bundle") or {}).get("model_selection"),
           "seeds": sorted(evaluation.seeds),
           "training_algorithm": training.get("algorithm"),
           "model_selection_seed": training.get("evaluation_seed")}
    identity = manifest.get("scenario_identity") or {}
    training_identity = training.get("scenario_identity") or {}
    for field in _SHA_FIELDS:
        key[f"scenario_{field}"] = identity.get(field)
        key[f"training_scenario_{field}"] = training_identity.get(field)
    for setting in _TRAINING_SETTINGS:
        key[f"training_{setting}"] = hyper.get(setting)
    if key["model_selection"] not in ("final", "best", None):
        key["bundle_num_timesteps"] = (manifest.get("bundle") or {}).get("num_timesteps")
    for policy, block in (manifest.get("policies") or {}).items():
        if isinstance(block, dict) and "baseline" in block:
            baseline = block["baseline"] or {}
            key[f"baseline_{policy}_fingerprint"] = baseline.get("fingerprint")
    return key


def _check_group(label: str, members: list) -> None:
    first, reference = members[0], _group_key(members[0])
    for evaluation in members[1:]:
        key = _group_key(evaluation)
        differing = sorted(name for name in set(reference) | set(key)
                           if reference.get(name) != key.get(name))
        if differing:
            raise ComparisonError(
                f"label {label!r} groups evaluations that differ in {differing}: "
                f"{first.eval_dir} and {evaluation.eval_dir}")
    for name, values in (("training.seed", [e.training_seed for e in members]),
                         ("bundle.model_sha256", [e.model_sha256 for e in members])):
        present = [v for v in values if v is not None]
        if len(set(present)) != len(present):
            raise ComparisonError(
                f"label {label!r} needs a distinct {name} per training run, got {values}")


def _group_comparison(used: list, baseline: str, metric: str) -> dict:
    usable = {evaluation.eval_dir: {pair["seed"]: pair["difference"]
                                    for pair in _pairs(evaluation, baseline, metric)[0]}
              for evaluation in used}
    seen = [set(pairs) for pairs in usable.values()]
    common = sorted(set.intersection(*seen)) if seen else []
    dropped = sorted(set.union(*seen) - set(common)) if seen else []
    per_run = [{"training_seed": evaluation.training_seed,
                "eval_dir": evaluation.eval_dir,
                "mean_difference": (sum(usable[evaluation.eval_dir][s] for s in common)
                                    / len(common)) if common else None}
               for evaluation in used]
    values = [entry["mean_difference"] for entry in per_run if entry["mean_difference"] is not None]
    result = {"baseline": baseline, "metric": metric, "common_seeds": common,
              "seeds_dropped_for_commonality": dropped, "per_run": per_run,
              "mean_difference": (sum(values) / len(values)) if values else None,
              "std_across_runs": _sample_std(values) if len(values) > 1 else None}
    result.update(_interval(values, ACROSS_RUNS_KIND, "training_runs", False,
                            "fewer than 2 usable runs"))
    return result


def _groups(evaluations: list, missing: list, baselines: list, runs_expected) -> list:
    groups = []
    for label in sorted({e.label for e in evaluations if e.label is not None}):
        members = [e for e in evaluations if e.label == label]
        _check_group(label, members)
        excluded_runs = [{"training_seed": entry.get("training_seed"), "reason": "missing"}
                         for entry in missing if entry.get("label") == label]
        used = []
        for evaluation in members:
            usable = any(_pairs(evaluation, baseline, metric)[0]
                         for baseline in baselines for metric in GROUP_METRICS)
            if evaluation.held_out is not True:
                excluded_runs.append({"training_seed": evaluation.training_seed,
                                      "reason": "seed_overlap"})
            elif not usable:
                excluded_runs.append({"training_seed": evaluation.training_seed,
                                      "reason": "no_usable_pairs"})
            else:
                used.append(evaluation)
        expected = (runs_expected.get(label, len(members))
                    if runs_expected is not None else len(members))
        groups.append({"label": label, "runs_expected": expected,
                       "runs_used": len(used), "excluded_runs": excluded_runs,
                       "comparisons": [_group_comparison(used, baseline, metric)
                                       for baseline in baselines
                                       for metric in GROUP_METRICS]})
    return groups


def build_comparison(evaluations: list, missing=(), baselines=None,
                     runs_expected=None) -> dict:
    """Assemble the full comparison payload; raises ComparisonError on a refusal."""
    missing = list(missing)
    if baselines is None:
        baselines = sorted({name for e in evaluations for name in e.records
                            if name != "model"})
    baselines = list(baselines)
    blocks = []
    for evaluation in evaluations:
        blocks.append({"eval_dir": evaluation.eval_dir, "label": evaluation.label,
                       "training_seed": evaluation.training_seed,
                       "held_out": evaluation.held_out,
                       "health": _health(evaluation),
                       "comparisons": [_comparison(evaluation, baseline, metric)
                                       for baseline in baselines
                                       for metric in sorted(METRICS)]})
    incomplete = bool(missing) or any(
        comparison["n_used"] != comparison["n_expected"]
        for block in blocks for comparison in block["comparisons"])
    return {"comparison_version": COMPARISON_VERSION,
            "status": "incomplete" if incomplete else "complete",
            "metric_source": dict(METRIC_SOURCE), "primary_metric": PRIMARY_METRIC,
            "metrics": {name: {"higher_is_better": higher,
                               "comparable_across_reward_definitions": shared}
                        for name, (higher, shared) in METRICS.items()},
            "inputs": [{"eval_dir": e.eval_dir, "eval_manifest_sha256": e.sha256,
                        "manifest_status": e.manifest.get("status"), "label": e.label,
                        "training_seed": e.training_seed,
                        "model_sha256": e.model_sha256, "held_out": e.held_out}
                       for e in evaluations],
            "missing_evaluations": missing,
            "evaluations": blocks,
            "groups": _groups(evaluations, missing, baselines, runs_expected)}


def exit_code(comparison: dict) -> int:
    """0 complete and clean, 2 health counters or seed overlap, 1 incomplete."""
    if comparison["status"] != "complete":
        return 1
    for block in comparison["evaluations"]:
        if block["held_out"] is False:
            return 2
        if any(sum(counters.values()) for counters in block["health"].values()):
            return 2
    return 0
