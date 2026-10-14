"""Gaps in policy/compare.py and policy/compare_inputs.py on hand-built eval manifests.

Contract sources: scripts/rl/policy/README.md ("Comparison", "Reading the output",
exit_code table), scripts/rl/policy/CLAUDE.md, scripts/rl/CLAUDE.md (version
mirroring). Differences are always model minus baseline; a seed is dropped when
either side is missing, not completed, or null; groups are keyed by scenario
digests and training settings and mixing them is an error.
"""

import copy
import json
import math
from pathlib import Path

import pytest

from scripts import stats
from scripts.rl.policy import compare_inputs
from scripts.rl.policy import evaluate as policy_evaluate
from scripts.rl.policy.compare import (ComparisonError, build_comparison, exit_code,
                                       load_evaluation, load_evaluations)

SEEDS = (11, 12, 13)


def _episode(seed, value, status="completed", **metric_overrides):
    """A completed episode whose metrics all equal `value` unless overridden."""
    if status != "completed" and value is not None:
        value = None
    metrics = {"delivery_ratio": value, "connectivity": value, "los_fraction": value,
               "unroutable_fraction": value, "first_all_los_decision": None,
               "travel_m_total": None, "displacement_m_final": None}
    metrics.update(metric_overrides)
    return {"seed": seed, "status": status, "error": None,
            "return": None if value is None else -float(value),
            "decisions": 0 if status == "not_run" else 10,
            "mask_violations": 0, "revalidated_slots_total": 0,
            "actions_sha256": "c" * 64, "episode_dir": f"/eval/{seed}",
            "summary_json": None, "metrics": metrics}


def _manifest(policies, *, seeds=SEEDS, label=None, training_seed=101,
              model_sha256="a" * 64, held_out=True, band="mmwave",
              model_selection="best", num_timesteps=256):
    """policies: {name: {seed: value | "failed" | "not_run" | None | dict}}."""
    blocks = {}
    for name, per_seed in policies.items():
        episodes = []
        for seed, value in per_seed.items():
            if isinstance(value, dict):
                episodes.append(value)
            elif isinstance(value, str):
                episodes.append(_episode(seed, None, status=value))
            else:
                episodes.append(_episode(seed, value))
        blocks[name] = {"episodes": episodes, "summary": {}}
    return {
        "eval_manifest_version": 2, "status": "completed", "label": label,
        "seeds": list(seeds),
        "seed_roles": {"training_seed": training_seed, "held_out": held_out},
        "training": {"algorithm": "MaskablePPO", "seed": training_seed,
                     "evaluation_seed": 201,
                     "scenario_identity": {"run_ini_sha256": "t" * 64,
                                           "nodes_json_sha256": "n" * 64,
                                           "buildings_json_sha256": None,
                                           "jammers_json_sha256": None},
                     "hyperparameters": {"total_timesteps": 256, "n_steps": 64,
                                         "gamma": 0.95, "ent_coef": 0.01,
                                         "eval_every_steps": 128, "eval_episodes": 1}},
        "bundle": {"model_selection": model_selection, "model_sha256": model_sha256,
                   "num_timesteps": num_timesteps},
        "selection": {"observation_preset": "local_links_v1",
                      "reward_components": ["delivery_ratio"], "reward_weights": [1.0]},
        "observation_schema": {"sha256": "o" * 64},
        "reward_schema": {"sha256": "r" * 64},
        "scenario_identity": {"run_ini_sha256": "s" * 64, "nodes_json_sha256": "n" * 64,
                              "buildings_json_sha256": None, "jammers_json_sha256": None},
        "band": band, "policies": blocks,
    }


def _write(path: Path, manifest: dict) -> str:
    path.mkdir(parents=True, exist_ok=True)
    (path / "eval_manifest.json").write_text(json.dumps(manifest, indent=2))
    return str(path)


def _load(tmp_path: Path, manifest: dict, name="eval"):
    return load_evaluation(_write(tmp_path / name, manifest))


def _pick(block, baseline="hold", metric="delivery_ratio"):
    return next(c for c in block["comparisons"]
                if c["baseline"] == baseline and c["metric"] == metric)


# 1. Version / constant mirroring ----------------------------------------------------

def test_required_manifest_version_tracks_the_writer():
    assert compare_inputs.REQUIRED_MANIFEST_VERSION == policy_evaluate.EVAL_MANIFEST_VERSION
    assert compare_inputs.EVAL_MANIFEST_NAME == policy_evaluate.EVAL_MANIFEST_NAME
    assert compare_inputs.METRIC_SOURCE == policy_evaluate.METRIC_SOURCE


def test_csv_metrics_are_the_metrics_evaluate_writes():
    assert compare_inputs.CSV_METRICS == policy_evaluate._METRIC_NAMES


# 2. load_evaluation validation ------------------------------------------------------


def test_evaluation_fields_come_from_their_documented_blocks(tmp_path):
    manifest = _manifest({"model": {s: 0.8 for s in SEEDS}}, label="L", training_seed=7,
                         model_sha256="f" * 64, held_out=False)
    evaluation = _load(tmp_path, manifest)
    raw = (tmp_path / "eval" / "eval_manifest.json").read_bytes()
    import hashlib
    assert evaluation.sha256 == hashlib.sha256(raw).hexdigest()
    assert (evaluation.label, evaluation.training_seed, evaluation.model_sha256,
            evaluation.held_out) == ("L", 7, "f" * 64, False)
    assert evaluation.seeds == SEEDS
    assert evaluation.eval_dir == str((tmp_path / "eval").resolve())


def test_load_evaluations_records_missing_and_refuses_the_same_dir_twice(tmp_path):
    present = _write(tmp_path / "a", _manifest({"model": {s: 0.8 for s in SEEDS}}))
    evaluations, missing = load_evaluations(
        [{"eval_dir": present, "label": "x", "training_seed": 1},
         {"eval_dir": str(tmp_path / "gone"), "label": "x", "training_seed": 2}])
    assert len(evaluations) == 1
    assert missing == [{"label": "x", "training_seed": 2,
                        "eval_dir": str((tmp_path / "gone").resolve()),
                        "reason": "no eval_manifest.json"}]
    with pytest.raises(ComparisonError, match="given twice"):
        load_evaluations([{"eval_dir": present},
                          {"eval_dir": str(Path(present) / ".." / "a")}])


def test_episode_rows_put_model_first_then_policies_lexicographically(tmp_path):
    manifest = _manifest({"random_valid": {s: 0.5 for s in SEEDS},
                          "hold": {s: 0.6 for s in SEEDS},
                          "model": {13: 0.9, 11: 0.8, 12: 0.7},
                          "a_custom": {s: 0.4 for s in SEEDS}})
    rows = compare_inputs.episode_rows([_load(tmp_path, manifest)])
    assert [row["policy"] for row in rows[::3]] == ["model", "a_custom", "hold",
                                                   "random_valid"]
    assert [row["seed"] for row in rows[:3]] == [11, 12, 13]
    assert all(list(row) == list(compare_inputs.CSV_COLUMNS) for row in rows)


# 3. Paired statistics: signs, exclusions, intervals ---------------------------------

def test_difference_is_model_minus_baseline_even_when_negative(tmp_path):
    evaluation = _load(tmp_path, _manifest({"model": {11: 0.5, 12: 0.4, 13: 0.3},
                                            "hold": {11: 0.7, 12: 0.7, 13: 0.7}}))
    entry = _pick(build_comparison([evaluation])["evaluations"][0])
    assert [p["difference"] for p in entry["pairs"]] == pytest.approx([-0.2, -0.3, -0.4])
    assert entry["mean_difference"] == pytest.approx(-0.3)
    assert entry["mean_difference"] == pytest.approx(entry["model_mean"]
                                                     - entry["baseline_mean"])
    assert entry["interval"]["low"] < entry["mean_difference"] < entry["interval"]["high"]


def test_lower_is_better_metric_is_not_sign_flipped(tmp_path):
    model = {s: _episode(s, 0.8, unroutable_fraction=0.1) for s in SEEDS}
    hold = {s: _episode(s, 0.6, unroutable_fraction=0.3) for s in SEEDS}
    payload = build_comparison([_load(tmp_path, _manifest({"model": model, "hold": hold}))])
    entry = _pick(payload["evaluations"][0], metric="unroutable_fraction")
    assert entry["mean_difference"] == pytest.approx(-0.2)
    assert payload["metrics"]["unroutable_fraction"]["higher_is_better"] is False


@pytest.mark.parametrize("model_value, hold_value, reasons", [
    (0.8, "absent", ["hold:missing"]),
    ("failed", "failed", ["model:failed", "hold:failed"]),
    ("not_run", 0.6, ["model:not_run"]),
    (None, None, ["model:metric_null", "hold:metric_null"]),
    ("absent", 0.6, ["model:missing"]),
])
def test_exclusion_reasons_name_each_side_model_first(tmp_path, model_value, hold_value,
                                                      reasons):
    model = {11: 0.8, 12: 0.7}
    hold = {11: 0.6, 12: 0.6}
    for side, value in ((model, model_value), (hold, hold_value)):
        if value == "absent":
            continue
        side[13] = value
    payload = build_comparison([_load(tmp_path, _manifest({"model": model, "hold": hold}))])
    entry = _pick(payload["evaluations"][0])
    assert entry["excluded"] == [{"seed": 13, "reasons": reasons}]
    assert (entry["n_expected"], entry["n_used"]) == (3, 2)
    assert payload["status"] == "incomplete"
    assert exit_code(payload) == 1


def test_two_pairs_use_df_one(tmp_path):
    evaluation = _load(tmp_path, _manifest({"model": {11: 0.9, 12: 0.7},
                                            "hold": {11: 0.6, 12: 0.6}}, seeds=(11, 12)))
    entry = _pick(build_comparison([evaluation])["evaluations"][0])
    interval = entry["interval"]
    std = math.sqrt(((0.3 - 0.2) ** 2 + (0.1 - 0.2) ** 2) / 1)
    assert (interval["n"], interval["df"], interval["level"]) == (2, 1, 0.95)
    assert interval["t"] == 12.706
    assert interval["half_width"] == pytest.approx(12.706 * std / math.sqrt(2))
    assert interval["low"] + interval["high"] == pytest.approx(2 * entry["mean_difference"])


@pytest.mark.parametrize("n", [31, 32, 45])
def test_paired_interval_uses_the_rounded_down_t_value(tmp_path, n):
    seeds = tuple(range(1, n + 1))
    model = {s: 0.5 + (s % 5) / 10 for s in seeds}
    hold = {s: 0.5 for s in seeds}
    evaluation = _load(tmp_path, _manifest({"model": model, "hold": hold}, seeds=seeds))
    entry = _pick(build_comparison([evaluation])["evaluations"][0])
    diffs = [model[s] - hold[s] for s in seeds]
    expected = stats.sample_stats(diffs, use_ci=True)
    assert entry["interval"]["t"] == stats.t_critical_95(n)
    assert entry["interval"]["half_width"] == pytest.approx(expected["ci95"])
    assert entry["std_difference"] == pytest.approx(expected["std"])


def test_an_explicit_baseline_absent_from_the_evaluation_is_all_missing(tmp_path):
    evaluation = _load(tmp_path, _manifest({"model": {s: 0.8 for s in SEEDS},
                                            "hold": {s: 0.6 for s in SEEDS}}))
    payload = build_comparison([evaluation], baselines=["random_valid"])
    entry = _pick(payload["evaluations"][0], baseline="random_valid")
    assert entry["n_used"] == 0 and entry["mean_difference"] is None
    assert entry["excluded"] == [{"seed": s, "reasons": ["random_valid:missing"]}
                                 for s in SEEDS]
    assert payload["status"] == "incomplete"


def test_default_baselines_are_every_non_model_policy_sorted(tmp_path):
    evaluation = _load(tmp_path, _manifest({"model": {s: 0.8 for s in SEEDS},
                                            "random_valid": {s: 0.5 for s in SEEDS},
                                            "hold": {s: 0.6 for s in SEEDS}}))
    block = build_comparison([evaluation])["evaluations"][0]
    baselines = [c["baseline"] for c in block["comparisons"]]
    assert baselines == sorted(baselines)
    assert set(baselines) == {"hold", "random_valid"}
    assert [c["metric"] for c in block["comparisons"][:5]] == sorted(compare_inputs.METRICS)


# 4. exit_code -----------------------------------------------------------------------

def _clean(tmp_path, name="eval", **kwargs):
    return _load(tmp_path, _manifest({"model": {s: 0.8 for s in SEEDS},
                                      "hold": {s: 0.6 for s in SEEDS}}, **kwargs), name)


def test_exit_code_zero_for_a_clean_complete_comparison(tmp_path):
    assert exit_code(build_comparison([_clean(tmp_path)])) == 0


def test_exit_code_incomplete_takes_priority_over_seed_overlap(tmp_path):
    manifest = _manifest({"model": {11: 0.8, 12: 0.8, 13: "failed"},
                          "hold": {s: 0.6 for s in SEEDS}}, held_out=False)
    payload = build_comparison([_load(tmp_path, manifest)])
    assert payload["status"] == "incomplete"
    assert exit_code(payload) == 1


def test_exit_code_two_for_seed_overlap(tmp_path):
    assert exit_code(build_comparison([_clean(tmp_path, held_out=False)])) == 2


@pytest.mark.parametrize("counter", ["mask_violations", "revalidated_slots_total"])
def test_exit_code_two_for_a_baseline_health_counter(tmp_path, counter):
    manifest = _manifest({"model": {s: 0.8 for s in SEEDS},
                          "hold": {s: 0.6 for s in SEEDS}})
    manifest["policies"]["hold"]["episodes"][2][counter] = 1
    payload = build_comparison([_load(tmp_path, manifest)])
    assert payload["status"] == "complete"
    assert exit_code(payload) == 2


def test_null_health_counters_count_as_zero(tmp_path):
    manifest = _manifest({"model": {s: 0.8 for s in SEEDS},
                          "hold": {s: 0.6 for s in SEEDS}})
    for episode in manifest["policies"]["hold"]["episodes"]:
        episode["mask_violations"] = None
        episode["revalidated_slots_total"] = None
    payload = build_comparison([_load(tmp_path, manifest)])
    assert payload["evaluations"][0]["health"]["hold"] == {"mask_violations": 0,
                                                           "revalidated_slots": 0}
    assert exit_code(payload) == 0


def test_a_missing_evaluation_makes_the_comparison_incomplete(tmp_path):
    payload = build_comparison([_clean(tmp_path)],
                               missing=[{"label": None, "training_seed": 9,
                                         "eval_dir": "/x", "reason": "no eval_manifest.json"}])
    assert payload["status"] == "incomplete"
    assert exit_code(payload) == 1


# 5. Groups across training runs ------------------------------------------------------

def _run(tmp_path, index, model, *, label="a", seeds=(11, 12), **kwargs):
    kwargs.setdefault("training_seed", 101 + index)
    kwargs.setdefault("model_sha256", chr(97 + index) * 64)
    manifest = _manifest({"model": model, "hold": {s: 0.6 for s in seeds}},
                         seeds=seeds, label=label, **kwargs)
    return _load(tmp_path, manifest, f"run-{index}")


def test_unlabelled_evaluations_form_no_group_and_labels_are_sorted(tmp_path):
    runs = [_run(tmp_path, 0, {11: 0.8, 12: 0.8}, label="zeta"),
            _run(tmp_path, 1, {11: 0.8, 12: 0.8}, label=None),
            _run(tmp_path, 2, {11: 0.8, 12: 0.8}, label="alpha")]
    payload = build_comparison(runs)
    assert [group["label"] for group in payload["groups"]] == ["alpha", "zeta"]
    assert len(payload["evaluations"]) == 3


def test_group_mean_difference_is_the_mean_of_per_run_means(tmp_path):
    runs = [_run(tmp_path, 0, {11: 0.7, 12: 0.9}),    # mean diff 0.2
            _run(tmp_path, 1, {11: 0.5, 12: 0.5})]    # mean diff -0.1
    entry = _pick(build_comparison(runs)["groups"][0])
    assert [r["mean_difference"] for r in entry["per_run"]] == pytest.approx([0.2, -0.1])
    assert entry["mean_difference"] == pytest.approx(0.05)
    assert entry["std_across_runs"] == pytest.approx(math.sqrt(0.045))
    assert entry["interval"]["df"] == 1 and entry["interval"]["t"] == 12.706
    assert entry["interval"]["fixed_model"] is False


def test_runs_without_a_common_usable_seed_yield_no_group_value(tmp_path):
    runs = [_run(tmp_path, 0, {11: "failed", 12: 0.8}),
            _run(tmp_path, 1, {11: 0.8, 12: "failed"})]
    payload = build_comparison(runs)
    group = payload["groups"][0]
    assert group["runs_used"] == 2
    entry = _pick(group)
    assert entry["common_seeds"] == []
    assert entry["seeds_dropped_for_commonality"] == [11, 12]
    assert [r["mean_difference"] for r in entry["per_run"]] == [None, None]
    assert entry["mean_difference"] is None and entry["interval"] is None
    assert entry["interval_omitted"] == "fewer than 2 usable runs"


def test_a_run_with_no_usable_pairs_is_excluded_from_its_group(tmp_path):
    runs = [_run(tmp_path, 0, {11: 0.8, 12: 0.8}),
            _run(tmp_path, 1, {11: "failed", 12: "failed"})]
    group = build_comparison(runs)["groups"][0]
    assert group["runs_used"] == 1 and group["runs_expected"] == 2
    assert group["excluded_runs"] == [{"training_seed": 102, "reason": "no_usable_pairs"}]


def test_runs_expected_can_be_given_per_label(tmp_path):
    runs = [_run(tmp_path, 0, {11: 0.8, 12: 0.8})]
    group = build_comparison(runs, runs_expected={"a": 5})["groups"][0]
    assert (group["runs_expected"], group["runs_used"]) == (5, 1)


def test_runs_without_a_training_seed_may_share_a_label(tmp_path):
    runs = [_run(tmp_path, 0, {11: 0.8, 12: 0.8}, training_seed=None),
            _run(tmp_path, 1, {11: 0.8, 12: 0.8}, training_seed=None)]
    assert build_comparison(runs)["groups"][0]["runs_used"] == 2


def test_final_runs_ignore_bundle_timesteps_in_the_group_key(tmp_path):
    runs = [_run(tmp_path, 0, {11: 0.8, 12: 0.8}, model_selection="final",
                 num_timesteps=100),
            _run(tmp_path, 1, {11: 0.8, 12: 0.8}, model_selection="final",
                 num_timesteps=None)]
    assert build_comparison(runs)["groups"][0]["runs_used"] == 2


def _mutated_pair(tmp_path, mutate):
    first = _manifest({"model": {11: 0.8, 12: 0.8}, "hold": {11: 0.6, 12: 0.6}},
                      seeds=(11, 12), label="a")
    second = copy.deepcopy(first)
    second["training"]["seed"] = 102
    second["bundle"]["model_sha256"] = "b" * 64
    mutate(second)
    return [_load(tmp_path, first, "run-0"), _load(tmp_path, second, "run-1")]


def test_different_labels_may_differ_in_settings(tmp_path):
    def other_label(manifest):
        manifest["label"] = "b"
        manifest["band"] = "sub-6"

    payload = build_comparison(_mutated_pair(tmp_path, other_label))
    assert [g["label"] for g in payload["groups"]] == ["a", "b"]
