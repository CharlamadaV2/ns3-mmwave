"""Shared statistics, paired comparison, grouping, and the compare CLI on hand-built manifests."""

import csv
import json
from pathlib import Path

import pytest

from scripts import stats
from scripts.rl import compare as compare_cli
from scripts.rl.policy import compare as policy_compare
from scripts.rl.policy.compare import (ACROSS_RUNS_KIND, CSV_COLUMNS, PAIRED_KIND,
                                       ComparisonError, build_comparison, load_evaluation)

SEEDS = (11, 12, 13)


def _episode(seed: int, value, status: str = "completed") -> dict:
    """One episode record; a non-float `value` names a non-completed status."""
    if isinstance(value, str):
        status, value = value, None
    metrics = {"delivery_ratio": value,
               "connectivity": None if value is None else 1.0,
               "los_fraction": None if value is None else 0.5,
               "unroutable_fraction": None if value is None else 0.0,
               "first_all_los_decision": None if value is None else 4}
    return {"seed": seed, "status": status,
            "error": "RuntimeError: sim died" if status == "failed" else None,
            "return": None if status == "not_run" else -1.0 - seed / 100.0,
            "decisions": 0 if status == "not_run" else 20,
            "mask_violations": None if value is None else 0,
            "revalidated_slots_total": None if value is None else 0,
            "actions_sha256": None if value is None else "c" * 64,
            "episode_dir": f"/eval/{seed}", "summary_json": None, "metrics": metrics}


def _policy_block(values: dict) -> dict:
    episodes = [_episode(seed, value) for seed, value in values.items()]
    completed = sum(1 for e in episodes if e["status"] == "completed")
    return {"summary": {"expected_episodes": len(episodes),
                        "completed_episodes": completed},
            "episodes": episodes}


def write_eval(path: Path, model=None, baselines=None, *, seeds=SEEDS, label=None,
               training_seed=101, model_sha256="a" * 64, held_out=True, version=2,
               reward_sha="r" * 64, training_sha="t" * 64, gamma=0.99,
               selection_seed=201, mutate=None) -> str:
    """Write one eval_manifest.json built from per-seed delivery_ratio values."""
    baselines = {"hold": {s: 0.6 for s in seeds}} if baselines is None else baselines
    policies = {}
    if model is not None:
        policies["model"] = _policy_block(model)
    for name, values in baselines.items():
        policies[name] = _policy_block(values)
    manifest = {
        "eval_manifest_version": version, "status": "completed", "label": label,
        "seeds": list(seeds), "episodes_expected": len(seeds) * len(policies),
        "episodes_completed": len(seeds) * len(policies),
        "metric_source": {"kind": "telemetry_window", "warmup_excluded": False},
        "seed_roles": {"training_seed": training_seed,
                       "model_selection_seed": selection_seed,
                       "selection_seed_used_for_model": True,
                       "held_out_seeds": list(seeds), "overlap": [],
                       "overlap_allowed": not held_out, "held_out": held_out},
        "training": {"algorithm": "MaskablePPO", "seed": training_seed,
                     "seed_source": "cli", "evaluation_seed": selection_seed,
                     "evaluation_episodes": 1,
                     "scenario_identity": {"run_ini_sha256": training_sha,
                                           "nodes_json_sha256": "n" * 64,
                                           "buildings_json_sha256": None,
                                           "jammers_json_sha256": None},
                     "hyperparameters": {"total_timesteps": 256, "n_steps": 64,
                                         "gamma": gamma, "ent_coef": 0.0,
                                         "eval_every_steps": 128, "eval_episodes": 1,
                                         "verbose": 0}},
        "bundle": {"run_dir": "/train", "model_selection": "best",
                   "model_path": "/train/best_model.zip", "model_sha256": model_sha256,
                   "num_timesteps": 256},
        "selection": {"observation_preset": "local_links_v1",
                      "reward_components": ["delivery_ratio"], "reward_weights": [1.0]},
        "observation_schema": {"sha256": "o" * 64},
        "reward_schema": {"sha256": reward_sha},
        "scenario_identity": {"run_ini_sha256": "s" * 64, "nodes_json_sha256": "n" * 64,
                              "buildings_json_sha256": None, "jammers_json_sha256": None},
        "band": "mmwave", "policies": policies,
    }
    if mutate is not None:
        mutate(manifest)
    path.mkdir(parents=True, exist_ok=True)
    (path / "eval_manifest.json").write_text(json.dumps(manifest, indent=2))
    return str(path)


def _run(out_dir: Path, *eval_dirs: str, extra=()) -> int:
    return compare_cli.main(["--eval-dirs", *eval_dirs, "--output-dir", str(out_dir),
                             *extra])


def _outputs(out_dir: Path) -> tuple:
    payload = json.loads((out_dir / "comparison.json").read_text())
    rows = list(csv.DictReader((out_dir / "episodes.csv").read_text().splitlines()))
    return payload, rows


def _pick(block: dict, baseline: str = "hold", metric: str = "delivery_ratio") -> dict:
    return next(c for c in block["comparisons"]
                if c["baseline"] == baseline and c["metric"] == metric)


# 1. Shared statistics -------------------------------------------------------------

@pytest.mark.parametrize("n, expected", [(2, 12.706), (13, 2.179), (35, 2.042),
                                         (200, 1.980), (1, 0.0)])
def test_t_critical_values(n, expected):
    assert stats.t_critical_95(n) == pytest.approx(expected, abs=1e-6)


def test_t_critical_is_non_increasing_and_never_below_normal():
    values = [stats.t_critical_95(n) for n in range(2, 201)]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))
    assert min(values) >= 1.96


def test_sample_stats_reports_mean_std_and_ci():
    result = stats.sample_stats([1, 2, 3], True)
    assert result["mean"] == pytest.approx(2.0, abs=1e-6)
    assert result["std"] == pytest.approx(1.0, abs=1e-6)
    assert result["ci95"] == pytest.approx(2.484338, abs=1e-6)


def test_plotting_uses_the_shared_statistics():
    pytest.importorskip("pandas")
    from scripts.plotting import aggregation

    assert aggregation.sample_stats is stats.sample_stats
    assert aggregation.t_critical_95 is stats.t_critical_95


# 2. Paired comparison within one evaluation ---------------------------------------

def test_paired_difference_across_evaluation_seeds(tmp_path):
    eval_dir = write_eval(tmp_path / "eval", {11: 0.80, 12: 0.70, 13: 0.90},
                          {"hold": {11: 0.60, 12: 0.65, 13: 0.70}})
    out = tmp_path / "cmp"
    assert _run(out, eval_dir) == 0
    payload, rows = _outputs(out)

    entry = _pick(payload["evaluations"][0])
    assert [p["difference"] for p in entry["pairs"]] == pytest.approx(
        [0.20, 0.05, 0.20], abs=1e-6)
    assert entry["mean_difference"] == pytest.approx(0.15, abs=1e-6)
    assert entry["std_difference"] == pytest.approx(0.0866025, abs=1e-6)
    assert entry["model_mean"] - entry["baseline_mean"] == pytest.approx(0.15, abs=1e-6)
    interval = entry["interval"]
    assert interval["kind"] == PAIRED_KIND
    assert interval["variability_source"] == "evaluation_seeds"
    assert interval["fixed_model"] is True
    assert interval["df"] == 2
    assert interval["half_width"] == pytest.approx(0.215150, abs=1e-6)
    assert interval["low"] == pytest.approx(-0.065150, abs=1e-6)
    assert interval["high"] == pytest.approx(0.365150, abs=1e-6)
    assert payload["status"] == "complete"
    assert list(rows[0]) == list(CSV_COLUMNS)
    assert [row["policy"] for row in rows] == ["model"] * 3 + ["hold"] * 3
    assert [row["seed"] for row in rows[:3]] == ["11", "12", "13"]
    assert {row["metric_source"] for row in rows} == {"telemetry_window"}
    assert payload["metric_source"] == {"kind": "telemetry_window",
                                        "warmup_excluded": False}


def test_a_health_counter_downgrades_a_clean_comparison(tmp_path):
    def violate(manifest):
        manifest["policies"]["model"]["episodes"][0]["mask_violations"] = 2

    eval_dir = write_eval(tmp_path / "eval", {s: 0.8 for s in SEEDS}, mutate=violate)
    out = tmp_path / "cmp"
    assert _run(out, eval_dir) == 2
    payload = _outputs(out)[0]
    assert payload["status"] == "complete"
    assert payload["evaluations"][0]["health"]["model"]["mask_violations"] == 2


def test_failed_episode_is_excluded_by_seed(tmp_path, capsys):
    eval_dir = write_eval(tmp_path / "eval", seeds=(11, 12, 13, 14),
                          model={11: 0.80, 12: 0.70, 13: 0.90, 14: "failed"},
                          baselines={"hold": {11: 0.60, 12: 0.65, 13: 0.70, 14: 0.75}})
    out = tmp_path / "cmp"
    assert _run(out, eval_dir) == 1
    payload, rows = _outputs(out)

    entry = _pick(payload["evaluations"][0])
    assert (entry["n_expected"], entry["n_used"]) == (4, 3)
    assert entry["mean_difference"] == pytest.approx(0.15, abs=1e-6)
    assert entry["interval"]["half_width"] == pytest.approx(0.215150, abs=1e-6)
    assert entry["excluded"] == [{"seed": 14, "reasons": ["model:failed"]}]
    assert payload["status"] == "incomplete"
    failed = next(r for r in rows if r["policy"] == "model" and r["seed"] == "14")
    assert failed["delivery_ratio"] == "" and failed["error"]
    assert "excluded:" in capsys.readouterr().out


def test_null_metric_excludes_that_seed(tmp_path):
    eval_dir = write_eval(tmp_path / "eval", {11: 0.80, 12: 0.70, 13: 0.90},
                          {"hold": {11: 0.60, 12: 0.65, 13: None}})
    out = tmp_path / "cmp"
    assert _run(out, eval_dir) == 1
    entry = _pick(_outputs(out)[0]["evaluations"][0])
    assert entry["n_used"] == 2
    assert entry["mean_difference"] == pytest.approx(0.125, abs=1e-6)
    assert entry["std_difference"] == pytest.approx(0.1060660, abs=1e-6)
    assert entry["interval"]["t"] == pytest.approx(12.706, abs=1e-6)
    assert entry["interval"]["half_width"] == pytest.approx(0.952950, abs=1e-6)
    assert entry["excluded"] == [{"seed": 13, "reasons": ["hold:metric_null"]}]


def test_single_seed_reports_no_interval(tmp_path, capsys):
    eval_dir = write_eval(tmp_path / "eval", {11: 0.80}, {"hold": {11: 0.60}},
                          seeds=(11,))
    out = tmp_path / "cmp"
    assert _run(out, eval_dir) == 0
    entry = _pick(_outputs(out)[0]["evaluations"][0])
    assert entry["mean_difference"] == pytest.approx(0.20, abs=1e-6)
    assert entry["interval"] is None
    assert entry["interval_omitted"] == "fewer than 2 usable pairs"
    assert "half_width" not in (out / "comparison.json").read_text()
    assert "single seed; no interval" in capsys.readouterr().out


def test_zero_variance_is_flagged(tmp_path):
    eval_dir = write_eval(tmp_path / "eval", {s: 0.8 for s in SEEDS},
                          {"hold": {s: 0.7 for s in SEEDS}})
    out = tmp_path / "cmp"
    assert _run(out, eval_dir) == 0
    entry = _pick(_outputs(out)[0]["evaluations"][0])
    assert entry["zero_variance"] is True
    assert entry["interval"]["half_width"] == pytest.approx(0.0, abs=1e-6)


def test_record_order_inside_a_manifest_does_not_matter(tmp_path):
    values = {11: 0.80, 12: 0.70, 13: 0.90}
    baselines = {"hold": {11: 0.60, 12: 0.65, 13: 0.70}}
    ordered = write_eval(tmp_path / "ordered", values, baselines)

    def shuffle(manifest):
        manifest["policies"] = {name: {**block, "episodes": list(reversed(block["episodes"]))}
                                for name, block in reversed(list(manifest["policies"].items()))}

    shuffled = write_eval(tmp_path / "shuffled", values, baselines, mutate=shuffle)
    first, second = tmp_path / "cmp-a", tmp_path / "cmp-b"
    assert _run(first, ordered) == 0
    assert _run(second, shuffled) == 0

    left, right = _outputs(first)[0], _outputs(second)[0]
    assert left["evaluations"][0]["comparisons"] == right["evaluations"][0]["comparisons"]
    csv_left = (first / "episodes.csv").read_text().replace("ordered", "X")
    csv_right = (second / "episodes.csv").read_text().replace("shuffled", "X")
    assert csv_left == csv_right


def test_baseline_only_evaluation_reports_a_missing_model(tmp_path):
    eval_dir = write_eval(tmp_path / "eval", None,
                          {"hold": {s: 0.6 for s in SEEDS},
                           "random_valid": {s: 0.5 for s in SEEDS}})
    out = tmp_path / "cmp"
    assert _run(out, eval_dir) == 1
    payload, rows = _outputs(out)
    entry = _pick(payload["evaluations"][0])
    assert entry["n_used"] == 0
    assert entry["mean_difference"] is None and entry["model_mean"] is None
    assert entry["excluded"] == [{"seed": s, "reasons": ["model:missing"]} for s in SEEDS]
    assert len(rows) == 6


def test_return_is_compared_only_inside_one_evaluation(tmp_path, capsys):
    first = write_eval(tmp_path / "a", {s: 0.8 for s in SEEDS}, label="delivery")
    second = write_eval(tmp_path / "b", {s: 0.7 for s in SEEDS}, label="connectivity",
                        training_seed=102, model_sha256="b" * 64, reward_sha="q" * 64)
    out = tmp_path / "cmp"
    assert _run(out, first, second) == 0
    payload = _outputs(out)[0]
    assert payload["metrics"]["return"]["comparable_across_reward_definitions"] is False
    for block in payload["evaluations"]:
        assert _pick(block, metric="return")["n_used"] == 3
    for group in payload["groups"]:
        assert all(c["metric"] != "return" for c in group["comparisons"])
    assert "return" not in capsys.readouterr().out


# 3. Across independently trained models -------------------------------------------

def _labelled_runs(tmp_path, *, drop_seed_12_in_run_2=False):
    dirs = []
    for index, offset in enumerate((0.05, 0.15, 0.25)):
        model = {11: 0.6 + offset, 12: 0.6 + offset + 0.10}
        if drop_seed_12_in_run_2 and index == 1:
            model[12] = "failed"
        dirs.append(write_eval(tmp_path / f"run-{index}", model,
                               {"hold": {11: 0.6, 12: 0.6}}, seeds=(11, 12), label="a",
                               training_seed=101 + index,
                               model_sha256=chr(97 + index) * 64))
    return dirs


def test_interval_across_training_runs(tmp_path):
    out = tmp_path / "cmp"
    assert _run(out, *_labelled_runs(tmp_path)) == 0
    payload = _outputs(out)[0]
    group = payload["groups"][0]
    entry = _pick(group)
    assert [run["mean_difference"] for run in entry["per_run"]] == pytest.approx(
        [0.10, 0.20, 0.30], abs=1e-6)
    assert entry["mean_difference"] == pytest.approx(0.20, abs=1e-6)
    assert entry["std_across_runs"] == pytest.approx(0.10, abs=1e-6)
    assert entry["interval"]["kind"] == ACROSS_RUNS_KIND
    assert entry["interval"]["variability_source"] == "training_runs"
    assert entry["interval"]["n"] == 3
    assert entry["interval"]["half_width"] == pytest.approx(0.248434, abs=1e-6)
    assert (group["runs_expected"], group["runs_used"]) == (3, 3)
    assert PAIRED_KIND not in json.dumps(payload["groups"])
    assert ACROSS_RUNS_KIND not in json.dumps(payload["evaluations"])


def test_group_uses_only_seeds_common_to_every_run(tmp_path):
    out = tmp_path / "cmp"
    assert _run(out, *_labelled_runs(tmp_path, drop_seed_12_in_run_2=True)) == 1
    entry = _pick(_outputs(out)[0]["groups"][0])
    assert entry["common_seeds"] == [11]
    assert entry["seeds_dropped_for_commonality"] == [12]
    assert [run["mean_difference"] for run in entry["per_run"]] == pytest.approx(
        [0.05, 0.15, 0.25], abs=1e-6)
    assert entry["mean_difference"] == pytest.approx(0.15, abs=1e-6)
    assert entry["interval"]["half_width"] == pytest.approx(0.248434, abs=1e-6)


def test_single_labelled_run_has_no_group_interval(tmp_path):
    eval_dir = write_eval(tmp_path / "run", {s: 0.8 for s in SEEDS}, label="a")
    out = tmp_path / "cmp"
    assert _run(out, eval_dir) == 0
    entry = _pick(_outputs(out)[0]["groups"][0])
    assert entry["interval"] is None
    assert entry["interval_omitted"] == "fewer than 2 usable runs"


def test_seed_overlap_is_flagged_and_excluded_from_the_group(tmp_path):
    eval_dir = write_eval(tmp_path / "run", {s: 0.8 for s in SEEDS}, label="a",
                          held_out=False)
    out = tmp_path / "cmp"
    assert _run(out, eval_dir) == 2
    payload = _outputs(out)[0]
    assert payload["evaluations"][0]["held_out"] is False
    assert _pick(payload["evaluations"][0])["n_used"] == 3
    group = payload["groups"][0]
    assert group["runs_used"] == 0
    assert group["excluded_runs"] == [{"training_seed": 101, "reason": "seed_overlap"}]


# 4. Refusals ----------------------------------------------------------------------

def _refuse_same_dir(tmp_path):
    eval_dir = write_eval(tmp_path / "run", {s: 0.8 for s in SEEDS})
    return [eval_dir, eval_dir]


def _refuse_pair(tmp_path, **overrides):
    first = write_eval(tmp_path / "run-0", {s: 0.8 for s in SEEDS}, label="a")
    second = write_eval(tmp_path / "run-1", {s: 0.8 for s in SEEDS}, label="a",
                        **{"training_seed": 102, "model_sha256": "b" * 64, **overrides})
    return [first, second]


def _refuse_same_model(tmp_path):
    return _refuse_pair(tmp_path, model_sha256="a" * 64)


def _refuse_reward_schema(tmp_path):
    return _refuse_pair(tmp_path, reward_sha="q" * 64)


def _refuse_training_scenario(tmp_path):
    return _refuse_pair(tmp_path, training_sha="u" * 64)


def _refuse_training_setting(tmp_path):
    return _refuse_pair(tmp_path, gamma=0.5)


def _refuse_selection_seed(tmp_path):
    return _refuse_pair(tmp_path, selection_seed=202)


def _refuse_v1(tmp_path):
    return [write_eval(tmp_path / "run", {s: 0.8 for s in SEEDS}, version=1)]


def _refuse_duplicate_record(tmp_path):
    def duplicate(manifest):
        episodes = manifest["policies"]["model"]["episodes"]
        episodes.append(dict(episodes[0]))

    return [write_eval(tmp_path / "run", {s: 0.8 for s in SEEDS}, mutate=duplicate)]


def _refuse_unexpected_seed(tmp_path):
    def rename(manifest):
        manifest["policies"]["model"]["episodes"][0]["seed"] = 99

    return [write_eval(tmp_path / "run", {s: 0.8 for s in SEEDS}, mutate=rename)]


def _refuse_non_finite(tmp_path):
    def infinite(manifest):
        manifest["policies"]["model"]["episodes"][0]["metrics"]["delivery_ratio"] = float("inf")

    return [write_eval(tmp_path / "run", {s: 0.8 for s in SEEDS}, mutate=infinite)]


@pytest.mark.parametrize("case", [
    _refuse_same_dir, _refuse_same_model, _refuse_reward_schema,
    _refuse_training_scenario, _refuse_training_setting, _refuse_selection_seed,
    _refuse_v1, _refuse_duplicate_record, _refuse_unexpected_seed, _refuse_non_finite])
def test_refusals_write_nothing(tmp_path, case, capsys):
    eval_dirs = case(tmp_path)
    out = tmp_path / "cmp"
    assert _run(out, *eval_dirs) == 1
    assert not out.exists()
    assert capsys.readouterr().err.strip()


def test_v1_manifest_names_the_remedy(tmp_path, capsys):
    assert _run(tmp_path / "cmp", *_refuse_v1(tmp_path)) == 1
    assert "re-evaluate with current tooling" in capsys.readouterr().err


def test_output_dir_with_an_evaluation_manifest_is_refused(tmp_path):
    eval_dir = write_eval(tmp_path / "run", {s: 0.8 for s in SEEDS})
    out = Path(write_eval(tmp_path / "busy", {s: 0.8 for s in SEEDS}))
    assert _run(out, eval_dir) == 1
    assert not (out / "comparison.json").exists()


# 5. CLI behavior ------------------------------------------------------------------

def test_repeated_collection_is_byte_identical(tmp_path):
    eval_dir = write_eval(tmp_path / "eval", {11: 0.80, 12: 0.70, 13: 0.90})
    out = tmp_path / "cmp"
    assert _run(out, eval_dir) == 0
    first = [(out / name).read_bytes() for name in ("episodes.csv", "comparison.json")]
    assert _run(out, eval_dir) == 0
    assert _run(out, eval_dir) == 0
    second = [(out / name).read_bytes() for name in ("episodes.csv", "comparison.json")]
    assert first == second
    assert sorted(p.name for p in out.iterdir()) == ["comparison.json", "episodes.csv"]


def test_baselines_flag_selects_the_compared_policies(tmp_path):
    eval_dir = write_eval(tmp_path / "eval", {s: 0.8 for s in SEEDS},
                          {"hold": {s: 0.6 for s in SEEDS},
                           "random_valid": {s: 0.5 for s in SEEDS}})
    out = tmp_path / "cmp"
    assert _run(out, eval_dir, extra=["--baselines", "random_valid"]) == 0
    block = _outputs(out)[0]["evaluations"][0]
    assert {c["baseline"] for c in block["comparisons"]} == {"random_valid"}


def test_json_flag_prints_the_comparison(tmp_path, capsys):
    eval_dir = write_eval(tmp_path / "eval", {s: 0.8 for s in SEEDS})
    out = tmp_path / "cmp"
    assert _run(out, eval_dir, extra=["--json"]) == 0
    assert json.loads(capsys.readouterr().out) == _outputs(out)[0]


def test_plan_mode_counts_a_missing_evaluation(tmp_path):
    present = write_eval(tmp_path / "run-0", {11: 0.8, 12: 0.9},
                         {"hold": {11: 0.6, 12: 0.6}}, seeds=(11, 12), label="a")
    plan = tmp_path / "experiment_plan.json"
    plan.write_text(json.dumps({"evaluations": [
        {"label": "a", "training_seed": 101, "eval_dir": present},
        {"label": "a", "training_seed": 102, "eval_dir": str(tmp_path / "run-1")}]}))
    out = tmp_path / "cmp"
    assert compare_cli.main(["--plan", str(plan), "--output-dir", str(out)]) == 1
    payload, rows = _outputs(out)
    assert len(payload["missing_evaluations"]) == 1
    assert payload["missing_evaluations"][0]["training_seed"] == 102
    group = payload["groups"][0]
    assert (group["runs_expected"], group["runs_used"]) == (2, 1)
    assert {"training_seed": 102, "reason": "missing"} in group["excluded_runs"]
    assert len(rows) == 4


# 6. Placement provenance ----------------------------------------------------------

_PLACEMENT_BASELINES = {"hold": {s: 0.6 for s in SEEDS},
                        "geometric": {s: 0.65 for s in SEEDS}}


def _with_placement(fingerprint: str):
    def add(manifest):
        manifest["policies"]["geometric"]["baseline"] = {"method": "geometric",
                                                         "fingerprint": fingerprint}
    return add


def _placement_pair(tmp_path, first: str, second: str) -> list:
    left = write_eval(tmp_path / "run-0", {s: 0.8 for s in SEEDS}, _PLACEMENT_BASELINES,
                      label="a", mutate=_with_placement(first))
    right = write_eval(tmp_path / "run-1", {s: 0.8 for s in SEEDS}, _PLACEMENT_BASELINES,
                       label="a", training_seed=102, model_sha256="b" * 64,
                       mutate=_with_placement(second))
    return [left, right]


def test_equal_placement_fingerprints_group_together(tmp_path):
    out = tmp_path / "cmp"
    assert _run(out, *_placement_pair(tmp_path, "f" * 64, "f" * 64)) == 0
    group = _outputs(out)[0]["groups"][0]
    assert (group["runs_expected"], group["runs_used"]) == (2, 2)
    assert {c["baseline"] for c in group["comparisons"]} == {"geometric", "hold"}


def test_differing_placement_fingerprints_are_refused(tmp_path, capsys):
    eval_dirs = _placement_pair(tmp_path, "f" * 64, "e" * 64)
    evaluations = [load_evaluation(eval_dir) for eval_dir in eval_dirs]
    with pytest.raises(ComparisonError, match="baseline_geometric_fingerprint"):
        build_comparison(evaluations)

    out = tmp_path / "cmp"
    assert _run(out, *eval_dirs) == 1
    assert not out.exists()
    assert "baseline_geometric_fingerprint" in capsys.readouterr().err


def test_group_key_without_placement_blocks_is_unchanged(tmp_path):
    plain = load_evaluation(write_eval(tmp_path / "plain", {s: 0.8 for s in SEEDS},
                                       _PLACEMENT_BASELINES))
    placed = load_evaluation(write_eval(tmp_path / "placed", {s: 0.8 for s in SEEDS},
                                        _PLACEMENT_BASELINES,
                                        mutate=_with_placement("f" * 64)))
    shas = ("run_ini_sha256", "nodes_json_sha256", "buildings_json_sha256",
            "jammers_json_sha256")
    settings = ("total_timesteps", "n_steps", "gamma", "ent_coef", "eval_every_steps",
                "eval_episodes")
    expected = {"observation_preset", "reward_components", "reward_weights",
                "observation_schema_sha256", "reward_schema_sha256", "band",
                "model_selection", "seeds", "training_algorithm", "model_selection_seed",
                *(f"scenario_{sha}" for sha in shas),
                *(f"training_scenario_{sha}" for sha in shas),
                *(f"training_{setting}" for setting in settings)}

    plain_key = policy_compare._group_key(plain)
    assert set(plain_key) == expected
    assert policy_compare._group_key(placed) == {
        **plain_key, "baseline_geometric_fingerprint": "f" * 64}
