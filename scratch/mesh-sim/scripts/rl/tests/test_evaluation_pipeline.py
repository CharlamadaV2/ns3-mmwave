"""End-to-end train, evaluate, compare, and matrix runs driven by fake_sim.py."""

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.baselines.tests.conftest import NODES as BASELINE_NODES
from scripts.baselines.tests.conftest import install_stub_planner, write_scenario
from scripts.rl import compare as compare_cli
from scripts.rl import evaluate as evaluate_cli
from scripts.rl import experiment
from scripts.rl.env.config import read_scenario_identity
from scripts.rl.policy import evaluate as policy_evaluate
from scripts.rl.policy.compare import ACROSS_RUNS_KIND, PAIRED_KIND, PRIMARY_METRIC
from scripts.sim_support import find_mesh_root

requires_sb3 = pytest.mark.skipif(importlib.util.find_spec("sb3_contrib") is None,
                                  reason="sb3_contrib not installed")


def _train(sim_binary, run_config, out_dir, monkeypatch, *extra: str) -> int:
    from scripts.rl import train

    monkeypatch.setattr(sys, "argv", [
        "train", "--sim-binary", sim_binary, "--run-config", run_config,
        "--output-dir", str(out_dir), "--verbose", "0",
        "m-ppo", "--total-timesteps", "16", "--n-steps", "16", "--seed", "1",
        *extra])
    return train.main()


def _primary(block: dict, baseline: str) -> dict:
    entry, = [c for c in block["comparisons"]
              if c["baseline"] == baseline and c["metric"] == PRIMARY_METRIC]
    return entry


def _csv_rows(path: Path) -> list:
    with path.open(newline="") as handle:
        return list(csv.reader(handle))[1:]


@requires_sb3
def test_failed_seed_survives_evaluation_and_narrows_the_comparison(
        sim_binary, multi_run_config, tmp_path, monkeypatch):
    run_dir = tmp_path / "train"
    assert _train(sim_binary, multi_run_config, run_dir, monkeypatch) == 0

    eval_dir = tmp_path / "eval"
    monkeypatch.setenv("FAKE_SIM_FAIL_SEEDS", "12")
    assert evaluate_cli.main([
        "--sim-binary", sim_binary, "--run-dir", str(run_dir),
        "--output-dir", str(eval_dir), "--seeds", "11-13",
        "--policies", "model,hold,random_valid", "--label", "pipeline"]) == 1

    manifest = json.loads((eval_dir / "eval_manifest.json").read_text())
    assert manifest["status"] == "partial"
    assert manifest["episodes_expected"] == 9 and manifest["episodes_completed"] == 6

    out_dir = tmp_path / "comparison"
    assert compare_cli.main(["--eval-dirs", str(eval_dir),
                             "--output-dir", str(out_dir)]) == 1

    comparison = json.loads((out_dir / "comparison.json").read_text())
    assert comparison["status"] == "incomplete"
    block, = comparison["evaluations"]
    for baseline in ("hold", "random_valid"):
        entry = _primary(block, baseline)
        assert (entry["n_expected"], entry["n_used"]) == (3, 2)
        assert [pair["seed"] for pair in entry["pairs"]] == [11, 13]
        assert entry["excluded"] == [
            {"seed": 12, "reasons": ["model:failed", f"{baseline}:failed"]}]
        assert entry["interval"]["kind"] == PAIRED_KIND

    assert len(_csv_rows(out_dir / "episodes.csv")) == 9


@requires_sb3
def test_experiment_run_groups_two_training_runs(sim_binary, multi_run_config, tmp_path):
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps({
        "matrix_version": 1,
        "name": "pipeline-smoke",
        "description": "Two training seeds on the fake simulator; smoke-sized budgets.",
        "run_config": multi_run_config,
        "band": None,
        "seeds": {"training": [1, 2], "model_selection": 5, "held_out": "11-12"},
        "training": {"total_timesteps": 16, "n_steps": 16, "eval_every_steps": 8},
        "evaluation": {"model": "best", "policies": ["model", "hold"]},
        "rows": [{"name": "local-delivery", "observation_preset": "local_links_v1",
                  "action_profile": "move_2d", "reward_components": ["delivery_ratio"],
                  "reward_weights": [1.0]}],
    }))

    root = tmp_path / "root"
    code = experiment.main(["run", "--matrix", str(matrix_path),
                            "--output-root", str(root), "--sim-binary", sim_binary])
    assert code in (0, 2), f"experiment run exited {code}"

    plan = experiment.load_plan(root)
    states = {step["id"]: experiment.step_state(step)[0] for step in plan["steps"]}
    assert set(states.values()) == {"done"}, states

    comparison = json.loads((root / "comparison" / "comparison.json").read_text())
    assert len(comparison["evaluations"]) == 2
    for block in comparison["evaluations"]:
        entry = _primary(block, "hold")
        assert (entry["n_expected"], entry["n_used"]) == (2, 2)
        assert entry["interval"]["kind"] == PAIRED_KIND

    group, = comparison["groups"]
    assert group["label"] == "local-delivery"
    assert (group["runs_expected"], group["runs_used"]) == (2, 2)
    assert group["excluded_runs"] == []
    entry = _primary(group, "hold")
    assert entry["common_seeds"] == [11, 12]
    assert entry["interval"]["kind"] == ACROSS_RUNS_KIND


def _placement_run_config(tmp_path, monkeypatch, **overrides) -> str:
    """Synthetic baseline scenario with the stub planner installed."""
    install_stub_planner(monkeypatch)
    ini = write_scenario(tmp_path / "placement-scenario", **overrides)
    ini.write_text(ini.read_text().replace("controlled_nodes = uav-a, uav-b, uav-c, walker",
                                           "controlled_nodes = uav-a, uav-c"))
    return str(ini)


def _record_manifest_writes(monkeypatch) -> list:
    """Snapshot every eval_manifest.json write made by the evaluation loop."""
    snapshots = []
    real = policy_evaluate.write_json

    def record(path, payload):
        if Path(path).name == policy_evaluate.EVAL_MANIFEST_NAME:
            snapshots.append(json.loads(json.dumps(payload)))
        return real(path, payload)

    monkeypatch.setattr(policy_evaluate, "write_json", record)
    return snapshots


def _decision_zero(episode_dir: str) -> dict:
    """Node id -> (x, y, z) from the decision-0 facts of one episode."""
    path = Path(episode_dir) / "steps.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = lines[0]["contract"]["node_ids"]
    first = next(line for line in lines[1:]
                 if line.get("type") == "step" and line["decision"] == 0)
    return {node_id: tuple(row[:3]) for node_id, row in zip(ids, first["facts"]["nodes"])}


def _assert_placement_blocks(policies: dict, placement: tuple) -> None:
    for name, block in policies.items():
        if name in placement:
            assert block["baseline"]["method"] == name
            assert block["baseline"]["executor"] == "hold"
        else:
            assert "baseline" not in block
            assert set(block) == {"episodes", "summary"}


def test_placement_policies_start_from_their_plans(sim_binary, tmp_path, monkeypatch):
    run_config = _placement_run_config(tmp_path, monkeypatch)
    snapshots = _record_manifest_writes(monkeypatch)
    eval_dir = tmp_path / "eval"
    assert evaluate_cli.main([
        "--sim-binary", sim_binary, "--run-config", run_config,
        "--output-dir", str(eval_dir), "--seeds", "1,2",
        "--policies", "hold,geometric,optimization"]) == 0

    manifest = json.loads((eval_dir / "eval_manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["eval_manifest_version"] == 2
    assert manifest["run_config"] == str(Path(run_config).resolve())
    assert manifest["scenario_identity"] == read_scenario_identity(run_config)
    assert list(manifest["policies"]) == ["hold", "geometric", "optimization"]
    _assert_placement_blocks(manifest["policies"], ("geometric", "optimization"))

    episode_writes = [s for s in snapshots if s["policies"]]
    assert len(episode_writes) >= 6
    for snapshot in episode_writes:
        _assert_placement_blocks(snapshot["policies"], ("geometric", "optimization"))

    original = {node["id"]: tuple(node["position"][axis] for axis in ("x", "y", "z"))
                for node in BASELINE_NODES}
    for episode in manifest["policies"]["hold"]["episodes"]:
        assert _decision_zero(episode["episode_dir"]) == pytest.approx(original)

    for method in ("geometric", "optimization"):
        block = manifest["policies"][method]["baseline"]
        assert block["manifest"] == f"{method}/baseline/baseline_manifest.json"
        assert block["plan"] == f"{method}/baseline/effective-inputs/baseline-plan.json"
        prep = json.loads((eval_dir / block["manifest"]).read_text())
        assert prep["status"] == "prepared" and prep["fingerprint"] == block["fingerprint"]
        assert (eval_dir / method / "baseline" / prep["eval_manifest"]).resolve() \
            == (eval_dir / "eval_manifest.json").resolve()
        plan = json.loads((eval_dir / block["plan"]).read_text())
        selected = {node["id"] for node in plan["nodes"] if node["selected"]}
        assert selected == {"uav-a", "uav-c"}
        assert all(node["displacement_m"] > 1.0 for node in plan["nodes"]
                   if node["selected"])
        for episode in manifest["policies"][method]["episodes"]:
            assert Path(episode["episode_dir"]).parent.name == method
            facts = _decision_zero(episode["episode_dir"])
            for node in plan["nodes"]:
                expected = (tuple(node["planned"][axis] for axis in ("x", "y", "z"))
                            if node["selected"] else original[node["id"]])
                assert facts[node["id"]] == pytest.approx(expected), node["id"]


def test_placement_block_survives_a_failed_seed(sim_binary, tmp_path, monkeypatch):
    run_config = _placement_run_config(tmp_path, monkeypatch)
    snapshots = _record_manifest_writes(monkeypatch)
    monkeypatch.setenv("FAKE_SIM_FAIL_SEEDS", "2")
    eval_dir = tmp_path / "eval"
    assert evaluate_cli.main([
        "--sim-binary", sim_binary, "--run-config", run_config,
        "--output-dir", str(eval_dir), "--seeds", "1,2",
        "--policies", "hold,geometric"]) == 1

    manifest = json.loads((eval_dir / "eval_manifest.json").read_text())
    assert manifest["status"] == "partial"
    statuses = [e["status"] for e in manifest["policies"]["geometric"]["episodes"]]
    assert statuses[0] == "completed" and statuses[1] != "completed"
    _assert_placement_blocks(manifest["policies"], ("geometric",))
    failed_writes = [s for s in snapshots if len(
        (s["policies"].get("geometric") or {}).get("episodes", [])) == 2]
    assert failed_writes
    for snapshot in snapshots:
        _assert_placement_blocks(snapshot["policies"], ("geometric",))


def test_aborted_evaluation_keeps_the_placement_block(tmp_path):
    metadata = {"method": "geometric", "fingerprint": "f" * 64}
    hold = policy_evaluate.Prepared(policy_evaluate.HoldPolicy())
    spec = policy_evaluate.PolicySpec("geometric", lambda env, seed: hold, metadata=metadata)

    def make_env(name):
        raise RuntimeError("simulator unavailable")

    with pytest.raises(RuntimeError, match="simulator unavailable"):
        policy_evaluate.evaluate(make_env, [spec], [1, 2], tmp_path / "eval", {})
    manifest = json.loads((tmp_path / "eval" / "eval_manifest.json").read_text())
    assert manifest["status"] == "failed"
    block = manifest["policies"]["geometric"]
    assert block["baseline"] == metadata
    assert [e["status"] for e in block["episodes"]] == ["not_run", "not_run"]


def test_default_policies_are_unchanged():
    args = evaluate_cli._build_parser().parse_args([
        "--sim-binary", "sim", "--output-dir", "out", "--seeds", "1"])
    assert args.policies == "model,hold,random_valid"
    assert evaluate_cli._parse_policies(
        "model,hold,random_valid,geometric,optimization") == [
        "model", "hold", "random_valid", "geometric", "optimization"]


def test_evaluation_module_does_not_import_baselines():
    code = ("import sys, scripts.rl.evaluate; "
            "sys.exit(any(m.startswith('scripts.baselines') for m in sys.modules))")
    assert subprocess.run([sys.executable, "-c", code],
                          cwd=str(find_mesh_root())).returncode == 0


def test_failed_preparation_runs_no_episode(sim_binary, tmp_path, monkeypatch, capsys):
    run_config = _placement_run_config(tmp_path, monkeypatch, drop=("seed",))
    eval_dir = tmp_path / "eval"
    assert evaluate_cli.main([
        "--sim-binary", sim_binary, "--run-config", run_config,
        "--output-dir", str(eval_dir), "--seeds", "1,2",
        "--policies", "hold,geometric,optimization"]) == 1

    assert "BaselinePreparationError" in capsys.readouterr().err
    assert not (eval_dir / "eval_manifest.json").exists()
    assert not list(eval_dir.rglob("episode-*"))
    assert not (eval_dir / "hold").exists()
    failed = json.loads(
        (eval_dir / "optimization" / "baseline" / "baseline_manifest.json").read_text())
    assert failed["status"] == "failed" and "seed" in failed["error"]
