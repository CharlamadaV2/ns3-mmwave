"""`scripts.rl.compare` refuses a --label group whose placement fingerprints differ.

Contract: scripts/baselines/README.md manifest table, `fingerprint` row: "scripts.rl.compare
rejects a --label group whose placement fingerprints differ"; scripts/rl/policy/CLAUDE.md:
"mixing groups is an error, not a warning". Unit cases use hand-built eval manifests;
the last case produces real ones with `scripts.rl.evaluate` on fake_sim.py and the stub
planner.
"""

import json
import sys
from pathlib import Path

import pytest

from scripts.baselines.tests.conftest import install_stub_planner, write_scenario
from scripts.rl import compare as compare_cli
from scripts.rl import evaluate as evaluate_cli
from scripts.rl.policy.compare import build_comparison, load_evaluation
from scripts.rl.tests.conftest import FAKE_SIM

SEEDS = (11, 12, 13)


def _episode(seed: int, value: float) -> dict:
    return {"seed": seed, "status": "completed", "error": None, "return": -1.0,
            "decisions": 20, "mask_violations": 0, "revalidated_slots_total": 0,
            "actions_sha256": "c" * 64, "episode_dir": f"/eval/{seed}",
            "summary_json": None,
            "metrics": {"delivery_ratio": value, "connectivity": 1.0, "los_fraction": 0.5,
                        "unroutable_fraction": 0.0, "first_all_los_decision": 4}}


def _write_eval(path: Path, *, label, training_seed, placements: dict,
                model_sha256: str) -> str:
    """eval_manifest.json with model + hold and one block per placement policy."""
    policies = {"model": {"episodes": [_episode(s, 0.8) for s in SEEDS], "summary": {}},
                "hold": {"episodes": [_episode(s, 0.6) for s in SEEDS], "summary": {}}}
    for name, baseline in placements.items():
        policies[name] = {"episodes": [_episode(s, 0.65) for s in SEEDS], "summary": {},
                          "baseline": baseline}
    manifest = {
        "eval_manifest_version": 2, "status": "completed", "label": label,
        "seeds": list(SEEDS),
        "seed_roles": {"training_seed": training_seed, "model_selection_seed": 201,
                       "held_out_seeds": list(SEEDS), "overlap": [], "held_out": True},
        "training": {"algorithm": "MaskablePPO", "seed": training_seed,
                     "evaluation_seed": 201, "hyperparameters": {"gamma": 0.99}},
        "bundle": {"model_selection": "best", "model_sha256": model_sha256,
                   "num_timesteps": 256},
        "selection": {"observation_preset": "p", "reward_components": ["delivery_ratio"],
                      "reward_weights": [1.0]},
        "observation_schema": {"sha256": "o" * 64}, "reward_schema": {"sha256": "r" * 64},
        "scenario_identity": {"run_ini_sha256": "s" * 64}, "band": "mmwave",
        "policies": policies,
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "eval_manifest.json").write_text(json.dumps(manifest, indent=2))
    return str(path)


def _block(method: str, fingerprint: str, **extra) -> dict:
    return {"method": method, "fingerprint": fingerprint,
            "manifest": f"{method}/baseline/baseline_manifest.json",
            "plan": f"{method}/baseline/effective-inputs/baseline-plan.json", **extra}


def _pair(tmp_path, first: dict, second: dict, labels=("a", "a")) -> list:
    return [load_evaluation(_write_eval(tmp_path / "run-0", label=labels[0],
                                        training_seed=101, placements=first,
                                        model_sha256="a" * 64)),
            load_evaluation(_write_eval(tmp_path / "run-1", label=labels[1],
                                        training_seed=102, placements=second,
                                        model_sha256="b" * 64))]


def test_equal_fingerprints_with_different_paths_and_outcomes_group(tmp_path):
    first = {"geometric": _block("geometric", "1" * 64, initial_displacement_m_total=3.0)}
    second = {"geometric": {**_block("geometric", "1" * 64,
                                     initial_displacement_m_total=9.0),
                            "manifest": "elsewhere/baseline_manifest.json",
                            "plan": "elsewhere/plan.json"}}
    comparison = build_comparison(_pair(tmp_path, first, second))
    group = comparison["groups"][0]
    assert (group["label"], group["runs_used"]) == ("a", 2)


def test_unlabeled_evaluations_are_never_grouped_so_never_refused(tmp_path):
    evaluations = _pair(tmp_path, {"geometric": _block("geometric", "1" * 64)},
                        {"geometric": _block("geometric", "2" * 64)}, labels=(None, None))
    comparison = build_comparison(evaluations)
    assert comparison["groups"] == []
    out = tmp_path / "cmp"
    assert compare_cli.main(["--eval-dirs", evaluations[0].eval_dir, evaluations[1].eval_dir,
                             "--output-dir", str(out)]) == 0
    assert (out / "comparison.json").is_file()


def test_each_label_is_checked_on_its_own(tmp_path):
    evaluations = _pair(tmp_path, {"geometric": _block("geometric", "1" * 64)},
                        {"geometric": _block("geometric", "2" * 64)}, labels=("x", "y"))
    comparison = build_comparison(evaluations)
    assert [g["label"] for g in comparison["groups"]] == ["x", "y"]


# --- end to end ------------------------------------------------------------------------

@pytest.fixture
def sim_binary(tmp_path) -> str:
    shim = tmp_path / "fake-sim"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_SIM}" "$@"\n')
    shim.chmod(0o755)
    return str(shim)


def _placement_eval(tmp_path, sim_binary, name: str, objective: str) -> str:
    ini = write_scenario(tmp_path / f"scenario-{name}", overrides={"objective": objective})
    ini.write_text(ini.read_text().replace("controlled_nodes = uav-a, uav-b, uav-c, walker",
                                           "controlled_nodes = uav-a, uav-c"))
    out = tmp_path / f"eval-{name}"
    assert evaluate_cli.main(["--sim-binary", sim_binary, "--run-config", str(ini),
                              "--output-dir", str(out), "--seeds", "1",
                              "--policies", "hold,geometric", "--label", "shared"]) == 0
    return str(out)


def test_evaluations_with_different_plans_cannot_share_a_label(tmp_path, sim_binary,
                                                               monkeypatch, capsys):
    install_stub_planner(monkeypatch)
    same_a = _placement_eval(tmp_path, sim_binary, "a", "coverage")
    same_b = _placement_eval(tmp_path, sim_binary, "b", "coverage")
    other = _placement_eval(tmp_path, sim_binary, "c", "balanced")
    fingerprints = [json.loads((Path(d) / "eval_manifest.json").read_text())["policies"][
        "geometric"]["baseline"]["fingerprint"] for d in (same_a, same_b, other)]
    assert fingerprints[0] == fingerprints[1] != fingerprints[2]

    refused = tmp_path / "cmp-refused"
    assert compare_cli.main(["--eval-dirs", same_a, other,
                             "--output-dir", str(refused)]) == 1
    assert "baseline_geometric_fingerprint" in capsys.readouterr().err
    assert not refused.exists()

    accepted = tmp_path / "cmp-accepted"
    compare_cli.main(["--eval-dirs", same_a, same_b, "--output-dir", str(accepted)])
    comparison = json.loads((accepted / "comparison.json").read_text())
    assert [g["label"] for g in comparison["groups"]] == ["shared"]
