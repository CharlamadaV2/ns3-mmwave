"""Gap tests for the scripts/rl/train.py CLI against fake_sim.py.

Sources: scripts/rl/README.md ("MaskablePPO smoke run", "Train" flag table,
"Manifests" row for train_manifest.json v4), scripts/rl/CLAUDE.md (MANIFEST_VERSION
mirrored in policy/bundle.py; DIRECT_DEPS recorded in every manifest), and
train.py's own argument checks. Training runs are tiny (16-32 timesteps).
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from scripts.rl import train
from scripts.rl.bootstrap_venv import DIRECT_DEPS
from scripts.rl.cli_common import sha256_file
from scripts.rl.env.config import read_scenario_identity

requires_sb3 = pytest.mark.skipif(importlib.util.find_spec("sb3_contrib") is None,
                                  reason="sb3_contrib not installed")


def _main(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["train", *argv])
    return train.main()


def _argv(sim_binary, run_config, out_dir, *ppo: str, pre=()) -> list[str]:
    return ["--sim-binary", sim_binary, "--run-config", run_config,
            "--output-dir", str(out_dir), "--verbose", "0", *pre,
            "m-ppo", "--total-timesteps", "16", "--n-steps", "16", *ppo]


# 1. Refusals before anything is written -------------------------------------------


# 2. Manifest contents of tiny runs ------------------------------------------------

def test_manifest_version_is_mirrored_in_the_bundle_reader():
    from scripts.rl.policy import bundle

    assert train.MANIFEST_VERSION == bundle.MANIFEST_VERSION == 4


@requires_sb3
def test_tiny_run_writes_a_complete_v4_manifest(sim_binary, multi_run_config, tmp_path,
                                                monkeypatch, capsys):
    out_dir = tmp_path / "out"
    assert _main(monkeypatch, _argv(sim_binary, multi_run_config, out_dir)) == 0
    manifest = json.loads((out_dir / "train_manifest.json").read_text())

    assert manifest["manifest_version"] == 4
    assert manifest["status"] == "completed"
    assert manifest["started_at"] and manifest["ended_at"]
    assert "error" not in manifest
    # No --seed -> [scenario] seed = 1 from run.ini.
    assert (manifest["seed"], manifest["seed_source"]) == (1, "run.ini")
    assert manifest["algorithm"] == "MaskablePPO"
    assert manifest["control_mode"] == "centralized"
    assert manifest["contract"]["contract"] == "mesh_move_2d_v1"
    assert manifest["band"] is None
    assert manifest["evaluation"] is None
    assert manifest["checkpoints"] == []
    assert manifest["best_model_path"] is None and manifest["best_mean_reward"] is None
    assert manifest["sim_binary"] == os.path.abspath(sim_binary)
    assert manifest["output_dir"] == str(out_dir.resolve())
    assert manifest["scenario_identity"] == read_scenario_identity(multi_run_config)
    assert set(manifest["package_versions"]) == set(DIRECT_DEPS)

    hyper = manifest["hyperparameters"]
    assert (hyper["total_timesteps"], hyper["n_steps"]) == (16, 16)
    assert (hyper["gamma"], hyper["ent_coef"]) == (0.95, 0.01)
    assert (hyper["checkpoint_every_steps"], hyper["keep_checkpoints"]) == (0, 3)
    assert (hyper["eval_every_steps"], hyper["eval_episodes"]) == (0, 1)

    model = Path(manifest["model_path"])
    assert model == (out_dir / "maskable_ppo_mesh.zip").resolve()
    assert manifest["model_sha256"] == sha256_file(model)
    assert "Done. Model + logs in" in capsys.readouterr().out

    episodes = sorted(out_dir.glob("episode-*/rl_episode.json"))
    assert episodes
    first = json.loads(episodes[0].read_text())
    assert first["seed"] == 1


@requires_sb3
def test_cadence_defaults_eval_seed_and_records_checkpoints(sim_binary, multi_run_config,
                                                            tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    argv = ["--sim-binary", sim_binary, "--run-config", multi_run_config,
            "--output-dir", str(out_dir), "--verbose", "0", "--band", "sub-6",
            "m-ppo", "--total-timesteps", "32", "--n-steps", "16", "--seed", "5",
            "--checkpoint-every-steps", "8", "--keep-checkpoints", "1",
            "--eval-every-steps", "16"]
    assert _main(monkeypatch, argv) == 0
    manifest = json.loads((out_dir / "train_manifest.json").read_text())

    assert (manifest["seed"], manifest["seed_source"]) == (5, "cli")
    assert manifest["band"] == "sub-6"
    evaluation = manifest["evaluation"]
    # README: --eval-seed defaults to the training seed + 1.
    assert evaluation["seed"] == 6 and evaluation["seed_source"] == "eval"
    assert evaluation["every_steps"] == 16 and evaluation["episodes"] == 1
    assert evaluation["output_dir"] == str((out_dir / "eval").resolve())

    checkpoints = manifest["checkpoints"]
    assert len(checkpoints) == 1                            # --keep-checkpoints 1
    for entry in checkpoints:
        assert Path(entry["path"]).is_file()
        assert entry["sha256"] == sha256_file(entry["path"])
        assert entry["num_timesteps"] > 0
    assert len(list((out_dir / "checkpoints").glob("*.zip"))) == 1

    best = manifest["best_model_path"]
    assert best and Path(best).is_file()
    assert manifest["best_model_sha256"] == sha256_file(best)

    eval_episodes = sorted((out_dir / "eval").glob("episode-*/rl_episode.json"))
    assert eval_episodes
    seeds = {json.loads(p.read_text())["seed"] for p in eval_episodes}
    assert seeds == {6}


@requires_sb3
def test_simulator_failure_is_recorded_in_the_manifest(sim_binary, multi_run_config,
                                                       tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_SIM_MODE", "exit3")
    out_dir = tmp_path / "out"
    assert _main(monkeypatch, _argv(sim_binary, multi_run_config, out_dir,
                                    "--seed", "2")) == 1
    assert "Training failed" in capsys.readouterr().err
    manifest = json.loads((out_dir / "train_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"] and len(manifest["error"]) <= 1000
    assert manifest["ended_at"] is not None
    assert manifest["model_path"] is None and manifest["model_sha256"] is None
    assert not (out_dir / "maskable_ppo_mesh.zip").exists()
    # A failed run still blocks reuse of its directory.
    monkeypatch.delenv("FAKE_SIM_MODE")
    assert _main(monkeypatch, _argv(sim_binary, multi_run_config, out_dir,
                                    "--seed", "2")) == 1
