"""CLI tests for validate_config and inspect_model, driven by fake_sim.py."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from scripts.rl import inspect_model, validate_config

FAKE_SIM = Path(__file__).resolve().parent / "fake_sim.py"

# Centralized fixture mirroring inputs/baselines/centralized-multi-smoke.
MULTI_RUN_INI = """[scenario]
name = fake-multi
seed = 1
duration_s = 1.0
tick_s = 0.1
nodes_file = nodes.json

[rl]
enabled = true
controlled_nodes = node-b, node-c
max_controlled_nodes = 3
action_profile = move_2d
decision_interval_s = 0.5
action_type = discrete
reward_type = all_links_los
step_size_m = 1.0
x_min = 0.0
x_max = 100.0
y_min = -50.0
y_max = 100.0
z_min = 0.0
z_max = 50.0
"""

NODES_JSON = """[
  {"id": "node-a", "role": "peer", "mobility": "fixed", "node_type": "drone",
   "position": {"x": 50.0, "y": 20.0, "z": 10.0}},
  {"id": "node-b", "role": "peer", "mobility": "waypoint", "node_type": "drone",
   "waypoints": [{"t": 0.0, "x": 100.0, "y": 0.0, "z": 10.0},
                 {"t": 1.0, "x": 100.0, "y": 40.0, "z": 10.0}]},
  {"id": "node-c", "role": "peer", "mobility": "constant_velocity", "node_type": "drone",
   "position": {"x": 97.0, "y": 50.0, "z": 10.0},
   "velocity": {"vx": 0.0, "vy": 0.0, "vz": 0.0}}
]
"""


@pytest.fixture
def sim_binary(tmp_path: Path) -> str:
    """Executable shim so the CLIs can spawn the fake sim like a real binary."""
    shim = tmp_path / "fake-sim"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_SIM}" "$@"\n')
    shim.chmod(0o755)
    return str(shim)


@pytest.fixture
def multi_run_config(tmp_path: Path) -> str:
    path = tmp_path / "run.ini"
    path.write_text(MULTI_RUN_INI)
    (tmp_path / "nodes.json").write_text(NODES_JSON)
    return str(path)


def _validate_argv(sim_binary: str, run_config: str, *extra: str) -> list[str]:
    return ["--sim-binary", sim_binary, "--run-config", run_config, *extra]


# 1. validate_config static checks ------------------------------------------------

def test_static_validation_passes(sim_binary, multi_run_config, capsys):
    code = validate_config.main(_validate_argv(sim_binary, multi_run_config, "--json"))
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["status"] == "ok"
    assert payload["launch"] is None
    names = {check["check"]: check for check in payload["checks"]}
    assert names["control_mode"]["detail"] == "centralized"
    assert names["seed"]["detail"].startswith("1 (source: run.ini)")
    assert all(check["kind"] == "static" for check in payload["checks"])


def test_unknown_preset_is_rejected(sim_binary, multi_run_config, capsys):
    code = validate_config.main(_validate_argv(sim_binary, multi_run_config,
                                               "--observation-preset", "nope_v9"))
    captured = capsys.readouterr()
    assert code == 1
    assert "not registered" in captured.err


def test_missing_seed_is_rejected(sim_binary, tmp_path, capsys):
    path = tmp_path / "noseed.ini"
    path.write_text(MULTI_RUN_INI.replace("seed = 1\n", ""))
    (tmp_path / "nodes.json").write_text(NODES_JSON)
    code = validate_config.main(_validate_argv(sim_binary, str(path)))
    captured = capsys.readouterr()
    assert code == 1
    assert "No seed available" in captured.err


def test_non_executable_binary_is_rejected(multi_run_config, tmp_path, capsys):
    plain = tmp_path / "not-a-binary"
    plain.write_text("#!/bin/sh\n")
    plain.chmod(0o644)
    code = validate_config.main(_validate_argv(str(plain), multi_run_config))
    captured = capsys.readouterr()
    assert code == 1
    assert "not executable" in captured.err


def test_launch_requires_output_dir(sim_binary, multi_run_config, capsys):
    code = validate_config.main(_validate_argv(sim_binary, multi_run_config, "--launch"))
    assert code == 1
    assert "--launch requires --output-dir" in capsys.readouterr().err


# 2. validate_config --launch -----------------------------------------------------

def test_launch_reports_the_live_contract(sim_binary, multi_run_config, tmp_path,
                                          capsys):
    out_dir = tmp_path / "validate-run"
    code = validate_config.main(_validate_argv(
        sim_binary, multi_run_config, "--output-dir", str(out_dir),
        "--launch", "--json"))
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    launch = payload["launch"]
    assert launch["contract"] == "mesh_move_2d_v1"
    assert launch["num_decisions"] == 2
    assert launch["num_mesh_nodes"] == 3 and launch["max_controlled_nodes"] == 3
    assert launch["cpp_obs_dim"] == launch["preset_obs_dim"]
    assert launch["observation_schema_sha256"] and launch["reward_schema_sha256"]
    assert any(check["kind"] == "launch" for check in payload["checks"])

    episode = json.loads((out_dir / "validate" / "episode-0000"
                          / "rl_episode.json").read_text())
    assert episode["status"] == "interrupted"


# 3. inspect_model ----------------------------------------------------------------

def test_missing_manifest_is_reported(tmp_path, capsys):
    code = inspect_model.main(["--run-dir", str(tmp_path)])
    assert code == 1
    assert "Cannot read" in capsys.readouterr().err


def test_missing_model_file_exits_two(tmp_path, capsys):
    manifest = {
        "manifest_version": 4, "status": "completed", "algorithm": "MaskablePPO",
        "seed": 1, "seed_source": "run.ini", "control_mode": "centralized",
        "model_path": str(tmp_path / "maskable_ppo_mesh.zip"),
        "model_sha256": "0" * 64, "checkpoints": [],
        "package_versions": {}, "python_version": "3.12.0",
        "platform": {"system": "Darwin", "machine": "arm64"},
    }
    (tmp_path / "train_manifest.json").write_text(json.dumps(manifest))
    code = inspect_model.main(["--run-dir", str(tmp_path)])
    assert code == 2
    assert "Model problem" in capsys.readouterr().err


@pytest.mark.skipif(importlib.util.find_spec("sb3_contrib") is None,
                    reason="sb3_contrib not installed")
def test_inspect_a_training_run(sim_binary, multi_run_config, tmp_path, monkeypatch,
                                capsys):
    from scripts.rl import train

    out_dir = tmp_path / "train"
    argv = ["train", "--sim-binary", sim_binary, "--run-config", multi_run_config,
            "--output-dir", str(out_dir), "--verbose", "0",
            "m-ppo", "--total-timesteps", "16", "--n-steps", "16"]
    monkeypatch.setattr(sys, "argv", argv)
    assert train.main() == 0
    capsys.readouterr()

    assert inspect_model.main(["--run-dir", str(out_dir), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["manifest_version"] == 4
    assert report["status"] == "completed"
    assert report["contract"]["contract"] == "mesh_move_2d_v1"
    assert report["observation_schema"]["note"] == "bounds not structural for raw_links_v1"
    final = [entry for entry in report["models"] if entry["role"] == "final"]
    assert len(final) == 1
    assert final[0]["exists"] and final[0]["digest_ok"] is True
    assert all(row["match"] for row in report["package_versions"])

    assert inspect_model.main(["--run-dir", str(out_dir)]) == 0
    assert "digest=ok" in capsys.readouterr().out

    model = out_dir / "maskable_ppo_mesh.zip"
    model.write_bytes(model.read_bytes()[:64])
    assert inspect_model.main(["--run-dir", str(out_dir)]) == 2
    assert "digest_ok=False" in capsys.readouterr().err
