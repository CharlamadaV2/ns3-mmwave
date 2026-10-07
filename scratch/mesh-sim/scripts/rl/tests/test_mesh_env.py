"""Contract tests for MeshRlEnv and the training CLI, driven by fake_sim.py."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import gymnasium
import numpy as np
import pytest

from scripts.rl.bootstrap_venv import DIRECT_DEPS
from scripts.rl.env.mesh_env import MeshRlEnv
from scripts.rl.env.selection import resolve_selection
from scripts.rl.env.telemetry import replay_file

MESH_SIM_ROOT = Path(__file__).resolve().parents[3]
FAKE_SIM = Path(__file__).resolve().parent / "fake_sim.py"

RUN_INI = """[scenario]
name = fake
seed = 1
duration_s = 0.4
tick_s = 0.1

[rl]
enabled = true
controlled_node_id = relay
action_type = discrete
reward_type = all_links_los
step_size_m = 5.0
"""

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
def run_config(tmp_path: Path) -> str:
    path = tmp_path / "run.ini"
    path.write_text(RUN_INI)
    (tmp_path / "nodes.json").write_text(NODES_JSON)
    return str(path)


@pytest.fixture
def sim_binary(tmp_path: Path) -> str:
    """Executable shim so MeshRlEnv can spawn the fake sim like a real binary."""
    shim = tmp_path / "fake-sim"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_SIM}" "$@"\n')
    shim.chmod(0o755)
    return str(shim)


def _episode_manifest(out_dir: Path, index: int) -> dict:
    return json.loads((out_dir / f"episode-{index:04d}" / "rl_episode.json").read_text())


# 1. Episode directory allocation ------------------------------------------------

def test_two_resets_allocate_distinct_episodes(sim_binary, run_config, tmp_path):
    out_dir = tmp_path / "train"
    env = MeshRlEnv(sim_binary, run_config, output_dir=str(out_dir))
    env.reset()
    first = _episode_manifest(out_dir, 0)
    env.reset()
    env.close()

    second = _episode_manifest(out_dir, 1)
    assert (out_dir / "episode-0000").is_dir() and (out_dir / "episode-0001").is_dir()
    assert first["episode"] == 0 and second["episode"] == 1
    assert _episode_manifest(out_dir, 0)["command"] == first["command"]
    for manifest in (first, second):
        assert manifest["seed"] == 1
        assert manifest["seed_source"] == "run.ini"
        assert manifest["manifest_version"] == 1
    for index in (0, 1):
        manifest = _episode_manifest(out_dir, index)
        assert manifest["status"] == "interrupted"
        assert manifest["exit_code"] == 0


def test_done_episode_is_completed(sim_binary, run_config, tmp_path):
    out_dir = tmp_path / "train"
    env = MeshRlEnv(sim_binary, run_config, output_dir=str(out_dir))
    env.reset()
    for _ in range(4):
        _, _, done, _, _ = env.step(6)
    assert done
    env.close()
    manifest = _episode_manifest(out_dir, 0)
    assert manifest["status"] == "completed" and manifest["steps"] == 4


# 2. Seed resolution -------------------------------------------------------------

def test_reset_seed_override_persists(sim_binary, run_config, tmp_path):
    out_dir = tmp_path / "train"
    env = MeshRlEnv(sim_binary, run_config, output_dir=str(out_dir))
    env.reset(seed=7)
    env.reset()
    env.close()

    assert _episode_manifest(out_dir, 0)["seed"] == 7
    assert _episode_manifest(out_dir, 1)["seed"] == 7
    assert _episode_manifest(out_dir, 1)["seed_source"] == "gym"


def test_reset_with_resolved_seed_keeps_source(sim_binary, run_config, tmp_path):
    out_dir = tmp_path / "train"
    env = MeshRlEnv(sim_binary, run_config, output_dir=str(out_dir))
    env.reset(seed=1)     # same value the run.ini already resolved
    env.close()

    manifest = _episode_manifest(out_dir, 0)
    assert manifest["seed"] == 1
    assert manifest["seed_source"] == "run.ini"


@pytest.mark.parametrize("options,expected_source", [
    ({"seed_source": "eval"}, "eval"),
    (None, "run.ini"),
])
def test_reset_options_label_the_seed_source(sim_binary, multi_run_config, tmp_path,
                                             options, expected_source):
    out_dir = tmp_path / "train"
    env = MeshRlEnv(sim_binary, multi_run_config, output_dir=str(out_dir))
    env.reset(seed=1, options=options)     # seed equals the run.ini value
    env.close()

    assert env.seed_value == 1 and env.seed_source == expected_source
    manifest = _episode_manifest(out_dir, 0)
    assert manifest["seed"] == 1
    assert manifest["seed_source"] == expected_source


def test_missing_seed_is_rejected(sim_binary, tmp_path):
    ini = tmp_path / "noseed.ini"
    ini.write_text("[scenario]\nduration_s = 0.4\ntick_s = 0.1\n")
    with pytest.raises(ValueError, match="seed"):
        MeshRlEnv(sim_binary, str(ini), output_dir=str(tmp_path / "train"))


@pytest.mark.parametrize("marker", ["#", ";"])
def test_seed_and_bounds_strip_inline_comments(sim_binary, run_config, tmp_path, marker):
    path = Path(run_config)
    path.write_text(RUN_INI.replace("seed = 1", f"seed = 1{marker}seed note") +
                    f"x_min = -10{marker}low\nx_max = 10 {marker}high\n"
                    f"y_min = -20{marker}low\ny_max = 20 {marker}high\n"
                    f"z_min = 0{marker}low\nz_max = 30 {marker}high\n")
    env = MeshRlEnv(sim_binary, run_config, output_dir=str(tmp_path / "train"))
    env.reset()
    assert env.seed_value == 1
    assert env._x_range == (-10.0, 10.0)
    assert env._y_range == (-20.0, 20.0)
    assert env._z_range == (0.0, 30.0)
    env.close()


# 3. Protocol failures -----------------------------------------------------------

def test_partial_bounds_override_one_endpoint(tmp_path):
    from scripts.rl.env.config import read_rl_bounds

    ini = tmp_path / "run.ini"
    ini.write_text("[rl]\nx_min = -10 # one endpoint only\n")
    assert read_rl_bounds(str(ini)) == (
        (-10.0, 2000.0), (-1000.0, 1000.0), (0.0, 100.0)
    )

def test_premature_exit_reports_code_and_stderr(sim_binary, run_config, tmp_path,
                                                monkeypatch):
    monkeypatch.setenv("FAKE_SIM_MODE", "exit3")
    env = MeshRlEnv(sim_binary, run_config, output_dir=str(tmp_path / "train"))
    env.reset()
    with pytest.raises(RuntimeError) as err:
        env.step(6)
    message = str(err.value)
    assert "exit code 3" in message
    assert "fake-sim: simulated fatal error" in message


def test_malformed_output_reports_line_and_text(sim_binary, run_config, tmp_path,
                                                monkeypatch):
    monkeypatch.setenv("FAKE_SIM_MODE", "malformed")
    env = MeshRlEnv(sim_binary, run_config, output_dir=str(tmp_path / "train"))
    env.reset()
    with pytest.raises(RuntimeError) as err:
        env.step(6)
    message = str(err.value)
    assert "message line 2" in message
    assert "this is not json {" in message


# 4. Band forwarding and action masks --------------------------------------------

@pytest.mark.parametrize("band", ["sub-6", None])
def test_band_flag_forwarding(sim_binary, run_config, tmp_path, band):
    env = MeshRlEnv(sim_binary, run_config, output_dir=str(tmp_path / "train"),
                    band=band)
    env.reset()
    env.close()
    flags = [a for a in env._cmd if a.startswith("--band")]
    assert flags == ([f"--band={band}"] if band else [])


def test_bound_defaults_match_cpp_and_mask_z(sim_binary, run_config, tmp_path):
    env = MeshRlEnv(sim_binary, run_config, output_dir=str(tmp_path / "train"))
    env.reset()
    assert env._x_range == (-1000.0, 2000.0)
    assert env._y_range == (-1000.0, 1000.0)
    assert env._z_range == (0.0, 100.0)

    mask = env.action_masks()
    assert not mask[4]      # at z_min -> -Z illegal
    assert mask[5] and mask[6]
    env.close()


# 5. Registration, CLI shape, and a tiny training run -----------------------------

def test_gym_registration_resolves(sim_binary, run_config, tmp_path):
    import scripts.rl  # noqa: F401  (performs the registration)

    env = gymnasium.make(
        "mesh_sim/MeshEnv-v0",
        disable_env_checker=True,
        sim_binary=sim_binary,
        run_config=run_config,
        output_dir=str(tmp_path / "train"),
    )
    assert isinstance(env.unwrapped, MeshRlEnv)
    env.close()


@pytest.mark.parametrize("argv,expected", [
    (["--help"], "--sim-binary"),
    (["--sim-binary", "x", "--run-config", "y", "m-ppo", "--help"], "--total-timesteps"),
])
def test_cli_help(argv, expected):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.rl.train", *argv],
        cwd=MESH_SIM_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    if argv == ["--help"]:
        assert "--total-timesteps" not in result.stdout


@pytest.mark.skipif(importlib.util.find_spec("sb3_contrib") is None,
                    reason="sb3_contrib not installed")
def test_tiny_training_run(sim_binary, run_config, tmp_path, monkeypatch):
    from scripts.rl import train

    out_dir = tmp_path / "train"
    argv = ["train", "--sim-binary", sim_binary, "--run-config", run_config,
            "--output-dir", str(out_dir), "--verbose", "0",
            "m-ppo", "--total-timesteps", "16", "--n-steps", "16"]
    monkeypatch.setattr(sys, "argv", argv)
    assert train.main() == 0

    manifest = json.loads((out_dir / "train_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["seed"] == 1 and manifest["seed_source"] == "run.ini"
    assert manifest["package_versions"]["sb3_contrib"]
    assert manifest["manifest_version"] == 4
    assert manifest["control_mode"] == "legacy"
    assert all(manifest[key] is None for key in
               ("selection", "observation_schema", "reward_schema", "telemetry"))
    assert set(manifest["platform"]) == {"system", "machine"}
    assert manifest["checkpoints"] == [] and manifest["evaluation"] is None
    assert manifest["best_model_path"] is None
    assert set(manifest["package_versions"]) == set(DIRECT_DEPS)
    assert len(manifest["package_versions"]) == 8
    assert all(manifest["package_versions"].values())
    assert (out_dir / "maskable_ppo_mesh.zip").is_file()

    # Every simulator episode, including the ones SB3 opens, must agree with the manifest.
    episodes = sorted(out_dir.glob("episode-*/rl_episode.json"))
    assert episodes
    for path in episodes:
        episode = json.loads(path.read_text())
        assert episode["seed"] == manifest["seed"], path
        assert episode["seed_source"] == manifest["seed_source"], path
        if episode["steps"] == 0:
            assert episode["status"] == "interrupted", path

    monkeypatch.setattr(sys, "argv", argv)
    assert train.main() == 1      # refuses to reuse a populated output directory


# 6. Centralized multi-node control ----------------------------------------------

SINGLE_SLOT_INI = (MULTI_RUN_INI
                   .replace("controlled_nodes = node-b, node-c",
                            "controlled_nodes = node-a")
                   .replace("max_controlled_nodes = 3", "max_controlled_nodes = 0"))


@pytest.fixture
def multi_run_config(tmp_path: Path) -> str:
    path = tmp_path / "run.ini"
    path.write_text(MULTI_RUN_INI)
    (tmp_path / "nodes.json").write_text(NODES_JSON)
    return str(path)


def _multi_env(sim_binary: str, run_config: str, tmp_path: Path) -> MeshRlEnv:
    return MeshRlEnv(sim_binary, run_config, output_dir=str(tmp_path / "train"))


def _mask_tuple(env: MeshRlEnv) -> tuple:
    flat = np.asarray(env.action_masks(), dtype=np.int8)
    return tuple(flat[i * 5:(i + 1) * 5] for i in range(len(env.action_space.nvec)))


def test_centralized_spaces_contract_and_padding(sim_binary, multi_run_config, tmp_path):
    env = _multi_env(sim_binary, multi_run_config, tmp_path)
    obs, info = env.reset()

    assert env.control_mode == "centralized"
    assert env.action_space == gymnasium.spaces.MultiDiscrete([5, 5, 5])
    assert env.observation_space.shape == (24,)
    assert obs.shape == (24,) and obs.dtype == np.float64

    contract = env.contract
    assert contract["contract"] == "mesh_move_2d_v1"
    assert contract["slot_node_ids"] == ["node-b", "node-c", None]
    assert (contract["obs_dim"], contract["mask_dim"]) == (24, 15)
    assert (contract["num_ticks"], contract["decision_interval_ticks"],
            contract["num_decisions"]) == (10, 5, 2)

    assert info == {"tick": 0, "time_s": 0.0, "decision": 0, "ticks_in_step": 1,
                    "revalidated_slots": []}
    assert list(obs[16:24]) == [0.0] * 8               # padded slot is all zeros
    mask = env.action_masks()
    assert mask.shape == (15,) and mask.dtype == bool
    assert list(mask[10:15]) == [False, False, False, False, True]
    env.close()


def test_centralized_single_slot_spaces(sim_binary, tmp_path):
    path = tmp_path / "run.ini"
    path.write_text(SINGLE_SLOT_INI)
    (tmp_path / "nodes.json").write_text(NODES_JSON)
    env = _multi_env(sim_binary, str(path), tmp_path)
    obs, _ = env.reset()

    assert env.action_space == gymnasium.spaces.MultiDiscrete([5])
    assert obs.shape == (8,)
    assert env.contract["slot_node_ids"] == ["node-a"]
    assert env.action_masks().shape == (5,)
    env.close()


def test_centralized_joint_action_is_forwarded(sim_binary, multi_run_config, tmp_path):
    env = _multi_env(sim_binary, multi_run_config, tmp_path)
    env.reset()
    obs, reward, terminated, truncated, info = env.step([2, 1, 4])

    # node-b drove south for 5 ticks; node-c drove east and clipped at x_max.
    assert (obs[1], obs[2]) == (100.0, -5.0)
    assert (obs[9], obs[10]) == (100.0, 50.0)
    assert list(obs[16:24]) == [0.0] * 8
    assert info == {"tick": 5, "time_s": 0.5, "decision": 1, "ticks_in_step": 5,
                    "revalidated_slots": []}
    assert reward == 1.0 and not terminated and not truncated

    mask = env.action_masks()
    assert not mask[1] and not mask[6]        # both controlled nodes sit at x_max
    assert mask[4] and mask[9] and mask[14]   # hold is always valid
    env.close()


def test_centralized_rejects_a_scalar_action(sim_binary, multi_run_config, tmp_path):
    env = _multi_env(sim_binary, multi_run_config, tmp_path)
    env.reset()
    with pytest.raises(ValueError, match="scalar"):
        env.step(4)
    with pytest.raises(ValueError, match="entries"):
        env.step([4, 4])
    env.close()


def test_centralized_masked_random_run_completes(sim_binary, multi_run_config, tmp_path):
    out_dir = tmp_path / "train"
    env = _multi_env(sim_binary, multi_run_config, tmp_path)
    env.reset()
    env.action_space.seed(3)

    done, steps = False, 0
    while not done:
        action = env.action_space.sample(mask=_mask_tuple(env))
        _, _, done, _, info = env.step(action)
        steps += 1
    assert steps == 2 and info["tick"] == 10

    env.close()
    manifest = _episode_manifest(out_dir, 0)
    assert manifest["manifest_version"] == 3
    assert manifest["status"] == "completed" and manifest["exit_code"] == 0
    assert manifest["control_mode"] == "centralized"
    assert manifest["contract"]["contract"] == "mesh_move_2d_v1"
    assert manifest["decisions"] == 2 and manifest["last_tick"] == 10
    assert manifest["stop_reason"] == "done" and manifest["escalation"] == "exited"


@pytest.mark.parametrize("mode,pattern", [
    ("bad_contract", "Unknown init contract"),
    ("bad_meanings", "action_meanings"),
    ("bad_counts", "num_controlled"),
    ("bad_obs_len", "obs length 23 != obs_dim 24"),
    ("bad_mask", r"mask\[0\] is not 0 or 1"),
    ("non_finite", r"obs\[1\] is not a finite number"),
])
def test_centralized_faults_are_rejected(sim_binary, multi_run_config, tmp_path,
                                         monkeypatch, mode, pattern):
    monkeypatch.setenv("FAKE_SIM_MODE", mode)
    out_dir = tmp_path / "train"
    env = _multi_env(sim_binary, multi_run_config, tmp_path)
    with pytest.raises(RuntimeError, match=pattern):
        env.reset()

    manifest = _episode_manifest(out_dir, 0)
    assert manifest["status"] == "failed"
    assert manifest["exit_code"] is not None
    assert "message line" in manifest["error"]
    assert env._proc is None

    # A later close must not relabel the recorded failure.
    env.close()
    assert _episode_manifest(out_dir, 0)["status"] == "failed"


def test_reset_signature_drift_fails_before_replacing_spaces(sim_binary,
                                                             multi_run_config, tmp_path):
    env = _multi_env(sim_binary, multi_run_config, tmp_path)
    env.reset()
    first_action, first_obs = env.action_space, env.observation_space

    Path(multi_run_config).write_text(
        MULTI_RUN_INI.replace("max_controlled_nodes = 3", "max_controlled_nodes = 4"))
    with pytest.raises(RuntimeError, match="contract changed between resets"):
        env.reset()
    assert env.action_space is first_action and env.observation_space is first_obs
    assert list(env.action_space.nvec) == [5, 5, 5]
    env.close()


def test_reset_mode_drift_is_rejected(sim_binary, multi_run_config, tmp_path):
    env = _multi_env(sim_binary, multi_run_config, tmp_path)
    env.reset()
    Path(multi_run_config).write_text(RUN_INI)
    with pytest.raises(RuntimeError, match="control_mode"):
        env.reset()
    assert isinstance(env.action_space, gymnasium.spaces.MultiDiscrete)
    env.close()


# 7. Bounded cleanup -------------------------------------------------------------

def _drain_threads() -> list:
    return [t for t in threading.enumerate() if t.name == "mesh-sim-drain"]


def test_long_stdout_is_drained_on_close(sim_binary, multi_run_config, tmp_path,
                                         monkeypatch):
    monkeypatch.setenv("FAKE_SIM_MODE", "long")
    out_dir = tmp_path / "train"
    before = threading.active_count()
    env = _multi_env(sim_binary, multi_run_config, tmp_path)
    env.reset()
    assert env.contract["num_ticks"] == 3000       # far more output than a pipe holds

    proc = env._proc
    env.close()

    assert proc.poll() is not None                 # reaped, not leaked
    assert env._proc is None
    assert _drain_threads() == []
    assert threading.active_count() == before
    manifest = _episode_manifest(out_dir, 0)
    assert manifest["status"] == "interrupted" and manifest["stop_reason"] == "close"
    assert manifest["exit_code"] == 0


def test_close_is_idempotent_and_reset_still_works(sim_binary, multi_run_config,
                                                   tmp_path):
    out_dir = tmp_path / "train"
    env = _multi_env(sim_binary, multi_run_config, tmp_path)
    env.reset()
    env.close()
    env.close()
    assert _episode_manifest(out_dir, 0)["status"] == "interrupted"

    env.reset()
    env.close()
    assert _episode_manifest(out_dir, 1)["status"] == "interrupted"
    assert _drain_threads() == []


def test_completed_episode_leaves_no_process_or_reader(sim_binary, multi_run_config,
                                                       tmp_path):
    before = threading.active_count()
    env = _multi_env(sim_binary, multi_run_config, tmp_path)
    env.reset()
    proc = env._proc
    env.step([4, 4, 4])
    _, _, done, _, _ = env.step([4, 4, 4])
    assert done
    assert proc.poll() == 0 and env._proc is None
    env.close()
    assert _drain_threads() == []
    assert threading.active_count() == before


# 8. Scenario identity -----------------------------------------------------------

def test_scenario_identity_uses_relative_nodes_and_inline_comments(tmp_path):
    from scripts.rl.env.config import read_scenario_identity

    nodes = tmp_path / "custom-nodes.json"
    nodes.write_text(NODES_JSON)
    ini = tmp_path / "run.ini"
    ini.write_text(MULTI_RUN_INI.replace(
        "nodes_file = nodes.json", "nodes_file = custom-nodes.json # relative"))

    identity = read_scenario_identity(str(ini))
    assert identity["run_config"] == str(ini.resolve())
    assert identity["run_ini_sha256"] == hashlib.sha256(ini.read_bytes()).hexdigest()
    assert identity["nodes_json_sha256"] == hashlib.sha256(nodes.read_bytes()).hexdigest()


def test_scenario_identity_accepts_absolute_nodes_path(tmp_path):
    from scripts.rl.env.config import read_scenario_identity

    nodes = tmp_path / "elsewhere" / "nodes.json"
    nodes.parent.mkdir()
    nodes.write_text(NODES_JSON)
    ini = tmp_path / "run.ini"
    ini.write_text(MULTI_RUN_INI.replace("nodes_file = nodes.json",
                                         f"nodes_file = {nodes}"))
    identity = read_scenario_identity(str(ini))
    assert identity["nodes_json_sha256"] == hashlib.sha256(nodes.read_bytes()).hexdigest()


def test_scenario_identity_reports_missing_nodes_file(tmp_path):
    from scripts.rl.env.config import read_scenario_identity

    ini = tmp_path / "run.ini"
    ini.write_text(MULTI_RUN_INI)
    with pytest.raises(FileNotFoundError, match="nodes file not found"):
        read_scenario_identity(str(ini))


BUILDINGS_JSON = """[
  {"id": "wall",
   "bounds": {"x_min": 90.0, "x_max": 110.0, "y_min": -20.0, "y_max": 20.0,
              "z_min": 0.0, "z_max": 20.0},
   "type": "Office", "ext_walls": "ConcreteWithWindows"}
]
"""

JAMMERS_JSON = """[
  {"id": "jammer-0", "enabled": true, "type": "constant",
   "target_freq": [2400.0], "tx_power_dbm": 30.0,
   "position": {"x": 50.0, "y": 0.0, "z": 10.0}}
]
"""


def _identity_ini(tmp_path: Path, *extra_scenario_lines: str) -> Path:
    """run.ini with nodes.json plus the given verbatim [scenario] lines."""
    ini = tmp_path / "run.ini"
    ini.write_text(MULTI_RUN_INI.replace(
        "nodes_file = nodes.json",
        "".join(["nodes_file = nodes.json\n", *extra_scenario_lines]).rstrip("\n")))
    (tmp_path / "nodes.json").write_text(NODES_JSON)
    return ini


def test_scenario_identity_hashes_buildings_and_jammers(tmp_path):
    from scripts.rl.env.config import read_scenario_identity

    buildings = tmp_path / "buildings.json"
    buildings.write_text(BUILDINGS_JSON)
    jammers = tmp_path / "elsewhere" / "jammers.json"
    jammers.parent.mkdir()
    jammers.write_text(JAMMERS_JSON)
    ini = _identity_ini(tmp_path, "buildings_file = buildings.json # blockage\n",
                        f"jammers_file = {jammers}\n")

    identity = read_scenario_identity(str(ini))
    assert identity["buildings_json_sha256"] == hashlib.sha256(
        buildings.read_bytes()).hexdigest()
    assert identity["jammers_json_sha256"] == hashlib.sha256(
        jammers.read_bytes()).hexdigest()


@pytest.mark.parametrize("extra", [
    ("buildings_file =\n", "jammers_file =   ; none\n"),
    (),
])
def test_scenario_identity_omits_blank_or_absent_optional_files(tmp_path, extra):
    from scripts.rl.env.config import read_scenario_identity

    identity = read_scenario_identity(str(_identity_ini(tmp_path, *extra)))
    assert identity["buildings_json_sha256"] is None
    assert identity["jammers_json_sha256"] is None


@pytest.mark.parametrize("key,label", [("buildings_file", "buildings file"),
                                       ("jammers_file", "jammers file")])
def test_scenario_identity_reports_missing_optional_file(tmp_path, key, label):
    from scripts.rl.env.config import read_scenario_identity

    ini = _identity_ini(tmp_path, f"{key} = missing.json\n")
    with pytest.raises(FileNotFoundError, match=f"{label} not found"):
        read_scenario_identity(str(ini))


# 9. Centralized training metadata and error-path cleanup ------------------------

@pytest.mark.skipif(importlib.util.find_spec("sb3_contrib") is None,
                    reason="sb3_contrib not installed")
def test_centralized_tiny_training_run(sim_binary, multi_run_config, tmp_path,
                                       monkeypatch):
    from scripts.rl import train

    out_dir = tmp_path / "train"
    argv = ["train", "--sim-binary", sim_binary, "--run-config", multi_run_config,
            "--output-dir", str(out_dir), "--verbose", "0",
            "m-ppo", "--total-timesteps", "16", "--n-steps", "16"]
    monkeypatch.setattr(sys, "argv", argv)
    assert train.main() == 0

    manifest = json.loads((out_dir / "train_manifest.json").read_text())
    assert manifest["manifest_version"] == 4
    assert manifest["status"] == "completed"
    assert manifest["control_mode"] == "centralized"
    assert manifest["contract"]["contract"] == "mesh_move_2d_v1"
    assert manifest["contract"]["max_controlled_nodes"] == 3
    assert set(manifest["platform"]) == {"system", "machine"}
    assert manifest["model_sha256"] == hashlib.sha256(
        (out_dir / "maskable_ppo_mesh.zip").read_bytes()).hexdigest()

    # Cadence off by default: no checkpoints, no during-training evaluation.
    assert manifest["checkpoints"] == []
    assert manifest["evaluation"] is None
    assert manifest["best_model_path"] is None
    assert manifest["best_model_sha256"] is None
    assert manifest["best_mean_reward"] is None
    assert not (out_dir / "checkpoints").exists()
    assert not (out_dir / "eval").exists()
    assert manifest["hyperparameters"]["checkpoint_every_steps"] == 0
    assert manifest["hyperparameters"]["eval_every_steps"] == 0

    identity = manifest["scenario_identity"]
    assert identity["run_config"] == str(Path(multi_run_config).resolve())
    assert identity["run_ini_sha256"] == hashlib.sha256(
        Path(multi_run_config).read_bytes()).hexdigest()
    assert identity["nodes_json_sha256"] == hashlib.sha256(
        (Path(multi_run_config).parent / "nodes.json").read_bytes()).hexdigest()

    episodes = sorted(out_dir.glob("episode-*/rl_episode.json"))
    assert episodes
    for path in episodes:
        episode = json.loads(path.read_text())
        assert episode["manifest_version"] == 3, path
        assert episode["status"] in ("completed", "interrupted"), path
        if episode["status"] == "completed":
            assert episode["decisions"] == 2 and episode["exit_code"] == 0, path
    assert _drain_threads() == []


@pytest.mark.skipif(importlib.util.find_spec("sb3_contrib") is None,
                    reason="sb3_contrib not installed")
def test_training_failure_closes_the_environment(sim_binary, multi_run_config,
                                                 tmp_path, monkeypatch):
    from scripts.rl import train

    monkeypatch.setenv("FAKE_SIM_MODE", "exit3")
    out_dir = tmp_path / "train"
    argv = ["train", "--sim-binary", sim_binary, "--run-config", multi_run_config,
            "--output-dir", str(out_dir), "--verbose", "0",
            "m-ppo", "--total-timesteps", "16", "--n-steps", "16"]
    monkeypatch.setattr(sys, "argv", argv)
    assert train.main() == 1

    manifest = json.loads((out_dir / "train_manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["error"]
    assert manifest["control_mode"] == "centralized"

    episodes = [json.loads(p.read_text())
                for p in sorted(out_dir.glob("episode-*/rl_episode.json"))]
    assert episodes and any(e["status"] == "failed" for e in episodes)
    assert _drain_threads() == []


def test_episode_allocation_scans_existing_directories_once(tmp_path, monkeypatch):
    from scripts.rl.env.episode import EpisodeSession

    (tmp_path / "episode-0007").mkdir()
    (tmp_path / "episode-10000").mkdir()
    original_iterdir = Path.iterdir
    scans = []

    def counted_iterdir(path):
        if path == tmp_path:
            scans.append(path)
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", counted_iterdir)
    session = EpisodeSession("sim", "run.ini", tmp_path, None)
    first, first_index = session._allocate_episode_dir()
    second, second_index = session._allocate_episode_dir()

    assert (first.name, first_index) == ("episode-10001", 10001)
    assert (second.name, second_index) == ("episode-10002", 10002)
    assert len(scans) == 1


# Configured observations, rewards, and telemetry ---------------------------------

CUSTOM_SELECTION = {"observation_preset": "local_links_v1",
                    "reward_components": "delivery_ratio,connectivity"}


def _custom_env(sim_binary: str, run_config: str, tmp_path: Path, **overrides) -> MeshRlEnv:
    selection = resolve_selection(run_config, **dict(CUSTOM_SELECTION, **overrides))
    return MeshRlEnv(sim_binary, run_config, output_dir=str(tmp_path / "train"),
                     selection=selection)


def _records(out_dir: Path, index: int = 0) -> tuple[dict, list[dict]]:
    lines = (out_dir / f"episode-{index:04d}" / "steps.jsonl").read_text().splitlines()
    parsed = [json.loads(line) for line in lines]
    return parsed[0], parsed[1:]


def _run_to_done(env: MeshRlEnv) -> None:
    done = False
    while not done:
        _, _, done, _, _ = env.step([4, 4, 4])


def test_custom_preset_and_reward_block(sim_binary, multi_run_config, tmp_path):
    env = _custom_env(sim_binary, multi_run_config, tmp_path)
    obs, info = env.reset()

    assert env.observation_space.shape == (36,)
    assert env.observation_space.dtype == np.float32
    assert obs.shape == (36,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert "reward" not in info

    obs, reward, terminated, _, info = env.step([4, 4, 4])
    block = info["reward"]
    assert set(block) == {"total", "components", "valid", "weights", "legacy"}
    assert reward == block["total"] == pytest.approx(1.5)
    assert block["components"]["delivery_ratio"] == pytest.approx(0.5)
    assert block["components"]["connectivity"] == pytest.approx(1.0)
    assert block["valid"] == {"delivery_ratio": 1, "connectivity": 1}
    assert block["legacy"] == 1.0
    assert obs.shape == (36,) and not terminated
    env.close()


@pytest.mark.parametrize("components", [None, "legacy"])
def test_negative_legacy_reward_window(sim_binary, multi_run_config, tmp_path,
                                       monkeypatch, components):
    monkeypatch.setenv("FAKE_SIM_MODE", "negative_reward")
    env = _custom_env(sim_binary, multi_run_config, tmp_path,
                      observation_preset=None, reward_components=components)
    try:
        env.reset()
        _, reward, _, _, info = env.step([4, 4, 4])
        assert reward == -1.0
        if components is None:
            assert "reward" not in info
        else:
            assert info["reward"]["total"] == info["reward"]["legacy"] == -1.0
    finally:
        env.close()


@pytest.mark.parametrize("telemetry,expected", [(None, False), ("steps", True)])
def test_telemetry_file_only_when_selected(sim_binary, multi_run_config, tmp_path,
                                           telemetry, expected):
    out_dir = tmp_path / "train"
    env = _custom_env(sim_binary, multi_run_config, tmp_path, telemetry=telemetry)
    env.reset()
    _run_to_done(env)
    env.close()

    assert (out_dir / "episode-0000" / "steps.jsonl").is_file() is expected
    manifest = _episode_manifest(out_dir, 0)
    assert (manifest["telemetry"] is not None) is expected


def test_telemetry_default_cadence_replays(sim_binary, multi_run_config, tmp_path):
    before = threading.active_count()
    out_dir = tmp_path / "train"
    env = _custom_env(sim_binary, multi_run_config, tmp_path, telemetry="steps")
    env.reset()
    _run_to_done(env)
    env.close()

    header, records = _records(out_dir)
    assert header["type"] == "header"
    assert len(records) == env.contract["num_decisions"] + 1 == 3
    assert [record["decision"] for record in records] == [0, 1, 2]
    assert records[0]["reward"] is None and records[0]["action_sent"] is None
    assert records[1]["action_sent"] == [4, 4, 4]
    assert records[1]["reward"]["total"] == pytest.approx(1.5)

    summary = replay_file(out_dir / "episode-0000" / "steps.jsonl")
    assert (summary.records, summary.obs_mismatches,
            summary.reward_mismatches) == (3, 0, 0)

    manifest = _episode_manifest(out_dir, 0)
    assert manifest["telemetry"] == {"file": "steps.jsonl", "records": 3, "every": 1}
    assert manifest["cumulative_reward"] == pytest.approx(3.0)
    assert _drain_threads() == []
    assert threading.active_count() == before


def test_telemetry_stride_keeps_full_manifest_totals(sim_binary, multi_run_config,
                                                     tmp_path):
    out_dir = tmp_path / "train"
    env = _custom_env(sim_binary, multi_run_config, tmp_path, telemetry="steps",
                      telemetry_every=2)
    env.reset()
    _run_to_done(env)
    env.close()

    _, records = _records(out_dir)
    assert [record["decision"] for record in records] == [0, 2]   # reset + terminal
    summary = replay_file(out_dir / "episode-0000" / "steps.jsonl")
    assert (summary.obs_mismatches, summary.reward_mismatches) == (0, 0)

    manifest = _episode_manifest(out_dir, 0)
    assert manifest["telemetry"] == {"file": "steps.jsonl", "records": 2, "every": 2}
    # Decision 1 was not saved, but both policy steps count in every total.
    assert manifest["decisions"] == 2
    assert manifest["cumulative_reward"] == pytest.approx(3.0)
    assert manifest["reward_components_sum"] == pytest.approx(
        {"delivery_ratio": 1.0, "connectivity": 2.0})


def test_zero_demand_masks_delivery_ratio(sim_binary, tmp_path):
    path = tmp_path / "run.ini"
    path.write_text(MULTI_RUN_INI + "\n[traffic]\ndemand_mbps = 0\n")
    (tmp_path / "nodes.json").write_text(NODES_JSON)
    env = _custom_env(sim_binary, str(path), tmp_path)
    env.reset()
    _, reward, _, _, info = env.step([4, 4, 4])

    block = info["reward"]
    assert block["valid"] == {"delivery_ratio": 0, "connectivity": 1}
    assert block["components"]["delivery_ratio"] == 0.0
    assert reward == pytest.approx(1.0)        # connectivity alone
    env.close()


@pytest.mark.parametrize("overrides", [
    {"observation_preset": None, "reward_components": None},
    {},
])
def test_binary_without_facts_is_rejected(sim_binary, multi_run_config, tmp_path,
                                          monkeypatch, overrides):
    monkeypatch.setenv("FAKE_SIM_MODE", "no_facts")
    out_dir = tmp_path / "train"
    env = _custom_env(sim_binary, multi_run_config, tmp_path, **overrides)
    with pytest.raises(RuntimeError, match="facts_schema missing"):
        env.reset()

    error = _episode_manifest(out_dir, 0)["error"]
    assert "requires a simulator binary emitting" in error
    env.close()


@pytest.mark.skipif(importlib.util.find_spec("sb3_contrib") is None,
                    reason="sb3_contrib not installed")
def test_training_records_selection_and_schema_hashes(sim_binary, multi_run_config,
                                                      tmp_path, monkeypatch):
    from scripts.rl import train

    out_dir = tmp_path / "train"
    argv = ["train", "--sim-binary", sim_binary, "--run-config", multi_run_config,
            "--output-dir", str(out_dir), "--verbose", "0",
            "--observation-preset", "local_links_v1",
            "--reward-components", "delivery_ratio,connectivity",
            "--reward-weights", "1.0,0.5",
            "--telemetry", "steps", "--telemetry-every", "2",
            "m-ppo", "--total-timesteps", "16", "--n-steps", "16"]
    monkeypatch.setattr(sys, "argv", argv)
    assert train.main() == 0

    manifest = json.loads((out_dir / "train_manifest.json").read_text())
    assert manifest["manifest_version"] == 4
    selection = manifest["selection"]
    assert selection["observation_preset"] == "local_links_v1"
    assert selection["reward_components"] == ["delivery_ratio", "connectivity"]
    assert selection["reward_weights"] == [1.0, 0.5]
    assert set(selection["source"].values()) == {"cli"}
    assert manifest["observation_schema"]["obs_dim"] == 36
    assert manifest["reward_schema"]["zero_demand_rule"] == "masked"
    assert manifest["telemetry"] == {"mode": "steps", "every": 2}

    obs_sha = manifest["observation_schema"]["sha256"]
    reward_sha = manifest["reward_schema"]["sha256"]
    episodes = sorted(out_dir.glob("episode-*/rl_episode.json"))
    assert episodes
    for path in episodes:
        episode = json.loads(path.read_text())
        assert episode["manifest_version"] == 3, path
        assert episode["observation_schema_sha256"] == obs_sha, path
        assert episode["reward_schema_sha256"] == reward_sha, path
        header = json.loads((path.parent / "steps.jsonl").read_text().splitlines()[0])
        assert header["observation_schema"]["sha256"] == obs_sha, path
        assert header["reward_schema"]["sha256"] == reward_sha, path
        assert header["selection"]["telemetry_every"] == 2, path
    assert _drain_threads() == []


def test_default_selection_telemetry_replays_via_raw_links(sim_binary, multi_run_config,
                                                         tmp_path):
    out_dir = tmp_path / "train"
    env = _custom_env(sim_binary, multi_run_config, tmp_path, observation_preset=None,
                      reward_components=None, telemetry="steps")
    obs, _ = env.reset()
    assert obs.shape == (24,) and obs.dtype == np.float64
    _run_to_done(env)
    env.close()

    header, records = _records(out_dir)
    assert header["observation_schema"]["schema_id"] == "raw_links_v1"
    assert header["reward_schema"]["authority"] == "cpp"
    assert records[1]["reward"] == {"total": 1.0, "source": "cpp"}
    summary = replay_file(out_dir / "episode-0000" / "steps.jsonl")
    assert (summary.obs_mismatches, summary.reward_mismatches) == (0, 0)
    assert _episode_manifest(out_dir, 0)["reward_components_sum"] == {}


# 12. Checkpoint retention and during-training evaluation ------------------------

@pytest.mark.skipif(importlib.util.find_spec("sb3_contrib") is None,
                    reason="sb3_contrib not installed")
def test_checkpoint_and_eval_cadence(sim_binary, multi_run_config, tmp_path, monkeypatch):
    from scripts.rl import train

    out_dir = tmp_path / "train"
    argv = ["train", "--sim-binary", sim_binary, "--run-config", multi_run_config,
            "--output-dir", str(out_dir), "--verbose", "0",
            "m-ppo", "--total-timesteps", "16", "--n-steps", "16",
            "--checkpoint-every-steps", "8", "--keep-checkpoints", "1",
            "--eval-every-steps", "8", "--eval-seed", "5"]
    monkeypatch.setattr(sys, "argv", argv)
    assert train.main() == 0

    manifest = json.loads((out_dir / "train_manifest.json").read_text())
    assert manifest["manifest_version"] == 4 and manifest["status"] == "completed"
    assert manifest["hyperparameters"]["checkpoint_every_steps"] == 8
    assert manifest["hyperparameters"]["keep_checkpoints"] == 1
    assert manifest["hyperparameters"]["eval_episodes"] == 1

    # Only the newest checkpoint survives pruning, and its digest is recorded.
    kept = sorted((out_dir / "checkpoints").glob("*.zip"))
    assert [path.name for path in kept] == ["checkpoint_16_steps.zip"]
    entry, = manifest["checkpoints"]
    assert Path(entry["path"]).resolve() == kept[0].resolve()
    assert entry["sha256"] == hashlib.sha256(kept[0].read_bytes()).hexdigest()
    assert entry["num_timesteps"] == 16

    best = out_dir / "best_model.zip"
    assert best.is_file() and (out_dir / "evaluations.npz").is_file()
    assert Path(manifest["best_model_path"]).resolve() == best.resolve()
    assert manifest["best_model_sha256"] == hashlib.sha256(best.read_bytes()).hexdigest()
    assert manifest["best_mean_reward"] is not None
    assert manifest["model_sha256"] == hashlib.sha256(
        (out_dir / "maskable_ppo_mesh.zip").read_bytes()).hexdigest()

    evaluation = manifest["evaluation"]
    assert [evaluation[key] for key in ("every_steps", "episodes", "seed", "seed_source")
            ] == [8, 1, 5, "eval"]
    assert Path(evaluation["output_dir"]).resolve() == (out_dir / "eval").resolve()
    assert Path(evaluation["log_path"]).resolve() == (out_dir / "evaluations.npz").resolve()

    eval_episodes = sorted((out_dir / "eval").glob("episode-*/rl_episode.json"))
    assert eval_episodes
    for path in eval_episodes:
        episode = json.loads(path.read_text())
        assert episode["seed"] == 5 and episode["seed_source"] == "eval", path
    training_episodes = sorted(out_dir.glob("episode-*/rl_episode.json"))
    assert training_episodes
    for path in training_episodes:
        assert json.loads(path.read_text())["seed"] == 1, path
    assert _drain_threads() == []


@pytest.mark.parametrize("flags", [
    ["--eval-episodes", "0"],
    ["--keep-checkpoints", "0"],
    ["--checkpoint-every-steps", "-1"],
    ["--eval-every-steps", "-8"],
])
def test_invalid_cadence_flags_exit_before_launch(sim_binary, multi_run_config, tmp_path,
                                                  monkeypatch, capsys, flags):
    from scripts.rl import train

    out_dir = tmp_path / "train"
    monkeypatch.setattr(sys, "argv",
                        ["train", "--sim-binary", sim_binary,
                         "--run-config", multi_run_config, "--output-dir", str(out_dir),
                         "--verbose", "0", "m-ppo", "--total-timesteps", "16", *flags])
    assert train.main() == 1
    assert capsys.readouterr().err.strip()
    assert not out_dir.exists()
