"""MeshRlEnv / EpisodeSession lifecycle: exit paths, manifests, leaks, signatures, masks.

Expectations: scripts/rl/env/README.md ("Failure and close", "Control Modes",
"Output"), scripts/rl/env/CLAUDE.md ("Every exit path must leave no child
process or reader thread"; "A later reset whose contract signature differs is a
protocol error, and the original spaces stay"), and src/rl/README.md
("Message fields": signature contents).
"""

import json
from pathlib import Path

import gymnasium
import numpy as np
import pytest

from scripts.rl.env import episode as episode_mod
from scripts.rl.env.mesh_env import MeshRlEnv

from ._helpers import (CENTRAL_INI, LEGACY_INI, ScriptedPeer, centralized_script,
                       drain_threads, episode_manifest, episode_steps, legacy_script,
                       make_init, make_legacy_step, write_ini)


@pytest.fixture
def peer(tmp_path, monkeypatch) -> ScriptedPeer:
    return ScriptedPeer(tmp_path, monkeypatch)


@pytest.fixture
def out_dir(tmp_path) -> Path:
    return tmp_path / "train"


@pytest.fixture
def central_ini(tmp_path) -> str:
    return write_ini(tmp_path, CENTRAL_INI)


@pytest.fixture
def legacy_ini(tmp_path) -> str:
    return write_ini(tmp_path, LEGACY_INI)


@pytest.fixture
def short_waits(monkeypatch):
    monkeypatch.setattr(episode_mod, "_NATURAL_EXIT_S", 0.3)
    monkeypatch.setattr(episode_mod, "_ESCALATION_WAIT_S", 1.0)


def _env(binary, ini, out_dir, **kwargs) -> MeshRlEnv:
    return MeshRlEnv(str(binary), ini, output_dir=str(out_dir), **kwargs)


def _assert_clean(env: MeshRlEnv, proc=None) -> None:
    assert env._proc is None
    if proc is not None:
        assert proc.poll() is not None, "simulator child was left running"
    assert drain_threads() == [], "stdout drain thread leaked"


def _legacy_steps(count: int, links: int = 2, **kwargs) -> list[dict]:
    return [make_legacy_step(t, links=links, done=t == count - 1, **kwargs)
            for t in range(count)]


# Lifecycle without an episode --------------------------------------------------

def test_close_before_reset_is_a_noop(peer, central_ini, out_dir):
    env = _env(peer.binary, central_ini, out_dir)
    env.close()
    env.close()
    assert not out_dir.exists() or not list(out_dir.glob("episode-*"))
    assert peer.argvs() == []
    assert env.action_space is None and env.observation_space is None


# Launch failures -----------------------------------------------------------------


def test_launch_command_shape(peer, central_ini, out_dir):
    peer.script(centralized_script(make_init()))
    env = _env(peer.binary, central_ini, out_dir, seed=9)
    env.reset()
    env.close()
    (argv,) = peer.argvs()
    assert argv == [f"--run-config={central_ini}", "--rl-mode", "--seed=9",
                    f"--output-dir={out_dir / 'episode-0000'}"]


# Early exit / bad first line -----------------------------------------------------

def test_exit_before_first_message_reports_code_and_stderr(peer, central_ini, out_dir):
    peer.script([{"stderr": "peer: cannot open nodes.json"}, {"exit": 7}])
    env = _env(peer.binary, central_ini, out_dir)
    with pytest.raises(RuntimeError) as err:
        env.reset()
    text = str(err.value)
    assert "ended unexpectedly (exit code 7)" in text
    assert "peer: cannot open nodes.json" in text
    manifest = episode_manifest(out_dir, 0)
    assert manifest["status"] == "failed" and manifest["exit_code"] == 7
    assert manifest["error"] == "sim process ended unexpectedly"
    _assert_clean(env)

    # A corrected simulator works on the next reset, in a fresh episode dir.
    peer.script(centralized_script(make_init()))
    obs, info = env.reset()
    assert obs.shape == (24,) and info["decision"] == 0
    env.close()
    assert episode_manifest(out_dir, 1)["status"] == "interrupted"
    _assert_clean(env)


# Mid-episode failures ------------------------------------------------------------


def test_fail_seed_mid_episode_then_recover(sim_binary, multi_run_config, out_dir,
                                            monkeypatch):
    monkeypatch.setenv("FAKE_SIM_FAIL_SEEDS", "5")
    env = _env(sim_binary, multi_run_config, out_dir)
    env.reset(seed=5)
    proc = env._proc
    _, _, done, _, _ = env.step([4, 4, 4])     # decision 1 arrives, then exit 3
    assert not done
    with pytest.raises(RuntimeError) as err:
        env.step([4, 4, 4])
    assert "exit code 3" in str(err.value)
    assert "fake-sim: simulated fatal error" in str(err.value)
    failed = episode_manifest(out_dir, 0)
    assert failed["status"] == "failed" and failed["exit_code"] == 3
    assert failed["steps"] == 1 and failed["decisions"] == 1
    _assert_clean(env, proc)

    env.reset(seed=1)
    done = False
    while not done:
        _, _, done, _, _ = env.step([4, 4, 4])
    completed = episode_manifest(out_dir, 1)
    assert completed["status"] == "completed" and completed["seed"] == 1
    env.close()
    _assert_clean(env)


def test_done_with_nonzero_exit_is_failed(peer, central_ini, out_dir):
    init = make_init()
    peer.script(centralized_script(init) + [{"exit": 3}])
    env = _env(peer.binary, central_ini, out_dir)
    env.reset()
    env.step([4, 4, 4])
    _, _, done, _, _ = env.step([4, 4, 4])
    assert done
    manifest = episode_manifest(out_dir, 0)
    assert manifest["status"] == "failed" and manifest["exit_code"] == 3
    assert manifest["stop_reason"] == "done"
    _assert_clean(env)


def test_completed_manifest_fields(peer, central_ini, out_dir):
    init = make_init()
    peer.script(centralized_script(init))
    env = _env(peer.binary, central_ini, out_dir)
    env.reset()
    rewards = [env.step([4, 4, 4])[1] for _ in range(2)]
    manifest = episode_manifest(out_dir, 0)
    assert manifest["status"] == "completed" and manifest["exit_code"] == 0
    assert manifest["stop_reason"] == "done" and manifest["escalation"] == "exited"
    assert manifest["steps"] == 2 and manifest["decisions"] == 2
    assert manifest["last_tick"] == 10
    assert manifest["cumulative_reward"] == pytest.approx(sum(rewards))
    assert manifest["contract"] == init
    assert manifest["started_at"] <= manifest["ended_at"]
    assert peer.actions() == [{"action": [4, 4, 4]}, {"action": [4, 4, 4]}]
    _assert_clean(env)


# Close / reset interrupt -----------------------------------------------------------

def test_reset_interrupts_running_centralized_episode(peer, central_ini, out_dir):
    peer.script(centralized_script(make_init()))
    env = _env(peer.binary, central_ini, out_dir)
    env.reset()
    first_proc = env._proc
    env.reset()
    assert first_proc.poll() is not None
    manifest = episode_manifest(out_dir, 0)
    assert manifest["status"] == "interrupted" and manifest["stop_reason"] == "reset"
    assert manifest["manifest_version"] == 3 and manifest["escalation"] == "exited"
    assert episode_manifest(out_dir, 1)["status"] == "running"
    env.close()
    second = episode_manifest(out_dir, 1)
    assert second["status"] == "interrupted" and second["stop_reason"] == "close"
    _assert_clean(env)


def test_legacy_close_manifest_is_version_1(peer, legacy_ini, out_dir):
    peer.script(legacy_script(_legacy_steps(5)))
    env = _env(peer.binary, legacy_ini, out_dir)
    env.reset()
    env.step(6)
    env.close()
    manifest = episode_manifest(out_dir, 0)
    assert manifest["manifest_version"] == 1
    assert manifest["status"] == "interrupted" and manifest["steps"] == 1
    assert "stop_reason" not in manifest and "contract" not in manifest
    _assert_clean(env)


# Escalation ----------------------------------------------------------------------

def test_close_terminates_simulator_that_ignores_eof(peer, legacy_ini, out_dir,
                                                     short_waits):
    peer.script([{"emit": make_legacy_step(0)}, {"hang": 30}])
    env = _env(peer.binary, legacy_ini, out_dir)
    env.reset()
    proc = env._proc
    env.close()
    manifest = episode_manifest(out_dir, 0)
    assert manifest["status"] == "interrupted"
    assert manifest["exit_code"] == -15
    _assert_clean(env, proc)


def test_close_kills_simulator_that_ignores_sigterm(peer, central_ini, out_dir,
                                                    short_waits):
    init = make_init()
    peer.script([{"ignore_term": True}, {"emit": init},
                 {"emit": episode_steps(init)[0]}, {"hang": 30}])
    env = _env(peer.binary, central_ini, out_dir)
    env.reset()
    proc = env._proc
    env.close()
    manifest = episode_manifest(out_dir, 0)
    assert manifest["status"] == "interrupted" and manifest["escalation"] == "killed"
    assert manifest["exit_code"] == -9
    _assert_clean(env, proc)


def test_terminated_escalation_is_recorded(peer, central_ini, out_dir, short_waits):
    init = make_init()
    peer.script([{"emit": init}, {"emit": episode_steps(init)[0]}, {"hang": 30}])
    env = _env(peer.binary, central_ini, out_dir)
    env.reset()
    env.close()
    manifest = episode_manifest(out_dir, 0)
    assert manifest["escalation"] == "terminated" and manifest["exit_code"] == -15
    _assert_clean(env)


# Actions and masks (centralized) ---------------------------------------------------


def test_centralized_mask_is_passed_through_unchanged(peer, central_ini, out_dir):
    init = make_init()
    steps = episode_steps(init)
    # Inside the arena, yet the simulator masks west/south: Python must not re-derive.
    steps[0]["mask"][0:5] = [0, 1, 0, 1, 1]
    steps[1]["mask"][5:10] = [1, 0, 1, 0, 1]
    peer.script(centralized_script(init, steps))
    env = _env(peer.binary, central_ini, out_dir)
    env.reset()
    mask = env.action_masks()
    assert mask.dtype == bool
    assert mask.astype(int).tolist() == steps[0]["mask"]
    mask[:] = False                          # caller mutation must not leak back
    assert env.action_masks().astype(int).tolist() == steps[0]["mask"]
    assert env.valid_action_mask().astype(int).tolist() == steps[0]["mask"]
    env.step([1, 4, 4])
    assert env.action_masks().astype(int).tolist() == steps[1]["mask"]
    env.close()


def test_info_reports_revalidated_slots(peer, central_ini, out_dir):
    init = make_init()
    steps = episode_steps(init)
    steps[1]["revalidated_slots"] = [2]
    peer.script(centralized_script(init, steps))
    env = _env(peer.binary, central_ini, out_dir)
    env.reset()
    _, _, _, _, info = env.step([4, 4, 0])
    assert info == {"tick": 5, "time_s": 0.5, "decision": 1, "ticks_in_step": 5,
                    "revalidated_slots": [2]}
    env.close()


# Contract signature across resets --------------------------------------------------


# Legacy mode: Python-derived Discrete(7) mask ----------------------------------------

LEGACY_EDGE_INI = LEGACY_INI + """x_min = 0.0
x_max = 5.0
y_min = -5.0
y_max = 0.0
z_min = 0.0
z_max = 5.0
"""


def test_legacy_mask_at_bound_edges(sim_binary, tmp_path, out_dir):
    # fake_sim legacy starts at (0,0,0) and moves 5 m per action (no clamping).
    ini = write_ini(tmp_path, LEGACY_EDGE_INI, "edge.ini")
    env = _env(sim_binary, ini, out_dir)
    env.reset()
    expected = {
        None: [0, 1, 1, 0, 0, 1, 1],     # x=xmin, y=ymax, z=zmin
        1: [1, 0, 1, 0, 0, 1, 1],        # +X -> x=xmax
        5: [1, 0, 1, 0, 1, 0, 1],        # +Z -> z=zmax
        2: [1, 0, 0, 1, 1, 0, 1],        # -Y -> y=ymin
        4: [1, 0, 0, 1, 0, 1, 1],        # legacy 4 is -Z -> z=zmin
    }
    for action, mask in expected.items():
        if action is not None:
            env.step(action)
        assert env.action_masks().astype(int).tolist() == mask, action
    env.close()
    _assert_clean(env)


def test_legacy_mask_inside_bounds_all_valid(peer, legacy_ini, out_dir):
    peer.script(legacy_script([make_legacy_step(0, pos=(10.0, 10.0, 10.0)),
                               make_legacy_step(1, pos=(-1000.0, 1000.0, 100.0),
                                                done=True)]))
    env = _env(peer.binary, legacy_ini, out_dir)
    env.reset()
    assert env.action_masks().all()
    env.step(6)
    # Defaults (-1000,2000)/(-1000,1000)/(0,100): at x_min, y_max, z_max.
    assert env.action_masks().astype(int).tolist() == [0, 1, 1, 0, 1, 0, 1]
    env.close()


def test_legacy_mask_beyond_bounds_masks_outward(peer, tmp_path, out_dir):
    ini = write_ini(tmp_path, LEGACY_EDGE_INI, "edge.ini")
    peer.script(legacy_script([make_legacy_step(0, pos=(7.0, -9.0, 6.0))]))
    env = _env(peer.binary, ini, out_dir)
    env.reset()
    # Outside the box on +x, -y, +z: only the inward directions and stay remain.
    assert env.action_masks().astype(int).tolist() == [1, 0, 0, 1, 1, 0, 1]
    env.close()


def test_legacy_mask_without_z_coordinate_leaves_z_valid(peer, tmp_path, out_dir):
    ini = write_ini(tmp_path, LEGACY_EDGE_INI, "edge.ini")
    peer.script(legacy_script([make_legacy_step(0, pos=(0.0, 0.0))]))
    env = _env(peer.binary, ini, out_dir)
    env.reset()
    assert env.action_masks().astype(int).tolist() == [0, 1, 1, 0, 1, 1, 1]
    env.close()


def test_legacy_continuous_action_space_and_forwarding(peer, legacy_ini, out_dir):
    peer.script(legacy_script(_legacy_steps(3, action_type="continuous")))
    env = _env(peer.binary, legacy_ini, out_dir)
    env.reset()
    assert isinstance(env.action_space, gymnasium.spaces.Box)
    assert env.action_space.shape == (2,)
    env.step(np.array([0.5, -1]))
    assert peer.actions() == [{"action": [0.5, -1.0]}]
    env.close()


def test_legacy_reset_info_and_obs(peer, legacy_ini, out_dir):
    peer.script(legacy_script([make_legacy_step(0, pos=(1.0, 2.0, 3.0))]))
    env = _env(peer.binary, legacy_ini, out_dir)
    obs, info = env.reset()
    assert obs.tolist() == [1.0, 2.0, 3.0, 15.0, 80.0, 15.0, 80.0]
    assert info == {"time_s": 0.0, "tick": 0}
    assert env.control_mode == "legacy" and env.contract is None
    assert env.observation_space.shape == (7,)
    env.close()


def test_manifest_json_is_valid_on_every_status(peer, central_ini, out_dir):
    # Every manifest written must be strict JSON (no NaN) with an ISO end time.
    peer.script(centralized_script(make_init()))
    env = _env(peer.binary, central_ini, out_dir)
    env.reset()
    env.reset()
    env.close()
    for index in (0, 1):
        text = (out_dir / f"episode-{index:04d}" / "rl_episode.json").read_text()
        manifest = json.loads(text, parse_constant=lambda c: pytest.fail(c))
        assert manifest["ended_at"] and manifest["status"] == "interrupted"


# Telemetry on failure, and EpisodeSession no-op paths -------------------------------

def test_failed_episode_flushes_telemetry_and_counts(peer, central_ini, out_dir):
    from scripts.rl.env.selection import resolve_selection
    from scripts.rl.env.telemetry import replay_file

    init = make_init()
    steps = episode_steps(init)
    bad = dict(steps[2], decision=5)
    peer.script(centralized_script(init, [steps[0], steps[1], bad]))
    selection = resolve_selection(central_ini, telemetry="steps")
    env = _env(peer.binary, central_ini, out_dir, selection=selection)
    env.reset()
    env.step([4, 4, 4])
    with pytest.raises(RuntimeError, match="decision 5 does not follow 1"):
        env.step([4, 4, 4])
    manifest = episode_manifest(out_dir, 0)
    assert manifest["status"] == "failed"
    assert manifest["telemetry"] == {"file": "steps.jsonl", "records": 2, "every": 1}
    lines = (out_dir / "episode-0000" / "steps.jsonl").read_text().splitlines()
    assert len(lines) == 3                     # header + reset + decision 1
    summary = replay_file(out_dir / "episode-0000" / "steps.jsonl")
    assert (summary.records, summary.obs_mismatches) == (2, 0)
    _assert_clean(env)
