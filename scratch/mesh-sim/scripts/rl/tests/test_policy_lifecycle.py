"""Compatibility, bundle, policy, and evaluation-loop tests driven by fake_sim.py."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.rl import evaluate as evaluate_cli
from scripts.rl.cli_common import write_json
from scripts.rl.env.observations import SchemaMismatchError, observation_schema
from scripts.rl.env.rewards import reward_schema
from scripts.rl.env.selection import RlSelection
from scripts.rl.policy import evaluate as policy_evaluate
from scripts.rl.policy.bundle import (eval_selection, read_bundle, seed_roles,
                                      selection_from_manifest, training_provenance)
from scripts.rl.policy.compat import (BundleError, RewardMismatchError,
                                      ScenarioMismatchError, StructuralMismatchError,
                                      check_compatibility)
from scripts.sim_support import parse_seed_spec

FAKE_SIM = Path(__file__).resolve().parent / "fake_sim.py"

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

FOURTH_NODE = """,
  {"id": "node-d", "role": "peer", "mobility": "fixed", "node_type": "drone",
   "position": {"x": 10.0, "y": 10.0, "z": 10.0}}
]
"""


@pytest.fixture
def sim_binary(tmp_path: Path) -> str:
    """Executable shim so MeshRlEnv can spawn the fake sim like a real binary."""
    shim = tmp_path / "fake-sim"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_SIM}" "$@"\n')
    shim.chmod(0o755)
    return str(shim)


@pytest.fixture
def multi_run_config(tmp_path: Path) -> str:
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    (scenario / "run.ini").write_text(MULTI_RUN_INI)
    (scenario / "nodes.json").write_text(NODES_JSON)
    return str(scenario / "run.ini")


# 1. Compatibility checks on saved/live dictionaries -------------------------------

def _contract(node_ids=("node-a", "node-b", "node-c"),
              slot_node_ids=("node-b", "node-c", None), reward_type="all_links_los"):
    nodes, slots = len(node_ids), len(slot_node_ids)
    return {
        "contract": "mesh_move_2d_v1",
        "dimensions": 2,
        "action_meanings": ["west", "east", "south", "north", "hold"],
        "max_controlled_nodes": slots,
        "num_mesh_nodes": nodes,
        "num_links": nodes * (nodes - 1) // 2,
        "node_ids": list(node_ids),
        "slot_node_ids": list(slot_node_ids),
        "obs_dim": slots * (4 + 2 * (nodes - 1)),
        "mask_dim": 5 * slots,
        "facts_schema": "mesh_facts_v1",
        "bounds": {"x_min": 0.0, "x_max": 100.0, "y_min": -50.0, "y_max": 100.0,
                   "z_min": 0.0, "z_max": 50.0},
        "reward_type": reward_type,
        "reward_window": "mean",
    }


IDENTITY = {"run_config": "/saved/run.ini", "run_ini_sha256": "a" * 64,
            "nodes_json_sha256": "b" * 64, "buildings_json_sha256": "c" * 64,
            "jammers_json_sha256": None}


class _StubEnv:
    """Minimal stand-in exposing what check_compatibility reads from MeshRlEnv."""

    def __init__(self, contract, preset="raw_links_v1", components=(), weights=()):
        self.contract = contract
        self.observation_schema = observation_schema(preset, contract)
        self.reward_schema = reward_schema(components, weights,
                                           reward_type=contract["reward_type"],
                                           reward_window=contract["reward_window"])


def _manifest(contract, preset="raw_links_v1", components=(), weights=(),
              identity=None, band="mmwave"):
    return {
        "manifest_version": 4,
        "status": "completed",
        "control_mode": "centralized",
        "contract": contract,
        "band": band,
        "scenario_identity": dict(identity or IDENTITY),
        "selection": RlSelection(preset, tuple(components), tuple(weights)).describe(),
        "observation_schema": observation_schema(preset, contract),
        "reward_schema": reward_schema(components, weights,
                                       reward_type=contract["reward_type"],
                                       reward_window=contract["reward_window"]),
    }


def _check(manifest, env, identity=None, band="mmwave", allow=False):
    return check_compatibility(manifest, env, live_identity=dict(identity or IDENTITY),
                               live_band=band, allow_different_scenario=allow)


def test_matching_bundle_and_env_are_compatible():
    contract = _contract()
    report = _check(_manifest(contract), _StubEnv(contract))
    assert report.scenario == "ok" and report.warnings == []
    assert report.describe()["note"] == "matching checks do not imply transfer"


@pytest.mark.parametrize("field,value", [
    ("contract", "mesh_move_3d_v1"),
    ("dimensions", 3),
    ("action_meanings", ["north", "south", "east", "west", "hold"]),
    ("max_controlled_nodes", 4),
    ("num_mesh_nodes", 4),
    ("obs_dim", 99),
    ("mask_dim", 20),
    ("facts_schema", "mesh_facts_v2"),
])
def test_structural_field_change_is_refused(field, value):
    saved = _contract()
    live = dict(saved)
    live[field] = value
    env = _StubEnv(saved)
    env.contract = live
    with pytest.raises(StructuralMismatchError) as excinfo:
        _check(_manifest(saved), env)
    assert excinfo.value.fields == [field]
    assert field in str(excinfo.value)


def test_different_observation_preset_is_refused():
    contract = _contract()
    with pytest.raises(SchemaMismatchError) as excinfo:
        _check(_manifest(contract, preset="raw_links_v1"),
               _StubEnv(contract, preset="local_links_v1"))
    assert "schema_id" in excinfo.value.fields


def test_different_reward_composition_is_refused():
    contract = _contract()
    with pytest.raises(RewardMismatchError, match="sha256"):
        _check(_manifest(contract, components=["delivery_ratio"], weights=[1.0]),
               _StubEnv(contract, components=["connectivity"], weights=[1.0]))


def test_legacy_component_requires_the_same_cpp_reward():
    saved = _contract(reward_type="all_links_los")
    live = _contract(reward_type="delivery_ratio")
    with pytest.raises(RewardMismatchError, match="reward_type"):
        _check(_manifest(saved, components=["legacy"], weights=[1.0]),
               _StubEnv(live, components=["legacy"], weights=[1.0]))


def test_python_reward_without_legacy_ignores_the_cpp_reward_type():
    saved = _contract(reward_type="all_links_los")
    live = _contract(reward_type="delivery_ratio")
    report = _check(_manifest(saved, components=["delivery_ratio"], weights=[1.0]),
                    _StubEnv(live, components=["delivery_ratio"], weights=[1.0]))
    assert report.scenario == "ok"


def _scenario_case(name):
    saved = _contract()
    if name == "slot_node_ids":
        return saved, _contract(slot_node_ids=("node-c", "node-b", None)), IDENTITY, "mmwave"
    if name == "node_ids":
        return saved, _contract(node_ids=("node-a", "node-b", "node-z"),
                                slot_node_ids=("node-b", "node-z", None)), IDENTITY, "mmwave"
    if name == "band":
        return saved, saved, IDENTITY, "sub-6"
    identity = dict(IDENTITY)
    identity[name] = "d" * 64
    return saved, saved, identity, "mmwave"


SCENARIO_CASES = ["slot_node_ids", "node_ids", "run_ini_sha256", "nodes_json_sha256",
                  "buildings_json_sha256", "jammers_json_sha256", "band"]


@pytest.mark.parametrize("case", SCENARIO_CASES)
def test_scenario_identity_change_is_refused(case):
    saved, live, identity, band = _scenario_case(case)
    with pytest.raises(ScenarioMismatchError) as excinfo:
        _check(_manifest(saved), _StubEnv(live), identity=identity, band=band)
    assert any(case in difference for difference in excinfo.value.differences)


@pytest.mark.parametrize("case", SCENARIO_CASES)
def test_scenario_identity_change_can_be_overridden(case):
    saved, live, identity, band = _scenario_case(case)
    report = _check(_manifest(saved), _StubEnv(live), identity=identity, band=band,
                    allow=True)
    assert report.scenario == "overridden" and report.warnings
    assert report.describe()["scenario"] == "overridden"


def test_structural_mismatch_is_reported_before_scenario_identity():
    saved = _contract()
    live = dict(saved)
    live["num_mesh_nodes"] = 4
    env = _StubEnv(saved)
    env.contract = live
    identity = dict(IDENTITY, run_ini_sha256="d" * 64)
    with pytest.raises(StructuralMismatchError):
        _check(_manifest(saved), env, identity=identity)


# 2. Model bundles ----------------------------------------------------------------

def _write_run_dir(tmp_path: Path, **overrides) -> Path:
    run_dir = tmp_path / "train"
    (run_dir / "checkpoints").mkdir(parents=True)
    model = run_dir / "maskable_ppo_mesh.zip"
    model.write_bytes(b"model-bytes")
    checkpoint = run_dir / "checkpoints" / "checkpoint_16_steps.zip"
    checkpoint.write_bytes(b"checkpoint-bytes")
    from scripts.rl.cli_common import sha256_file

    manifest = _manifest(_contract())
    manifest.update({
        "run_config": str(tmp_path / "scenario" / "run.ini"),
        "model_path": str(model),
        "model_sha256": sha256_file(model),
        "best_model_path": None,
        "best_model_sha256": None,
        "best_mean_reward": None,
        "checkpoints": [{"path": str(checkpoint), "sha256": sha256_file(checkpoint),
                         "num_timesteps": 16}],
    })
    manifest.update(overrides)
    (run_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
    return run_dir


def test_read_bundle_accepts_final_and_checkpoint(tmp_path):
    run_dir = _write_run_dir(tmp_path)
    final = read_bundle(run_dir)
    assert final.selection == "final" and final.model_path.name.endswith(".zip")
    assert final.describe()["train_manifest_sha256"]

    checkpoint = read_bundle(run_dir, "checkpoints/checkpoint_16_steps.zip")
    assert checkpoint.selection == "checkpoint" and checkpoint.num_timesteps == 16


@pytest.mark.parametrize("overrides,model,pattern", [
    ({"status": "running"}, "final", "running"),
    ({"status": "failed"}, "final", "failed"),
    ({"manifest_version": 3}, "final", "retrain with current tooling"),
    ({"control_mode": "legacy"}, "final",
     "legacy control mode is not supported by this loader"),
    ({}, "best", "no best model"),
    ({}, "checkpoints/missing.zip", "not listed"),
    ({}, "../outside.zip", "must be"),
    ({}, "nested/dir/file.zip", "must be"),
])
def test_read_bundle_refuses_unusable_runs(tmp_path, overrides, model, pattern):
    run_dir = _write_run_dir(tmp_path, **overrides)
    with pytest.raises(BundleError, match=pattern):
        read_bundle(run_dir, model)


def test_read_bundle_refuses_a_missing_or_invalid_manifest(tmp_path):
    with pytest.raises(BundleError, match="train_manifest.json"):
        read_bundle(tmp_path / "absent")
    run_dir = _write_run_dir(tmp_path)
    (run_dir / "train_manifest.json").write_text("{not json")
    with pytest.raises(BundleError, match="readable JSON"):
        read_bundle(run_dir)


def test_read_bundle_refuses_a_missing_file_or_digest_mismatch(tmp_path):
    run_dir = _write_run_dir(tmp_path)
    model = run_dir / "maskable_ppo_mesh.zip"
    model.write_bytes(b"model-byte")            # truncated copy keeps the recorded sha
    with pytest.raises(BundleError, match="digest mismatch"):
        read_bundle(run_dir)
    model.unlink()
    with pytest.raises(BundleError, match="missing"):
        read_bundle(run_dir)


def test_selection_from_manifest_round_trips_and_eval_forces_telemetry(tmp_path):
    manifest = _manifest(_contract(), components=["delivery_ratio"], weights=[2.0])
    selection = selection_from_manifest(manifest)
    assert selection.reward_components == ("delivery_ratio",)
    assert selection.source["observation_preset"] == "manifest"

    forced = eval_selection(selection)
    assert forced.telemetry == "steps" and forced.telemetry_every == 1
    described = forced.describe()
    assert described["source"]["telemetry"] == "eval"
    assert described["source"]["telemetry_every"] == "eval"
    assert described["source"]["reward_components"] == "manifest"


@pytest.mark.parametrize("key,value,pattern", [
    ("observation_preset", "nope_v9", "not registered"),
    ("reward_components", ["nope"], "unknown entries"),
    ("telemetry", "verbose", "not valid"),
    ("telemetry_every", 0, ">= 1"),
])
def test_selection_from_manifest_revalidates(key, value, pattern):
    manifest = _manifest(_contract(), components=["delivery_ratio"], weights=[1.0])
    manifest["selection"][key] = value
    with pytest.raises(BundleError, match=pattern):
        selection_from_manifest(manifest)


# 3. Policies ---------------------------------------------------------------------

def _mask(*rows) -> np.ndarray:
    return np.asarray([bit for row in rows for bit in row], dtype=bool)


def test_hold_policy_holds_every_slot():
    mask = _mask([1, 0, 1, 1, 1], [0, 1, 1, 0, 1])
    action = policy_evaluate.HoldPolicy().act(np.zeros(4), mask, {})
    assert list(action) == [4, 4]
    assert all(mask[slot * 5 + choice] for slot, choice in enumerate(action))


def test_random_valid_policy_never_picks_a_masked_action():
    mask = _mask([1, 0, 0, 1, 1], [0, 0, 1, 0, 1])
    policy = policy_evaluate.RandomValidPolicy()
    policy.start_episode(3)
    for _ in range(50):
        action = policy.act(np.zeros(4), mask, {})
        assert all(mask[slot * 5 + choice] for slot, choice in enumerate(action))


def test_random_valid_policy_is_seeded_per_episode():
    mask = _mask([1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
    policy = policy_evaluate.RandomValidPolicy()

    def sequence(seed):
        policy.start_episode(seed)
        return [list(policy.act(np.zeros(4), mask, {})) for _ in range(20)]

    assert sequence(1) == sequence(1)
    assert sequence(1) != sequence(2)


# 4. Evaluation loop with the fake simulator --------------------------------------

def _eval_argv(sim_binary, run_config, out_dir, **extra):
    argv = ["--sim-binary", sim_binary, "--run-config", run_config,
            "--output-dir", str(out_dir), "--seeds", "1,2",
            "--policies", "hold,random_valid"]
    for flag, value in extra.items():
        argv.extend([f"--{flag.replace('_', '-')}", str(value)])
    return argv


def test_baseline_evaluation_writes_a_completed_manifest(sim_binary, multi_run_config,
                                                         tmp_path):
    out_dir = tmp_path / "eval"
    assert evaluate_cli.main(_eval_argv(sim_binary, multi_run_config, out_dir)) == 0

    manifest = json.loads((out_dir / "eval_manifest.json").read_text())
    assert manifest["eval_manifest_version"] == 2 and manifest["status"] == "completed"
    assert manifest["metric_source"] == {"kind": "telemetry_window",
                                         "warmup_excluded": False}
    assert manifest["label"] is None and manifest["training"] is None
    assert manifest["episodes_expected"] == 4 and manifest["episodes_completed"] == 4
    assert manifest["seed_roles"]["held_out"] is None
    assert manifest["seeds"] == [1, 2] and manifest["seed_source"] == "eval"
    assert manifest["deterministic"] is True and manifest["bundle"] is None
    assert manifest["selection"]["telemetry"] == "steps"
    assert manifest["selection"]["telemetry_every"] == 1
    assert manifest["contract"]["contract"] == "mesh_move_2d_v1"
    assert set(manifest["policies"]) == {"hold", "random_valid"}

    for name, block in manifest["policies"].items():
        assert block["summary"]["completed_episodes"] == 2
        assert block["summary"]["revalidated_slots_total"] == 0
        assert block["summary"]["mask_violations_total"] == 0
        for episode, seed in zip(block["episodes"], [1, 2]):
            assert episode["seed"] == seed and episode["status"] == "completed"
            assert episode["actions_sha256"]
            assert episode["metrics"]["connectivity"] is not None
            assert episode["metrics"]["los_fraction"] is not None
            saved = json.loads(
                (Path(episode["episode_dir"]) / "rl_episode.json").read_text())
            assert saved["seed_source"] == "eval" and saved["seed"] == seed
            assert episode["return"] == pytest.approx(saved["cumulative_reward"], abs=1e-9)
            assert Path(episode["episode_dir"]).parent.name == name


def test_json_output_is_the_eval_manifest(sim_binary, multi_run_config, tmp_path,
                                          capsys):
    out_dir = tmp_path / "eval"
    argv = _eval_argv(sim_binary, multi_run_config, out_dir) + ["--json"]
    assert evaluate_cli.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads((out_dir / "eval_manifest.json").read_text())


def test_evaluation_refuses_to_write_into_a_training_run(sim_binary, multi_run_config,
                                                         tmp_path):
    run_dir = tmp_path / "train"
    run_dir.mkdir()
    argv = _eval_argv(sim_binary, multi_run_config, run_dir / "eval")
    argv.extend(["--run-dir", str(run_dir)])
    assert evaluate_cli.main(argv) == 1

    populated = tmp_path / "eval"
    populated.mkdir()
    (populated / "train_manifest.json").write_text("{}")
    assert evaluate_cli.main(_eval_argv(sim_binary, multi_run_config, populated)) == 1


def test_model_policy_requires_a_run_dir(sim_binary, multi_run_config, tmp_path):
    argv = ["--sim-binary", sim_binary, "--run-config", multi_run_config,
            "--output-dir", str(tmp_path / "eval"), "--seeds", "1",
            "--policies", "model"]
    assert evaluate_cli.main(argv) == 1


@pytest.mark.parametrize("flag,value", [("--seeds", "1,1"), ("--seeds", "x"),
                                        ("--seeds", "3-1"), ("--policies", "greedy")])
def test_invalid_seed_and_policy_lists_are_refused(sim_binary, multi_run_config,
                                                   tmp_path, flag, value):
    argv = _eval_argv(sim_binary, multi_run_config, tmp_path / "eval")
    argv[argv.index(flag) + 1] = value
    assert evaluate_cli.main(argv) == 1


class _MaskedChoicePolicy:
    """Deliberately sends a masked index so revalidation and violations are counted."""

    name = "masked"

    def act(self, obs, mask, contract):
        flat = np.asarray(mask, dtype=bool)
        choices = []
        for slot in range(len(flat) // 5):
            window = flat[slot * 5:(slot + 1) * 5]
            invalid = np.flatnonzero(~window)
            choices.append(int(invalid[0]) if invalid.size else 4)
        return np.asarray(choices, dtype=np.int64)


def test_a_masked_action_is_counted_and_reported_as_exit_code_two(sim_binary,
                                                                  multi_run_config,
                                                                  tmp_path):
    from scripts.rl.env.mesh_env import MeshRlEnv
    from scripts.rl.env.selection import resolve_selection

    out_dir = tmp_path / "eval"
    selection = eval_selection(resolve_selection(multi_run_config))
    spec = policy_evaluate.PolicySpec(
        "masked", lambda env, seed: policy_evaluate.Prepared(_MaskedChoicePolicy()))

    def make_env(name):
        return MeshRlEnv(sim_binary, multi_run_config, seed=1,
                         output_dir=str(out_dir / name), selection=selection)

    manifest = policy_evaluate.evaluate(make_env, [spec], [1], out_dir, {})
    summary = manifest["policies"]["masked"]["summary"]
    assert summary["mask_violations_total"] > 0
    assert summary["revalidated_slots_total"] > 0
    assert evaluate_cli._exit_code(manifest, 1) == 2


# 5. Model reload end to end ------------------------------------------------------

def _train(sim_binary, run_config, out_dir, monkeypatch) -> int:
    from scripts.rl import train

    monkeypatch.setattr(sys, "argv", [
        "train", "--sim-binary", sim_binary, "--run-config", run_config,
        "--output-dir", str(out_dir), "--verbose", "0",
        "m-ppo", "--total-timesteps", "16", "--n-steps", "16", "--seed", "1"])
    return train.main()


@pytest.mark.skipif(importlib.util.find_spec("sb3_contrib") is None,
                    reason="sb3_contrib not installed")
def test_model_evaluation_reloads_a_trained_policy(sim_binary, multi_run_config,
                                                   tmp_path, monkeypatch):
    run_dir = tmp_path / "train"
    assert _train(sim_binary, multi_run_config, run_dir, monkeypatch) == 0

    out_dir = tmp_path / "eval"
    argv = ["--sim-binary", sim_binary, "--run-dir", str(run_dir),
            "--output-dir", str(out_dir), "--seeds", "11,12", "--policies", "model"]
    assert evaluate_cli.main(argv) == 0

    manifest = json.loads((out_dir / "eval_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["compatibility"]["scenario"] == "ok"
    assert manifest["compatibility"]["note"] == "matching checks do not imply transfer"
    assert manifest["bundle"]["model_selection"] == "final"
    assert manifest["bundle"]["train_manifest_sha256"]
    episodes = manifest["policies"]["model"]["episodes"]
    assert [episode["seed"] for episode in episodes] == [11, 12]
    # The compatibility reset is reused, so no interrupted zero-step episode is left.
    statuses = [json.loads(path.read_text())["status"]
                for path in sorted((out_dir / "model").glob("episode-*/rl_episode.json"))]
    assert statuses == ["completed", "completed"]


@pytest.mark.skipif(importlib.util.find_spec("sb3_contrib") is None,
                    reason="sb3_contrib not installed")
def test_model_evaluation_refuses_a_different_node_count(sim_binary, multi_run_config,
                                                         tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "train"
    assert _train(sim_binary, multi_run_config, run_dir, monkeypatch) == 0

    other = tmp_path / "scenario-4"
    other.mkdir()
    (other / "run.ini").write_text(MULTI_RUN_INI)
    (other / "nodes.json").write_text(NODES_JSON.rstrip()[:-1].rstrip() + FOURTH_NODE)

    argv = ["--sim-binary", sim_binary, "--run-dir", str(run_dir),
            "--run-config", str(other / "run.ini"),
            "--output-dir", str(tmp_path / "eval-4"), "--seeds", "11",
            "--policies", "model"]
    assert evaluate_cli.main(argv) == 1
    assert "StructuralMismatchError" in capsys.readouterr().err

    failed = json.loads((tmp_path / "eval-4" / "eval_manifest.json").read_text())
    assert failed["status"] == "failed"
    assert failed["error"].startswith("StructuralMismatchError")
    assert [episode["status"] for episode in failed["policies"]["model"]["episodes"]] \
        == ["not_run"]


# 6. Seed specifications ----------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("1-3,7", [1, 2, 3, 7]),
    ("301-303, 310", [301, 302, 303, 310]),
    (" 5 ", [5]),
    ("7-7", [7]),
])
def test_parse_seed_spec_expands_ranges_in_order(raw, expected):
    assert parse_seed_spec(raw) == expected


@pytest.mark.parametrize("raw", ["3-1", "1,1", "", "   ", "x", "1-x"])
def test_parse_seed_spec_refuses_bad_specifications(raw):
    with pytest.raises(ValueError):
        parse_seed_spec(raw)


def test_parse_seed_spec_keeps_repeats_when_not_distinct():
    assert parse_seed_spec("1,1-2", distinct=False) == [1, 1, 2]


# 7. Seed roles and training provenance -------------------------------------------

def _train_manifest(seed=101, evaluation=None):
    manifest = _manifest(_contract())
    manifest.update({"algorithm": "MaskablePPO", "seed": seed, "seed_source": "cli",
                     "evaluation": evaluation,
                     "hyperparameters": {"total_timesteps": 256, "n_steps": 64}})
    return manifest


def test_seed_roles_flag_training_and_selection_overlap():
    manifest = _train_manifest(evaluation={"seed": 201, "episodes": 1})
    roles = seed_roles(manifest, "best", [101, 201, 301])
    assert roles["training_seed"] == 101 and roles["model_selection_seed"] == 201
    assert roles["selection_seed_used_for_model"] is True
    assert roles["overlap"] == [101, 201] and roles["held_out"] is False
    assert roles["overlap_allowed"] is False


def test_seed_roles_without_an_evaluation_block_only_checks_the_training_seed():
    roles = seed_roles(_train_manifest(), "final", [201, 301])
    assert roles["model_selection_seed"] is None
    assert roles["selection_seed_used_for_model"] is False
    assert roles["overlap"] == [] and roles["held_out"] is True


def test_seed_roles_without_a_bundle_are_unknown():
    roles = seed_roles(None, None, [301, 302])
    assert roles["training_seed"] is None and roles["model_selection_seed"] is None
    assert roles["held_out"] is None and roles["held_out_seeds"] == [301, 302]


def test_training_provenance_copies_only_portable_fields():
    provenance = training_provenance(_train_manifest(
        evaluation={"seed": 201, "episodes": 1}))
    assert provenance["seed"] == 101 and provenance["evaluation_seed"] == 201
    assert provenance["evaluation_episodes"] == 1
    assert provenance["hyperparameters"]["total_timesteps"] == 256
    assert set(provenance["scenario_identity"]) == {
        "run_ini_sha256", "nodes_json_sha256", "buildings_json_sha256",
        "jammers_json_sha256"}


def test_evaluation_refuses_held_out_seeds_used_in_training(sim_binary, tmp_path,
                                                            capsys):
    run_dir = _write_run_dir(tmp_path, seed=1, evaluation={"seed": 5, "episodes": 1})
    out_dir = tmp_path / "eval"
    argv = ["--sim-binary", sim_binary, "--run-dir", str(run_dir), "--model", "final",
            "--output-dir", str(out_dir), "--seeds", "1,5,11", "--policies", "model"]
    assert evaluate_cli.main(argv) == 1

    err = capsys.readouterr().err
    assert "held-out seed 1 is the training seed" in err
    assert "held-out seed 5 is the model-selection seed" in err
    assert "--allow-seed-overlap" in err
    assert not out_dir.exists()


@pytest.mark.skipif(importlib.util.find_spec("sb3_contrib") is None,
                    reason="sb3_contrib not installed")
def test_allow_seed_overlap_warns_and_records_the_flag(sim_binary, multi_run_config,
                                                       tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "train"
    assert _train(sim_binary, multi_run_config, run_dir, monkeypatch) == 0

    out_dir = tmp_path / "eval"
    argv = ["--sim-binary", sim_binary, "--run-dir", str(run_dir),
            "--output-dir", str(out_dir), "--seeds", "1", "--policies", "model",
            "--label", "overlap-check", "--allow-seed-overlap"]
    assert evaluate_cli.main(argv) == 0
    assert "WARNING:" in capsys.readouterr().err

    manifest = json.loads((out_dir / "eval_manifest.json").read_text())
    assert manifest["label"] == "overlap-check"
    assert manifest["seed_roles"]["overlap"] == [1]
    assert manifest["seed_roles"]["overlap_allowed"] is True
    assert manifest["seed_roles"]["held_out"] is False
    assert manifest["training"]["seed"] == 1
    assert manifest["training"]["algorithm"] == "MaskablePPO"


# 8. Failure tolerance and atomic manifest writes ----------------------------------

def test_a_failed_episode_keeps_the_other_records(sim_binary, multi_run_config,
                                                  tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SIM_FAIL_SEEDS", "2")
    out_dir = tmp_path / "eval"
    argv = ["--sim-binary", sim_binary, "--run-config", multi_run_config,
            "--output-dir", str(out_dir), "--seeds", "1-3",
            "--policies", "hold,random_valid"]
    assert evaluate_cli.main(argv) == 1

    manifest = json.loads((out_dir / "eval_manifest.json").read_text())
    assert manifest["status"] == "partial"
    assert manifest["episodes_expected"] == 6 and manifest["episodes_completed"] == 4
    for block in manifest["policies"].values():
        episodes = block["episodes"]
        assert [episode["seed"] for episode in episodes] == [1, 2, 3]
        assert [episode["status"] for episode in episodes] == [
            "completed", "failed", "completed"]
        assert block["summary"]["expected_episodes"] == 3
        assert block["summary"]["completed_episodes"] == 2

        failed = episodes[1]
        assert set(failed["metrics"].values()) == {None}
        assert failed["error"]
        assert failed["summary_json"] is None
        saved = json.loads(
            (Path(failed["episode_dir"]) / "rl_episode.json").read_text())
        assert saved["status"] == "failed" and saved["seed"] == 2
        assert episodes[0]["metrics"]["unroutable_fraction"] == 0.0
        assert Path(episodes[0]["summary_json"]).is_file()


class _ManifestCountingPolicy:
    """Records how many of its own episode records the manifest holds on disk."""

    name = "counting"

    def __init__(self, manifest_path: Path):
        self._path = manifest_path
        self.counts: list[int] = []

    def start_episode(self, seed: int) -> None:
        manifest = json.loads(self._path.read_text())
        block = manifest["policies"].get(self.name) or {"episodes": []}
        self.counts.append(len(block["episodes"]))

    def act(self, obs, mask, contract):
        return np.full(len(mask) // 5, 4, dtype=np.int64)


def test_each_episode_record_is_written_before_the_next_episode(sim_binary,
                                                                multi_run_config,
                                                                tmp_path, monkeypatch):
    from scripts.rl.env.mesh_env import MeshRlEnv
    from scripts.rl.env.selection import resolve_selection

    monkeypatch.setenv("FAKE_SIM_FAIL_SEEDS", "2")
    out_dir = tmp_path / "eval"
    selection = eval_selection(resolve_selection(multi_run_config))
    policy = _ManifestCountingPolicy(out_dir / "eval_manifest.json")
    spec = policy_evaluate.PolicySpec(
        policy.name, lambda env, seed: policy_evaluate.Prepared(policy))

    def make_env(name):
        return MeshRlEnv(sim_binary, multi_run_config, seed=1,
                         output_dir=str(out_dir / name), selection=selection)

    manifest = policy_evaluate.evaluate(make_env, [spec], [1, 2, 3], out_dir, {})
    assert policy.counts == [0, 1, 2]
    assert manifest["status"] == "partial"


class _FinishedEpisodeEnv:
    """An env whose episode directory holds a manifest claiming the episode completed."""

    def __init__(self, episode_dir: Path):
        self._cmd = ["sim", f"--output-dir={episode_dir}"]


def test_a_raised_episode_is_recorded_as_failed_despite_its_manifest(tmp_path):
    episode_dir = tmp_path / "episode-0000"
    episode_dir.mkdir()
    write_json(episode_dir / "rl_episode.json",
               {"seed": 7, "status": "completed", "exit_code": 0, "decisions": 3,
                "cumulative_reward": -1.5})

    result = policy_evaluate._failed_result(
        _FinishedEpisodeEnv(episode_dir), 7,
        RuntimeError("Episode return 0.0 does not match cumulative_reward -1.5"),
        set())
    assert result.status == "failed"
    assert result.episode_dir == str(episode_dir)
    assert result.total_return == -1.5 and result.decisions == 3
    assert "does not match" in result.error
    assert set(result.metrics.values()) == {None}

    summary = policy_evaluate.summarize([result], expected=1)
    assert summary["completed_episodes"] == 0
    assert summary["mean_return"] is None


def test_write_json_replaces_the_file_atomically(tmp_path):
    path = tmp_path / "manifest.json"
    write_json(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert list(tmp_path.glob("*.tmp")) == []

    with pytest.raises(TypeError):
        write_json(path, {"a": object()})
    assert json.loads(path.read_text()) == {"a": 1}
    assert list(tmp_path.glob("*.tmp")) == []
