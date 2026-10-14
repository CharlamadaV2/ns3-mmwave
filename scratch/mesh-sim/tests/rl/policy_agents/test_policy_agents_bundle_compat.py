"""Gaps in policy/bundle.py and policy/compat.py on hand-built manifests and model files.

Contract sources: scripts/rl/policy/CLAUDE.md and README.md ("Bundle Verification",
"Compatibility Checks"): read_bundle accepts only a completed, centralized manifest at
MANIFEST_VERSION (== train.MANIFEST_VERSION) and verifies the chosen file's SHA-256,
with no bypass; checks run structural -> observation schema -> reward -> scenario and
only the scenario step is overridable.
"""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.rl.cli_common import sha256_file
from scripts.rl.env.observations import SchemaMismatchError, observation_schema
from scripts.rl.env.rewards import reward_schema
from scripts.rl.env.selection import RlSelection
from scripts.rl.policy import bundle
from scripts.rl.policy.bundle import (eval_selection, policy_weights_sha256, read_bundle, seed_roles, selection_from_manifest)
from scripts.rl.policy.compat import (BundleError, RewardMismatchError, ScenarioMismatchError, StructuralMismatchError, check_compatibility)


def _contract(node_ids=("node-a", "node-b", "node-c"),
              slot_node_ids=("node-b", "node-c", None), reward_type="all_links_los",
              reward_window="mean", x_max=100.0):
    nodes, slots = len(node_ids), len(slot_node_ids)
    return {"contract": "mesh_move_2d_v1", "dimensions": 2,
            "action_meanings": ["west", "east", "south", "north", "hold"],
            "max_controlled_nodes": slots, "num_mesh_nodes": nodes,
            "num_links": nodes * (nodes - 1) // 2, "node_ids": list(node_ids),
            "slot_node_ids": list(slot_node_ids),
            "obs_dim": slots * (4 + 2 * (nodes - 1)), "mask_dim": 5 * slots,
            "facts_schema": "mesh_facts_v1",
            "bounds": {"x_min": 0.0, "x_max": x_max, "y_min": -50.0, "y_max": 100.0,
                       "z_min": 0.0, "z_max": 50.0},
            "reward_type": reward_type, "reward_window": reward_window}


IDENTITY = {"run_ini_sha256": "a" * 64, "nodes_json_sha256": "b" * 64,
            "buildings_json_sha256": None, "jammers_json_sha256": None}


class _Env:
    """What check_compatibility reads from a live MeshRlEnv."""

    def __init__(self, contract, preset="raw_links_v1", components=("delivery_ratio",),
                 weights=(1.0,)):
        self.contract = contract
        self.observation_schema = observation_schema(preset, contract)
        self.reward_schema = reward_schema(components, weights,
                                           reward_type=contract["reward_type"],
                                           reward_window=contract["reward_window"])


def _saved(contract, preset="raw_links_v1", components=("delivery_ratio",), weights=(1.0,),
           band="mmwave", identity=None):
    return {"manifest_version": 4, "status": "completed", "control_mode": "centralized",
            "contract": contract, "band": band,
            "scenario_identity": dict(identity or IDENTITY),
            "selection": RlSelection(preset, tuple(components), tuple(weights)).describe(),
            "observation_schema": observation_schema(preset, contract),
            "reward_schema": reward_schema(components, weights,
                                           reward_type=contract["reward_type"],
                                           reward_window=contract["reward_window"])}


def _check(manifest, env, identity=None, band="mmwave", allow=False):
    return check_compatibility(manifest, env, live_identity=dict(identity or IDENTITY),
                               live_band=band, allow_different_scenario=allow)


# 1. Compatibility ordering and override scope ---------------------------------------


def test_structural_wins_over_every_later_step_even_with_override():
    saved = _contract()
    live = _contract(node_ids=("node-a", "node-b", "node-c", "node-d"),
                     slot_node_ids=("node-b", "node-d", None), reward_type="delivery_ratio")
    env = _Env(live, preset="local_links_v1", components=("connectivity",))
    with pytest.raises(StructuralMismatchError):
        _check(_saved(saved), env, identity=dict(IDENTITY, run_ini_sha256="z" * 64),
               band="sub-6", allow=True)


def test_observation_schema_wins_over_reward_and_scenario_even_with_override():
    contract = _contract()
    env = _Env(contract, preset="local_links_v1", components=("connectivity",))
    with pytest.raises(SchemaMismatchError):
        _check(_saved(contract), env, identity=dict(IDENTITY, run_ini_sha256="z" * 64),
               band="sub-6", allow=True)


def test_reward_mismatch_is_not_overridable():
    contract = _contract()
    env = _Env(contract, components=("connectivity",))
    with pytest.raises(RewardMismatchError):
        _check(_saved(contract), env, identity=dict(IDENTITY, run_ini_sha256="z" * 64),
               band="sub-6", allow=True)


def test_tuple_and_list_values_compare_equal():
    saved = _contract()
    live = dict(saved, action_meanings=tuple(saved["action_meanings"]),
                node_ids=tuple(saved["node_ids"]),
                slot_node_ids=tuple(saved["slot_node_ids"]))
    env = _Env(saved)
    env.contract = live
    assert _check(_saved(saved), env).scenario == "ok"


def test_every_scenario_difference_is_listed_and_overridable():
    contract = _contract()
    identity = dict(IDENTITY, run_ini_sha256="z" * 64, jammers_json_sha256="j" * 64)
    with pytest.raises(ScenarioMismatchError) as excinfo:
        _check(_saved(contract), _Env(contract), identity=identity, band="sub-6")
    labels = [d.split(":")[0] for d in excinfo.value.differences]
    assert labels == ["scenario_identity.run_ini_sha256",
                      "scenario_identity.jammers_json_sha256", "band"]
    assert "--allow-different-scenario" in str(excinfo.value)

    report = _check(_saved(contract), _Env(contract), identity=identity, band="sub-6",
                    allow=True)
    described = report.describe()
    assert described["scenario"] == "overridden"
    assert {described[k] for k in ("structural", "observation_schema", "reward")} == {"ok"}
    assert described["warnings"] == excinfo.value.differences


def test_observation_schema_node_changes_are_folded_into_the_scenario_step():
    saved = _contract()
    live = _contract(slot_node_ids=("node-c", "node-b", None))
    with pytest.raises(ScenarioMismatchError) as excinfo:
        _check(_saved(saved), _Env(live))
    assert any(d.startswith("slot_node_ids changed") for d in excinfo.value.differences)
    report = _check(_saved(saved), _Env(live), allow=True)
    assert report.scenario == "overridden"


def test_bounds_matter_only_to_presets_that_use_coordinates():
    saved, live = _contract(), _contract(x_max=200.0)
    assert _check(_saved(saved), _Env(live)).scenario == "ok"
    with pytest.raises(SchemaMismatchError):
        _check(_saved(saved, preset="local_links_v1"), _Env(live, preset="local_links_v1"),
               allow=True)


def test_cpp_authored_reward_requires_the_same_reward_window():
    saved = _contract(reward_window="mean")
    live = _contract(reward_window="sum")
    with pytest.raises(RewardMismatchError, match="reward_window"):
        _check(_saved(saved, components=("legacy",)), _Env(live, components=("legacy",)),
               allow=True)


# 2. Bundles --------------------------------------------------------------------------

def _run_dir(tmp_path: Path, checkpoints=(16,), best=False, **overrides) -> Path:
    run_dir = tmp_path / "train"
    (run_dir / "checkpoints").mkdir(parents=True)
    model = run_dir / "maskable_ppo_mesh.zip"
    model.write_bytes(b"final-bytes")
    entries = []
    for steps in checkpoints:
        path = run_dir / "checkpoints" / f"checkpoint_{steps}_steps.zip"
        path.write_bytes(f"checkpoint-{steps}".encode())
        entries.append({"path": str(path), "sha256": sha256_file(path),
                        "num_timesteps": steps})
    manifest = _saved(_contract())
    manifest.update({"model_path": str(model), "model_sha256": sha256_file(model),
                     "best_model_path": None, "best_model_sha256": None,
                     "checkpoints": entries})
    if best:
        best_path = run_dir / "best_model.zip"
        best_path.write_bytes(b"best-bytes")
        manifest.update({"best_model_path": str(best_path),
                         "best_model_sha256": sha256_file(best_path)})
    manifest.update(overrides)
    (run_dir / "train_manifest.json").write_text(json.dumps(manifest))
    return run_dir


def test_bundle_version_mirrors_the_train_writer():
    pytest.importorskip("sb3_contrib")
    from scripts.rl import train

    assert bundle.MANIFEST_VERSION == train.MANIFEST_VERSION


def test_best_model_is_selectable_and_digest_checked(tmp_path):
    run_dir = _run_dir(tmp_path, best=True)
    best = read_bundle(run_dir, "best")
    assert best.selection == "best" and best.num_timesteps is None
    assert best.model_path.name == "best_model.zip"
    (run_dir / "best_model.zip").write_bytes(b"tampered")
    with pytest.raises(BundleError, match="digest mismatch"):
        read_bundle(run_dir, "best")
    assert read_bundle(run_dir).selection == "final"   # final is unaffected


def test_checkpoint_selection_matches_by_path_not_by_list_position(tmp_path):
    run_dir = _run_dir(tmp_path, checkpoints=(100, 20))
    hundred = read_bundle(run_dir, "checkpoints/checkpoint_100_steps.zip")
    twenty = read_bundle(run_dir, "checkpoints/checkpoint_20_steps.zip")
    assert (hundred.num_timesteps, twenty.num_timesteps) == (100, 20)
    assert hundred.model_sha256 == sha256_file(run_dir / "checkpoints"
                                               / "checkpoint_100_steps.zip")
    assert hundred.selection == twenty.selection == "checkpoint"


def test_describe_reports_the_manifest_digest(tmp_path):
    run_dir = _run_dir(tmp_path)
    described = read_bundle(run_dir).describe()
    assert described["train_manifest_sha256"] == sha256_file(run_dir / "train_manifest.json")
    assert described["model_sha256"] == sha256_file(run_dir / "maskable_ppo_mesh.zip")
    assert described["model_selection"] == "final"
    assert described["num_timesteps"] is None


# 3. policy_weights_sha256 ------------------------------------------------------------

def _zip(path: Path, entries: dict) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def test_policy_weights_digest_ignores_other_zip_entries(tmp_path):
    first = _zip(tmp_path / "a.zip", {"policy.pth": b"weights", "data": b"one"})
    second = _zip(tmp_path / "b.zip", {"data": b"two", "policy.pth": b"weights"})
    third = _zip(tmp_path / "c.zip", {"policy.pth": b"other"})
    assert policy_weights_sha256(first) == hashlib.sha256(b"weights").hexdigest()
    assert policy_weights_sha256(first) == policy_weights_sha256(second)
    assert policy_weights_sha256(third) != policy_weights_sha256(first)
    assert sha256_file(first) != sha256_file(second)


# 4. Selection and seed roles ---------------------------------------------------------


def test_selection_defaults_and_eval_override_leave_the_original_alone():
    selection = selection_from_manifest({"selection": {"observation_preset": "raw_links_v1"}})
    assert (selection.reward_components, selection.reward_weights) == ((), ())
    assert (selection.telemetry, selection.telemetry_every) == ("none", 1)
    forced = eval_selection(selection)
    assert (forced.telemetry, forced.telemetry_every) == ("steps", 1)
    assert (selection.telemetry, selection.source["telemetry"]) == ("none", "manifest")
    assert forced.observation_preset == selection.observation_preset


def test_seed_roles_converts_seeds_and_lists_overlap_in_request_order():
    manifest = {"seed": "5", "evaluation": {"seed": 6}}
    roles = seed_roles(manifest, "best", ["6", 7, "5"])
    assert roles["held_out_seeds"] == [6, 7, 5]
    assert roles["overlap"] == [6, 5]
    assert roles["held_out"] is False
    assert roles["selection_seed_used_for_model"] is True
    assert (roles["training_seed"], roles["model_selection_seed"]) == (5, 6)


def test_seed_roles_with_disjoint_seeds_are_held_out():
    roles = seed_roles({"seed": 1, "evaluation": None}, "final", [2, 3])
    assert roles["held_out"] is True and roles["overlap"] == []
    assert roles["selection_seed_used_for_model"] is False
