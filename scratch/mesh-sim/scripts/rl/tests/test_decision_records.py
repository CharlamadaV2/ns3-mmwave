"""Opt-in per-episode decision records, driven by fake_sim.py; map in src/rl/decision-records.md."""

import hashlib
import json
import re
import sys
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.rl import evaluate as evaluate_cli
from scripts.rl.env.decisions import (DECISIONS_FILE, DECISIONS_MANIFEST,
                                      DecisionContext, DecisionRecorder,
                                      DecisionRecording, applied_action,
                                      coverage_gaps, coverage_ranges, mask_sha256,
                                      masked_probs, resolve_decision_records)
from scripts.rl.env.mesh_env import MeshRlEnv
from scripts.rl.env.selection import resolve_selection
from scripts.rl.env.telemetry import obs_sha256
from scripts.rl.tests.conftest import FAKE_SIM, MULTI_RUN_INI, NODES_JSON

MESH_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = MESH_ROOT / "docs" / "schemas" / "decision_record.v1.schema.json"
SCHEMA_V1_SHA256 = "2855e03eacb33acdb654d7115ee77b23500428e5916f1942459350def5a12f79"

DEFAULT_MAX_BYTES = 67108864
HOLD = 4
SLOT_ACTIONS = 5
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PREPROCESS_NOTE = "stable_baselines3.common.preprocessing.preprocess_obs: obs.float()"
MODEL_INPUT_CONVERSION = "torch.Tensor.float() == numpy.astype(float32)"
FLAGS = {"enabled": "--decision-records",
         "obs_vector": "--decision-records-obs-vector",
         "preferences": "--decision-records-preferences",
         "record_every": "--decision-records-every",
         "max_bytes_per_episode": "--decision-records-max-bytes"}
SCRIPTED = DecisionContext(mode="evaluation", source="evaluate", policy="scripted")

LEGACY_RUN_INI = """[scenario]
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


def _write_scenario(directory: Path, duration_s: str = "1.0") -> str:
    directory.mkdir(parents=True)
    assert "duration_s = 1.0" in MULTI_RUN_INI
    (directory / "run.ini").write_text(
        MULTI_RUN_INI.replace("duration_s = 1.0", f"duration_s = {duration_s}"))
    (directory / "nodes.json").write_text(NODES_JSON)
    return str(directory / "run.ini")


def _write_shim(directory: Path) -> str:
    shim = directory / "fake-sim"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE_SIM}" "$@"\n')
    shim.chmod(0o755)
    return str(shim)


@pytest.fixture
def long_run_config(tmp_path: Path) -> str:
    """MULTI_RUN_INI stretched to 4.5 s: nine decisions of five ticks."""
    return _write_scenario(tmp_path / "long-scenario", "4.5")


def _settings(**overrides):
    return resolve_decision_records(enabled=True, **overrides)


def _env(sim_binary: str, run_config: str, out_dir: Path, *, telemetry="steps",
         every=None, records=None, context=None, preset=None) -> MeshRlEnv:
    selection = resolve_selection(run_config, observation_preset=preset,
                                  telemetry=telemetry, telemetry_every=every)
    recording = (None if records is None
                 else DecisionRecording(records, context or SCRIPTED))
    return MeshRlEnv(sim_binary, run_config, output_dir=str(out_dir),
                     selection=selection, decision_records=recording)


def _cycle(env: MeshRlEnv, decision: int) -> list[int]:
    """Varied but always-valid joint action, so nothing is revalidated."""
    mask = env.action_masks()
    action = []
    for slot in range(len(mask) // SLOT_ACTIONS):
        row = mask[slot * SLOT_ACTIONS:(slot + 1) * SLOT_ACTIONS]
        order = [(decision + slot + k) % 4 for k in range(4)]
        action.append(next((a for a in order if row[a]), HOLD))
    return action


def _run(env: MeshRlEnv, actions=_cycle) -> list[dict]:
    """Reset, then step to done with a list (hold when exhausted) or a callable."""
    env.reset()
    infos, done, decision = [], False, 0
    while not done:
        decision += 1
        if callable(actions):
            action = actions(env, decision)
        else:
            action = actions[decision - 1] if decision <= len(actions) else [HOLD] * 3
        _, _, done, _, info = env.step(action)
        infos.append(info)
    return infos


def _episode(out_dir: Path, index: int = 0) -> Path:
    return Path(out_dir) / f"episode-{index:04d}"


def _rl_episode(episode_dir: Path) -> dict:
    return json.loads((episode_dir / "rl_episode.json").read_text())


def _raw_lines(episode_dir: Path) -> list[bytes]:
    return (episode_dir / DECISIONS_FILE).read_bytes().splitlines(keepends=True)


def _read_sidecar(episode_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((episode_dir / DECISIONS_MANIFEST).read_text())
    path = episode_dir / DECISIONS_FILE
    lines = ([json.loads(line) for line in path.read_text().splitlines()]
             if path.is_file() else [])
    return manifest, lines


def _steps(episode_dir: Path) -> dict[int, dict]:
    lines = (episode_dir / "steps.jsonl").read_text().splitlines()
    return {record["decision"]: record for record in map(json.loads, lines[1:])}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _expand(ranges: list[list[int]]) -> list[int]:
    return [n for low, high in ranges for n in range(low, high + 1)]


def _check_sidecar(episode_dir: Path) -> tuple[dict, list[dict]]:
    """File identity plus the coverage partition rules shared by every closed sidecar."""
    manifest, lines = _read_sidecar(episode_dir)
    raw = (episode_dir / DECISIONS_FILE).read_bytes()
    assert manifest["status"] != "writing"
    assert manifest["jsonl_sha256"] == _sha256(raw)
    assert manifest["bytes"] == len(raw)

    coverage = manifest["coverage"]
    expected = coverage["decisions_expected"]
    assert expected == manifest["contract_identity"]["num_decisions"]
    assert coverage["records"] == len(lines)
    assert coverage["decision_0_retained"] is True
    assert lines[0]["type"] == "reset" and lines[0]["decision"] == 0
    saved = [line["decision"] for line in lines[1:]]
    assert all(line["type"] == "decision" for line in lines[1:])
    assert _expand(coverage["saved_ranges"]) == saved
    assert coverage["saved_ranges"] == coverage_ranges(saved)
    assert coverage["gaps"] == coverage_gaps(saved, expected)
    assert sorted(saved + _expand(coverage["gaps"])) == list(range(1, expected + 1))
    if manifest["status"] == "complete" and manifest["settings"]["record_every"] == 1:
        assert coverage["gaps"] == []
    if manifest["status"] == "capped" and manifest["settings"]["record_every"] == 1:
        assert len(coverage["gaps"]) == 1 and coverage["gaps"][0][1] == expected
    return manifest, lines


def _stable(manifest: dict) -> dict:
    return {key: value for key, value in manifest.items()
            if key not in ("command", "started_at", "ended_at")}


# 1. Settings ---------------------------------------------------------------------

def _raises_naming(key: str, **kwargs) -> None:
    with pytest.raises(ValueError) as excinfo:
        resolve_decision_records(**kwargs)
    message = str(excinfo.value)
    assert key in message or FLAGS[key] in message, message


def test_settings_resolution_and_dependent_flags():
    assert (DECISIONS_FILE, DECISIONS_MANIFEST) == (
        "policy_decisions.jsonl", "policy_decisions_manifest.json")

    defaults = resolve_decision_records()
    assert defaults.enabled is False
    assert (defaults.obs_vector, defaults.preferences) == ("auto", "auto")
    assert (defaults.record_every, defaults.max_bytes_per_episode) == (1, DEFAULT_MAX_BYTES)
    described = defaults.describe()
    assert described["source"] == {key: "default" for key in FLAGS}

    given = resolve_decision_records(enabled=True, obs_vector="always", preferences="off",
                                     record_every=3, max_bytes_per_episode=0)
    assert (given.enabled, given.obs_vector, given.preferences, given.record_every,
            given.max_bytes_per_episode) == (True, "always", "off", 3, 0)
    assert given.describe() == {"enabled": True, "obs_vector": "always",
                                "preferences": "off", "record_every": 3,
                                "max_bytes_per_episode": 0,
                                "source": {key: "cli" for key in FLAGS}}
    partial = resolve_decision_records(enabled=True, record_every=2).describe()["source"]
    assert partial == {"enabled": "cli", "obs_vector": "default", "preferences": "default",
                       "record_every": "cli", "max_bytes_per_episode": "default"}

    _raises_naming("record_every", enabled=True, record_every=0)
    _raises_naming("max_bytes_per_episode", enabled=True, max_bytes_per_episode=-1)
    _raises_naming("obs_vector", enabled=True, obs_vector="sometimes")
    _raises_naming("preferences", enabled=True, preferences="logits")
    for key, value in (("obs_vector", "always"), ("preferences", "off"),
                       ("record_every", 2), ("max_bytes_per_episode", 0)):
        _raises_naming(key, **{key: value})


# 2. Disabled parity --------------------------------------------------------------

def test_disabled_by_default_writes_nothing(sim_binary, long_run_config, tmp_path):
    runs = {"absent": None, "disabled": resolve_decision_records()}
    for name, records in runs.items():
        env = _env(sim_binary, long_run_config, tmp_path / name, records=records)
        try:
            _run(env)
        finally:
            env.close()

    manifests = {}
    for name in runs:
        episode_dir = _episode(tmp_path / name)
        assert not list((tmp_path / name).rglob("policy_decisions*"))
        manifests[name] = _rl_episode(episode_dir)
        assert "decision_records" not in manifests[name]
        assert manifests[name]["status"] == "completed"
    assert _stable(manifests["absent"]) == _stable(manifests["disabled"])
    assert ((_episode(tmp_path / "absent") / "steps.jsonl").read_bytes()
            == (_episode(tmp_path / "disabled") / "steps.jsonl").read_bytes())


# 3-7. Join, sampling, vectors, dtypes --------------------------------------------

def _join_run(sim_binary, run_config, out_dir, **settings) -> Path:
    env = _env(sim_binary, run_config, out_dir, records=_settings(**settings))
    try:
        _run(env)
    finally:
        env.close()
    return _episode(out_dir)


def test_records_join_previous_step_at_every_one(sim_binary, long_run_config, tmp_path):
    out_dir = tmp_path / "rl"
    records = _settings()
    env = _env(sim_binary, long_run_config, out_dir, records=records)
    try:
        infos = _run(env)
    finally:
        env.close()
    assert len(infos) == 9
    episode_dir = _episode(out_dir)
    manifest, lines = _check_sidecar(episode_dir)
    steps = _steps(episode_dir)
    rl_episode = _rl_episode(episode_dir)
    assert rl_episode["decision_records"] == {"manifest": DECISIONS_MANIFEST,
                                              "file": DECISIONS_FILE}

    reset = lines[0]
    assert (reset["tick"], reset["time_s"], reset["obs_dtype"]) == (0, 0.0, "float64")
    assert reset["obs_sha256"] == steps[0]["obs_sha256"]
    assert reset["mask"] == steps[0]["mask"]
    assert reset["mask_sha256"] == mask_sha256(reset["mask"])
    assert reset["steps_ref"] == {"file": "steps.jsonl", "decision": 0,
                                  "obs_sha256": steps[0]["obs_sha256"]}
    assert "action" not in reset and "outcome" not in reset

    assert [line["decision"] for line in lines] == list(range(10))
    for record in lines[1:]:
        n = record["decision"]
        source, step = record["input"], record["outcome"]
        assert source["source_decision"] == n - 1
        assert source["obs_sha256"] == steps[n - 1]["obs_sha256"]
        assert source["mask"] == steps[n - 1]["mask"]
        assert source["steps_ref"] == {"file": "steps.jsonl", "decision": n - 1,
                                       "obs_sha256": steps[n - 1]["obs_sha256"]}
        assert source["obs_vector"] is None
        assert source["tick"] == steps[n - 1]["tick"]
        assert source["time_s"] == pytest.approx(steps[n - 1]["time_s"], abs=1e-9)
        assert source["tick"] == step["tick"] - step["ticks_in_step"]
        assert (step["tick"], step["ticks_in_step"]) == (steps[n]["tick"],
                                                         steps[n]["ticks_in_step"])
        assert step["interval_s"] == {"start_exclusive": source["time_s"],
                                      "end_inclusive": step["time_s"]}
        assert step["done"] is (n == 9)
        assert step["legacy_reward"] == steps[n]["legacy_reward"]
        assert step["reward"] == {"total": 1.0, "source": "cpp"}
        assert step["reward_schema_sha256"] == rl_episode["reward_schema_sha256"]
        assert record["action"]["requested"] == steps[n]["action_sent"]
        assert record["action"]["revalidated_slots"] == [] == steps[n]["revalidated_slots"]
        assert record["preferences"] is None

    assert manifest["contract"] == "decision_record" and manifest["version"] == 1
    assert manifest["status"] == "complete" and manifest["error"] is None
    assert manifest["settings"] == records.describe()
    assert manifest["coverage"] == {"decisions_expected": 9, "decision_0_retained": True,
                                    "terminal_decision": 9, "saved_ranges": [[1, 9]],
                                    "gaps": [], "records": 10}
    episode = manifest["episode"]
    assert episode == {"dir_name": "episode-0000", "index": 0, "seed": 1,
                       "seed_source": "run.ini", "mode": "evaluation",
                       "source": "evaluate", "policy": "scripted", "status": "completed",
                       "rl_episode_manifest": "rl_episode.json",
                       "steps_file": "steps.jsonl", "steps_telemetry_every": 1}
    assert manifest["observation_schema"]["sha256"] == rl_episode["observation_schema_sha256"]
    assert manifest["reward_schema_sha256"] == rl_episode["reward_schema_sha256"]
    assert manifest["model"] is None
    assert manifest["preferences"]["captured"] is False
    assert manifest["writer"]["written_after_outcome"] is True


def test_sampled_steps_are_never_relabelled(sim_binary, long_run_config, tmp_path):
    out_dir = tmp_path / "rl"
    env = _env(sim_binary, long_run_config, out_dir, every=2, records=_settings())
    try:
        _run(env)
    finally:
        env.close()
    episode_dir = _episode(out_dir)
    _, lines = _check_sidecar(episode_dir)
    steps = _steps(episode_dir)
    assert sorted(steps) == [0, 2, 4, 6, 8, 9]

    for record in lines[1:]:
        n, source = record["decision"], record["input"]
        if n % 2 == 0:
            assert n - 1 not in steps
            assert source["steps_ref"] is None
            vector = np.asarray(source["obs_vector"], dtype=source["obs_dtype"])
            assert obs_sha256(vector, source["obs_dtype"]) == source["obs_sha256"]
            assert obs_sha256(vector, "float32") == source["model_input_sha256"]
        else:
            assert source["steps_ref"]["decision"] == n - 1
            assert source["steps_ref"]["obs_sha256"] == steps[n - 1]["obs_sha256"]
            assert source["obs_vector"] is None


def test_no_telemetry_stores_exact_vector(sim_binary, long_run_config, tmp_path):
    out_dir = tmp_path / "rl"
    env = _env(sim_binary, long_run_config, out_dir, telemetry="none", records=_settings())
    try:
        _run(env)
    finally:
        env.close()
    episode_dir = _episode(out_dir)
    assert not (episode_dir / "steps.jsonl").exists()
    manifest, lines = _check_sidecar(episode_dir)
    assert manifest["episode"]["steps_file"] is None
    assert manifest["episode"]["steps_telemetry_every"] is None
    assert lines[0]["steps_ref"] is None
    for record in lines[1:]:
        source = record["input"]
        assert source["steps_ref"] is None
        vector = np.asarray(source["obs_vector"], dtype=source["obs_dtype"])
        assert obs_sha256(vector, source["obs_dtype"]) == source["obs_sha256"]


def test_obs_vector_always(sim_binary, multi_run_config, tmp_path):
    episode_dir = _join_run(sim_binary, multi_run_config, tmp_path / "rl",
                            obs_vector="always")
    _, lines = _check_sidecar(episode_dir)
    assert len(lines) == 3
    for record in lines[1:]:
        source = record["input"]
        assert source["steps_ref"] is not None
        vector = np.asarray(source["obs_vector"], dtype=source["obs_dtype"])
        assert obs_sha256(vector, source["obs_dtype"]) == source["obs_sha256"]
        assert source["steps_ref"]["obs_sha256"] == source["obs_sha256"]


@pytest.mark.parametrize("preset,dtype,lossless", [("local_links_v1", "float32", True),
                                                    ("raw_links_v1", "float64", False)])
def test_float32_preset_hashes_coincide(sim_binary, multi_run_config, tmp_path, preset,
                                        dtype, lossless):
    out_dir = tmp_path / "rl"
    env = _env(sim_binary, multi_run_config, out_dir, preset=preset, records=_settings())
    try:
        _run(env)
    finally:
        env.close()
    manifest, lines = _check_sidecar(_episode(out_dir))
    assert manifest["observation_schema"]["schema_id"] == preset
    assert manifest["observation_schema"]["dtype"] == dtype
    assert manifest["model_input"] == {"dtype": "float32", "conversion": PREPROCESS_NOTE,
                                       "lossless_from_obs": lossless}
    for record in lines[1:]:
        source = record["input"]
        assert source["obs_dtype"] == dtype
        assert source["model_input_dtype"] == "float32"
        assert source["model_input_conversion"] == MODEL_INPUT_CONVERSION
        assert (source["model_input_sha256"] == source["obs_sha256"]) is lossless


# 8-9. Actions and masks ----------------------------------------------------------

def test_requested_revalidated_applied(sim_binary, multi_run_config, tmp_path):
    assert applied_action([1, 3, 4], [0]) == [4, 3, 4]
    assert applied_action([2, 1, 4], []) == [2, 1, 4]

    out_dir = tmp_path / "rl"
    env = _env(sim_binary, multi_run_config, out_dir, records=_settings())
    try:
        env.reset()
        assert env.action_masks()[1] == 0          # node-b starts at x_max
        _, _, done, _, info = env.step([1, 3, 4])
        assert info["revalidated_slots"] == [0] and not done
        _, _, done, _, info = env.step([4, 4, 1])
        assert info["revalidated_slots"] == [2] and done
    finally:
        env.close()

    episode_dir = _episode(out_dir)
    manifest, lines = _check_sidecar(episode_dir)
    steps = _steps(episode_dir)
    first, second = lines[1]["action"], lines[2]["action"]
    assert lines[1]["input"]["mask"][1] == 0
    assert first == {"requested": [1, 3, 4], "revalidated_slots": [0],
                     "applied": [4, 3, 4], "applied_status": "derived", "hold_index": 4}
    assert second == {"requested": [4, 4, 1], "revalidated_slots": [2],
                      "applied": [4, 4, 4], "applied_status": "derived", "hold_index": 4}
    assert [steps[n]["revalidated_slots"] for n in (1, 2)] == [[0], [2]]
    assert manifest["status"] == "complete"


def test_mask_hash_is_over_int_json(sim_binary, multi_run_config, tmp_path):
    episode_dir = _join_run(sim_binary, multi_run_config, tmp_path / "rl")
    _, lines = _check_sidecar(episode_dir)
    for record in lines:
        holder = record if record["type"] == "reset" else record["input"]
        mask = holder["mask"]
        assert all(type(m) is int and m in (0, 1) for m in mask)
        as_ints = _sha256(json.dumps(mask, separators=(",", ":")).encode())
        as_bools = _sha256(json.dumps([bool(m) for m in mask],
                                      separators=(",", ":")).encode())
        assert holder["mask_sha256"] == as_ints == mask_sha256(mask)
        assert mask_sha256(np.asarray(mask, dtype=np.int8)) == as_ints
        assert as_bools != as_ints


# 10-12a. Caps and sampling -------------------------------------------------------

def _settled_run(sim_binary, run_config, out_dir, **settings) -> Path:
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return _join_run(sim_binary, run_config, out_dir, **settings)


def _same_episode(left: Path, right: Path) -> None:
    assert (left / "steps.jsonl").read_bytes() == (right / "steps.jsonl").read_bytes()
    a, b = _rl_episode(left), _rl_episode(right)
    for key in ("status", "steps", "decisions", "cumulative_reward", "telemetry",
                "last_tick", "exit_code"):
        assert a[key] == b[key], key


def test_cap_stops_at_record_boundary(sim_binary, long_run_config, tmp_path):
    full = _join_run(sim_binary, long_run_config, tmp_path / "full",
                     max_bytes_per_episode=0)
    sizes = [len(line) for line in _raw_lines(full)]
    cap = sizes[0] + sizes[1] + 1

    capped = _join_run(sim_binary, long_run_config, tmp_path / "capped",
                       max_bytes_per_episode=cap)
    manifest, lines = _check_sidecar(capped)
    assert manifest["status"] == "capped"
    assert manifest["coverage"]["saved_ranges"] == [[1, 1]]
    assert manifest["coverage"]["gaps"] == [[2, 9]]
    assert manifest["coverage"]["records"] == 2
    assert manifest["coverage"]["terminal_decision"] == 9
    assert manifest["bytes"] == sizes[0] + sizes[1]
    assert manifest["bytes"] <= cap < manifest["bytes"] + sizes[2]
    assert _raw_lines(capped) == _raw_lines(full)[:2]
    assert manifest["episode"]["status"] == "completed"
    _same_episode(full, capped)


def test_record_every_two_partition(sim_binary, long_run_config, tmp_path):
    assert coverage_ranges([2, 4, 6, 8, 9]) == [[2, 2], [4, 4], [6, 6], [8, 9]]
    assert coverage_gaps([2, 4, 6, 8, 9], 9) == [[1, 1], [3, 3], [5, 5], [7, 7]]

    episode_dir = _join_run(sim_binary, long_run_config, tmp_path / "rl", record_every=2)
    manifest, lines = _check_sidecar(episode_dir)
    assert [line["decision"] for line in lines] == [0, 2, 4, 6, 8, 9]
    assert manifest["status"] == "complete"
    assert manifest["coverage"]["saved_ranges"] == [[2, 2], [4, 4], [6, 6], [8, 9]]
    assert manifest["coverage"]["gaps"] == [[1, 1], [3, 3], [5, 5], [7, 7]]
    assert manifest["coverage"]["records"] == 6
    assert manifest["settings"]["record_every"] == 2
    for record in lines[1:]:
        assert record["input"]["source_decision"] == record["decision"] - 1


def test_zero_cap_means_no_cap(sim_binary, long_run_config, tmp_path):
    episode_dir = _join_run(sim_binary, long_run_config, tmp_path / "rl",
                            max_bytes_per_episode=0)
    manifest, lines = _check_sidecar(episode_dir)
    assert manifest["settings"]["max_bytes_per_episode"] == 0
    assert manifest["status"] == "complete"
    assert manifest["coverage"]["records"] == 10 == len(lines)
    assert manifest["coverage"]["gaps"] == []


def test_cap_smaller_than_reset(sim_binary, long_run_config, tmp_path):
    full = _join_run(sim_binary, long_run_config, tmp_path / "full")
    reset_size = len(_raw_lines(full)[0])
    small = _settled_run(sim_binary, long_run_config, tmp_path / "small",
                         max_bytes_per_episode=reset_size - 1)

    manifest = json.loads((small / DECISIONS_MANIFEST).read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]
    assert manifest["bytes"] == 0
    assert manifest["coverage"]["records"] == 0
    assert manifest["coverage"]["decision_0_retained"] is False
    path = small / DECISIONS_FILE
    assert not path.exists() or path.stat().st_size == 0
    _same_episode(full, small)
    assert _rl_episode(small)["status"] == "completed"


# 13-15. Failure isolation, interruption, identity --------------------------------

def test_writer_failure_is_isolated(sim_binary, long_run_config, tmp_path, monkeypatch):
    clean = _join_run(sim_binary, long_run_config, tmp_path / "clean")

    original = DecisionRecorder._write_line
    calls = []

    def flaky(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 3:                     # reset, decision 1, then decision 2
            raise OSError("injected decision-record write failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DecisionRecorder, "_write_line", flaky)
    out_dir = tmp_path / "broken"
    env = _env(sim_binary, long_run_config, out_dir, records=_settings())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            infos = _run(env)
        finally:
            env.close()
    assert len(infos) == 9 and infos[-1]["decision"] == 9

    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime) == 1
    broken = _episode(out_dir)
    assert _rl_episode(broken)["status"] == "completed"
    _same_episode(clean, broken)

    manifest, lines = _read_sidecar(broken)
    raw = (broken / DECISIONS_FILE).read_bytes()
    assert manifest["status"] == "failed"
    assert "injected decision-record write failure" in manifest["error"]
    assert manifest["jsonl_sha256"] == _sha256(raw)
    assert manifest["bytes"] == len(raw)
    assert [line["decision"] for line in lines] == [0, 1]


def test_interrupted_episode_status(sim_binary, long_run_config, tmp_path):
    out_dir = tmp_path / "rl"
    env = _env(sim_binary, long_run_config, out_dir, records=_settings())
    try:
        env.reset()
        env.step(_cycle(env, 1))
        infos = _run(env)
    finally:
        env.close()
    assert len(infos) == 9

    first, first_lines = _check_sidecar(_episode(out_dir, 0))
    assert first["status"] == "interrupted"
    assert first["episode"]["status"] == "interrupted"
    assert _rl_episode(_episode(out_dir, 0))["status"] == "interrupted"
    assert first["coverage"]["terminal_decision"] == 1
    assert first["coverage"]["saved_ranges"] == [[1, 1]]
    assert first["coverage"]["gaps"] == [[2, 9]]
    assert [line["decision"] for line in first_lines] == [0, 1]

    second, second_lines = _check_sidecar(_episode(out_dir, 1))
    assert second["status"] == "complete"
    assert (second["episode"]["dir_name"], second["episode"]["index"]) == ("episode-0001", 1)
    assert second["episode"]["status"] == "completed"
    assert [line["decision"] for line in second_lines] == list(range(10))
    assert second_lines[1]["input"]["source_decision"] == 0


def test_string_ids_and_null_padding(sim_binary, multi_run_config, tmp_path):
    episode_dir = _join_run(sim_binary, multi_run_config, tmp_path / "rl")
    manifest, lines = _check_sidecar(episode_dir)
    identity = manifest["contract_identity"]
    assert identity == {"contract_id": "mesh_move_2d_v1",
                        "node_ids": ["node-a", "node-b", "node-c"],
                        "slot_node_ids": ["node-b", "node-c", None],
                        "action_meanings": ["west", "east", "south", "north", "hold"],
                        "hold_index": 4, "tick_s": 0.1, "decision_interval_ticks": 5,
                        "num_decisions": 2}
    for record in lines[1:]:
        assert record["targets"] == ["node-b", "node-c"]
        assert record["action"]["hold_index"] == 4


# 16-17. Saved-model evaluation (sb3) ---------------------------------------------

@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """One tiny MaskablePPO run on the nine-decision scenario, shared by tests 16, 17, 20."""
    pytest.importorskip("sb3_contrib")
    from scripts.rl import train

    root = tmp_path_factory.mktemp("decision-records-model")
    shim = _write_shim(root)
    run_config = _write_scenario(root / "scenario", "4.5")
    run_dir = root / "train"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", [
            "train", "--sim-binary", shim, "--run-config", run_config,
            "--output-dir", str(run_dir), "--verbose", "0",
            "m-ppo", "--total-timesteps", "16", "--n-steps", "16", "--seed", "1"])
        assert train.main() == 0
    return SimpleNamespace(sim_binary=shim, run_config=run_config, run_dir=run_dir)


def _evaluate(trained, out_dir: Path, *extra: str) -> dict:
    code = evaluate_cli.main(["--sim-binary", trained.sim_binary,
                              "--run-dir", str(trained.run_dir),
                              "--output-dir", str(out_dir), "--seeds", "11",
                              "--policies", "model,hold", *extra])
    assert code == 0
    return json.loads((out_dir / "eval_manifest.json").read_text())


def _episode_dir_of(eval_manifest: dict, policy: str) -> Path:
    episode, = eval_manifest["policies"][policy]["episodes"]
    return Path(episode["episode_dir"])


def _masked_argmax(row: np.ndarray, mask_row) -> int:
    return int(np.argmax(np.where(np.asarray(mask_row, dtype=bool), row, -np.inf)))


def _softmax_rounded(row: np.ndarray, mask_row) -> list[float]:
    valid = np.asarray(mask_row, dtype=bool)
    shifted = row.astype(np.float64) - row[valid].max()
    weights = np.where(valid, np.exp(shifted), 0.0)
    return list(weights / weights.sum())


def test_model_evaluation_records_preferences_and_identity(trained_run, tmp_path):
    from scripts.rl.policy.bundle import policy_weights_sha256, read_bundle

    eval_manifest = _evaluate(trained_run, tmp_path / "eval", "--decision-records")
    bundle = read_bundle(trained_run.run_dir, "final")
    with zipfile.ZipFile(bundle.model_path) as archive:
        weights = _sha256(archive.read("policy.pth"))

    model_dir = _episode_dir_of(eval_manifest, "model")
    manifest, lines = _check_sidecar(model_dir)
    assert manifest["status"] == "complete"
    assert manifest["episode"]["policy"] == "model"
    assert manifest["episode"]["mode"] == "evaluation"
    assert manifest["episode"]["source"] == "evaluate"
    assert manifest["preferences"] == {"captured": True,
                                       "representation": "raw_action_net_logits_float32",
                                       "capture": "action_net_forward_hook", "error": None}
    model = manifest["model"]
    assert model["model_sha256"] == bundle.model_sha256
    assert model["policy_weights_sha256"] == weights == policy_weights_sha256(
        bundle.model_path)
    assert model["model_path_recorded"] == "maskable_ppo_mesh.zip"
    assert model["model_selection"] == "final"
    assert model["num_timesteps"] is None
    assert model["train_manifest_sha256"] == _sha256(
        (trained_run.run_dir / "train_manifest.json").read_bytes())
    assert model["inference"]["deterministic"] is True
    assert model["inference"]["device"] == "cpu"

    for record in lines[1:]:
        preferences = record["preferences"]
        assert preferences["representation"] == "raw_action_net_logits_float32"
        assert preferences["capture"] == "action_net_forward_hook"
        per_slot = np.asarray(preferences["per_slot"], dtype=np.float32)
        assert per_slot.shape == (3, SLOT_ACTIONS)
        assert all(float(np.float32(v)) == v for row in preferences["per_slot"] for v in row)
        mask = record["input"]["mask"]
        rows = [mask[s * SLOT_ACTIONS:(s + 1) * SLOT_ACTIONS] for s in range(3)]
        assert record["action"]["requested"] == [
            _masked_argmax(per_slot[s], rows[s]) for s in range(3)]
        assert preferences["masked_probs"] == masked_probs(per_slot, mask)
        for s, probs in enumerate(preferences["masked_probs"]):
            assert all(p == 0 for p, valid in zip(probs, rows[s]) if not valid)
            assert sum(probs) == pytest.approx(1.0, abs=1e-5)
            assert probs == pytest.approx(_softmax_rounded(per_slot[s], rows[s]), abs=2e-6)

    hold_manifest, hold_lines = _check_sidecar(_episode_dir_of(eval_manifest, "hold"))
    assert hold_manifest["preferences"]["captured"] is False
    assert hold_manifest["model"] is None
    assert hold_manifest["episode"]["policy"] == "hold"
    assert all(record["preferences"] is None for record in hold_lines[1:])

    schema = _load_schema()
    for episode_dir in (model_dir, _episode_dir_of(eval_manifest, "hold")):
        _assert_documents_valid(schema, *_read_sidecar(episode_dir))


def _state_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _snapshot(model) -> tuple:
    import torch

    numpy_state = np.random.get_state()
    return (torch.get_rng_state().clone(), numpy_state[0], numpy_state[1].copy(),
            numpy_state[2:], _state_digest(model), model.policy.training)


def _same_snapshot(left: tuple, right: tuple) -> bool:
    """Torch RNG, NumPy RNG, weights digest and training flag all unchanged."""
    import torch

    return (torch.equal(left[0], right[0]) and left[1] == right[1]
            and np.array_equal(left[2], right[2]) and left[3:] == right[3:])


def test_capture_on_off_parity(trained_run, tmp_path):
    from scripts.rl.policy.bundle import (eval_selection, load_model, read_bundle,
                                          selection_from_manifest)
    from scripts.rl.policy.evaluate import ModelPolicy
    from scripts.rl.policy.preferences import PreferenceCapture

    off = _evaluate(trained_run, tmp_path / "off")
    on = _evaluate(trained_run, tmp_path / "on", "--decision-records",
                   "--decision-records-obs-vector", "always")
    for policy in ("model", "hold"):
        off_episode, = off["policies"][policy]["episodes"]
        on_episode, = on["policies"][policy]["episodes"]
        assert off_episode["actions_sha256"] == on_episode["actions_sha256"], policy
        assert ((Path(off_episode["episode_dir"]) / "steps.jsonl").read_bytes()
                == (Path(on_episode["episode_dir"]) / "steps.jsonl").read_bytes()), policy
        assert not list(Path(off_episode["episode_dir"]).glob("policy_decisions*"))

    manifest, lines = _check_sidecar(_episode_dir_of(on, "model"))
    assert manifest["preferences"]["captured"] is True
    inputs = [(np.asarray(r["input"]["obs_vector"], dtype=r["input"]["obs_dtype"]),
               np.asarray(r["input"]["mask"], dtype=bool)) for r in lines[1:]]
    assert len(inputs) == 9

    bundle = read_bundle(trained_run.run_dir, "final")
    selection = eval_selection(selection_from_manifest(bundle.manifest))
    env = MeshRlEnv(trained_run.sim_binary, trained_run.run_config, seed=11,
                    output_dir=str(tmp_path / "unit"), selection=selection)
    try:
        env.reset(seed=11, options={"seed_source": "eval"})
        model = load_model(bundle, env, evaluate_cli.mask_fn)
    finally:
        env.close()

    model.predict(inputs[0][0], action_masks=inputs[0][1], deterministic=True)
    before = _snapshot(model)
    control = ModelPolicy(model)
    plain = [control.act(obs, mask, {}).tolist() for obs, mask in inputs]
    assert _same_snapshot(before, _snapshot(model))

    capture = PreferenceCapture()
    hooked_policy = ModelPolicy(model, capture)
    try:
        assert capture.registered is True and capture.error is None
        hooked, captured = [], []
        for obs, mask in inputs:
            hooked.append(hooked_policy.act(obs, mask, {}).tolist())
            captured.append(capture.take())
        after = _snapshot(model)
    finally:
        capture.detach()
    assert hooked == plain
    assert plain == [r["action"]["requested"] for r in lines[1:]]
    assert _same_snapshot(before, after)
    for logits, record in zip(captured, lines[1:]):
        assert logits is not None and logits.dtype == np.float32
        assert logits.shape == (3 * SLOT_ACTIONS,)
        np.testing.assert_array_equal(
            logits.reshape(3, SLOT_ACTIONS),
            np.asarray(record["preferences"]["per_slot"], dtype=np.float32))


# 18-19. Schema -------------------------------------------------------------------

SCHEMA_DEFS = ("manifest", "reset_record", "decision_record", "steps_ref", "reward",
               "preferences", "model", "coverage", "settings")
VALIDATION_KEYWORDS = {"type", "const", "enum", "required", "properties",
                       "additionalProperties", "items", "prefixItems", "minItems",
                       "maxItems", "minimum", "pattern", "oneOf", "$ref"}
ANNOTATION_KEYWORDS = {"$schema", "$defs", "title", "description", "$comment"}
_JSON_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool))
    or (isinstance(v, float) and v.is_integer()),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _keywords(schema, found: set) -> set:
    """Every keyword the schema uses, walking only schema positions."""
    if isinstance(schema, bool):
        return found
    for key, value in schema.items():
        found.add(key)
        if key in ("properties", "$defs"):
            for sub in value.values():
                _keywords(sub, found)
        elif key in ("items", "additionalProperties"):
            _keywords(value, found)
        elif key in ("prefixItems", "oneOf"):
            for sub in value:
                _keywords(sub, found)
    return found


def _same(a, b) -> bool:
    """JSON equality: booleans never equal numbers."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _errors(value, schema, root: dict, path: str = "$") -> list[str]:
    """Mini draft 2020-12 validator for the keyword subset the v1 schema may use."""
    if schema is True:
        return []
    if schema is False:
        return [f"{path}: not allowed"]
    errors = []
    if "$ref" in schema:
        ref = schema["$ref"]
        assert ref.startswith("#/$defs/"), ref
        errors += _errors(value, root["$defs"][ref[len("#/$defs/"):]], root, path)
    if "oneOf" in schema:
        matched = sum(not _errors(value, sub, root, path) for sub in schema["oneOf"])
        if matched != 1:
            errors.append(f"{path}: matched {matched} oneOf branches")
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_JSON_TYPES[t](value) for t in types):
            return errors + [f"{path}: {value!r} is not {types}"]
    if "const" in schema and not _same(value, schema["const"]):
        errors.append(f"{path}: {value!r} != const {schema['const']!r}")
    if "enum" in schema and not any(_same(value, c) for c in schema["enum"]):
        errors.append(f"{path}: {value!r} not in {schema['enum']!r}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        errors += [f"{path}: missing {key}" for key in schema.get("required", [])
                   if key not in value]
        for key, item in value.items():
            if key in properties:
                errors += _errors(item, properties[key], root, f"{path}.{key}")
            elif "additionalProperties" in schema:
                errors += _errors(item, schema["additionalProperties"], root,
                                  f"{path}.{key}")
    if isinstance(value, list):
        prefix = schema.get("prefixItems", [])
        for index, item in enumerate(value):
            if index < len(prefix):
                errors += _errors(item, prefix[index], root, f"{path}[{index}]")
            elif "items" in schema:
                errors += _errors(item, schema["items"], root, f"{path}[{index}]")
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
    if (_JSON_TYPES["number"](value) and "minimum" in schema
            and value < schema["minimum"]):
        errors.append(f"{path}: {value} < {schema['minimum']}")
    if isinstance(value, str) and "pattern" in schema and not re.search(
            schema["pattern"], value):
        errors.append(f"{path}: {value!r} does not match {schema['pattern']}")
    return errors


def _semantic_errors(record: dict) -> list[str]:
    """Cross-field rules a JSON Schema cannot state."""
    if record.get("type") != "decision":
        return []
    errors = []
    source, outcome, action = record["input"], record["outcome"], record["action"]
    if source["source_decision"] != record["decision"] - 1:
        errors.append("input.source_decision != decision - 1")
    if source["tick"] != outcome["tick"] - outcome["ticks_in_step"]:
        errors.append("input.tick != outcome.tick - ticks_in_step")
    if action["applied"] != applied_action(action["requested"],
                                           action["revalidated_slots"]):
        errors.append("action.applied is not derived from requested")
    return errors


def _document_errors(schema: dict, document: dict) -> list[str]:
    return _errors(document, schema, schema) + _semantic_errors(document)


def _assert_documents_valid(schema: dict, manifest: dict, lines: list[dict]) -> None:
    assert _errors(manifest, schema["$defs"]["manifest"], schema) == []
    assert _document_errors(schema, manifest) == []
    for line in lines:
        name = "reset_record" if line["type"] == "reset" else "decision_record"
        assert _errors(line, schema["$defs"][name], schema) == [], line["decision"]
        assert _document_errors(schema, line) == [], line["decision"]


def _produced_documents(sim_binary, multi_run_config, long_run_config,
                        tmp_path) -> list[tuple[dict, list[dict]]]:
    """Sidecars from the join, masked, interrupted, sampled and vector scenarios."""
    episodes = [
        _join_run(sim_binary, long_run_config, tmp_path / "join"),
        _join_run(sim_binary, long_run_config, tmp_path / "every", record_every=2),
    ]
    env = _env(sim_binary, multi_run_config, tmp_path / "masked", records=_settings())
    try:
        _run(env, [[1, 3, 4], [4, 4, 1]])
        env.reset()
        env.step([4, 4, 4])
    finally:
        env.close()
    episodes += [_episode(tmp_path / "masked", 0), _episode(tmp_path / "masked", 1)]
    env = _env(sim_binary, long_run_config, tmp_path / "vectors", telemetry="none",
               preset="local_links_v1", records=_settings())
    try:
        _run(env)
    finally:
        env.close()
    episodes.append(_episode(tmp_path / "vectors"))
    return [_read_sidecar(path) for path in episodes]


def _mutants(manifest: dict, decision: dict) -> list[tuple[str, dict, bool]]:
    """(label, document, schema_expressible) for documents every check must reject."""
    def copy(doc):
        return json.loads(json.dumps(doc))

    extra = copy(decision)
    extra["influence"] = 0.0
    manifest_extra = copy(manifest)
    manifest_extra["coverage"]["extra"] = 1
    bool_mask = copy(decision)
    bool_mask["input"]["mask"] = [bool(m) for m in bool_mask["input"]["mask"]]
    non_hex = copy(decision)
    non_hex["input"]["obs_sha256"] = non_hex["input"]["obs_sha256"].upper()
    bad_manifest_hash = copy(manifest)
    bad_manifest_hash["jsonl_sha256"] = "abc"
    shifted = copy(decision)
    shifted["input"]["source_decision"] = decision["decision"]
    return [("extra key", extra, True), ("manifest extra key", manifest_extra, True),
            ("bool mask", bool_mask, True), ("non-hex hash", non_hex, True),
            ("short manifest hash", bad_manifest_hash, True),
            ("source_decision != decision-1", shifted, False)]


def test_schema_well_formed_and_documents_validate(sim_binary, multi_run_config,
                                                   long_run_config, tmp_path):
    schema = _load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "$id" not in schema
    assert schema.get("title") and schema.get("description")
    assert set(SCHEMA_DEFS) <= set(schema["$defs"])
    assert len(schema["oneOf"]) == 3
    unsupported = _keywords(schema, set()) - VALIDATION_KEYWORDS - ANNOTATION_KEYWORDS
    assert unsupported == set()

    documents = _produced_documents(sim_binary, multi_run_config, long_run_config,
                                    tmp_path)
    statuses = {manifest["status"] for manifest, _ in documents}
    assert {"complete", "interrupted"} <= statuses
    for manifest, lines in documents:
        _assert_documents_valid(schema, manifest, lines)

    manifest, lines = documents[0]
    for label, mutant, _ in _mutants(manifest, lines[1]):
        assert _document_errors(schema, mutant), label


def test_added_reward_components_produce_valid_decision_records(
        sim_binary, multi_run_config, tmp_path):
    selection = resolve_selection(
        multi_run_config, telemetry="steps",
        reward_components="service_success,sinr_quality",
        reward_weights="1,0.5")
    env = MeshRlEnv(
        sim_binary, multi_run_config, output_dir=str(tmp_path / "extended-reward"),
        selection=selection,
        decision_records=DecisionRecording(_settings(), SCRIPTED))
    try:
        _run(env)
    finally:
        env.close()

    manifest, lines = _read_sidecar(_episode(tmp_path / "extended-reward"))
    assert manifest["status"] == "complete"
    reward = lines[1]["outcome"]["reward"]
    assert set(reward["components"]) == {"service_success", "sinr_quality"}
    assert set(reward["valid"]) == set(reward["components"])
    _assert_documents_valid(_load_schema(), manifest, lines)


def test_schema_matches_jsonschema_when_installed(sim_binary, multi_run_config,
                                                  long_run_config, tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    documents = _produced_documents(sim_binary, multi_run_config, long_run_config,
                                    tmp_path)
    for manifest, lines in documents:
        for document in (manifest, *lines):
            assert list(validator.iter_errors(document)) == []
            assert _errors(document, schema, schema) == []
    manifest, lines = documents[0]
    for label, mutant, expressible in _mutants(manifest, lines[1]):
        if expressible:
            assert not validator.is_valid(mutant), label
            assert _errors(mutant, schema, schema), label


def test_schema_v1_is_frozen():
    if SCHEMA_V1_SHA256.startswith("<"):
        pytest.fail("SCHEMA_V1_SHA256 is not pinned yet: fill at publication with "
                    "`shasum -a 256 docs/schemas/decision_record.v1.schema.json`")
    assert HEX64.match(SCHEMA_V1_SHA256)
    assert _sha256(SCHEMA_PATH.read_bytes()) == SCHEMA_V1_SHA256


# 20-22. Training context, legacy refusal, CLI ------------------------------------

def test_training_context_records(trained_run, sim_binary, multi_run_config, tmp_path,
                                  monkeypatch):
    from scripts.rl import train

    out_dir = tmp_path / "train"
    monkeypatch.setattr(sys, "argv", [
        "train", "--sim-binary", sim_binary, "--run-config", multi_run_config,
        "--output-dir", str(out_dir), "--verbose", "0", "--decision-records",
        "m-ppo", "--total-timesteps", "32", "--n-steps", "16", "--seed", "1",
        "--eval-every-steps", "16", "--eval-episodes", "1"])
    assert train.main() == 0

    manifest = json.loads((out_dir / "train_manifest.json").read_text())
    baseline = json.loads((trained_run.run_dir / "train_manifest.json").read_text())
    assert set(manifest) == set(baseline)
    assert set(manifest["hyperparameters"]) == set(baseline["hyperparameters"])

    schema = _load_schema()
    training = sorted(out_dir.glob("episode-*"))
    assert len(training) > 1
    first, _ = _check_sidecar(training[0])
    assert first["status"] == "interrupted"
    assert first["episode"]["status"] == "interrupted"
    assert _rl_episode(training[0])["status"] == "interrupted"
    statuses = set()
    for episode_dir in training:
        sidecar, lines = _check_sidecar(episode_dir)
        statuses.add(sidecar["status"])
        assert (sidecar["episode"]["mode"], sidecar["episode"]["source"],
                sidecar["episode"]["policy"]) == ("training", "train", None)
        assert sidecar["model"] is None
        assert sidecar["preferences"]["captured"] is False
        assert all(record["preferences"] is None for record in lines[1:])
        _assert_documents_valid(schema, sidecar, lines)
    assert "complete" in statuses

    evaluation = sorted((out_dir / "eval").glob("episode-*"))
    assert evaluation
    statuses = set()
    for episode_dir in evaluation:
        sidecar, lines = _check_sidecar(episode_dir)
        statuses.add(sidecar["status"])
        assert (sidecar["episode"]["mode"], sidecar["episode"]["source"],
                sidecar["episode"]["policy"]) == ("evaluation", "train_eval", "model")
        assert sidecar["episode"]["seed_source"] == "eval"
        assert sidecar["model"] is None
        assert sidecar["preferences"]["captured"] is False
        _assert_documents_valid(schema, sidecar, lines)
    assert "complete" in statuses


def test_legacy_mode_is_refused(sim_binary, tmp_path):
    run_config = tmp_path / "legacy.ini"
    run_config.write_text(LEGACY_RUN_INI)
    with pytest.raises(ValueError):
        MeshRlEnv(sim_binary, str(run_config), output_dir=str(tmp_path / "rl"),
                  decision_records=DecisionRecording(_settings(), SCRIPTED))
    assert not (tmp_path / "rl").exists() or not list((tmp_path / "rl").iterdir())

    disabled = MeshRlEnv(sim_binary, str(run_config), output_dir=str(tmp_path / "off"),
                         decision_records=DecisionRecording(resolve_decision_records(),
                                                            SCRIPTED))
    disabled.close()


def test_cli_dependent_flag_without_enable_exits_one(sim_binary, multi_run_config,
                                                     tmp_path, capsys):
    out_dir = tmp_path / "eval"
    code = evaluate_cli.main(["--sim-binary", sim_binary, "--run-config", multi_run_config,
                              "--output-dir", str(out_dir), "--seeds", "1",
                              "--policies", "hold", "--decision-records-every", "2"])
    assert code == 1
    err = capsys.readouterr().err
    assert "--decision-records-every" in err
    assert re.search(r"--decision-records(?![-\w])", err), err
    assert not out_dir.exists()
