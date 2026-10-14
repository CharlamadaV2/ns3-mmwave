"""Decision-record helpers, recorder file formats, caps, and failure isolation.

Spec: src/rl/decision-records.md (Files, Masks and hashes, Requested/revalidated/
applied, Sampling/caps/coverage/status, Failure isolation) and scripts/rl/env/
CLAUDE.md ("recorder failures must never raise into the env").
"""

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from scripts.rl.env import decisions
from scripts.rl.env.decisions import (DECISIONS_FILE, DECISIONS_MANIFEST, DecisionContext, DecisionRecorder, DecisionRecording, coverage_gaps, coverage_ranges, mask_sha256, masked_probs, reset_record, resolve_decision_records)
from scripts.rl.env.mesh_env import MeshRlEnv
from scripts.rl.env.selection import resolve_selection
from scripts.rl.env.telemetry import obs_sha256

HOLD = 4
TRAINING = DecisionContext(mode="training", source="train")
EVALUATION = DecisionContext(mode="evaluation", source="evaluate", policy="hold")
SLOT_NODE_IDS = ["node-b", "node-c", None]
CONTRACT = {
    "contract": "mesh_move_2d_v1",
    "node_ids": ["node-a", "node-b", "node-c"],
    "slot_node_ids": SLOT_NODE_IDS,
    "action_meanings": ["west", "east", "south", "north", "hold"],
    "tick_s": 0.1,
    "decision_interval_ticks": 5,
    "num_decisions": 4,
}
OBS_SCHEMA = {"schema_id": "local_links_v1", "dtype": "float32", "obs_dim": 4,
              "sha256": "0" * 64}
REWARD_SCHEMA = {"sha256": "1" * 64}
EPISODE = {"dir_name": "episode-0000", "index": 0, "seed": 7, "seed_source": "cli"}
MASK = [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]


def recorder(tmp_path: Path, context=TRAINING, telemetry="none", **settings):
    return DecisionRecorder(
        tmp_path, resolve_decision_records(enabled=True, **settings), context,
        episode=EPISODE, contract=CONTRACT,
        selection_describe={"telemetry": telemetry, "telemetry_every": 1},
        observation_schema=OBS_SCHEMA, reward_schema=REWARD_SCHEMA)


def reset_msg():
    return {"decision": 0, "tick": 0, "time_s": 0.0, "mask": MASK}


def step_msg(decision: int, done: bool = False, revalidated=()):
    return {"decision": decision, "tick": 5 * decision, "time_s": 0.5 * decision,
            "ticks_in_step": 5, "revalidated_slots": list(revalidated), "done": done,
            "reward": 0.25}


def pre(decision: int, preferences=None):
    return {"obs": np.asarray([decision, 0.5, -0.25, 1.0], dtype=np.float32),
            "mask": MASK, "requested": [0, 1, HOLD], "preferences": preferences}


def manifest(tmp_path: Path) -> dict:
    return json.loads((tmp_path / DECISIONS_MANIFEST).read_text())


def lines(tmp_path: Path) -> list[str]:
    return (tmp_path / DECISIONS_FILE).read_text().splitlines()


def run_episode(rec, decisions_to_run: int, steps_saved=False):
    obs = np.zeros(4, dtype=np.float32)
    rec.record_reset(reset_msg(), obs, steps_saved)
    for n in range(1, decisions_to_run + 1):
        rec.record_decision(step_msg(n, done=n == CONTRACT["num_decisions"]), pre(n - 1),
                            0.25, steps_saved)


# --- Settings ---------------------------------------------------------------------


def test_resolve_settings_normalizes_strings():
    settings = resolve_decision_records(enabled=True, obs_vector=" always ",
                                        record_every=" 3 ", max_bytes_per_episode="0")
    assert (settings.obs_vector, settings.record_every,
            settings.max_bytes_per_episode) == ("always", 3, 0)


def test_explicit_disable_without_dependents_is_cli_sourced():
    settings = resolve_decision_records(enabled=False)
    assert settings.enabled is False
    assert settings.describe()["source"]["enabled"] == "cli"


# --- Pure helpers -------------------------------------------------------------------


@pytest.mark.parametrize("saved,ranges", [
    ([], []),
    ([3], [[3, 3]]),
    ([5, 1, 2, 2, 3], [[1, 3], [5, 5]]),
    ([1, 3, 5], [[1, 1], [3, 3], [5, 5]]),
])
def test_coverage_ranges(saved, ranges):
    assert coverage_ranges(saved) == ranges


@pytest.mark.parametrize("saved,expected,gaps", [
    ([], 0, []),
    ([], 3, [[1, 3]]),
    ([1, 2, 3], 3, []),
    ([0, 2, 9], 3, [[1, 1], [3, 3]]),   # 0 (reset) and out-of-range ignored
    ([3], 3, [[1, 2]]),
])
def test_coverage_gaps_partition(saved, expected, gaps):
    assert coverage_gaps(saved, expected) == gaps


def test_masked_probs_edge_cases():
    logits = np.asarray([[0, 0, 0, 0, 0],
                         [1000.0, 1001.0, -1000.0, 5.0, 0.0],
                         [3.0, 1.0, 2.0, 0.0, 9.0]], dtype=np.float32)
    mask = [1, 1, 1, 1, 1,
            1, 1, 0, 0, 0,
            0, 0, 0, 0, 0]
    probs = masked_probs(logits, mask)
    assert probs[0] == [0.2] * 5
    assert probs[1][2:] == [0.0, 0.0, 0.0]
    assert probs[1][0] + probs[1][1] == pytest.approx(1.0, abs=2e-6)
    assert probs[1][1] > probs[1][0]
    assert probs[2] == [0.0] * 5          # a fully masked slot has no distribution
    single = masked_probs(np.zeros(5), [0, 0, 1, 0, 0])
    assert single == [[0.0, 0.0, 1.0, 0.0, 0.0]]


def test_reset_record_format():
    obs = np.asarray([1.0, 2.0], dtype=np.float64)
    record = reset_record(dict(reset_msg(), mask=[True, False]), obs, "float64", False)
    assert record["type"] == "reset" and record["mask"] == [1, 0]
    assert record["mask_sha256"] == mask_sha256([1, 0])
    assert record["steps_ref"] is None
    assert record["obs_sha256"] == obs_sha256(obs, "float64")
    saved = reset_record(reset_msg(), obs, "float64", True)
    assert saved["steps_ref"] == {"file": "steps.jsonl", "decision": 0,
                                  "obs_sha256": obs_sha256(obs)}


# --- Recorder: files and formats --------------------------------------------------

def test_open_writes_manifest_atomically_with_writing_status(tmp_path):
    recorder(tmp_path)
    data = manifest(tmp_path)
    assert data["status"] == "writing" and data["contract"] == "decision_record"
    assert data["coverage"]["records"] == 0
    assert data["coverage"]["decision_0_retained"] is False
    assert (tmp_path / DECISIONS_FILE).read_bytes() == b""
    assert not list(tmp_path.glob("*.tmp"))


def test_lines_are_compact_sorted_json_and_manifest_hash_matches(tmp_path):
    rec = recorder(tmp_path)
    run_episode(rec, 4)
    rec.close("completed", "done", None)
    raw = (tmp_path / DECISIONS_FILE).read_bytes()
    for line in raw.decode().splitlines():
        parsed = json.loads(line)
        assert line == json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    data = manifest(tmp_path)
    assert data["status"] == "complete"
    assert data["jsonl_sha256"] == hashlib.sha256(raw).hexdigest()
    assert data["bytes"] == len(raw)
    assert data["coverage"]["saved_ranges"] == [[1, 4]] and data["coverage"]["gaps"] == []


def test_decision_record_joins_previous_message(tmp_path):
    rec = recorder(tmp_path)
    run_episode(rec, 2)
    record = json.loads(lines(tmp_path)[2])
    assert record["decision"] == 2
    assert record["targets"] == ["node-b", "node-c"]
    assert record["input"]["source_decision"] == 1
    assert record["input"]["tick"] == record["outcome"]["tick"] - \
        record["outcome"]["ticks_in_step"]
    assert record["outcome"]["interval_s"] == {"start_exclusive": 0.5,
                                               "end_inclusive": 1.0}
    assert record["input"]["steps_ref"] is None
    # No steps.jsonl row exists, so the exact float32 vector is stored.
    assert record["input"]["obs_vector"] == pre(1)["obs"].tolist()
    assert record["outcome"]["reward"] == {"total": 0.25, "source": "cpp"}
    assert record["outcome"]["reward_schema_sha256"] == REWARD_SCHEMA["sha256"]


def test_revalidated_slots_become_hold_in_applied(tmp_path):
    rec = recorder(tmp_path)
    rec.record_reset(reset_msg(), np.zeros(4, dtype=np.float32), False)
    rec.record_decision(step_msg(1, revalidated=[1]), pre(0), 0.0, False)
    action = json.loads(lines(tmp_path)[1])["action"]
    assert action == {"requested": [0, 1, HOLD], "revalidated_slots": [1],
                      "applied": [0, HOLD, HOLD], "applied_status": "derived",
                      "hold_index": HOLD}


def test_terminal_decision_is_saved_off_cadence(tmp_path):
    rec = recorder(tmp_path, record_every=3)
    run_episode(rec, 4)
    rec.close("completed", "done", None)
    assert [json.loads(line)["decision"] for line in lines(tmp_path)] == [0, 3, 4]
    assert manifest(tmp_path)["coverage"]["gaps"] == [[1, 2]]


def test_preferences_only_recorded_in_evaluation_context(tmp_path):
    logits = np.arange(15, dtype=np.float32)
    train_dir, eval_dir = tmp_path / "train", tmp_path / "eval"
    train_dir.mkdir()
    eval_dir.mkdir()
    for directory, context in ((train_dir, TRAINING), (eval_dir, EVALUATION)):
        rec = recorder(directory, context=context)
        rec.record_reset(reset_msg(), np.zeros(4, dtype=np.float32), False)
        rec.record_decision(step_msg(1), pre(0, preferences=logits), 0.0, False)
    assert json.loads(lines(train_dir)[1])["preferences"] is None
    prefs = json.loads(lines(eval_dir)[1])["preferences"]
    assert prefs["per_slot"][0] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert prefs["masked_probs"][2][:4] == [0.0, 0.0, 0.0, 0.0]


# --- Recorder: ordering errors (raised to the caller, which isolates them) ----------


# --- Recorder: byte cap boundaries --------------------------------------------------

def _line_sizes(tmp_path: Path) -> list[int]:
    probe = tmp_path / "probe"
    probe.mkdir()
    rec = recorder(probe, max_bytes_per_episode=0)
    run_episode(rec, 4)
    rec.close("completed", "done", None)
    return [len(line) + 1 for line in lines(probe)]


def test_cap_equal_to_written_bytes_is_not_exceeded(tmp_path):
    sizes = _line_sizes(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    rec = recorder(out, max_bytes_per_episode=sizes[0] + sizes[1])
    run_episode(rec, 4)
    rec.close("completed", "done", None)
    data = manifest(out)
    assert [json.loads(line)["decision"] for line in lines(out)] == [0, 1]
    assert data["status"] == "capped"
    assert data["bytes"] == sizes[0] + sizes[1]
    # Decisions are still counted after the cap.
    assert data["coverage"]["terminal_decision"] == 4
    assert data["coverage"]["gaps"] == [[2, 4]]


# --- Recorder: close / fail lifecycle -----------------------------------------------

def test_close_is_idempotent_and_fail_after_close_is_noop(tmp_path):
    rec = recorder(tmp_path)
    run_episode(rec, 4)
    rec.close("completed", "done", None)
    first = (tmp_path / DECISIONS_MANIFEST).read_bytes()
    rec.close("interrupted", "close", "late")
    rec.fail("late failure")
    assert rec.status == "complete"
    assert (tmp_path / DECISIONS_MANIFEST).read_bytes() == first


def test_fail_truncates_error_and_close_keeps_failed(tmp_path):
    rec = recorder(tmp_path)
    run_episode(rec, 1)
    rec.fail("x" * 5000)
    rec.close("completed", "done", None)
    data = manifest(tmp_path)
    assert data["status"] == "failed" and len(data["error"]) == 1000
    raw = (tmp_path / DECISIONS_FILE).read_bytes()
    assert data["jsonl_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("episode_error,expected", [
    (None, "episode interrupted (close)"),
    ("simulator exited 3", "simulator exited 3"),
])
def test_interrupted_status_and_error(tmp_path, episode_error, expected):
    rec = recorder(tmp_path)
    run_episode(rec, 2)
    rec.close("interrupted", "close", episode_error)
    data = manifest(tmp_path)
    assert data["status"] == "interrupted" and data["error"] == expected
    assert data["episode"]["status"] == "interrupted"
    assert data["coverage"]["terminal_decision"] == 2


def test_failed_episode_maps_to_interrupted_sidecar(tmp_path):
    # Table: interrupted = "Episode reset, closed, or failed before done".
    rec = recorder(tmp_path)
    run_episode(rec, 1)
    rec.close("failed", "error", "protocol error")
    assert manifest(tmp_path)["status"] == "interrupted"


# --- Failure isolation through MeshRlEnv (fake simulator) -------------------------

def _env(sim_binary, run_config, out_dir):
    selection = resolve_selection(run_config, telemetry="steps")
    records = DecisionRecording(resolve_decision_records(enabled=True), TRAINING)
    return MeshRlEnv(sim_binary, run_config, output_dir=str(out_dir),
                     selection=selection, decision_records=records)


def _run_to_done(env) -> int:
    env.reset()
    steps, done = 0, False
    while not done:
        _, _, done, _, _ = env.step([HOLD, HOLD, HOLD])
        steps += 1
    return steps


def _isolated_run(sim_binary, run_config, out_dir):
    env = _env(sim_binary, run_config, out_dir)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            steps = _run_to_done(env)
        finally:
            env.close()
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)
               and "Decision records" in str(w.message)]
    episode = Path(out_dir) / "episode-0000"
    rl_episode = json.loads((episode / "rl_episode.json").read_text())
    return steps, runtime, episode, rl_episode


def test_recorder_open_failure_does_not_reach_the_env(sim_binary, multi_run_config,
                                                     tmp_path, monkeypatch):
    real_open = open

    def failing_open(path, *args, **kwargs):
        if str(path).endswith(DECISIONS_FILE):
            raise PermissionError("injected open failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(decisions, "open", failing_open, raising=False)
    steps, runtime, episode, rl_episode = _isolated_run(sim_binary, multi_run_config,
                                                        tmp_path / "rl")
    assert steps == 2 and rl_episode["status"] == "completed"
    assert len(runtime) == 1
    data = json.loads((episode / DECISIONS_MANIFEST).read_text())
    assert data["status"] == "failed" and "injected open failure" in data["error"]
    assert (episode / "steps.jsonl").is_file()


def test_close_failure_does_not_reach_the_env(sim_binary, multi_run_config, tmp_path,
                                              monkeypatch):
    def broken_close(self, *args, **kwargs):
        raise OSError("injected close failure")

    monkeypatch.setattr(DecisionRecorder, "close", broken_close)
    steps, runtime, episode, rl_episode = _isolated_run(sim_binary, multi_run_config,
                                                        tmp_path / "rl")
    assert steps == 2 and rl_episode["status"] == "completed"
    assert len(runtime) == 1
    data = json.loads((episode / DECISIONS_MANIFEST).read_text())
    assert data["status"] == "failed" and "injected close failure" in data["error"]
