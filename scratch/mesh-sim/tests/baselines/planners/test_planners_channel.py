"""ChannelScorer (mesh_channel_query_v1 client) edge cases: fake_query.py and custom workers.

No real simulator is started. Gaps over scripts/baselines/tests/test_channel.py: batch
splits at exactly max_layouts / the byte limit, probe-limit boundary, init type and
tolerance edges, framing (split lines, stray lines, echoed ids), the worker's own
`connected` flags, worker death between/mid requests, and cleanup on exceptions.
"""

import json
import time

import numpy as np
import pytest

from scripts.baselines.effective_inputs import start_position
from scripts.baselines.planners.channel import ChannelScorer, PlannerError
from scripts.baselines.tests.conftest import NODES, SCENARIO_SECTIONS, fake_query_binary

from ._planner_support import (CUSTOM_ROSTER, CUSTOM_STARTS, assert_gone, make_worker,
                               read_records)

ROSTER = [node["id"] for node in NODES]
STARTS = np.array([start_position(node) for node in NODES])
FAKE_ENV = ("FAKE_QUERY_FAULT", "FAKE_QUERY_FAULT_AT", "FAKE_QUERY_INIT_PATCH",
            "FAKE_QUERY_PAD_BYTES", "FAKE_QUERY_RANGE_M", "FAKE_QUERY_RECORD",
            "FAKE_QUERY_PID_FILE", "CUSTOM_PID", "CUSTOM_RECORD")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Scenario for fake_query.py (seed 7, run_id 1), shim, record/PID files and log."""
    for name in FAKE_ENV:
        monkeypatch.delenv(name, raising=False)
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    (scenario / "run.ini").write_text(SCENARIO_SECTIONS.replace("seed = 1", "seed = 7"))
    (scenario / "nodes.json").write_text(json.dumps(NODES, indent=2) + "\n")
    paths = {"ini": scenario / "run.ini", "bin": fake_query_binary(tmp_path / "bin"),
             "record": tmp_path / "record.jsonl", "pids": tmp_path / "pids.json",
             "custom_pid": tmp_path / "custom.pid", "custom_record": tmp_path / "custom.jsonl",
             "log": tmp_path / "planner.log", "tmp": tmp_path}
    monkeypatch.setenv("FAKE_QUERY_RECORD", str(paths["record"]))
    monkeypatch.setenv("FAKE_QUERY_PID_FILE", str(paths["pids"]))
    monkeypatch.setenv("CUSTOM_PID", str(paths["custom_pid"]))
    monkeypatch.setenv("CUSTOM_RECORD", str(paths["custom_record"]))
    with open(paths["log"], "w", encoding="utf-8") as log:
        paths["stream"] = log
        yield paths


def _fake(env, *, mode="standalone", band=None, roster=None, starts=None, **kwargs):
    return ChannelScorer(env["bin"], env["ini"], 7, 1, band, mode,
                         ROSTER if roster is None else roster,
                         STARTS if starts is None else starts, 7, env["stream"], **kwargs)


def _custom(env, body, **kwargs):
    binary = make_worker(env["tmp"] / "custom", body)
    return ChannelScorer(binary, env["ini"], 7, 1, None, "standalone", CUSTOM_ROSTER,
                         CUSTOM_STARTS, 7, env["stream"], **kwargs)


def _custom_pid(env) -> int:
    deadline = time.monotonic() + 5.0
    while not env["custom_pid"].exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    return int(env["custom_pid"].read_text())


def _patch_init(monkeypatch, patch: dict) -> None:
    monkeypatch.setenv("FAKE_QUERY_INIT_PATCH", json.dumps(patch))


def _expected_connected(layout) -> np.ndarray:
    layout = np.asarray(layout, dtype=float)
    distance = np.linalg.norm(layout[:, None, :] - layout[None, :, :], axis=2)
    connected = distance <= 100.0
    np.fill_diagonal(connected, False)
    return connected


# ---------------------------------------------------------------------------------------
# Init handshake edges.


def test_init_band_from_ini_is_fine_when_no_band_was_requested(env, monkeypatch):
    _patch_init(monkeypatch, {"band": "sub-6", "band_source": "run.ini"})
    with _fake(env) as scorer:
        assert scorer.band == "sub-6"


@pytest.mark.parametrize("body,needle", [
    ("emit('hello')", "malformed JSON in init line"),
    ("emit('{\"type\": \"init\", \"sinr_threshold_db\": NaN}')", "malformed JSON"),
    ("emit([1, 2, 3])", "did not start with an init line"),
    ("sys.exit(5)", "ended before the init line; it exited with code 5"),
])
def test_bad_or_missing_init_line_aborts_and_reaps(env, body, needle):
    with pytest.raises(PlannerError, match=needle):
        _custom(env, body)
    assert_gone(_custom_pid(env))


def test_silent_worker_times_out_on_init_and_is_reaped(env):
    started = time.monotonic()
    with pytest.raises(PlannerError, match="timed out waiting for the init line"):
        _custom(env, "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                     "while True:\n    time.sleep(0.1)",
                init_timeout_s=0.5, terminate_wait_s=0.3)
    assert_gone(_custom_pid(env))
    assert time.monotonic() - started < 15


def test_worker_stderr_reaches_the_planner_log(env):
    body = ("sys.stderr.write('custom-worker-diagnostic\\n'); sys.stderr.flush()\n"
            "emit(init_obj())\nfor req in requests():\n    emit(result(req))")
    with _custom(env, body):
        pass
    env["stream"].flush()
    assert "custom-worker-diagnostic" in env["log"].read_text()


# ---------------------------------------------------------------------------------------
# Batching boundaries.


def test_results_keep_input_order_across_batches_and_probes_ride_every_batch(env):
    probes = [[150.0, 180.0], [400.0, 400.0]]
    layouts = []
    for k in range(5):
        layout = STARTS.copy()
        layout[1, 0] += 40.0 * k   # uav-a drifts away from gw step by step
        layouts.append(layout)
    with _fake(env, max_layouts=2) as scorer:
        scorer.set_probes(probes)
        results = scorer.evaluate(layouts)
        stats = scorer.stats
    for layout, result in zip(layouts, results):
        assert np.array_equal(result.connected, _expected_connected(layout))
    records = read_records(env["record"], "evaluate")
    assert [r["layouts"] for r in records] == [2, 2, 1]
    assert all(r["probes"]["points"] == probes for r in records)
    assert stats["probe_links"] == 5 * len(ROSTER) * len(probes)


# ---------------------------------------------------------------------------------------
# Framing and response validation through custom workers.

SERVE = "emit(init_obj())\nfor req in requests():\n"


def test_response_split_across_many_writes_is_reassembled(env):
    body = SERVE + ("    text = json.dumps(result(req)) + '\\n'\n"
                    "    for k in range(0, len(text), 13):\n"
                    "        sys.stdout.write(text[k:k + 13]); sys.stdout.flush()\n"
                    "        time.sleep(0.001)\n")
    with _custom(env, body) as scorer:
        first, second = scorer.evaluate([CUSTOM_STARTS, CUSTOM_STARTS])
    assert np.array_equal(first.connected, _expected_connected(CUSTOM_STARTS))
    assert np.array_equal(first.sinr_db, second.sinr_db)


def test_worker_connected_flags_are_used_verbatim(env):
    # The worker says the strong 0-1 link is down and the weak 0-2 link is up; the client
    # must not re-derive `connected` from sinr_db or sinr_threshold_db.
    body = ("emit(init_obj(sinr_threshold_db=50.0))\nfor req in requests():\n"
            "    out = result(req)\n"
            "    for entry in out['layouts']:\n"
            "        entry['links'] = [[0, 1, 40.0, 1.0, True, False],\n"
            "                          [0, 2, -90.0, 0.0, False, True],\n"
            "                          [1, 2, -6.7, 0.0, False, False]]\n"
            "    emit(out)\n")
    with _custom(env, body) as scorer:
        assert scorer.sinr_threshold_db == 50.0
        (result,) = scorer.evaluate([CUSTOM_STARTS])
    assert result.connected.tolist() == [[False, False, True], [False, False, False],
                                         [True, False, False]]
    assert result.sinr_db[0, 1] == 40.0 and result.sinr_db[2, 0] == -90.0
    assert result.is_los[0, 1] and not result.is_los[0, 2]


# ---------------------------------------------------------------------------------------
# Worker death and cleanup.

def test_worker_exit_between_requests(env):
    body = SERVE + "    emit(result(req))\n    sys.exit(0)\n"
    scorer = _custom(env, body, terminate_wait_s=0.5)
    pid = scorer.pid
    (first,) = scorer.evaluate([CUSTOM_STARTS])
    assert first.connected[0, 1]
    with pytest.raises(PlannerError, match="ended before|stopped reading"):
        scorer.evaluate([CUSTOM_STARTS])
    assert scorer.closed
    scorer.close()  # idempotent, no exception after the abort
    assert_gone(pid)


def test_worker_death_mid_batch_returns_nothing(env):
    body = ("emit(init_obj())\n"
            "for n, req in enumerate(requests(), 1):\n"
            "    if n == 2:\n        os._exit(0)\n"
            "    emit(result(req))\n")
    with _custom(env, body, max_layouts=1) as scorer:
        with pytest.raises(PlannerError, match="ended before the response to request 2.*"
                                               "exited with code 0"):
            scorer.evaluate([CUSTOM_STARTS] * 3)
        assert scorer.stats["requests"] == 1
    assert len(read_records(env["custom_record"], "evaluate")) == 2


def test_close_kills_a_worker_that_ignores_shutdown_and_sigterm(env):
    body = ("emit(init_obj())\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "for req in requests():\n    emit(result(req))\n"
            "while True:\n    time.sleep(0.1)\n")
    scorer = _custom(env, body, shutdown_grace_s=0.3, terminate_wait_s=0.3)
    pid = scorer.pid
    scorer.evaluate([CUSTOM_STARTS])
    started = time.monotonic()
    scorer.close()
    assert time.monotonic() - started < 10
    assert read_records(env["custom_record"], "shutdown"), "shutdown was not sent"
    assert_gone(pid)
