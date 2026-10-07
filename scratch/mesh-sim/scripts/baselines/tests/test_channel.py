"""ChannelScorer against fake_query.py; no real simulator is started."""

import json
import math
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from scripts.baselines.effective_inputs import start_position
from scripts.baselines.planners.channel import (ChannelScorer, LayoutResult, PlannerError,
                                                query_command)
from scripts.baselines.tests.conftest import NODES, SCENARIO_SECTIONS

MESH_ROOT = Path(__file__).resolve().parents[3]
FAKE_QUERY = Path(__file__).with_name("fake_query.py")
ROSTER = [node["id"] for node in NODES]
STARTS = np.array([start_position(node) for node in NODES])
RANGE_M = 100.0
CHANNEL_RX_GAIN = 7.5
INI_TEXT = (SCENARIO_SECTIONS.replace("seed = 1", "seed = 11\nrun_id = 4") +
            f"\n[channel]\nband = mmwave\nrx_array_gain_dbi = {CHANNEL_RX_GAIN}\n")
PROBES = [[150.0, 180.0], [400.0, 400.0], [200.0, 200.0]]
GONE_WAIT_S = 10.0


def _binary(directory: Path) -> Path:
    path = directory / "fake-mesh-sim"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!{sys.executable}\nimport runpy\n"
                    f"runpy.run_path({str(FAKE_QUERY)!r}, run_name='__main__')\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Scenario, shim binary, worker log, request record and PID file paths."""
    scenario = tmp_path / "scenario"
    scenario.mkdir()
    (scenario / "run.ini").write_text(INI_TEXT)
    (scenario / "nodes.json").write_text(json.dumps(NODES, indent=2) + "\n")
    paths = {"ini": scenario / "run.ini", "bin": _binary(tmp_path / "bin"),
             "record": tmp_path / "record.jsonl", "pids": tmp_path / "pids.json",
             "log": tmp_path / "planner.log"}
    monkeypatch.setenv("FAKE_QUERY_RECORD", str(paths["record"]))
    monkeypatch.setenv("FAKE_QUERY_PID_FILE", str(paths["pids"]))
    for name in ("FAKE_QUERY_FAULT", "FAKE_QUERY_FAULT_AT", "FAKE_QUERY_INIT_PATCH",
                 "FAKE_QUERY_PAD_BYTES", "FAKE_QUERY_RANGE_M"):
        monkeypatch.delenv(name, raising=False)
    with open(paths["log"], "w", encoding="utf-8") as log:
        paths["stream"] = log
        yield paths


def _scorer(env, *, mode="standalone", band=None, seed=7, run_id=4, jammer_seed=7,
            roster=None, starts=None, **kwargs) -> ChannelScorer:
    return ChannelScorer(env["bin"], env["ini"], seed, run_id, band, mode,
                         ROSTER if roster is None else roster,
                         STARTS if starts is None else starts, jammer_seed, env["stream"],
                         **kwargs)


def _records(env, kind="evaluate") -> list:
    lines = env["record"].read_text().splitlines() if env["record"].exists() else []
    return [r for r in map(json.loads, lines) if r.get("type") == kind]


def _pids(env) -> dict:
    return json.loads(env["pids"].read_text())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _assert_gone(*pids: int) -> None:
    deadline = time.monotonic() + GONE_WAIT_S
    while any(_alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not [pid for pid in pids if _alive(pid)], "worker descendants still alive"


def _moved(dx: float) -> np.ndarray:
    layout = STARTS.copy()
    layout[1, 0] += dx
    return layout


def _expected(layout: np.ndarray):
    distance = np.linalg.norm(layout[:, None, :] - layout[None, :, :], axis=2)
    connected = distance <= RANGE_M
    np.fill_diagonal(connected, False)
    return distance, connected


def test_argv_follows_mode_and_band(tmp_path):
    base = ["/bin/sim", "--run-config=/x/run.ini", "--channel-query", "--seed=3"]
    assert query_command("/bin/sim", "/x/run.ini", 3, None, "standalone") == base
    assert query_command("/bin/sim", "/x/run.ini", 3, "sub-6", "evaluation") == [
        *base, "--band=sub-6", "--rl-mode"]


@pytest.mark.parametrize("mode,band", [("standalone", None), ("evaluation", "sub-6")])
def test_handshake_evaluate_and_graceful_close(env, mode, band):
    with _scorer(env, mode=mode, band=band) as scorer:
        assert scorer.init["contract"] == "mesh_channel_query_v1"
        assert scorer.band == (band or "mmwave")
        assert scorer.limits["max_layouts"] == scorer.max_layouts == 1024
        assert scorer.channel["rx_array_gain_dbi"] == CHANNEL_RX_GAIN
        layouts = [STARTS, _moved(120.0)]
        results = scorer.evaluate(layouts)
        worker = scorer.pid
    assert scorer.closed
    _assert_gone(worker)
    argv = json.loads(env["record"].read_text().splitlines()[0])["argv"]
    assert [env["bin"].as_posix(), *argv] == scorer.command
    assert ("--rl-mode" in argv) == (mode == "evaluation")
    assert _records(env, "shutdown"), "close() did not send shutdown"
    for layout, result in zip(layouts, results):
        distance, connected = _expected(layout)
        assert isinstance(result, LayoutResult)
        assert np.array_equal(result.connected, connected)
        assert np.array_equal(result.connected, result.connected.T)
        assert np.array_equal(result.sinr_db, result.sinr_db.T)
        assert np.all(np.isneginf(np.diag(result.sinr_db)))
        off = ~np.eye(len(ROSTER), dtype=bool)
        expected_sinr = -6.7 + 20 * np.log10(RANGE_M / np.maximum(distance, 1.0))
        assert np.allclose(result.sinr_db[off], expected_sinr[off])
        assert np.array_equal(result.connected[off], result.sinr_db[off] >= -6.7)
        assert result.is_los[off].all() and result.coverage is None
    assert not np.array_equal(results[0].connected, results[1].connected)


def _shift_start(init_starts):
    shifted = [list(p) for p in init_starts]
    shifted[2][1] += 1e-3
    return shifted


@pytest.mark.parametrize("patch,needle", [
    ({"contract": "mesh_channel_query_v0"}, "contract"),
    ({"node_ids": list(reversed(ROSTER))}, "node_ids"),
    ({"start_positions": _shift_start(STARTS.tolist())}, "start_positions"),
    ({"seed": 8}, "seed 8 != 7"),
    ({"run_id": 1}, "run_id"),
    ({"jammer_seed": 11}, "jammer_seed"),
    ({"rl_enabled": True}, "rl_enabled"),
    ({"band": "6g"}, "band"),
    ({"sinr_threshold_db": None}, "sinr_threshold_db"),
    ({"limits": {"max_request_line_bytes": 0}}, "limits"),
    ({"type": "result"}, "init line"),
])
def test_init_mismatch_is_a_planner_error(env, monkeypatch, patch, needle):
    monkeypatch.setenv("FAKE_QUERY_INIT_PATCH", json.dumps(patch))
    with pytest.raises(PlannerError, match=needle):
        _scorer(env)
    _assert_gone(_pids(env)["worker"])


def test_requested_band_must_come_from_the_cli(env, monkeypatch):
    monkeypatch.setenv("FAKE_QUERY_INIT_PATCH", json.dumps({"band_source": "run.ini"}))
    with pytest.raises(PlannerError, match="band"):
        _scorer(env, band="mmwave")


def test_malformed_init_and_missing_binary(env, monkeypatch):
    monkeypatch.setenv("FAKE_QUERY_INIT_PATCH", json.dumps({"node_ids": "not-a-list"}))
    with pytest.raises(PlannerError):
        _scorer(env)
    with pytest.raises(PlannerError, match="cannot start"):
        ChannelScorer(env["bin"].with_name("missing"), env["ini"], 7, 4, None, "standalone",
                      ROSTER, STARTS, 7, env["stream"])


def test_batches_respect_max_layouts_and_stats_accumulate(env, monkeypatch):
    monkeypatch.setenv("FAKE_QUERY_INIT_PATCH", json.dumps({"limits": {"max_layouts": 4}}))
    layouts = [_moved(float(k)) for k in range(7)]
    with _scorer(env, max_layouts=3) as scorer:
        assert scorer.max_layouts == 3
        scorer.set_probes(PROBES)
        first = scorer.evaluate(layouts)
        assert [r["layouts"] for r in _records(env)] == [3, 3, 1]
        assert [r["request_id"] for r in _records(env)] == [1, 2, 3]
        after_first = scorer.stats
        scorer.evaluate(layouts[:2])
        stats = scorer.stats
    assert len(first) == 7
    pairs = len(ROSTER) * (len(ROSTER) - 1) // 2
    assert after_first["requests"] == 3 and after_first["layouts"] == 7
    assert (stats["requests"], stats["layouts"], stats["links"], stats["probe_links"]) == (
        4, 9, 9 * pairs, 9 * len(ROSTER) * len(PROBES))
    assert stats["wall_s"] > after_first["wall_s"] > 0
    with _scorer(env) as scorer:
        scorer.evaluate(layouts)
    assert [r["layouts"] for r in _records(env)][4:] == [4, 3]


def test_batches_respect_the_request_byte_limit(env, monkeypatch):
    one = len(json.dumps(STARTS.tolist(), separators=(",", ":")))
    fixed = len('{"type":"evaluate","request_id":%d,"layouts":[]}\n' % 10 ** 18)
    limit = fixed + 2 * one + 1
    monkeypatch.setenv("FAKE_QUERY_INIT_PATCH",
                       json.dumps({"limits": {"max_request_line_bytes": limit}}))
    layouts = [_moved(0.5 * k) for k in range(5)]
    with _scorer(env) as scorer:
        results = scorer.evaluate(layouts)
    records = _records(env)
    assert len(results) == 5
    assert [r["layouts"] for r in records] == [2, 2, 1]
    assert all(r["bytes"] <= limit and r["problem"] is None for r in records)


def test_one_layout_over_the_byte_limit_aborts(env, monkeypatch):
    monkeypatch.setenv("FAKE_QUERY_INIT_PATCH",
                       json.dumps({"limits": {"max_request_line_bytes": 2000}}))
    with _scorer(env) as scorer:
        scorer.set_probes([[float(k), 0.0] for k in range(200)])
        with pytest.raises(PlannerError, match="max_request_line_bytes"):
            scorer.evaluate([STARTS])
        assert scorer.closed
    assert not _records(env)


def test_probe_grid_over_max_probes_is_rejected(env, monkeypatch):
    monkeypatch.setenv("FAKE_QUERY_INIT_PATCH", json.dumps({"limits": {"max_probes": 2}}))
    with _scorer(env) as scorer:
        with pytest.raises(PlannerError, match="max_probes"):
            scorer.set_probes(PROBES)


def test_probe_gain_resolves_from_init_and_coverage(env):
    with _scorer(env) as scorer:
        scorer.set_probes(PROBES, height_m=2.0)
        assert scorer.probe_rx_gain_dbi == CHANNEL_RX_GAIN
        with pytest.raises(ValueError, match="already set"):
            scorer.set_probes(PROBES)
        (result,) = scorer.evaluate([STARTS])
    sent = _records(env)[0]["probes"]
    assert sent == {"height_m": 2.0, "rx_gain_dbi": CHANNEL_RX_GAIN, "sinr_db": -6.7,
                    "points": PROBES}
    probes = np.array([[x, y, 2.0] for x, y in PROBES])
    for node, covered in enumerate(result.coverage):
        distance = np.linalg.norm(probes - STARTS[node], axis=1)
        assert covered == [k for k in range(len(PROBES)) if distance[k] <= RANGE_M]
    with _scorer(env) as scorer:
        scorer.evaluate([STARTS])
        with pytest.raises(ValueError, match="before the first"):
            scorer.set_probes(PROBES, rx_gain_dbi=3.0)
    with _scorer(env) as scorer:
        scorer.set_probes(PROBES, rx_gain_dbi=3.0, sinr_db=0.0)
        assert scorer.probe_rx_gain_dbi == 3.0 and scorer.probes["sinr_db"] == 0.0


def test_invalid_layout_input_is_a_value_error(env):
    with _scorer(env) as scorer:
        with pytest.raises(ValueError):
            scorer.evaluate([STARTS[:-1]])
        bad = STARTS.copy()
        bad[0, 0] = math.nan
        with pytest.raises(ValueError):
            scorer.evaluate([bad])
        assert not scorer.closed
        assert scorer.evaluate([]) == []


def test_output_larger_than_pipe_capacity_completes(env, monkeypatch):
    monkeypatch.setenv("FAKE_QUERY_FAULT", "oversized")
    monkeypatch.setenv("FAKE_QUERY_PAD_BYTES", str(2 * 1024 * 1024))
    with _scorer(env) as scorer:
        (result,) = scorer.evaluate([STARTS])
    assert np.array_equal(result.connected, _expected(STARTS)[1])


def test_response_over_the_line_limit_aborts(env, monkeypatch):
    monkeypatch.setenv("FAKE_QUERY_FAULT", "oversized")
    monkeypatch.setenv("FAKE_QUERY_INIT_PATCH",
                       json.dumps({"limits": {"max_child_response_bytes": 512 * 1024}}))
    with _scorer(env) as scorer:
        with pytest.raises(PlannerError, match="line limit"):
            scorer.evaluate([STARTS])
        worker = scorer.pid
    _assert_gone(worker)


@pytest.mark.parametrize("fault,needle", [
    ("malformed", "malformed JSON"),
    ("nonfinite", "malformed JSON|non-finite"),
    ("layout_error", "layout 1 failed.*injected layout error"),
    ("request_error", "rejected.*injected request error"),
    ("wrong_id", "echoes request_id"),
    ("short_links", "expected 15 links"),
    ("bad_coverage", "probe index"),
    ("crash", "exited with code 3"),
])
def test_worker_faults_abort_without_partial_results(env, monkeypatch, fault, needle):
    monkeypatch.setenv("FAKE_QUERY_FAULT", f"{fault},grandchild")
    monkeypatch.setenv("FAKE_QUERY_FAULT_AT", "2")
    with _scorer(env, max_layouts=1) as scorer:
        scorer.set_probes(PROBES)
        with pytest.raises(PlannerError, match=needle):
            scorer.evaluate([STARTS, _moved(1.0), _moved(2.0)])
        assert scorer.closed
        with pytest.raises(PlannerError, match="closed"):
            scorer.evaluate([STARTS])
    pids = _pids(env)
    _assert_gone(pids["worker"], pids["grandchild"])


def test_worker_layout_rule_error_aborts(env):
    layout = STARTS.copy()
    layout[0, 2] += 1.0
    with _scorer(env) as scorer:
        with pytest.raises(PlannerError, match="layout 1 failed.*z differs"):
            scorer.evaluate([STARTS, layout])


def test_timeout_kills_hung_worker_group(env, monkeypatch):
    monkeypatch.setenv("FAKE_QUERY_FAULT", "hang,ignore_term,grandchild_ignore_term")
    started = time.monotonic()
    with _scorer(env, request_timeout_s=1.0, terminate_wait_s=0.5) as scorer:
        pids = _pids(env)
        with pytest.raises(PlannerError, match="timed out"):
            scorer.evaluate([STARTS])
    _assert_gone(pids["worker"], pids["grandchild"])
    assert time.monotonic() - started < 30


@pytest.mark.parametrize("grandchild", ["grandchild", "grandchild_ignore_term"])
def test_close_reaps_the_whole_group(env, monkeypatch, grandchild):
    monkeypatch.setenv("FAKE_QUERY_FAULT", grandchild)
    scorer = _scorer(env, terminate_wait_s=0.5)
    pids = _pids(env)
    assert _alive(pids["grandchild"])
    scorer.evaluate([STARTS])
    scorer.close()
    scorer.close()
    _assert_gone(pids["worker"], pids["grandchild"])


class _Cancel(BaseException):
    pass


def _raise_cancel(signum, frame):
    raise _Cancel()


def test_cancellation_mid_request_cleans_up(env, monkeypatch):
    monkeypatch.setenv("FAKE_QUERY_FAULT", "hang,grandchild_ignore_term")
    previous = signal.signal(signal.SIGALRM, _raise_cancel)
    try:
        with _scorer(env, terminate_wait_s=0.5) as scorer:
            pids = _pids(env)
            signal.setitimer(signal.ITIMER_REAL, 0.5)
            with pytest.raises(_Cancel):
                scorer.evaluate([STARTS])
            assert scorer.closed
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    _assert_gone(pids["worker"], pids["grandchild"])


def test_dead_parent_does_not_leave_the_worker_alive(env, tmp_path):
    script = textwrap.dedent(f"""
        import os, signal, sys
        import numpy as np
        from scripts.baselines.planners.channel import ChannelScorer
        scorer = ChannelScorer({str(env['bin'])!r}, {str(env['ini'])!r}, 7, 4, None,
                               "standalone", {ROSTER!r}, np.array({STARTS.tolist()!r}),
                               7, sys.stderr)
        print(scorer.pid, flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
    """)
    parent = subprocess.run([sys.executable, "-c", script], cwd=MESH_ROOT, text=True,
                            capture_output=True, timeout=60)
    assert parent.returncode == -signal.SIGKILL, parent.stderr
    _assert_gone(int(parent.stdout.strip()))


def test_channel_module_imports_no_rl_or_planner_dependencies():
    result = subprocess.run([sys.executable, "-c", textwrap.dedent("""
        import sys
        import scripts.baselines.planners.channel
        banned = ('gymnasium', 'torch', 'pydantic', 'shapely', 'pyproj', 'yaml')
        print(sorted(m for m in sys.modules
                     if m.split('.')[0] in banned or m.startswith('scripts.rl')))
    """)], cwd=MESH_ROOT, text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "[]"
