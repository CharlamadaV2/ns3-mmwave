"""Standalone runner lifecycle against fake_child.py; no real simulator is started."""

import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import pytest

from scripts.baselines import artifacts, runner
from scripts.baselines.tests.conftest import fake_child_binary, write_scenario
from scripts.sim_support import stop_process

MESH_ROOT = Path(__file__).resolve().parents[3]
BUILD_LIB = MESH_ROOT.parent.parent / "build" / "lib"
REFERENCE_KEYS = ("plan", "planner_log", "sim_log", "source_inputs", "effective_inputs")


def _main(binary, ini, out, *extra) -> int:
    return runner.main(["--sim-binary", str(binary), "--run-config", str(ini),
                        "--output-dir", str(out), *extra])


def _manifest(out: Path) -> dict:
    return artifacts.read_json(out / artifacts.MANIFEST_NAME)


@pytest.fixture
def record(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "child-record.json"
    monkeypatch.setenv("FAKE_CHILD_RECORD", str(path))
    return path


def test_argv_environment_and_complete_run(tmp_path, fake_child, record, stub_planner,
                                           monkeypatch):
    monkeypatch.setenv("FAKE_CHILD_MARKER", "passed-through")
    ini = write_scenario(tmp_path / "s")
    out = tmp_path / "run"
    assert _main(fake_child, ini, out, "--seeds", "1-2", "--band", "sub-6") == 0

    run = out.resolve()
    seen = json.loads(record.read_text())
    assert seen["argv"] == [f"--run-config={run}/effective-inputs/run.ini",
                            f"--output-dir={run}", "--seeds=1,2", "--band=sub-6"]
    assert seen["env"]["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(BUILD_LIB)
    assert seen["env"]["FAKE_CHILD_MARKER"] == "passed-through"
    assert seen["stdin_devnull"] is True
    assert (seen["rl_enabled"], seen["nodes_file"]) == ("false", "nodes.json")

    manifest = _manifest(out)
    assert (manifest["status"], manifest["mode"], manifest["executor"]) == (
        "complete", "standalone", "none")
    assert (manifest["method"], manifest["requested_algorithm"]) == ("geometric", "geometric")
    assert manifest["simulation_seeds"] == [1, 2]
    assert manifest["seeds"] == [
        {"seed": 1, "status": "complete", "summary": "seed-1/summary.json"},
        {"seed": 2, "status": "complete", "summary": "seed-2/summary.json"}]
    assert manifest["sim_binary_sha256"] == artifacts.sha256_file(fake_child)
    assert manifest["baseline_manifest_version"] == 2
    assert (manifest["channel_scoring"]["band"],
            manifest["channel_scoring"]["band_source"]) == ("sub-6", "cli")
    assert manifest["channel_scoring"]["planning_seed"] == 1
    assert manifest["error"] is None and manifest["ended_at"] is not None
    assert "fake_child: mode=ok" in (out / "sim.log").read_text()
    assert {"baseline_manifest.json", "planner.log", "sim.log", "source-inputs",
            "effective-inputs", "inputs", "run.log", "seed-1", "seed-2"} <= {
        p.name for p in out.iterdir()}
    assert not list(out.rglob("rl_episode.json")) and not list(out.rglob("steps.jsonl"))

    plan = artifacts.read_json(out / "effective-inputs/baseline-plan.json")
    nodes = {n["id"]: n for n in json.loads((out / "effective-inputs/nodes.json").read_text())}
    for entry in plan["nodes"]:
        if entry["selected"]:
            assert nodes[entry["id"]]["position"]["x"] == entry["planned"]["x"]
            assert nodes[entry["id"]]["position"]["y"] == entry["planned"]["y"]


def test_ini_seed_and_algorithm_override(tmp_path, fake_child, record):
    ini = write_scenario(tmp_path / "s")
    out = tmp_path / "run"
    assert _main(fake_child, ini, out, "--algorithm", "none") == 0

    run = out.resolve()
    assert json.loads(record.read_text())["argv"] == [
        f"--run-config={run}/effective-inputs/run.ini", f"--output-dir={run}"]
    manifest = _manifest(out)
    assert (manifest["method"], manifest["requested_algorithm"]) == ("none", "geometric")
    assert manifest["channel_scoring"] is None
    assert manifest["simulation_seeds"] == [1]
    assert manifest["seeds"] == [{"seed": 1, "status": "complete",
                                  "summary": "seed-1/summary.json"}]
    assert (out / "effective-inputs/nodes.json").read_bytes() == (
        (ini.parent / "nodes.json").read_bytes())


def test_simulator_exit_code_is_propagated(tmp_path, fake_child, monkeypatch):
    monkeypatch.setenv("FAKE_CHILD_MODE", "fail")
    ini = write_scenario(tmp_path / "s", overrides={"algorithm": "none"})
    out = tmp_path / "run"
    assert _main(fake_child, ini, out, "--seeds", "1,2") == 7

    manifest = _manifest(out)
    assert manifest["status"] == "failed"
    assert "exited with code 7" in manifest["error"]
    assert "simulated failure" in manifest["error"]
    assert manifest["seeds"] == [
        {"seed": 1, "status": "complete", "summary": "seed-1/summary.json"},
        {"seed": 2, "status": "failed", "summary": None}]


def test_missing_summary_fails_the_run(tmp_path, fake_child, monkeypatch):
    monkeypatch.setenv("FAKE_CHILD_MODE", "missing")
    ini = write_scenario(tmp_path / "s", overrides={"algorithm": "none"})
    out = tmp_path / "run"
    assert _main(fake_child, ini, out, "--seeds", "3,4") == 1

    manifest = _manifest(out)
    assert manifest["status"] == "failed"
    assert "seed(s) 4 have no seed-N/summary.json" in manifest["error"]
    assert manifest["seeds"] == [
        {"seed": 3, "status": "complete", "summary": "seed-3/summary.json"},
        {"seed": 4, "status": "missing", "summary": None}]


def test_refuses_non_empty_output_and_bad_arguments(tmp_path, fake_child, record):
    ini = write_scenario(tmp_path / "s", overrides={"algorithm": "none"})
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "keep.txt").write_text("x")
    assert _main(fake_child, ini, busy) == 1
    assert [p.name for p in busy.iterdir()] == ["keep.txt"]

    as_file = tmp_path / "file-out"
    as_file.write_text("x")
    assert _main(fake_child, ini, as_file) == 1
    assert as_file.read_text() == "x"

    fresh = tmp_path / "fresh"
    assert _main(tmp_path / "no-such-binary", ini, fresh) == 1
    assert _main(fake_child, tmp_path / "missing.ini", fresh) == 1
    assert _main(fake_child, ini, fresh, "--seeds", "2,2") == 1
    assert not fresh.exists()
    assert not record.exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    assert _main(fake_child, ini, empty) == 0
    assert _manifest(empty)["status"] == "complete"


def test_preparation_failure_starts_no_child(tmp_path, fake_child, record, stub_planner):
    ini = write_scenario(tmp_path / "s", overrides={"objective": "fastest"})
    out = tmp_path / "run"
    assert _main(fake_child, ini, out) == 1
    manifest = _manifest(out)
    assert manifest["status"] == "failed" and "baseline.objective" in manifest["error"]
    assert not (out / "sim.log").exists()
    assert not record.exists()


def test_manifest_status_sequence(tmp_path, fake_child, record, stub_planner, monkeypatch):
    transitions = []
    original = artifacts.set_status

    def spy(path, status, *args, **kwargs):
        transitions.append((artifacts.read_json(path)["status"], status))
        return original(path, status, *args, **kwargs)

    monkeypatch.setattr(artifacts, "set_status", spy)
    ini = write_scenario(tmp_path / "s")
    assert _main(fake_child, ini, tmp_path / "run") == 0
    assert transitions == [("preparing", "prepared"), ("prepared", "running"),
                           ("running", "complete")]
    assert json.loads(record.read_text())["manifest_status"] == "running"


def test_relocated_run_remains_readable(tmp_path, fake_child, stub_planner):
    ini = write_scenario(tmp_path / "s")
    out = tmp_path / "run"
    assert _main(fake_child, ini, out, "--seeds", "1,2") == 0
    moved = tmp_path / "elsewhere" / "moved-run"
    moved.parent.mkdir()
    shutil.move(str(out), str(moved))

    manifest = _manifest(moved)
    for key in REFERENCE_KEYS:
        assert (moved / manifest[key]).exists(), key
    for entry in manifest["seeds"]:
        assert (moved / entry["summary"]).is_file()
    for key in ("source_scenario_identity", "effective_scenario_identity"):
        identity = manifest[key]
        assert artifacts.sha256_file(moved / identity["run_config"]) == (
            identity["run_ini_sha256"])
    portable = {k: v for k, v in manifest.items() if k != "source_run_config_abs"}
    assert str(out.resolve()) not in json.dumps(portable)


def _fresh_python(script: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=MESH_ROOT,
                          text=True, capture_output=True, check=True)


def test_none_imports_no_planner_module(tmp_path):
    ini = write_scenario(tmp_path / "s", overrides={"algorithm": "none"})
    binary = fake_child_binary(tmp_path / "bin")
    result = _fresh_python(f"""
        import sys
        from scripts.baselines import runner
        code = runner.main(["--sim-binary", {str(binary)!r}, "--run-config", {str(ini)!r},
                            "--output-dir", {str(tmp_path / 'run')!r}])
        loaded = sorted(m for m in sys.modules if m == 'scripts.baselines.solver'
                        or m.startswith('scripts.baselines.planners'))
        print(code, loaded)
    """)
    assert result.stdout.strip().splitlines()[-1] == "0 []"


def test_planner_source_option_is_gone(tmp_path, fake_child, capsys):
    ini = write_scenario(tmp_path / "s", overrides={"algorithm": "none"})
    with pytest.raises(SystemExit) as exit_info:
        _main(fake_child, ini, tmp_path / "run", "--planner-source", str(tmp_path))
    assert exit_info.value.code == 2
    assert "unrecognized arguments: --planner-source" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_seed_resolution(tmp_path):
    ini = write_scenario(tmp_path / "s")
    assert runner.resolve_seeds(None, ini) == [1]
    assert runner.resolve_seeds("3-4", ini) == [3, 4]
    bare = write_scenario(tmp_path / "b", ini_text="[scenario]\nname = x\n")
    assert runner.resolve_seeds(None, bare) == [runner.DEFAULT_SCENARIO_SEED] == [42]
    bad = write_scenario(tmp_path / "c", ini_text="[scenario]\nseed = -1\n")
    with pytest.raises(ValueError, match="seed"):
        runner.resolve_seeds(None, bad)


def test_runner_import_loads_no_rl_or_ml_stack():
    result = _fresh_python("""
        import sys
        import scripts.baselines.runner
        banned = ('gymnasium', 'torch', 'stable_baselines3', 'sb3_contrib')
        print(sorted(m for m in sys.modules
                     if m.split('.')[0] in banned or m.startswith('scripts.rl')))
    """)
    assert result.stdout.strip() == "[]"


def _wait_for(path: Path, timeout_s: float = 30.0) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file() and path.read_text().strip():
            return path.read_text().strip()
        time.sleep(0.05)
    raise AssertionError(f"{path} did not appear within {timeout_s}s")


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_interruption_stops_child_and_records_interrupted(tmp_path, signum):
    ini = write_scenario(tmp_path / "s", overrides={"algorithm": "none"})
    binary = fake_child_binary(tmp_path / "bin")
    out = tmp_path / "run"
    pid_file = tmp_path / "child.pid"
    env = {**os.environ, "FAKE_CHILD_MODE": "hang", "FAKE_CHILD_PID_FILE": str(pid_file)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "scripts.baselines.runner", "--sim-binary", str(binary),
         "--run-config", str(ini), "--output-dir", str(out), "--seeds", "1,2"],
        cwd=MESH_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        child_pid = int(_wait_for(pid_file))
        assert _manifest(out)["status"] == "running"
        proc.send_signal(signum)
        _, stderr = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == 128 + signum, stderr
    manifest = _manifest(out)
    assert manifest["status"] == "interrupted"
    assert manifest["error"] == f"interrupted by {signal.Signals(signum).name}"
    assert manifest["ended_at"] is not None
    assert [s["status"] for s in manifest["seeds"]] == ["missing", "missing"]
    deadline = time.monotonic() + 10
    while not _gone(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _gone(child_pid)


@pytest.mark.parametrize("mode,expected", [("hang", -signal.SIGTERM),
                                           ("hang-ignore-term", -signal.SIGKILL)])
def test_stop_child_terminates_then_kills_and_reaps(tmp_path, monkeypatch, mode, expected):
    ini = write_scenario(tmp_path / "s", overrides={"algorithm": "none"})
    pid_file = tmp_path / "child.pid"
    monkeypatch.setenv("FAKE_CHILD_MODE", mode)
    monkeypatch.setenv("FAKE_CHILD_PID_FILE", str(pid_file))
    out = tmp_path / "out"
    out.mkdir()
    proc = subprocess.Popen([str(fake_child_binary(tmp_path / "bin")),
                             f"--run-config={ini}", f"--output-dir={out}"],
                            stdin=subprocess.DEVNULL)
    try:
        _wait_for(pid_file)
        started = time.monotonic()
        runner.stop_child(proc, wait_s=0.5)
        assert proc.returncode == expected
        assert time.monotonic() - started < 10
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.mark.parametrize("mode,expected", [("hang", -signal.SIGTERM),
                                           ("hang-ignore-term", -signal.SIGKILL)])
def test_stop_process_escalates_and_reaps(tmp_path, monkeypatch, mode, expected):
    ini = write_scenario(tmp_path / "s", overrides={"algorithm": "none"})
    pid_file = tmp_path / "child.pid"
    monkeypatch.setenv("FAKE_CHILD_MODE", mode)
    monkeypatch.setenv("FAKE_CHILD_PID_FILE", str(pid_file))
    out = tmp_path / "out"
    out.mkdir()
    proc = subprocess.Popen([str(fake_child_binary(tmp_path / "bin")),
                             f"--run-config={ini}", f"--output-dir={out}"],
                            stdin=subprocess.DEVNULL)
    try:
        _wait_for(pid_file)
        started = time.monotonic()
        stop_process(proc, 0.5)
        assert proc.returncode == expected
        assert time.monotonic() - started < 10
        stop_process(proc, 0.5)
        assert proc.returncode == expected
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


GROUP_LEADER = """
import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c",
    "import signal, time\\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
    "while True: time.sleep(0.1)"])
open(sys.argv[1], "w").write(str(child.pid))
while True:
    time.sleep(0.1)
"""


@pytest.mark.parametrize("process_group", [False, True])
def test_stop_process_group_mode_reaches_grandchildren(tmp_path, process_group):
    pid_file = tmp_path / "grandchild.pid"
    proc = subprocess.Popen([sys.executable, "-c", GROUP_LEADER, str(pid_file)],
                            stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        grandchild = int(_wait_for(pid_file))
        stop_process(proc, 0.5, process_group=process_group)
        assert proc.returncode == -signal.SIGTERM
        deadline = time.monotonic() + 10
        while not _gone(grandchild) and time.monotonic() < deadline:
            time.sleep(0.05)
        # Default mode signals only the direct child, as runner.stop_child always has.
        assert _gone(grandchild) is process_group
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

EXITING_LEADER = """
import subprocess, sys
if sys.argv[1] == "orphan":
    child = subprocess.Popen([sys.executable, "-c",
        "import signal, time\\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\\n"
        "while True: time.sleep(0.1)"])
    open(sys.argv[2], "w").write(str(child.pid))
"""


def _wait_for_zombie(pid: int, timeout_s: float = 10.0) -> None:
    """Wait until pid has exited but is not reaped (ps state Z) without calling wait()."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                               text=True).stdout.strip()
        if state.startswith("Z"):
            return
        time.sleep(0.05)
    raise AssertionError(f"process {pid} did not exit within {timeout_s}s")


@pytest.mark.parametrize("case", ["zombie", "reaped", "orphan"])
def test_stop_process_group_mode_after_the_leader_exited(tmp_path, case):
    pid_file = tmp_path / "orphan.pid"
    proc = subprocess.Popen([sys.executable, "-c", EXITING_LEADER, case, str(pid_file)],
                            stdin=subprocess.DEVNULL, start_new_session=True)
    orphan = None
    try:
        if case == "orphan":
            orphan = int(_wait_for(pid_file))
        if case == "reaped":
            proc.wait(timeout=10)
        else:
            _wait_for_zombie(proc.pid)
        started = time.monotonic()
        stop_process(proc, 0.5, process_group=True)
        assert proc.returncode == 0
        assert time.monotonic() - started < 10
        if orphan is not None:
            deadline = time.monotonic() + 10
            while not _gone(orphan) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert _gone(orphan)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if orphan is not None and not _gone(orphan):
            os.kill(orphan, signal.SIGKILL)


def test_stop_process_group_mode_falls_back_when_killpg_is_refused(tmp_path, monkeypatch):
    def refused(pgid, signum):
        raise PermissionError(1, "Operation not permitted")

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        monkeypatch.setattr("scripts.sim_support.os.killpg", refused)
        stop_process(proc, 5.0, process_group=True)
        assert proc.returncode == -signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_automatic_output_root_naming(tmp_path):
    now = datetime(2026, 9, 29, 13, 5, 7)
    first = runner.automatic_output_root(tmp_path, now)
    assert first == tmp_path / "outputs/2026-09/29/13-05-07-baseline"
    first.mkdir(parents=True)
    assert runner.automatic_output_root(tmp_path, now) == first.with_name(
        "13-05-07-baseline-2")
