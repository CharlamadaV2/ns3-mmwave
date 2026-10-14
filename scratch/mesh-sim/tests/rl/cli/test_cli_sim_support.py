"""Gap tests for scripts/sim_support.py: roots, binaries, env, seeds, INI comments, processes.

Expectations come from the Doxygen blocks in scripts/sim_support.py, scripts/CLAUDE.md
("Shared modules"), and, for strip_inline_comment, the C++ INI parser it claims to
mirror (src/util/ini-parser.{h,cc} + trimStr in src/util/string-utils.cc).
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.sim_support import (find_mesh_root, find_sim_binary, parse_seed_spec,
                                 simulator_env, stop_process, strip_inline_comment,
                                 tail_lines)

MESH_ROOT = Path(__file__).resolve().parents[3]


# 1. find_mesh_root ----------------------------------------------------------------

def test_find_mesh_root_defaults_to_the_repository_root():
    assert find_mesh_root() == MESH_ROOT
    assert (find_mesh_root() / "sim.cc").is_file()


def test_find_mesh_root_accepts_a_file_or_a_directory(tmp_path):
    root = tmp_path / "mesh"
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    (root / "sim.cc").write_text("// marker\n")
    leaf = nested / "file.py"
    leaf.write_text("")
    assert find_mesh_root(leaf) == root.resolve()
    assert find_mesh_root(nested) == root.resolve()
    assert find_mesh_root(root) == root.resolve()
    assert find_mesh_root(str(root / "sim.cc")) == root.resolve()


# 2. find_sim_binary ---------------------------------------------------------------

def _fake_ns3_tree(tmp_path: Path) -> tuple[Path, Path]:
    mesh_root = tmp_path / "ns3" / "scratch" / "mesh-sim"
    mesh_root.mkdir(parents=True)
    build = tmp_path / "ns3" / "build" / "scratch" / "mesh-sim"
    return mesh_root, build


def test_find_sim_binary_returns_the_first_sorted_match(tmp_path):
    mesh_root, build = _fake_ns3_tree(tmp_path)
    build.mkdir(parents=True)
    for name in ("ns3.42-sim-optimized", "ns3.42-sim-debug", "ns3.42-sim-default"):
        (build / name).write_text("")
    found = find_sim_binary(mesh_root)
    assert found == str(build.resolve() / "ns3.42-sim-debug")
    assert isinstance(found, str)


# 3. simulator_env -----------------------------------------------------------------

@pytest.mark.parametrize("variable", ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"])
def test_simulator_env_prepends_lib_when_unset(tmp_path, monkeypatch, variable):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)
    mesh_root, _ = _fake_ns3_tree(tmp_path)
    env = simulator_env(mesh_root)
    assert env[variable] == str((tmp_path / "ns3" / "build" / "lib").resolve())


def test_simulator_env_keeps_existing_entries_and_drops_empty_ones(tmp_path, monkeypatch):
    mesh_root, _ = _fake_ns3_tree(tmp_path)
    lib = str((tmp_path / "ns3" / "build" / "lib").resolve())
    raw = os.pathsep.join(["", "/opt/a", "", "/opt/b", ""])
    monkeypatch.setenv("LD_LIBRARY_PATH", raw)
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/opt/c")
    env = simulator_env(mesh_root)
    assert env["LD_LIBRARY_PATH"] == os.pathsep.join([lib, "/opt/a", "/opt/b"])
    assert env["DYLD_LIBRARY_PATH"] == os.pathsep.join([lib, "/opt/c"])


def test_simulator_env_is_a_copy_and_resolves_relative_roots(tmp_path, monkeypatch):
    mesh_root, _ = _fake_ns3_tree(tmp_path)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/a")
    monkeypatch.setenv("MESH_SIM_TEST_MARKER", "kept")
    monkeypatch.chdir(mesh_root.parent)
    env = simulator_env("mesh-sim")              # relative to cwd
    lib = str((tmp_path / "ns3" / "build" / "lib").resolve())
    assert env["LD_LIBRARY_PATH"].split(os.pathsep)[0] == lib
    assert env["MESH_SIM_TEST_MARKER"] == "kept"
    assert os.environ["LD_LIBRARY_PATH"] == "/opt/a"   # caller's env untouched
    assert env is not os.environ


# 4. parse_seed_spec gaps ----------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("1,,2", [1, 2]),                 # empty tokens between commas are skipped
    (",3,", [3]),
    (" 5 - 7 ", [5, 6, 7]),           # whitespace around range endpoints
    ("5,1-2", [5, 1, 2]),             # order is preserved, not sorted
    ("0", [0]),
    ("0-2", [0, 1, 2]),
    (7, [7]),                         # non-string input is str()-ed
])
def test_parse_seed_spec_accepts(raw, expected):
    assert parse_seed_spec(raw) == expected


# 5. strip_inline_comment ----------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("7 # seed", "7"),
    ("7 ; seed", "7"),
    ("a;b", "a"),
    ("a#b;c", "a"),
    ("a;b#c", "a"),
    ("# all comment", ""),
    (";", ""),
    ("", ""),
    ("  v  ", "v"),
    ("\tv\t", "v"),
    ("a = b", "a = b"),
    ("path/to#frag", "path/to"),       # a # inside a value truncates, like C++
    ("node-b, node-c", "node-b, node-c"),
])
def test_strip_inline_comment(raw, expected):
    assert strip_inline_comment(raw) == expected


_CPP_HARNESS = r"""
#include "src/util/ini-parser.h"
#include <iostream>
int main(int argc, char** argv)
{
    auto ini = mesh_sim::parseIni(argv[1]);
    for (const auto& kv : ini["s"])
    {
        std::cout << kv.first << '\x1f' << kv.second << '\x1e';
    }
    return 0;
}
"""


@pytest.fixture(scope="module")
def cpp_ini_parser(tmp_path_factory):
    """Compile the ns-3-free C++ INI parser into a tiny dumper (no ns-3 build)."""
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no C++ compiler available for the INI parity harness")
    work = tmp_path_factory.mktemp("ini-harness")
    source = work / "dump.cc"
    source.write_text(_CPP_HARNESS)
    binary = work / "dump"
    result = subprocess.run(
        [compiler, "-std=c++17", "-I", str(MESH_ROOT), str(source),
         str(MESH_ROOT / "src/util/ini-parser.cc"),
         str(MESH_ROOT / "src/util/string-utils.cc"), "-o", str(binary)],
        capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"C++ harness did not compile: {result.stderr[-500:]}")
    return binary


def _cpp_values(binary: Path, tmp_path: Path, values: list[str]) -> list[str]:
    ini = tmp_path / "parity.ini"
    lines = ["[s]"] + [f"k{index:03d} = {value}" for index, value in enumerate(values)]
    ini.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    out = subprocess.run([str(binary), str(ini)], capture_output=True, check=True).stdout
    parsed = {}
    for record in out.decode("utf-8").split("\x1e"):
        if record:
            key, _, value = record.partition("\x1f")
            parsed[key] = value
    return [parsed.get(f"k{index:03d}", "<absent>") for index in range(len(values))]


_PARITY_VALUES = ["7 # seed", "7;x", "a#b;c", "  spaced  ", "\ttabbed\t",
                  "a = b = c", "x # y ; z", "node-b, node-c", "0.0 ; west edge",
                  "1e-3", "trailing\r"]


def test_strip_inline_comment_matches_the_cpp_parser(cpp_ini_parser, tmp_path):
    cpp = _cpp_values(cpp_ini_parser, tmp_path, _PARITY_VALUES)
    python = [strip_inline_comment(value) for value in _PARITY_VALUES]
    assert python == cpp


# 6. tail_lines --------------------------------------------------------------------

def test_tail_lines_keeps_the_last_lines(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text("".join(f"line {i}\n" for i in range(1, 6)))
    assert tail_lines(path, 2) == "line 4\nline 5"
    assert tail_lines(path, 10) == "\n".join(f"line {i}" for i in range(1, 6))
    assert tail_lines(path, 0) == ""
    assert tail_lines(str(path)).splitlines()[-1] == "line 5"


# 7. stop_process ------------------------------------------------------------------

_IGNORE_TERM = ("import signal, sys, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "print('ready', flush=True)\n"
                "time.sleep(60)\n")


def test_stop_process_on_an_exited_child_is_a_no_op():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    stop_process(proc, 0.5)
    assert proc.returncode == 0


def test_stop_process_terminates_a_running_child():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    stop_process(proc, 5.0)
    assert proc.returncode == -signal.SIGTERM


def test_stop_process_kills_a_child_that_ignores_sigterm():
    proc = subprocess.Popen([sys.executable, "-c", _IGNORE_TERM], stdout=subprocess.PIPE,
                            text=True)
    assert proc.stdout.readline().strip() == "ready"
    started = time.monotonic()
    stop_process(proc, 0.3)
    assert proc.returncode == -signal.SIGKILL
    assert time.monotonic() - started < 10
    proc.stdout.close()


def _alive(pid: int) -> bool:
    """True if pid exists and is not a zombie."""
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        try:
            return stat.read_text().split(")")[-1].split()[0] != "Z"
        except OSError:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_stop_process_group_mode_reaps_a_term_ignoring_grandchild():
    launcher = ("import subprocess, sys, time\n"
                f"child = subprocess.Popen([sys.executable, '-c', {_IGNORE_TERM!r}],"
                " stdout=subprocess.PIPE, text=True)\n"
                "child.stdout.readline()\n"
                "print(child.pid, flush=True)\n"
                "time.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, "-c", launcher], stdout=subprocess.PIPE,
                            text=True, start_new_session=True)
    grandchild = int(proc.stdout.readline().strip())
    assert _alive(grandchild)
    stop_process(proc, 0.3, process_group=True)
    assert proc.returncode is not None
    deadline = time.monotonic() + 5
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(grandchild)
    proc.stdout.close()


def test_stop_process_group_mode_tolerates_an_exited_leader():
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    proc.wait()
    stop_process(proc, 0.2, process_group=True)
    assert proc.returncode == 0
