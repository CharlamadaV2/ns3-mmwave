"""CSV metadata and coordinate comparison contracts for normalized snapshots."""
# test_regression_check.py
# pytest tests for regression_check.py / regression_snapshot.py. They also cover
# the launcher contract (stdin closed, clean library paths) shared with
# run_batch.py and scripts.sweep, and the missing-reference error message.
# Run from scratch/mesh-sim/ with: python -m pytest scripts/validation/tests
# No real simulator is started; subprocess.run is replaced by a fake.

from copy import deepcopy
import os
import subprocess
import sys

import pytest

from scripts.validation import regression_check
from scripts.validation.regression_check import compare_snapshots, load_table


## @fn test_positions_header_and_coordinates
# @brief `load_table` skips `#` metadata lines, keeps x/y/z numeric, and a change in any axis is reported.
#
# @param tmp_path  pytest temporary directory.
# @param prefix    Text placed before the header: none, or two `#` metadata lines.
# @return None; asserts on failure.
#
# Loads a one-row positions table, checks a snapshot equals a copy of itself, then
# adds 1 to each of x, y, z in turn and expects exactly one reported difference
# at `positions.csv[row 0].<axis>`.
@pytest.mark.parametrize("prefix", ["", "# scenario=test\n# frequency=28\n"])
def test_positions_header_and_coordinates(tmp_path, prefix):
    path = tmp_path / "positions.csv"
    path.write_text(prefix + "time_s,node_id,x,y,z,node_type,active\n"
                    "0,relay,1,2,3,drone,1\n")
    table = load_table(path)
    assert table["columns"] == ["time_s", "node_id", "x", "y", "z", "node_type", "active"]
    assert len(table["rows"]) == 1
    assert table["rows"][0]["node_id"] == "relay"
    assert [table["rows"][0][axis] for axis in "xyz"] == [1.0, 2.0, 3.0]

    baseline = {"tables": {"positions.csv": table}, "summary": {}}
    assert compare_snapshots(baseline, deepcopy(baseline))["match"]
    for axis in "xyz":
        candidate = deepcopy(baseline)
        candidate["tables"]["positions.csv"]["rows"][0][axis] += 1
        report = compare_snapshots(baseline, candidate)
        assert not report["match"] and report["counts"]["total"] == 1
        assert report["differences"][0]["where"] == f"positions.csv[row 0].{axis}"


## @fn test_no_header_is_empty
# @brief A file that is empty or has only `#` lines loads as an empty table.
#
# @param tmp_path  pytest temporary directory.
# @param text      File contents: empty, or one `#` metadata line.
# @return None; asserts on failure.
@pytest.mark.parametrize("text", ["", "# scenario=test\n"])
def test_no_header_is_empty(tmp_path, text):
    path = tmp_path / "positions.csv"
    path.write_text(text)
    assert load_table(path) == {"columns": [], "rows": []}


## @fn test_unattended_launchers_close_stdin_and_clean_loader_paths
# @brief Launchers pass `stdin=DEVNULL` and a library path without empty entries.
#
# @param tmp_path     pytest temporary directory.
# @param monkeypatch  pytest fixture used to set env vars and replace `subprocess.run`.
# @param existing     Pre-set `DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH` value: unset, or
#                     one with empty entries around `approved-lib`.
# @return None; asserts on failure.
#
# Runs `regression_check.cmd_capture`, `run_batch.main` and the sweep runner
# against a fake `subprocess.run` (three calls in total). Each call must close
# stdin, and each loader variable must contain no empty parts, with `approved-lib`
# kept last when it was set.
@pytest.mark.parametrize("existing", [None, os.pathsep + "approved-lib" + os.pathsep * 2])
def test_unattended_launchers_close_stdin_and_clean_loader_paths(tmp_path, monkeypatch, existing):
    from scripts.sweep.config import SweepConfig
    from scripts.sweep.runner import run_sweep
    from scripts.validation import run_batch

    for var in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        if existing is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, existing)
    calls = []

    ## @brief Stand-in for `subprocess.run` that asserts the launcher contract and records the command.
    # @param cmd     Command list the launcher tried to run.
    # @param kwargs  Keyword args passed to `subprocess.run`; needs `stdin` and `env`.
    # @return `CompletedProcess` with return code 0.
    def fake_run(cmd, **kwargs):
        assert kwargs["stdin"] == subprocess.DEVNULL
        for var in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
            parts = kwargs["env"][var].split(os.pathsep)
            assert all(parts)
            assert len(parts) == (1 if existing is None else 2)
            if existing is not None:
                assert parts[-1] == "approved-lib"
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    (tmp_path / "sim.cc").touch()
    scenario = tmp_path / "inputs" / "base"
    scenario.mkdir(parents=True)
    (scenario / "run.ini").write_text("[scenario]\nnodes_file = nodes.json\n")
    (scenario / "nodes.json").write_text("[]\n")
    args = regression_check.build_parser().parse_args([
        "capture", "--sim-binary", sys.executable, "--run-config", str(scenario / "run.ini"),
        "--band", "mmwave", "--seed", "1", "--family", "baseline", "--case", "fake",
        "--out", str(tmp_path / "capture"),
    ])
    assert regression_check.cmd_capture(args) == 0
    assert run_batch.main(["--scenarios-dir", str(scenario.parent), "--seeds", "1",
                           "--sim-binary", sys.executable, "--out", str(tmp_path / "batch")]) == 0
    sweep = tmp_path / "sweep.ini"
    sweep.write_text("[sweep.meta]\n")
    run_sweep(SweepConfig(str(scenario), [1], "none", "", "fake"),
              str(sweep), sim_binary=sys.executable)
    assert len(calls) == 3


## @fn test_missing_reference_explains_cloud_bundle
# @brief A missing reference snapshot raises an error that points to the cloud baseline bundle.
#
# @param tmp_path  pytest temporary directory (used as the mesh-sim root).
# @return None; asserts on failure.
def test_missing_reference_explains_cloud_bundle(tmp_path):
    manifest = {"cases": [{"snapshot": "tests/fixtures/regression/p0/missing.json"}]}
    with pytest.raises(ValueError, match="Ask the project team.*cloud baseline bundle"):
        regression_check.verify_manifest_snapshots(manifest, tmp_path)
