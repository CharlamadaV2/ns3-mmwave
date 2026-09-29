# smoke_check.py
# Data-free sanity check of the simulator binary. Runs the tiny synthetic
# scenario inputs/baselines/p0-jammer-smoke twice with seed 1, validates the
# output tables, and requires both runs to match. Used by CI; not a comparison
# against historical references (see regression_check.py for that).
# The docstring below doubles as the --help description.
"""Check a tiny real simulation and same-seed repeatability without cloud data."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from scripts.sim_support import find_mesh_root, simulator_env, tail_lines
from .regression_snapshot import build_snapshot, compare_snapshots, load_table


## @fn check_run
# @brief Validate the output contracts of one smoke run.
#
# @param run_dir  Directory holding `seed-1/` output of the simulator.
# @param nodes    Parsed `nodes.json` of the scenario (list of node dicts, each
#                 with a `position` of x/y/z in metres).
# @return None.
# @throws ValueError if `summary.json` has the wrong scenario/seed, a non-finite
#         or out-of-range network metric, zero throughput, positions or links
#         that are not exactly 5 frames (t = 0.0..0.4 s, 0.1 s apart) for every
#         node / node pair, moved stationary nodes, or non-finite SINR/capacity.
# @throws KeyError if an expected key or column is missing.
#
# Reads `summary.json`, `positions.csv` and `links.csv` from `seed-1/`. Expects
# scenario name `p0-jammer-smoke`. Nothing is written.
def check_run(run_dir: Path, nodes: list[dict]) -> None:
    """Check the shipped stationary three-node jammer scenario's output contracts."""
    seed_dir = run_dir / "seed-1"
    summary = json.loads((seed_dir / "summary.json").read_text())
    if summary["seed"] != 1 or summary["scenario"] != "p0-jammer-smoke":
        raise ValueError("Summary has the wrong scenario or seed")
    for key in ("mean_sinr_db", "sum_throughput_mbps", "connectivity"):
        value = summary["network"][key]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Non-finite network metric: {key}")
    if not 0 <= summary["network"]["connectivity"] <= 1:
        raise ValueError("Connectivity lies outside [0, 1]")
    if summary["network"]["sum_throughput_mbps"] <= 0:
        raise ValueError("The smoke scenario delivered no traffic")

    positions = load_table(seed_dir / "positions.csv")["rows"]
    links = load_table(seed_dir / "links.csv")["rows"]
    times = [round(index * 0.1, 6) for index in range(5)]
    expected_positions = {(time, index) for time in times for index in range(len(nodes))}
    actual_positions = {(row["time_s"], row["node_id"]) for row in positions}
    if actual_positions != expected_positions or len(positions) != len(expected_positions):
        raise ValueError("Positions do not contain exactly five frames for every node")
    for row in positions:
        expected = nodes[int(row["node_id"])]["position"]
        if any(not math.isclose(row[axis], expected[axis], abs_tol=1e-6) for axis in "xyz"):
            raise ValueError("Stationary smoke nodes changed position")
    expected_links = {(time, a, b) for time in times
                      for a in range(len(nodes)) for b in range(a + 1, len(nodes))}
    actual_links = {(row["time_s"], row["node_a"], row["node_b"]) for row in links}
    if actual_links != expected_links or len(links) != len(expected_links):
        raise ValueError("Links do not contain exactly one row per pair per frame")
    for row in links:
        if not math.isfinite(row["sinr_db"]) or not math.isfinite(row["capacity_mbps"]):
            raise ValueError("Non-finite link SINR or capacity")


## @fn main
# @brief CLI entry point: run the smoke scenario twice and compare the runs.
#
# @param argv  Argument list; defaults to `sys.argv[1:]` when None.
# @return 0 and prints `PASS` on success; 1 and prints `FAIL: <reason>` to stderr
#         on any I/O, validation, or timeout error.
#
# Flags: `--sim-binary` (required) and `--out` (required, must not exist yet).
# Creates `<out>/run-1` and `<out>/run-2`, each with a `console.log` and the
# simulator output. Each run uses `--band=sub-6 --seed=1` and a 60 s timeout,
# and is validated by `check_run`. The two normalized snapshots must then match
# (default tolerance of `compare_snapshots`).
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-binary", required=True)
    parser.add_argument("--out", required=True, help="new output directory for two smoke runs")
    args = parser.parse_args(argv)
    root = find_mesh_root(__file__)
    output = Path(args.out).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
        scenario = root / "inputs" / "baselines" / "p0-jammer-smoke"
        nodes = json.loads((scenario / "nodes.json").read_text())
        snapshots = []
        for index in (1, 2):
            run_dir = output / f"run-{index}"
            run_dir.mkdir()
            console = run_dir / "console.log"
            command = [str(Path(args.sim_binary).resolve()),
                       f"--run-config={scenario / 'run.ini'}", "--band=sub-6",
                       "--seed=1", f"--output-dir={run_dir}"]
            with console.open("w") as handle:
                result = subprocess.run(command, stdin=subprocess.DEVNULL,
                                        stdout=handle, stderr=subprocess.STDOUT,
                                        env=simulator_env(root), timeout=60)
            if result.returncode:
                raise ValueError(f"Simulator exited {result.returncode}:\n{tail_lines(console)}")
            check_run(run_dir, nodes)
            snapshots.append(build_snapshot(run_dir, [], root, "baseline", "smoke", 1, "sub-6"))
        report = compare_snapshots(*snapshots)
        if not report["match"]:
            raise ValueError(f"Same-seed results differ: {report['differences']}")
        print("PASS: synthetic output contracts and same-seed repeatability (two runs)")
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
