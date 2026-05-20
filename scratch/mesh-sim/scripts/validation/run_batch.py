## @file run_batch.py
# @brief Run mesh-sim across a directory of scenarios, multi-seed.
#
# Discovers all scenario subdirectories that contain a ``run.ini``, invokes
# the ns-3 sim binary for each with a configurable set of random seeds, and
# writes outputs to a timestamped batch directory.
#
# Optionally patches each scenario's ``nodes.json`` with real-GPS-derived
# waypoints before running (see @ref build_waypoints).
#
# **Output layout**
# @code
# outputs/<YYYY-MM>/<DD>/<HH-MM-SS>-validation/
#   <scenario>/
#     run.log
#     seed-<N>/         (written by the sim binary)
#   batch_manifest.json
# @endcode

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .build_waypoints import patch_scenario_waypoints

REPO_ROOT              = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS_DIR  = REPO_ROOT / "inputs" / "custom" / "sherpa" / "spring_lake"
DEFAULT_SEEDS          = "1,2,3,4,5"  ##< Default comma-separated seed list.


## @brief Extract the ``duration_s`` value from a ``run.ini`` ``[scenario]`` section.
#
# Parses the file line-by-line to avoid loading the entire configparser machinery
# for a single value. Returns ``None`` if the key is absent or unparseable.
#
# @param ini_path Path to the ``run.ini`` file.
# @return Duration in seconds as a float, or ``None``.
def _read_scenario_duration(ini_path: Path) -> float | None:
    if not ini_path.is_file():
        return None
    in_section = False
    for line in ini_path.read_text().splitlines():
        s = line.strip()
        if s.startswith("["):
            in_section = (s == "[scenario]")
            continue
        if in_section and s.startswith("duration_s"):
            _, _, rhs = s.partition("=")
            try:
                return float(rhs.strip().split()[0])
            except (ValueError, IndexError):
                return None
    return None


## @brief Locate the compiled ns-3 sim binary via glob.
#
# Searches for the pattern
# ``<repo_root>/../../build/scratch/mesh-sim/ns3*-sim-*`` and returns the
# lexicographically first match (which in practice is the only build).
#
# @return Absolute path string of the binary, or ``None`` if not found.
def _find_sim_binary() -> str | None:
    ns3_root = REPO_ROOT.parent.parent
    pattern  = str(ns3_root / "build" / "scratch" / "mesh-sim" / "ns3*-sim-*")
    matches  = sorted(glob.glob(pattern))
    return matches[0] if matches else None


## @brief Discover all scenario directories that contain a ``run.ini`` file.
#
# @param scenarios_dir Parent directory to search.
# @return Sorted list of qualifying subdirectory paths.
def _discover_scenarios(scenarios_dir: Path) -> list[Path]:
    if not scenarios_dir.is_dir():
        return []
    return sorted(p for p in scenarios_dir.iterdir()
                  if p.is_dir() and (p / "run.ini").is_file())


## @brief Invoke the sim binary for one scenario and capture its output to a log file.
#
# The command passes the scenario's ``run.ini``, the seed list, and the output
# directory as command-line arguments. Both stdout and stderr are written to
# ``<out_dir>/run.log``.
#
# @param sim_binary Path string of the ns-3 sim binary.
# @param scenario   Scenario directory (contains ``run.ini``).
# @param seeds      Comma-separated seed string (e.g. ``"1,2,3,4,5"``).
# @param out_dir    Destination for sim output files and the log.
# @param env        Environment variables dict for the subprocess.
# @return Tuple ``(returncode, log_path_str)``.
def _run_one(sim_binary: str, scenario: Path, seeds: str,
             out_dir: Path, env: dict[str, str]) -> tuple[int, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sim_binary,
        f"--run-config={scenario / 'run.ini'}",
        f"--seeds={seeds}",
        f"--output-dir={out_dir}",
    ]
    log_path = out_dir / "run.log"
    with open(log_path, "w") as log_f:
        log_f.write("Command: " + " ".join(cmd) + "\n\n")
        log_f.flush()
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    return result.returncode, str(log_path)


## @brief CLI entry point for the batch runner.
#
# Full workflow:
# -# Discover scenarios under ``--scenarios-dir``.
# -# Locate the sim binary (auto-detect or ``--sim-binary``).
# -# For each scenario (optionally filtered by ``--only``):
#    -# Optionally patch waypoints via @ref patch_scenario_waypoints.
#    -# Invoke the sim via @ref _run_one.
#    -# Append the result to ``batch_manifest.json`` (written after every run
#       so a crash doesn't lose partial results).
# -# Print a final OK/failed count.
#
# @param argv Argument list; defaults to ``sys.argv[1:]`` when ``None``.
# @return 0 if all scenarios succeeded, 1 if any failed.
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run mesh-sim across validation scenarios.")
    p.add_argument("--scenarios-dir", default=str(DEFAULT_SCENARIOS_DIR),
                   help="Parent dir containing one scenario subdir per run.ini")
    p.add_argument("--seeds", default=DEFAULT_SEEDS,
                   help="Comma-separated seed list passed to the sim")
    p.add_argument("--out", default=None,
                   help="Output root (default: outputs/YYYY-MM/DD/HH-MM-SS-validation/)")
    p.add_argument("--sim-binary", default=None,
                   help="Override auto-detected sim binary path")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would run, don't invoke the sim")
    p.add_argument("--only", default=None,
                   help="Run only this scenario name (matches scenario dir basename)")
    p.add_argument("--auto-waypoints", action="store_true",
                   help="Before each sim run, patch nodes.json with waypoints "
                        "derived from the field GPS trace (skips scenarios where "
                        "the field node is static). Mutates inputs/.../nodes.json.")
    p.add_argument("--waypoint-node", default="rab2",
                   help="Node id to author waypoints for when --auto-waypoints "
                        "is set (default: rab2)")
    p.add_argument("--waypoint-time-mode",
                   choices=("raw", "clip", "scale"), default="scale",
                   help="Time-axis handling for --auto-waypoints (default: scale)")
    args = p.parse_args(argv)

    scenarios_dir = Path(args.scenarios_dir).resolve()
    scenarios     = _discover_scenarios(scenarios_dir)
    if args.only:
        scenarios = [s for s in scenarios if s.name == args.only]
    if not scenarios:
        print(f"No scenarios found under {scenarios_dir}", file=sys.stderr)
        return 1

    sim_binary = args.sim_binary or _find_sim_binary()
    if not sim_binary and not args.dry_run:
        print("Could not auto-detect sim binary. Pass --sim-binary.", file=sys.stderr)
        return 1

    if args.out:
        batch_root = Path(args.out).resolve()
    else:
        now        = datetime.now()
        batch_root = (REPO_ROOT / "outputs" / now.strftime("%Y-%m")
                      / now.strftime("%d")
                      / (now.strftime("%H-%M-%S") + "-validation"))

    seed_count = len([s for s in args.seeds.split(",") if s.strip()])
    print(f"Batch root: {batch_root}")
    print(f"Sim binary: {sim_binary}")
    print(f"Seeds:      {args.seeds} ({seed_count} per scenario)")
    print(f"Scenarios:  {len(scenarios)}")
    for s in scenarios:
        print(f"  - {s.name}")

    if args.dry_run:
        print("\n(dry-run) exiting before invocation.")
        return 0

    batch_root.mkdir(parents=True, exist_ok=True)

    # Extend library search paths so the sim binary can find ns-3 shared libs.
    env     = os.environ.copy()
    ns3_root = REPO_ROOT.parent.parent
    lib_dir  = str(ns3_root / "build" / "lib")
    env["DYLD_LIBRARY_PATH"] = lib_dir + ":" + env.get("DYLD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"]   = lib_dir + ":" + env.get("LD_LIBRARY_PATH",   "")

    manifest = {
        "timestamp":     datetime.now().isoformat(),
        "scenarios_dir": str(scenarios_dir),
        "seeds":         args.seeds,
        "sim_binary":    sim_binary,
        "runs":          [],
    }

    n_ok = n_fail = 0
    for i, scen in enumerate(scenarios, 1):
        out_dir = batch_root / scen.name
        print(f"\n[{i}/{len(scenarios)}] {scen.name}")

        if args.auto_waypoints:
            # Read scenario duration so scale-mode targets the actual sim window.
            duration = _read_scenario_duration(scen / "run.ini")
            status   = patch_scenario_waypoints(
                scen,
                node=args.waypoint_node,
                time_mode=args.waypoint_time_mode,
                duration=duration,
            )
            print(f"  waypoints: {status}")

        rc, log  = _run_one(sim_binary, scen, args.seeds, out_dir, env)
        ok_flag  = rc == 0
        if ok_flag:
            n_ok += 1
            print(f"  ok  -> {out_dir}")
        else:
            n_fail += 1
            print(f"  FAILED (exit {rc}); see {log}")

        manifest["runs"].append({
            "scenario":     scen.name,
            "scenario_dir": str(scen),
            "output_dir":   str(out_dir),
            "status":       "ok" if ok_flag else "failed",
            "exit_code":    rc,
        })
        # Write after every scenario so a mid-batch crash doesn't lose results.
        with open(batch_root / "batch_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"\nDone: {n_ok} ok, {n_fail} failed")
    print(f"Manifest: {batch_root / 'batch_manifest.json'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())