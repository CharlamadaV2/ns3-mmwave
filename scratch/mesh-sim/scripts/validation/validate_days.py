'''validate_days.py'''
## @file validate_days.py
# @brief Orchestrates per-day sim validation, with single-day or multi-day modes.
#
# Wraps the three node-mode validation steps —
# @ref scripts.validation.sim_to_traces, @ref scripts.validation.compare, and
# @ref scripts.validation.scenario_fidelity — and runs them once per requested
# day's sim output directory (``outputs/calfex/<day>/``). After all requested
# days are processed, optionally calls @ref scripts.validation.compare_runs
# to build cross-day heatmaps from the per-day ``validation_summary.csv``
# files, giving a day-vs-day comparison equivalent to ``arpo_data``'s
# ``multi-day`` subcommand but for sim-vs-field validation results.
#
# **Expected directory layout**
# @code
# outputs/calfex/
#   2026-05-13/
#     inputs/         (nodes.json, run.ini)
#     seed-1/         (sim run output: positions.csv, links.csv, ...)
#     sim_traces/      <- written by sim_to_traces
#     validation/      <- written by compare
#     summary/         <- written by compare
# @endcode
#
# **Field traces are expected at**
# ``<field_root>/<day>/<node>/csvs/<node>/IH_<metric>__<node>_to_neighbors_trace.csv``
# (the per-day output of ``arpo_data.cli plot --nodes``).

from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path

from . import compare, compare_runs, scenario_fidelity, sim_to_traces

REPO_ROOT        = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "calfex"
DEFAULT_FIELD_ROOT = (REPO_ROOT / "data" / "arpo_extracted"
                      / "_plots" / "per_day")


## @brief Discover available per-day sim output directories.
#
# A directory qualifies if its name looks like ``YYYY-MM-DD`` and it
# contains an ``inputs/`` subdirectory.
#
# @param out_root  Root directory containing per-day sim output subdirs.
# @return          Sorted list of day strings (``"YYYY-MM-DD"``).
def _discover_days(out_root: Path) -> list[str]:
    days = []
    if not out_root.is_dir():
        return days
    for p in out_root.iterdir():
        if not p.is_dir() or not (p / "inputs").is_dir():
            continue
        parts = p.name.split("-")
        if len(parts) == 3 and all(s.isdigit() for s in parts):
            days.append(p.name)
    return sorted(days)


## @brief Locate the field GPS trace CSV for one day, checking known layouts.
#
# Checks, in order:
# -# ``<field_root>/<day>/gps_all_nodes_trace.csv``  — written directly by
#    ``arpo_data.cli plot --nodes`` per-day mode.
# -# ``<field_root>/by_day/gps_all_nodes_trace_<day>.csv``  — written by
#    ``arpo_data.cli split-day`` from a combined multi-day trace.
#
# @param field_root  Root of per-day field trace dirs.
# @param day         Day string (``"YYYY-MM-DD"``).
# @return            Path to the first candidate that exists, or ``None``.
def _resolve_field_gps(field_root: Path, day: str) -> Path | None:
    candidates = [
        field_root / day / "gps_all_nodes_trace.csv",
        field_root / "by_day" / f"gps_all_nodes_trace_{day}.csv",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


## @brief Run sim_to_traces, compare, and scenario_fidelity for one day.
#
# @param day_dir     Sim output directory for this day (``outputs/calfex/<day>/``).
# @param field_dir   Field traces directory for this day
#                    (``<field_root>/<day>/``).
# @param field_gps   Path to this day's GPS trace CSV, for scenario_fidelity.
# @param metrics     Comma-separated metric shorts to compare.
# @param tol_m       Pairwise distance tolerance in metres for fidelity checks.
# @param skip_fidelity Skip the scenario_fidelity step if ``True``.
# @return           0 on success, 1 if any step reported a hard error.
def _run_day(day_dir: Path, field_dir: Path, field_gps: Path | None,
            metrics: str, tol_m: float, skip_fidelity: bool,
            window_s: float) -> int:
    print(f"\n{'=' * 60}")
    print(f"  day: {day_dir.name}  (window: first {window_s:g} s)")
    print(f"{'=' * 60}")

    print("\n-- sim_to_traces --")
    sim_to_traces.convert_batch(day_dir, mode="node")

    print("\n-- compare --")
    rc = compare.main([
        "--batch_root", str(day_dir),
        "--mode", "node",
        "--field-root", str(field_dir),
        "--metrics", metrics,
        "--window", str(window_s,)
    ])
    if rc != 0:
        print(f"  compare reported an error for {day_dir.name}", file=sys.stderr)

    if not skip_fidelity:
        print("\n-- scenario_fidelity (mobility + geometry check) --")
        if field_gps is None or not field_gps.is_file():
            print(f"  SKIPPED: no field GPS trace available for {day_dir.name}",
                  file=sys.stderr)
        else:
            fid_rc = scenario_fidelity.main([
                str(day_dir),
                "--mode", "node",
                "--field-gps", str(field_gps),
                "--tol-m", str(tol_m),
                "--window", str(window_s),
            ])
            if fid_rc != 0:
                print(f"  scenario_fidelity flagged mismatches for {day_dir.name}",
                      file=sys.stderr)

    return rc


## @brief CLI entry point for the per-day validation orchestrator.
#
# @param argv Argument list; defaults to ``sys.argv[1:]`` when ``None``.
# @return     0 on success, 1 on error.
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run sim-vs-field validation for one day or every day, "
                    "then optionally compare across days.")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                   help=f"root of per-day sim output dirs (default: {DEFAULT_OUT_ROOT})")
    p.add_argument("--field-root", type=Path, default=DEFAULT_FIELD_ROOT,
                   help=f"root of per-day field trace dirs (default: {DEFAULT_FIELD_ROOT})")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--day", default=None,
                   help="validate a single day (YYYY-MM-DD)")
    g.add_argument("--all-days", action="store_true",
                   help="validate every day found under --out-root")
    p.add_argument("--time", "-t", type=float,
                   help="comparison window in seconds: compare only the first N "
                        "seconds of sim AND field. Default will compare entire day between sim and field")
    #Add-on commands
    p.add_argument("--metrics", default="snr,rcpi,mcs",
                   help="comma-separated metric shorts to compare (default: snr,rcpi,mcs)")
    p.add_argument("--tol-m", type=float, default=5.0,
                   help="pairwise distance tolerance in metres for fidelity checks")
    p.add_argument("--field-gps", type=Path, default=None,
                   help="explicit path to a single GPS trace CSV to use for "
                        "ALL requested days (overrides auto-detection). "
                        "Only sensible with --day, not --all-days.")
    p.add_argument("--skip-fidelity", action="store_true",
                   help="skip the scenario_fidelity geometry/mobility check")
    p.add_argument("--skip-cross-day", action="store_true",
                   help="skip the final compare_runs cross-day heatmap step "
                        "(only relevant with --all-days)")
    args = p.parse_args(argv)
    cfg = configparser.ConfigParser()

    if not args.out_root.is_dir():
        print(f"Error: out-root not found: {args.out_root}", file=sys.stderr)
        return 1

    if args.day:
        days = [args.day]
    else:
        days = _discover_days(args.out_root)
        if not days:
            print(f"Error: no per-day sim output found under {args.out_root}",
                  file=sys.stderr)
            return 1

    print(f"Validating {len(days)} day(s): {', '.join(days)}")

    n_errors = 0
    for day in days:
        day_dir   = args.out_root / day
        if not day_dir.is_dir(): #< Check for day in sim directory
            print(f"Error: no sim output for {day} at {day_dir}", file=sys.stderr)
            n_errors += 1
            continue

        field_dir = args.field_root / day
        if not field_dir.is_dir(): #< Check for day in field directory
            print(f"Error: no field traces for {day} at {field_dir}", file=sys.stderr)
            n_errors += 1
            continue

        if args.field_gps is not None: #< Specific gps trace
            field_gps = args.field_gps
            if not field_gps.is_file(): #< Check for field gps
                print(f"Error: --field-gps not found: {field_gps}", file=sys.stderr)
                n_errors += 1
                continue
        else: #< All gps traces (Default)
            field_gps = _resolve_field_gps(args.field_root, day)
            if field_gps is None and not args.skip_fidelity:
                print(f"  WARNING: no GPS trace found for {day} — checked "
                      f"{args.field_root / day / 'gps_all_nodes_trace.csv'} and "
                      f"{args.field_root / 'by_day' / f'gps_all_nodes_trace_{day}.csv'}",
                      file=sys.stderr)
                
        #If args.time is empty, read duration_s from run.ini for this day.
        if args.time is None:
            config_directory = day_dir / 'inputs' / 'run.ini'
            cfg.read(config_directory)
            compare_window = cfg.getfloat("scenario", "duration_s", fallback=60.0)
        else:
            compare_window = float(args.time)
            
        # Main logic        
        rc = _run_day(day_dir, field_dir, field_gps, 
                     args.metrics, args.tol_m, args.skip_fidelity, compare_window)
        if rc != 0:
            n_errors += 1

    #TODO: Update logic to use comparison window
    if len(days) > 1 and not args.skip_cross_day: #< Crossday comparison logic
        print(f"\n{'=' * 60}")
        print("  cross-day comparison (compare_runs)")
        print(f"{'=' * 60}\n")
        batch_dirs = [str(args.out_root / d) for d in days]
        cr_rc = compare_runs.main([
            *batch_dirs,
            "--metrics", args.metrics,
            "--out", str(args.out_root / "cross_day_summary"),
        ])
        if cr_rc != 0:
            print("  compare_runs reported an error", file=sys.stderr)
            n_errors += 1

    print(f"\n{'=' * 60}")
    print(f"  done — {len(days)} day(s) processed, {n_errors} error(s)")
    print(f"{'=' * 60}")
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())