'''split_trace_by_day.py'''
## @file split_trace_by_day.py
# @brief Split a combined multi-day GPS trace CSV into one file per calendar day.
#
# The calfex field collect spans multiple days (e.g. 2026-05-13 through
# 2026-05-17). Feeding the full combined trace into @ref build_waypoints
# produces waypoint timelines spanning the entire collect (tens of hours),
# which is far longer than any reasonable simulated duration. Splitting by
# day lets each day be simulated independently with a sensible duration.
#
# Input columns expected: ``label, node, t_utc, sec_since_origin, lat_deg,
# lon_deg, east_m, north_m`` (the format produced by
# ``arpo_data.cli plot --nodes``).
#
# Output: one CSV per day, with ``sec_since_origin`` re-zeroed so each day's
# trace starts at ``t=0`` (required for @ref build_waypoints time-mode logic
# to work correctly per day).

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


## @brief Split a combined GPS trace into one CSV per calendar day.
#
# Parses ``t_utc`` to extract the calendar date for each row, groups rows by
# date, and re-zeroes ``sec_since_origin`` within each day group so every
# day's trace starts at ``t=0`` (matching the convention @ref build_waypoints
# expects for a single-day trace).
#
# @param trace_path  Path to the combined multi-day trace CSV.
# @param out_dir     Directory to write one CSV per day into.
# @param prefix      Filename prefix for each output file (default: derived
#                    from the input filename stem).
# @return            Dict mapping date string (``YYYY-MM-DD``) to the
#                    written file path.
def split_trace_by_day(trace_path: Path, out_dir: Path,
                       prefix: str | None = None) -> dict[str, Path]:
    df = pd.read_csv(trace_path)
    if "t_utc" not in df.columns:
        raise ValueError(f"'t_utc' column not found in {trace_path}")

    df["t_utc"] = pd.to_datetime(df["t_utc"])
    df["__date__"] = df["t_utc"].dt.date.astype(str)

    if prefix is None:
        prefix = trace_path.stem

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for date_str, day_df in df.groupby("__date__"):
        day_df = day_df.sort_values("sec_since_origin").copy()
        # Re-zero sec_since_origin so each day's trace starts at t=0.
        day_df["sec_since_origin"] = (
            day_df["sec_since_origin"] - day_df["sec_since_origin"].min()
        )
        day_df = day_df.drop(columns=["__date__"])

        out_path = out_dir / f"{prefix}_{date_str}.csv"
        day_df.to_csv(out_path, index=False)
        written[date_str] = out_path

    return written


## @brief Print a per-day, per-node row-count summary.
#
# Useful for spotting days where a node has little or no data before
# running @ref build_waypoints on that day's file.
#
# @param trace_path Path to the combined multi-day trace CSV.
def print_day_node_summary(trace_path: Path) -> None:
    df = pd.read_csv(trace_path, usecols=["node", "t_utc"])
    df["t_utc"] = pd.to_datetime(df["t_utc"])
    df["date"]  = df["t_utc"].dt.date.astype(str)
    table = df.groupby(["date", "node"]).size().unstack(fill_value=0)
    print(table.to_string())


## @brief CLI entry point for the trace-splitting tool.
#
# @param argv Argument list; defaults to ``sys.argv[1:]`` when ``None``.
# @return     0 on success, 1 on error.
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Split a combined multi-day GPS trace CSV into one file per day.")
    p.add_argument("-i", "--input", type=Path, required=True,
                   help="path to the combined trace CSV (e.g. gps_all_nodes_trace.csv)")
    p.add_argument("-o", "--output", type=Path, required=True,
                   help="output directory for the per-day CSVs")
    p.add_argument("--prefix", default=None,
                   help="filename prefix for outputs (default: derived from input filename)")
    p.add_argument("--summary", action="store_true",
                   help="print a per-day, per-node row-count table and exit "
                        "without writing any files")
    args = p.parse_args(argv)

    if not args.input.is_file():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    if args.summary:
        print_day_node_summary(args.input)
        return 0

    written = split_trace_by_day(args.input, args.output, args.prefix)
    print(f"Split {args.input.name} into {len(written)} day files:")
    for date_str, path in sorted(written.items()):
        n_rows = sum(1 for _ in open(path)) - 1  # exclude header
        print(f"  {date_str}: {path}  ({n_rows} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())