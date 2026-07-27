#!/usr/bin/env python3
# split_trace_by_scenario.py
"""Extract one scenario (a wall-clock time window) from a per-day trace directory.

A per-day trace directory holds each node's GPS trace and its per-node metric
CSVs, and often contains several scenarios run back-to-back at different clock
times. This tool slices every CSV in the tree to one scenario's wall-clock
``[start, end)`` window and writes the survivors under an output directory,
preserving the input's sub-directory layout (so each node's files stay under
that node).

Selection is keyed on the time-of-day of each file's absolute ``t_utc``
timestamp (UTC) — the only column that means the same thing across files.
Files that instead carry ``sec_since_origin`` (0-based per day) are windowed
using an offset derived from a GPS file that has both columns. Files keyed only
on the radio ``sec`` clock (e.g. the ``IH_*`` metric traces, whose ``sec`` does
NOT start at midnight) cannot be mapped to wall-clock here; they are reported
and skipped rather than sliced incorrectly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


## @brief Convert an ``Hour.Minute`` float into seconds since midnight.
# @param value  Time as an ``HH.MM`` float, 24-hour.
# @return Seconds since midnight in ``[0, 86340]``.
# @throws ValueError on unparseable input or out-of-range hour/minute.
def _parse_clock(value: float) -> int:
    text = f"{float(value):.2f}"
    parts = text.split(".")
    if len(parts) != 2:
        raise ValueError(f"could not parse time {value!r}")
    hours, minutes = int(parts[0]), int(parts[1])
    if not 0 <= hours <= 23:
        raise ValueError(f"hour out of range [0, 23] in {value!r}")
    if not 0 <= minutes <= 59:
        raise ValueError(f"minute out of range [0, 59] in {value!r} "
                         f"(write minutes as two digits, e.g. 9.05 for 09:05)")
    return hours * 3600 + minutes * 60


## @brief Format seconds-since-midnight as a compact ``HHMM`` filename tag.
def _fmt_hhmm(total_s: int) -> str:
    return f"{total_s // 3600:02d}{(total_s % 3600) // 60:02d}"


## @brief Seconds-since-midnight (UTC) for each row of a ``t_utc`` column.
def _seconds_since_midnight(t_utc: pd.Series) -> pd.Series:
    t = pd.to_datetime(t_utc, utc=True)
    return (t.dt.hour * 3600 + t.dt.minute * 60
            + t.dt.second + t.dt.microsecond / 1e6)


## @brief Offset D with seconds_since_midnight ~= sec_since_origin + D.
def _day_origin_offset(df: pd.DataFrame) -> float | None:
    if not {"t_utc", "sec_since_origin"}.issubset(df.columns):
        return None
    tod = _seconds_since_midnight(df["t_utc"])
    return float((tod - df["sec_since_origin"]).median())


## @brief Re-zero the primary time column in place so the slice starts at t=0.
def _rezero_time(df: pd.DataFrame) -> None:
    for col in ("sec_since_origin", "sec"):
        if col in df.columns:
            df.sort_values(col, inplace=True)
            df[col] = df[col] - df[col].min()
            return


## @brief Slice one frame to ``[start_s, end_s)``, or return ``None``.
#
# Selection key, in priority order:
#  - ``t_utc``            -> time-of-day directly (exact wall-clock).
#  - ``sec_since_origin`` -> shifted by @p day_offset.
#  - ``sec`` (radio clock) -> elapsed from its first row (``sec - sec.min()``)
#    shifted by @p day_offset. This ASSUMES the trace's first sample is at the
#    same instant as the GPS trace's start (the metric ``sec`` carries no
#    wall-clock of its own); if the radio and GPS logs didn't co-start, the
#    metric window is shifted by that difference.
#
# Returns ``None`` only when no usable time column is present, or when a
# ``sec``-based file is seen but no @p day_offset could be derived.
def _slice_frame(df: pd.DataFrame, start_s: int, end_s: int,
                 day_offset: float | None) -> pd.DataFrame | None:
    if "t_utc" in df.columns:
        tod = _seconds_since_midnight(df["t_utc"])
        mask = (tod >= start_s) & (tod < end_s)
    elif "sec_since_origin" in df.columns and day_offset is not None:
        wall = df["sec_since_origin"] + day_offset
        mask = (wall >= start_s) & (wall < end_s)
    elif "sec" in df.columns and day_offset is not None:
        # No wall-clock in the file: anchor its first row to the GPS start.
        wall = (df["sec"] - df["sec"].min()) + day_offset
        mask = (wall >= start_s) & (wall < end_s)
    else:
        return None
    sub = df[mask].copy()
    if not sub.empty:
        _rezero_time(sub)
    return sub


## @brief Slice every CSV under a day directory to one scenario window.
#
# Walks @p day_path for ``*.csv`` recursively, slices each to
# ``[start_s, end_s)``, and writes survivors under ``out_dir/<HHMM-HHMM>/``,
# preserving each file's path relative to @p day_path (so node folders are
# kept). A GPS file carrying both ``t_utc`` and ``sec_since_origin`` is used to
# derive the wall-clock offset for any ``sec_since_origin``-only files.
#
# @param start_s   Window start, seconds since midnight (inclusive).
# @param end_s     Window end,   seconds since midnight (exclusive).
# @param day_path  Per-day trace directory (node sub-folders inside).
# @param out_dir   Output root; a ``<HHMM-HHMM>`` sub-dir is made under it.
# @param prefix    Optional scenario-dir name prefix (before the time tag).
# @return Total rows written across all sliced files (0 if nothing matched).
# @throws ValueError if @p day_path is not a directory or holds no CSVs.
def split_trace_by_scenario(start_s: int, end_s: int, day_path: Path,
                            out_dir: Path, prefix: str | None = None) -> int:
    day_path = Path(day_path)
    if not day_path.is_dir():
        raise ValueError(f"input is not a directory: {day_path}")

    tag = f"{_fmt_hhmm(start_s)}-{_fmt_hhmm(end_s)}"
    scen_dir = out_dir / (f"{prefix}_{tag}" if prefix else tag)
    scen_res = scen_dir.resolve()

    # Skip anything already sitting under the output dir (safe re-runs when
    # out_dir is nested in day_path).
    csvs = sorted(c for c in day_path.rglob("*.csv")
                  if scen_res not in c.resolve().parents)
    if not csvs:
        raise ValueError(f"no CSV files under {day_path}")

    # Derive the wall-clock offset from the first GPS file that has both keys.
    day_offset = None
    for c in csvs:
        off = _day_origin_offset(pd.read_csv(c))
        if off is not None:
            day_offset = off
            break
    if day_offset is None:
        print("  note: no GPS file with both t_utc and sec_since_origin found; "
              "files without t_utc cannot be placed on wall-clock and will be "
              "skipped", file=sys.stderr)

    total_rows = 0
    skipped: list[Path] = []
    anchored: list[Path] = []          # sec-only files placed via the GPS start
    for csv in csvs:
        rel = csv.relative_to(day_path)
        df = pd.read_csv(csv)
        sub = _slice_frame(df, start_s, end_s, day_offset)
        if sub is None:
            skipped.append(rel)
            continue
        if "t_utc" not in df.columns and "sec_since_origin" not in df.columns:
            anchored.append(rel)        # matched only on the radio 'sec' clock
            
        out_path = scen_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_path, index=False)
        total_rows += len(sub)
        extra = f", {sub['node'].nunique()} nodes" if "node" in sub.columns else ""
        print(f"  {rel}: {len(sub)} rows{extra}")

    if anchored:
        print(f"  note: {len(anchored)} metric file(s) keyed only on 'sec' were "
              f"windowed by anchoring their first row to the GPS start "
              f"(assumes radio and GPS logs co-started).", file=sys.stderr)
    if skipped:
        print(f"  skipped {len(skipped)} file(s) with no usable time column:",
              file=sys.stderr)
        for s in skipped:
            print(f"    {s}", file=sys.stderr)
    return total_rows


## @brief CLI entry point.
#
# @param argv Argument list; defaults to ``sys.argv[1:]`` when ``None``.
# @return 0 on success, 1 on error (bad args, missing dir, or empty window).
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Extract one scenario (clock-time window) from a per-day trace directory.")
    p.add_argument("-i", "--input", type=Path, required=True,
                   help="Path to the day trace DIRECTORY (node folders inside)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output root for the per-scenario dirs "
                        "(default: <input>/scenarios)")
    p.add_argument("--start", required=True, type=float,
                   help="Scenario start time as Hour.Minute, e.g. 9.30 (9.05 = 09:05)")
    p.add_argument("--end", required=True, type=float,
                   help="Scenario end time as Hour.Minute, e.g. 10.15")
    p.add_argument("--prefix", default=None,
                   help="Optional name prefix for the scenario sub-directory")
    args = p.parse_args(argv)

    #Input directory check
    if not args.input.is_dir():
        print(f"Error: input directory not found: {args.input}", file=sys.stderr)
        return 1

    #Valid time check
    try:
        start_s = _parse_clock(args.start)
        end_s = _parse_clock(args.end)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if start_s >= end_s:
        print(f"Error: start ({args.start}) must be before end ({args.end})",
              file=sys.stderr)
        return 1

    out_dir = args.output if args.output is not None else args.input / "scenarios"

    #Main Logic
    try:
        n_rows = split_trace_by_scenario(start_s, end_s, args.input, out_dir, args.prefix)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if n_rows == 0:
        print(f"Error: no samples in window {args.start}-{args.end} under {args.input}",
              file=sys.stderr)
        return 1
    print(f"Scenario generated ({n_rows} rows across sliced files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())