"""cli script"""

## @file cli.py
# @brief Main script to be ran for manipulating ARPO data as desired
#
# Provides a command line interface that can do one of the following commands:
# - Extract unzips data folder into extraction target direction
# - Multi-day compares day vs day data for multi day scenarios
# - Plot generates figures for specific scenario or all scenarios per individual day
# - Split-day splits a combined multi-day GPS trace CSV into one file per day

import argparse
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .paths import DatasetPaths
from .extract import extract
from .loaders import (
    # load_bh2_scenario, #< Uncomment to enable if data contains bh2_scenarios
    load_gps_scenario,
    load_gps_all,
    load_rf_scenario,
    load_node_id,
    filter_by_day,
    days_present,
)
from .multi_day import multi_day
from .split_trace_by_day import split_trace_by_day, print_day_node_summary
from .paths import CSV_ROOT, PER_DAY_DIR
from .plots import (
    # __Uncomment to enable if data contains bh2_scenarios__
    # plot_bh2_mcs, 
    # plot_bh2_per,
    # plot_bh2_rcpi,
    # plot_bh2_snr,
    # plot_bh2_throughput,
    plot_gps_tracks,
    plot_silvus_snr,
    plot_silvus_rcpi,
    plot_silvus_mcs,
    plot_silvus_throughput,
    plot_silvus_per,
)


## @brief Save a figure (and an optional trace CSV) returned from a plot fn.
#
# Plot fns may return either ``Figure`` or ``(Figure, trace_df)``. The PNG
# lands at ``png_path``; if a trace is present it goes to ``csv_path`` when
# supplied, otherwise next to the PNG as ``<stem>_trace.csv``.
#
# @param result    Figure or ``(Figure, trace_df)`` tuple returned by a plot fn.
# @param png_path  Destination path for the PNG file.
# @param csv_path  Destination path for the trace CSV; auto-derived if ``None``.
def _save(result, png_path: Path, csv_path: Path | None = None) -> None:
    if result is None:
        return
    if isinstance(result, tuple):
        fig, trace = result
    else:
        fig, trace = result, None
    if fig is None:
        return
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {png_path}")
    if trace is not None and not trace.empty:
        target = csv_path if csv_path is not None \
            else png_path.with_name(f"{png_path.stem}_trace.csv")
        target.parent.mkdir(parents=True, exist_ok=True)
        trace.to_csv(target, index=False)
        print(f"    wrote {target}")


## @brief Save ``(src, peer, fig, trace)`` tuples emitted by per-radio plot fns.
#
# PNGs land under ``pngs/<src>/`` and traces under ``csvs/<src>/`` so the
# outputs for each source rab live together.
#
# @param results   List of ``(src, peer, fig, trace_df)`` tuples.
# @param scen_dir  Output root directory for this scenario.
# @param base_name Metric prefix used in the output filename (e.g. ``"bh2_snr"``).
def _save_pairs(results, scen_dir: Path, base_name: str) -> None:
    if not results:
        return
    for src, peer, fig, trace in results:
        png = scen_dir / "pngs" / src / f"{base_name}__{src}_to_{peer}.png"
        csv = scen_dir / "csvs" / src / f"{base_name}__{src}_to_{peer}_trace.csv"
        _save((fig, trace), png, csv)


## @brief Plot all bh2 metrics and GPS tracks for one scenario directory.
#
# @param scen_dir    Path to the scenario directory.
# @param output_path Root output directory; a subdirectory named after the
#                    scenario is created inside it.
def _plot_one(node_dir: Path, output_path: Path) -> None:
    out  = output_path / node_dir.name
    pngs = out / "pngs"
    csvs = out / "csvs"
    print(f"  [{node_dir.name}]")

    name = node_dir.name
    # ------- OLD FORMAT ----------
    # bh2  = load_bh2_scenario(scen_dir)
    # if bh2 is not None:
    #     _save_pairs(plot_bh2_snr(bh2, name),        out, "bh2_snr")
    #     _save_pairs(plot_bh2_rcpi(bh2, name),       out, "bh2_rcpi")
    #     _save_pairs(plot_bh2_mcs(bh2, name),        out, "bh2_mcs")
    #     _save_pairs(plot_bh2_throughput(bh2, name), out, "bh2_throughput")
    #     _save_pairs(plot_bh2_per(bh2, name),        out, "bh2_per")

    gps = load_gps_scenario(node_dir)
    if gps is not None:
        _save(plot_gps_tracks(gps, name),
              pngs / "gps_track.png", csvs / "gps_track_trace.csv")
        
    rf_data = load_rf_scenario(node_dir)
    if rf_data is not None:
        _save_pairs(plot_silvus_snr(rf_data, node_dir.name),        out, "IH_snr")
        _save_pairs(plot_silvus_rcpi(rf_data, node_dir.name),       out, "IH_rcpi")
        _save_pairs(plot_silvus_mcs(rf_data, node_dir.name),        out, "IH_mcs")
        _save_pairs(plot_silvus_throughput(rf_data, node_dir.name), out, "IH_throughput")
        _save_pairs(plot_silvus_per(rf_data, node_dir.name),        out, "IH_per")


## @brief Load all valid node directories under a CSV root and their node-id map.
#
# A node is valid if it is a directory (excluding ``sdwan``) that contains
# both ``gps/`` and ``silvus/`` subdirectories and has a readable node id.
#
# @param csv_dir Directory containing per-node subdirs.
# @return Tuple ``(nodes, node_ids, node_id_to_name)`` where ``nodes`` is the
#         list of valid node directory paths, ``node_ids`` is the parallel
#         list of node id strings, and ``node_id_to_name`` maps every
#         discovered node id to its directory name.
def _discover_valid_nodes(
    csv_dir: Path,
) -> tuple[list[Path], list[str], dict[str, str]]:
    nodes: list[Path] = []
    node_ids: list[str] = []
    node_id_to_name: dict[str, str] = {}

    for node_dir in sorted(csv_dir.iterdir()):
        if not node_dir.is_dir() or node_dir.name == "sdwan":
            continue
        if not (node_dir / "gps").is_dir() or not (node_dir / "silvus").is_dir():
            print(f"  WARNING: {node_dir.name} missing gps or silvus directory, skipping",
                  file=sys.stderr)
            continue
        result = load_node_id(node_dir)
        if result is None:
            print(f"  WARNING: could not read node_id for {node_dir.name}, skipping",
                  file=sys.stderr)
            continue
        node_id_to_name.update(result)
        nodes.append(node_dir)
        node_ids.append(list(result.keys())[0])

    return nodes, node_ids, node_id_to_name


## @brief Plot one calendar day's worth of node GPS and RF data.
#
# Filters every node's GPS and RF DataFrame down to @p day before plotting,
# and writes output under ``output_path/<day>/``. Mirrors the un-split
# per-node and combined-GPS plotting that @ref _plot_nodes used to do across
# the whole dataset, but scoped to a single day.
#
# @param day             Calendar date to plot.
# @param nodes           List of valid node directory paths.
# @param node_ids        Parallel list of node id strings.
# @param node_id_to_name Map of node id to directory name.
# @param csv_dir         Original CSV root (used for the combined-GPS title).
# @param output_path     Root output directory; a ``<day>/`` subdirectory is
#                        created inside it.
# @return ``True`` if at least one node produced output for this day.
def _plot_nodes_for_day(
    day,
    nodes: list[Path],
    node_ids: list[str],
    node_id_to_name: dict[str, str],
    csv_dir: Path,
    output_path: Path,
) -> bool:
    day_str  = day.isoformat()
    day_root = output_path / day_str
    day_root.mkdir(parents=True, exist_ok=True)

    any_output = False
    gps_frames: list = []

    for node_dir in nodes:
        gps_full = load_gps_scenario(node_dir)
        gps_day  = filter_by_day(gps_full, day)
        if gps_day is None:
            print(f"  WARNING: no GPS data for {node_dir.name} on {day_str}, skipping",
                  file=sys.stderr)
        else:
            _save(
                plot_gps_tracks(gps_day, node_dir.name),
                day_root / node_dir.name / "gps_track.png",
                day_root / node_dir.name / "gps_track_trace.csv",
            )
            gps_frames.append(gps_day.assign(__node__=node_dir.name))
            any_output = True

        rf_full = load_rf_scenario(node_dir, node_ids, node_id_to_name)
        rf_day  = filter_by_day(rf_full, day)
        if rf_day is None:
            print(f"  WARNING: no rf data for {node_dir.name} on {day_str}, skipping",
                  file=sys.stderr)
            continue

        out = day_root / node_dir.name
        _save_pairs(plot_silvus_snr(rf_day, node_dir.name),        out, "IH_snr")
        _save_pairs(plot_silvus_rcpi(rf_day, node_dir.name),       out, "IH_rcpi")
        _save_pairs(plot_silvus_mcs(rf_day, node_dir.name),        out, "IH_mcs")
        _save_pairs(plot_silvus_throughput(rf_day, node_dir.name), out, "IH_throughput")
        _save_pairs(plot_silvus_per(rf_day, node_dir.name),        out, "IH_per")
        any_output = True

    if gps_frames:
        import pandas as pd
        combined = pd.concat(gps_frames, ignore_index=True)
        _save(
            plot_gps_tracks(combined, csv_dir.name),
            day_root / "gps_all_nodes.png",
            day_root / "gps_all_nodes_trace.csv",
        )

    return any_output


## @brief Plot individual node GPS tracks and all nodes combined, split by day.
#
# Loads each node's full GPS and RF data once, discovers every calendar day
# present across all nodes, then re-plots each day's filtered slice into its
# own ``output_path/<day>/`` subdirectory via @ref _plot_nodes_for_day.
# Silvus GPS is excluded and static nodes are filtered out automatically by
# @ref load_gps_all-style logic inside the per-day combined plot.
#
# @param csv_dir     Directory containing per-node subdirs.
# @param output_path Directory to write per-day PNGs and trace CSVs into.
# @param day         If given, only this calendar date is plotted. Otherwise
#                    every day discovered in the data is plotted.
# @return 0 on success, 1 if no GPS data is found in any node or the
#        requested day has no data.
def _plot_nodes(csv_dir: Path, output_path: Path, day=None) -> int:
    output_path.mkdir(parents=True, exist_ok=True)

    nodes, node_ids, node_id_to_name = _discover_valid_nodes(csv_dir)
    if not nodes:
        print(f"ERROR: no valid node directories found in {csv_dir}", file=sys.stderr)
        return 1

    # Discover every calendar day present, pooling across all nodes' GPS data.
    all_days: set = set()
    for node_dir in nodes:
        gps_full = load_gps_scenario(node_dir)
        all_days.update(days_present(gps_full))

    if not all_days:
        print(f"ERROR: no GPS data found in {csv_dir}", file=sys.stderr)
        return 1

    if day is not None:
        if day not in all_days:
            available = ", ".join(d.isoformat() for d in sorted(all_days))
            print(f"ERROR: no data for {day.isoformat()}; available days: {available}",
                  file=sys.stderr)
            return 1
        target_days = [day]
    else:
        target_days = sorted(all_days)

    wrote_any = False
    for d in target_days:
        print(f"[{d.isoformat()}]")
        if _plot_nodes_for_day(d, nodes, node_ids, node_id_to_name, csv_dir, output_path):
            wrote_any = True

    if not wrote_any:
        print("ERROR: no output was written for any requested day", file=sys.stderr)
        return 1
    return 0


## @brief Dispatch the plot subcommand.
#
# @param args  Parsed argument namespace from argparse.
# @return 0 on success, 1 on error.
def _cmd_plot(args: argparse.Namespace) -> int:
    if not args.input.exists():
        print(f"ERROR: {args.input} not found -- run `extract` first",
              file=sys.stderr)
        return 1

    available = sorted(p.name for p in args.input.iterdir() if p.is_dir())
    if not available:
        print(f"ERROR: no directories found in {args.input}", file=sys.stderr)
        return 1

    #Plot all scenarios
    if args.all:
        scenarios = sorted(p for p in args.input.iterdir() if p.is_dir())
    elif args.nodes:
        day = None
        if getattr(args, "day", None):
            try:
                day = date.fromisoformat(args.day)
            except ValueError:
                print(f"ERROR: --day must be YYYY-MM-DD, got {args.day!r}",
                      file=sys.stderr)
                return 1
        return _plot_nodes(args.input, args.output, day=day)
    else:
        #Plot specific scenario
        if args.scenario not in available:
            print(f"ERROR: scenario '{args.scenario}' not found", file=sys.stderr)
            print(f"available: {', '.join(available)}", file=sys.stderr)
            return 1
        scenarios = [args.input / args.scenario]

    args.output.mkdir(parents=True, exist_ok=True)
    for s in scenarios:
        _plot_one(s, args.output)
    print(f"\nDone. Figures in {args.output}/")
    return 0


## @brief Dispatch the split-day subcommand.
#
# Splits a combined multi-day GPS trace CSV (e.g. ``gps_all_nodes_trace.csv``
# produced by ``plot --nodes``) into one CSV per calendar day, with
# ``sec_since_origin`` re-zeroed for each day. With ``--summary``, prints a
# per-day per-node row-count table instead of writing any files.
#
# @param args  Parsed argument namespace from argparse.
# @return 0 on success, 1 on error.
def _cmd_split_day(args: argparse.Namespace) -> int:
    if not args.input.is_file():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    if args.summary:
        print_day_node_summary(args.input)
        return 0

    written = split_trace_by_day(args.input, args.output, args.prefix)
    print(f"Split {args.input.name} into {len(written)} day files:")
    for date_str, path in sorted(written.items()):
        with open(path) as f:
            n_rows = sum(1 for _ in f) - 1  # exclude header
        print(f"  {date_str}: {path}  ({n_rows} rows)")
    return 0


## @brief CLI entry point.
#
# Subcommands: ``extract``, ``plot``, ``load-config``, ``multi-day``, ``split-day``.
#
# @return Exit code: 0 on success, 1 on error.
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    # Extract arguements
    e = sub.add_parser("extract", help="Unzip the ARPO data bundle")
    e.add_argument("-i", "--input", help="Filepath for zip file")

    # Plot arguments
    pp = sub.add_parser("plot", help="Generate per-scenario figures")
    pp.add_argument("-i", "--input",  type=Path,
                    help="Filepath for csv folder", default=CSV_ROOT)
    pp.add_argument("-o", "--output", type=Path,
                    help="Filepath for output folder", default=PER_DAY_DIR)
    g = pp.add_mutually_exclusive_group(required=True)
    g.add_argument("--scenario", help="Scenario directory name under csv/")
    g.add_argument("--nodes",    action="store_true",
                   help="Plot all nodes GPS on one graph")
    g.add_argument("--all",      action="store_true",
                   help="Plot every scenario")
    pp.add_argument("--day", default=None,
                    help="Only plot this calendar day (YYYY-MM-DD); "
                         "used with --nodes. Default: plot every day found, "
                         "each into its own output subdirectory.")

    #Multi-day
    md = sub.add_parser(
        "multi-day",
        help="Day-vs-day distribution overlays + similarity table per scenario family",
    )
    md.add_argument("--family", help="Only process this scenario family")
    md.add_argument("--audit",  action="store_true",
                    help="Print raw-CSV vs parsed-bag row count + max diagnostics")

    sd = sub.add_parser(
        "split-day",
        help="Split a combined multi-day GPS trace CSV into one file per day",
    )
    sd.add_argument("-i", "--input", type=Path, required=True,
                     help="Path to the combined trace CSV (e.g. gps_all_nodes_trace.csv)")
    sd.add_argument("-o", "--output", type=Path,
                     help="Output directory for the per-day CSVs (required unless --summary)")
    sd.add_argument("--prefix", default=None,
                     help="Filename prefix for outputs (default: derived from input filename)")
    sd.add_argument("--summary", action="store_true",
                     help="Print a per-day, per-node row-count table and exit "
                          "without writing any files")

    args = p.parse_args()
    if args.cmd == "extract":
        return extract(
            paths=DatasetPaths(zip_path=Path(args.input))
        ) if args.input else extract()
    if args.cmd == "plot":
        return _cmd_plot(args)
    if args.cmd == "multi-day":
        return multi_day(audit=args.audit, family_filter=args.family)
    if args.cmd == "split-day":
        if not args.summary and args.output is None:
            print("ERROR: --output is required unless --summary is given",
                  file=sys.stderr)
            return 1
        return _cmd_split_day(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())