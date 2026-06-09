"""cli script"""

## @file cli.py
# @brief Main script to be ran for manipulating ARPO data as desired
#
# Provides a command line interface that can do one of the following commands:
# - Extract unzips data folder into extraction target direction
# - Multi-day compares day vs day data for multi day scenarios
# - Plot generates figures for specific scenario or all scenarios per individual day

import argparse
import sys
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
    load_config,
    load_rf_scenario,
    load_node_id,
)
from .multi_day import multi_day
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


## @brief Plot individual node GPS tracks and all nodes combined on one graph.
#
# For each node subdirectory, plots its GPS track individually. Then plots
# all nodes together on one combined graph. Silvus GPS is excluded and
# static nodes are filtered out automatically by @ref load_gps_all.
#
# @param csv_dir     Directory containing per-node subdirs.
# @param output_path Directory to write PNGs and trace CSVs into.
# @return 0 on success, 1 if no GPS data is found in any node.
def _plot_nodes(csv_dir: Path, output_path: Path) -> int:
    output_path.mkdir(parents=True, exist_ok=True)

    nodes: list[Path] = []  #< List of directories to parse through
    node_ids : list[str] = [] #< List of ids to keep
    node_id_to_name: dict[str, str] = {} #< List of node ids to keep for each dataset
    
    # Create a list of nodes with valid data sets
    for node_dir in sorted(csv_dir.iterdir()):
        # Check if directory exists
        if not node_dir.is_dir() or node_dir.name == "sdwan":
            continue
        # Check if node contains both gps and silvus directories
        if not (node_dir / "gps").is_dir() or not (node_dir / "silvus").is_dir():
            print(f"  WARNING: {node_dir.name} missing gps or silvus directory, skipping",
                file=sys.stderr)
            continue
        
        # Create a list of nodes to parse through with their id
        result = load_node_id(node_dir)
        if result is None:
            print(f"  WARNING: could not read node_id for {node_dir.name}, skipping",
                  file=sys.stderr)
            continue
        node_id_to_name.update(result)
        nodes.append(node_dir)
        node_ids.append(list(result.keys())[0])
        
    # Parses through valid nodes and plots their data
    for node_dir in nodes:
        # GPS Plotting
        gps = load_gps_scenario(node_dir)
        if gps is None:
            print(f"  WARNING: no GPS data for {node_dir.name}, skipping",
                  file=sys.stderr)
            continue
        _save(
            plot_gps_tracks(gps, node_dir.name),
            output_path / node_dir.name / "gps_track.png",
            output_path / node_dir.name / "gps_track_trace.csv",
        )
        
        # RF Quality Plotting
        out = output_path / node_dir.name
        rf_data = load_rf_scenario(node_dir, node_ids, node_id_to_name)
        if rf_data is None:
            print(f"  WARNING: no rf data for {node_dir.name}, skipping",
                file=sys.stderr)
            continue
        
        _save_pairs(plot_silvus_snr(rf_data, node_dir.name),        out, "IH_snr")
        _save_pairs(plot_silvus_rcpi(rf_data, node_dir.name),       out, "IH_rcpi")
        _save_pairs(plot_silvus_mcs(rf_data, node_dir.name),        out, "IH_mcs")
        _save_pairs(plot_silvus_throughput(rf_data, node_dir.name), out, "IH_throughput")
        _save_pairs(plot_silvus_per(rf_data, node_dir.name),        out, "IH_per")


    # Plot all nodes combined.
    df = load_gps_all(csv_dir)
    if df is None:
        print(f"ERROR: no GPS data found in {csv_dir}", file=sys.stderr)
        return 1

    _save(
        plot_gps_tracks(df, csv_dir.name),
        output_path / "gps_all_nodes.png",
        output_path / "gps_all_nodes_trace.csv",
    )
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
        return _plot_nodes(args.input, args.output)
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


## @brief CLI entry point.
#
# Subcommands: ``extract``, ``plot``, ``load-config``, ``multi-day``.
#
# @return Exit code: 0 on success, 1 on error.
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="Unzip the ARPO data bundle")
    e.add_argument("-i", "--input", help="Filepath for zip file")

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

    lc = sub.add_parser("load-config",
                         help="Load extracted node data into simulator config format")
    lc.add_argument("-i", "--input",  type=Path, required=True,
                    help="Filepath for node csv folder")
    lc.add_argument("-o", "--output", type=Path,
                    help="Filepath for output folder", default=Path("../../inputs"))

    md = sub.add_parser(
        "multi-day",
        help="Day-vs-day distribution overlays + similarity table per scenario family",
    )
    md.add_argument("--family", help="Only process this scenario family")
    md.add_argument("--audit",  action="store_true",
                    help="Print raw-CSV vs parsed-bag row count + max diagnostics")

    args = p.parse_args()
    if args.cmd == "extract":
        return extract(
            paths=DatasetPaths(zip_path=Path(args.input))
        ) if args.input else extract()
    if args.cmd == "plot":
        return _cmd_plot(args)
    if args.cmd == "load-config":
        return load_config(args.input)
    if args.cmd == "multi-day":
        return multi_day(audit=args.audit, family_filter=args.family)
    return 1


if __name__ == "__main__":
    sys.exit(main())