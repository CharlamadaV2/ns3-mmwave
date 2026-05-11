"""
Run from scratch/mesh-sim/:
    python -m scripts.arpo_data.cli extract
    python -m scripts.arpo_data.cli summarize
    python -m scripts.arpo_data.cli plot --scenario <name> [--gallery]
    python -m scripts.arpo_data.cli plot --all [--gallery]
    python -m scripts.arpo_data.cli compare
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .compare import compare
from .extract import extract
from .gallery import (
    plot_snr_box,
    plot_snr_heatmap,
    plot_snr_per_link,
    plot_snr_per_node,
    plot_topology_card,
)
from .loaders import load_bh2_scenario, load_gps_scenario, load_ping_scenario
from .paths import CSV_ROOT, PLOTS_DIR
from .plots import (
    filter_dominant_peers,
    plot_bh2_mcs,
    plot_bh2_snr,
    plot_bh2_throughput,
    plot_gps_tracks,
    plot_ping_latency,
)
from .summarize import summarize


def _save(result, path: Path) -> None:
    """
    Save a figure (and an optional trace CSV) returned from a plot fn.

    Plot fns may return either ``Figure`` or ``(Figure, trace_df)``. When a
    trace is present, it's written next to the PNG as ``<stem>_trace.csv``
    so a reader can grep any plotted point back to the source rows.
    """
    if result is None:
        return
    if isinstance(result, tuple):
        fig, trace = result
    else:
        fig, trace = result, None
    if fig is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # I believe the DPI is maxed
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {path}")
    if trace is not None and not trace.empty:
        trace_path = path.with_name(f"{path.stem}_trace.csv")
        trace.to_csv(trace_path, index=False)
        print(f"    wrote {trace_path}")


def _plot_one(scen_dir: Path, gallery: bool) -> None:
    out = PLOTS_DIR / scen_dir.name
    print(f"  [{scen_dir.name}]")

    name = scen_dir.name
    bh2 = load_bh2_scenario(scen_dir)
    if bh2 is not None:
        _save(plot_bh2_snr(bh2, name), out / "bh2_snr.png")
        _save(plot_bh2_mcs(bh2, name), out / "bh2_mcs.png")
        _save(plot_bh2_throughput(bh2, name), out / "bh2_throughput.png")

    ping = load_ping_scenario(scen_dir)
    if ping is not None:
        _save(plot_ping_latency(ping, name), out / "ping_latency.png")

    gps = load_gps_scenario(scen_dir)
    if gps is not None:
        _save(plot_gps_tracks(gps, name), out / "gps_track.png")

    if gallery and bh2 is not None:
        snr = filter_dominant_peers(bh2.dropna(subset=["field_snr"]))
        if not snr.empty:
            gout = out / "snr_gallery"
            _save(plot_topology_card(bh2.attrs.get("topology", {})), gout / "topology.png")
            _save(plot_snr_per_node(snr), gout / "style_a_per_node.png")
            _save(plot_snr_per_link(snr), gout / "style_b_per_link.png")
            _save(plot_snr_heatmap(snr), gout / "style_c_heatmap.png")
            _save(plot_snr_box(snr), gout / "style_d_box.png")


def _cmd_plot(args: argparse.Namespace) -> int:
    if not CSV_ROOT.exists():
        print(f"ERROR: {CSV_ROOT} not found -- run `extract` first", file=sys.stderr)
        return 1

    if args.all:
        scenarios = sorted(p for p in CSV_ROOT.iterdir() if p.is_dir())
    else:
        scen = CSV_ROOT / args.scenario
        if not scen.is_dir():
            available = sorted(p.name for p in CSV_ROOT.iterdir() if p.is_dir())
            print(f"ERROR: scenario '{args.scenario}' not found", file=sys.stderr)
            print(f"available: {', '.join(available)}", file=sys.stderr)
            return 1
        scenarios = [scen]

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for s in scenarios:
        _plot_one(s, gallery=args.gallery)
    print(f"\nDone. Figures in {PLOTS_DIR}/")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("extract", help="Unzip the ARPO data bundle")
    sub.add_parser("summarize", help="Report file presence and data anomalies")

    pp = sub.add_parser("plot", help="Generate per-scenario figures")
    g = pp.add_mutually_exclusive_group(required=True)
    g.add_argument("--scenario", help="Scenario directory name under csv/")
    g.add_argument("--all", action="store_true", help="Plot every scenario")
    pp.add_argument("--gallery", action="store_true",
                    help="Also emit the 4-style SNR comparison gallery")

    sub.add_parser("compare", help="Cross-scenario SNR heatmap")

    args = p.parse_args()
    if args.cmd == "extract":
        return extract()
    if args.cmd == "summarize":
        return summarize()
    if args.cmd == "plot":
        return _cmd_plot(args)
    if args.cmd == "compare":
        return compare()
    return 1


if __name__ == "__main__":
    sys.exit(main())
