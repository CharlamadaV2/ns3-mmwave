"""
Run from scratch/mesh-sim/:
    python -m scripts.arpo_data.cli extract
    python -m scripts.arpo_data.cli plot --scenario <name>
    python -m scripts.arpo_data.cli plot --all
    python -m scripts.arpo_data.cli multi-day
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .extract import extract
from .loaders import (
    load_bh2_scenario,
    load_gps_scenario,
)
from .multi_day import multi_day
from .paths import CSV_ROOT, PER_DAY_DIR
from .plots import (
    plot_bh2_mcs,
    plot_bh2_per,
    plot_bh2_rcpi,
    plot_bh2_snr,
    plot_bh2_throughput,
    plot_gps_tracks,
)


def _save(result, png_path: Path, csv_path: Path | None = None) -> None:
    """
    Save a figure (and an optional trace CSV) returned from a plot fn.

    Plot fns may return either ``Figure`` or ``(Figure, trace_df)``. The PNG
    lands at ``png_path``; if a trace is present it goes to ``csv_path`` when
    supplied, otherwise next to the PNG as ``<stem>_trace.csv``.
    """
    if result is None:
        return
    if isinstance(result, tuple):
        fig, trace = result
    else:
        fig, trace = result, None
    if fig is None:
        return
    png_path.parent.mkdir(parents=True, exist_ok=True)
    # I believe the DPI is maxed
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {png_path}")
    if trace is not None and not trace.empty:
        target = csv_path if csv_path is not None \
            else png_path.with_name(f"{png_path.stem}_trace.csv")
        target.parent.mkdir(parents=True, exist_ok=True)
        trace.to_csv(target, index=False)
        print(f"    wrote {target}")


def _save_pairs(results, scen_dir: Path, base_name: str) -> None:
    """
    Save the (src, peer, fig, trace) tuples emitted by per-radio plot fns.

    PNGs land under ``pngs/<src>/`` and traces under ``csvs/<src>/`` so the
    outputs for each source rab live together.
    """
    if not results:
        return
    for src, peer, fig, trace in results:
        png = scen_dir / "pngs" / src / f"{base_name}__{src}_to_{peer}.png"
        csv = scen_dir / "csvs" / src / f"{base_name}__{src}_to_{peer}_trace.csv"
        _save((fig, trace), png, csv)


def _plot_one(scen_dir: Path) -> None:
    out = PER_DAY_DIR / scen_dir.name
    pngs = out / "pngs"
    csvs = out / "csvs"
    print(f"  [{scen_dir.name}]")

    name = scen_dir.name
    bh2 = load_bh2_scenario(scen_dir)
    if bh2 is not None:
        _save_pairs(plot_bh2_snr(bh2, name),        out, "bh2_snr")
        _save_pairs(plot_bh2_rcpi(bh2, name),       out, "bh2_rcpi")
        _save_pairs(plot_bh2_mcs(bh2, name),        out, "bh2_mcs")
        _save_pairs(plot_bh2_throughput(bh2, name), out, "bh2_throughput")
        _save_pairs(plot_bh2_per(bh2, name),        out, "bh2_per")

    gps = load_gps_scenario(scen_dir)
    if gps is not None:
        _save(plot_gps_tracks(gps, name),
              pngs / "gps_track.png", csvs / "gps_track_trace.csv")


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

    PER_DAY_DIR.mkdir(parents=True, exist_ok=True)
    for s in scenarios:
        _plot_one(s)
    print(f"\nDone. Figures in {PER_DAY_DIR}/")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("extract", help="Unzip the ARPO data bundle")

    pp = sub.add_parser("plot", help="Generate per-scenario figures")
    g = pp.add_mutually_exclusive_group(required=True)
    g.add_argument("--scenario", help="Scenario directory name under csv/")
    g.add_argument("--all", action="store_true", help="Plot every scenario")

    md = sub.add_parser(
        "multi-day",
        help="Day-vs-day distribution overlays + similarity table per scenario family",
    )
    md.add_argument("--family",
                    help="Only process this scenario family")
    md.add_argument("--audit", action="store_true",
                    help="Print raw-CSV vs parsed-bag row count + max diagnostics")

    args = p.parse_args()
    if args.cmd == "extract":
        return extract()
    if args.cmd == "plot":
        return _cmd_plot(args)
    if args.cmd == "multi-day":
        return multi_day(audit=args.audit, family_filter=args.family)
    return 1


if __name__ == "__main__":
    sys.exit(main())
