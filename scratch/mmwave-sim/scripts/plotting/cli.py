#!/usr/bin/env python3
"""CLI entry point for mmwave-sim plotting.

Usage:
    python scripts/plotting/cli.py --config plot.ini
    python -m scripts.plotting.cli --config plot.ini
"""

import argparse
import configparser
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                    stream=sys.stderr)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate plots from mmwave-sim C++ output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example plot.ini:

  [data]
  data_dir = scratch/mmwave-sim/outputs/2026-03/25/14-30-00

  [plots]
  sinr_timeseries = true
  throughput_timeseries = true
  mcs_timeseries = true
  sim_runtime = true
  network_summary = true

  [output]
  format = png
  dpi = 150
""",
    )
    p.add_argument("--config", required=True, help="Path to plot.ini")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory from INI")
    return p.parse_args()


def _bool(val: str) -> bool:
    return val.strip().lower() in ("true", "1", "yes")


def main():
    args = parse_args()

    # --- Parse INI -----------------------------------------------------------
    ini = configparser.ConfigParser()
    ini.read(args.config)

    data_dir = ini.get("data", "data_dir")
    if not os.path.isabs(data_dir):
        # Resolve relative to INI file location
        ini_dir = os.path.dirname(os.path.abspath(args.config))
        data_dir = os.path.normpath(os.path.join(ini_dir, data_dir))

    out_dir = args.output_dir or os.path.join(data_dir, "figures")
    fmt = ini.get("output", "format", fallback="png")
    dpi = int(ini.get("output", "dpi", fallback="150"))

    # Which plots to generate
    plot_flags = {
        "sinr_timeseries":       _bool(ini.get("plots", "sinr_timeseries", fallback="true")),
        "throughput_timeseries": _bool(ini.get("plots", "throughput_timeseries", fallback="true")),
        "mcs_timeseries":       _bool(ini.get("plots", "mcs_timeseries", fallback="true")),
        "sim_runtime":           _bool(ini.get("plots", "sim_runtime", fallback="true")),
        "network_summary":       _bool(ini.get("plots", "network_summary", fallback="true")),
    }

    # --- Lazy imports (keeps startup fast if just checking --help) -----------
    from .loaders import (
        load_summary,
        load_links_csv,
        load_dl_pdcp_stats,
        load_rx_packet_trace,
        discover_seed_dirs,
        aggregate_summaries,
    )
    from .plots import (
        plot_sinr_timeseries,
        plot_throughput_timeseries,
        plot_mcs_timeseries,
        plot_sim_runtime,
        plot_network_summary,
    )

    # --- Discover seeds ------------------------------------------------------
    seed_dirs = discover_seed_dirs(data_dir)
    if not seed_dirs:
        log.error("No seed-N/ directories found in %s", data_dir)
        sys.exit(1)

    log.info("Found %d seed dir(s) in %s", len(seed_dirs), data_dir)

    # Load all summaries
    summaries = []
    for sd in seed_dirs:
        sp = os.path.join(sd, "summary.json")
        if os.path.isfile(sp):
            summaries.append(load_summary(sp))
        else:
            log.warning("summary.json not found in %s", sd)

    if not summaries:
        log.error("No summary.json files found")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    def _save(fig, name: str):
        path = os.path.join(out_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        log.info("Saved %s", path)
        import matplotlib.pyplot as plt
        plt.close(fig)

    # --- Aggregate across seeds (works for 1 or many) ------------------------
    agg = aggregate_summaries(summaries)

    if plot_flags["sim_runtime"]:
        _save(plot_sim_runtime(summaries), "sim_runtime")

    if plot_flags["network_summary"]:
        _save(plot_network_summary(agg), "network_summary")

    # --- SINR timeseries (all seeds) -----------------------------------------
    if plot_flags["sinr_timeseries"]:
        all_links = []
        for sd in seed_dirs:
            links_path = os.path.join(sd, "links.csv")
            if os.path.isfile(links_path):
                df = load_links_csv(links_path)
                seed_name = os.path.basename(sd)
                df["seed"] = seed_name
                all_links.append(df)
            else:
                log.warning("links.csv not found in %s, skipping", sd)

        if all_links:
            import pandas as pd
            combined = pd.concat(all_links, ignore_index=True)
            _save(plot_sinr_timeseries(combined,
                                       title=f"SINR Time Series ({len(all_links)} seeds)"),
                  "sinr_timeseries")
        else:
            log.warning("No links.csv files found, skipping timeseries")

    # --- Throughput timeseries (all seeds) ------------------------------------
    if plot_flags["throughput_timeseries"]:
        all_pdcp = []
        for sd in seed_dirs:
            pdcp_path = os.path.join(sd, "DlPdcpStats.txt")
            if os.path.isfile(pdcp_path):
                df = load_dl_pdcp_stats(pdcp_path)
                df["seed"] = os.path.basename(sd)
                all_pdcp.append(df)
            else:
                log.warning("DlPdcpStats.txt not found in %s, skipping", sd)

        if all_pdcp:
            import pandas as pd
            combined = pd.concat(all_pdcp, ignore_index=True)
            n = len(all_pdcp)
            _save(plot_throughput_timeseries(
                combined, title=f"DL Throughput Time Series ({n} seeds)"),
                "throughput_timeseries")
        else:
            log.warning("No DlPdcpStats.txt files found, skipping throughput")

    # --- MCS timeseries (all seeds) ------------------------------------------
    if plot_flags["mcs_timeseries"]:
        all_rx = []
        for sd in seed_dirs:
            rx_path = os.path.join(sd, "RxPacketTrace.txt")
            if os.path.isfile(rx_path):
                df = load_rx_packet_trace(rx_path)
                df["seed"] = os.path.basename(sd)
                all_rx.append(df)
            else:
                log.warning("RxPacketTrace.txt not found in %s, skipping", sd)

        if all_rx:
            import pandas as pd
            combined = pd.concat(all_rx, ignore_index=True)
            n = len(all_rx)
            _save(plot_mcs_timeseries(
                combined, title=f"MCS Time Series ({n} seeds)"),
                "mcs_timeseries")
        else:
            log.warning("No RxPacketTrace.txt files found, skipping MCS")

    log.info("All plots saved to %s", out_dir)


if __name__ == "__main__":
    main()
