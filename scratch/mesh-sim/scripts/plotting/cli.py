#!/usr/bin/env python3
"""CLI entry point for mesh-sim plotting.

Usage:
    python -m scripts.plotting.cli --config scripts/plotting/plot.example.ini
    python scripts/plotting/cli.py --config path/to/plot.ini
"""

import argparse
import configparser
import os
import sys
import warnings

import matplotlib.pyplot as plt

from .loaders import (
    discover_seed_dirs,
    load_flows_csv,
    load_links_csv,
    load_mcs_csv,
    load_positions_csv,
    load_routes_csv,
    load_rx_power_csv,
    load_summary,
)
from .aggregation import aggregate_summaries, aggregate_timeseries
from .plots_timeseries import (
    plot_capacity_timeseries,
    plot_derived_geometry,
    plot_latency_timeseries,
    plot_mcs_timeseries,
    plot_rx_power_timeseries,
    plot_sinr_timeseries,
    plot_throughput_timeseries,
)
from .plots_summary import (
    plot_network_summary,
    plot_per_flow_bars,
    plot_per_node_bars,
    plot_sim_runtime,
)


def _save(fig, name: str, out_dir: str, fmt: str, dpi: int):
    path = os.path.join(out_dir, f"{name}.{fmt}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def _enabled(cfg: configparser.ConfigParser, key: str) -> bool:
    return cfg.getboolean("plots", key, fallback=True)


def main():
    parser = argparse.ArgumentParser(description="mesh-sim plot generator")
    parser.add_argument("--config", required=True, help="Path to plot INI config")
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    data_dir = cfg.get("output", "data_dir")
    fmt = cfg.get("output", "format", fallback="png")
    dpi = cfg.getint("output", "dpi", fallback=150)

    seed_dirs = discover_seed_dirs(data_dir)
    if not seed_dirs:
        print(f"No seed directories found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(seed_dirs)} seed(s) in {data_dir}")

    out_dir = os.path.join(data_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    # --- Summary aggregation (always needed) ---
    summaries = []
    for sd in seed_dirs:
        s = load_summary(os.path.join(sd, "summary.json"))
        if s:
            summaries.append(s)

    agg_summary = aggregate_summaries(summaries)

    # --- Time-series: links.csv ---
    link_agg = None
    link_needed = any(_enabled(cfg, k) for k in
                      ["sinr_timeseries", "capacity_timeseries",
                       "throughput_timeseries"])
    if link_needed:
        link_agg = aggregate_timeseries(
            seed_dirs, load_links_csv, "links.csv", "time_s",
            ["node_a", "node_b"],
            ["sinr_db", "capacity_mbps", "delivered_mbps"])

    # --- Time-series: rx-power.csv ---
    rx_agg = None
    if _enabled(cfg, "rx_power_timeseries"):
        rx_agg = aggregate_timeseries(
            seed_dirs, load_rx_power_csv, "rx-power.csv", "time_s",
            ["node_a", "node_b"], ["rx_power_dbm"])

    # --- Time-series: mcs.csv ---
    mcs_agg = None
    if _enabled(cfg, "mcs_timeseries"):
        mcs_agg = aggregate_timeseries(
            seed_dirs, load_mcs_csv, "mcs.csv", "time_s",
            ["node_a", "node_b"], ["mcs_index"])

    # --- Time-series: flows.csv ---
    flow_agg = None
    flow_needed = any(_enabled(cfg, k) for k in
                      ["throughput_timeseries", "latency_timeseries"])
    if flow_needed:
        flow_agg = aggregate_timeseries(
            seed_dirs, load_flows_csv, "flows.csv", "time_s",
            ["src", "dst"],
            ["demand_mbps", "delivered_mbps", "latency_ms"])

    # --- Generate plots ---
    print("Generating plots...")

    if _enabled(cfg, "sinr_timeseries") and link_agg is not None:
        _save(plot_sinr_timeseries(link_agg),
              "sinr_timeseries", out_dir, fmt, dpi)

    if _enabled(cfg, "rx_power_timeseries") and rx_agg is not None:
        _save(plot_rx_power_timeseries(rx_agg),
              "rx_power_timeseries", out_dir, fmt, dpi)

    if _enabled(cfg, "capacity_timeseries") and link_agg is not None:
        _save(plot_capacity_timeseries(link_agg),
              "capacity_timeseries", out_dir, fmt, dpi)

    if _enabled(cfg, "mcs_timeseries") and mcs_agg is not None:
        _save(plot_mcs_timeseries(mcs_agg),
              "mcs_timeseries", out_dir, fmt, dpi)

    if _enabled(cfg, "throughput_timeseries"):
        if flow_agg is not None:
            _save(plot_throughput_timeseries(flow_agg),
                  "flow_throughput_timeseries", out_dir, fmt, dpi)
        if link_agg is not None:
            _save(plot_throughput_timeseries(link_agg),
                  "link_throughput_timeseries", out_dir, fmt, dpi)

    if _enabled(cfg, "latency_timeseries") and flow_agg is not None:
        _save(plot_latency_timeseries(flow_agg),
              "latency_timeseries", out_dir, fmt, dpi)

    if _enabled(cfg, "geometry"):
        positions_frames = []
        for sd in seed_dirs:
            p = load_positions_csv(os.path.join(sd, "positions.csv"))
            if p is not None:
                positions_frames.append(p)
        if positions_frames:
            import pandas as pd
            all_pos = pd.concat(positions_frames, ignore_index=True)
            _save(plot_derived_geometry(all_pos),
                  "geometry", out_dir, fmt, dpi)

    # Summary plots
    if _enabled(cfg, "per_node_sinr") and agg_summary["per_node"]:
        _save(plot_per_node_bars(agg_summary, "mean_sinr_db"),
              "per_node_sinr", out_dir, fmt, dpi)

    if _enabled(cfg, "per_node_throughput") and agg_summary["per_node"]:
        _save(plot_per_node_bars(agg_summary, "tx_throughput_mbps"),
              "per_node_tx_throughput", out_dir, fmt, dpi)
        _save(plot_per_node_bars(agg_summary, "rx_throughput_mbps"),
              "per_node_rx_throughput", out_dir, fmt, dpi)

    if _enabled(cfg, "per_flow_throughput") and agg_summary["per_flow"]:
        _save(plot_per_flow_bars(agg_summary, "delivered_mbps"),
              "per_flow_throughput", out_dir, fmt, dpi)

    if _enabled(cfg, "per_flow_latency") and agg_summary["per_flow"]:
        _save(plot_per_flow_bars(agg_summary, "latency_ms"),
              "per_flow_latency", out_dir, fmt, dpi)

    if _enabled(cfg, "network_summary"):
        _save(plot_network_summary(agg_summary),
              "network_summary", out_dir, fmt, dpi)

    if _enabled(cfg, "sim_runtime") and summaries:
        _save(plot_sim_runtime(summaries),
              "sim_runtime", out_dir, fmt, dpi)

    print(f"Done. Figures written to {out_dir}/")


if __name__ == "__main__":
    main()
