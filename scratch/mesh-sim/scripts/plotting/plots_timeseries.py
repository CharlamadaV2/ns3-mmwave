"""Time-series plot functions for mesh-sim outputs.

Each function takes a pre-aggregated DataFrame (output of aggregate_timeseries)
and returns a matplotlib Figure. No function calls savefig or plt.show.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _style_timeseries(ax, xlabel="Simulation Time (s)", ylabel="", title=""):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)


def _plot_per_link(agg_df: pd.DataFrame, val_col: str, ylabel: str,
                   title: str, step: bool = False) -> plt.Figure:
    """Generic per-link time-series with optional CI band."""
    fig, ax = plt.subplots(figsize=(10, 5))

    mean_col = f"{val_col}_mean"
    ci_col = f"{val_col}_ci95"

    pairs = agg_df.groupby(["node_a", "node_b"])
    for i, ((na, nb), g) in enumerate(pairs):
        c = _COLORS[i % len(_COLORS)]
        g = g.sort_values("time_s")
        label = f"{na}-{nb}"

        if step:
            ax.step(g["time_s"], g[mean_col], label=label, color=c,
                    linewidth=1.0, alpha=0.8, where="post")
        else:
            ax.plot(g["time_s"], g[mean_col], label=label, color=c,
                    linewidth=1.0, alpha=0.8)

        if ci_col in g.columns:
            ci = g[ci_col]
            if ci.max() > 0:
                ax.fill_between(g["time_s"],
                                g[mean_col] - ci,
                                g[mean_col] + ci,
                                color=c, alpha=0.15)

    _style_timeseries(ax, ylabel=ylabel, title=title)
    fig.tight_layout()
    return fig


def _plot_per_flow(agg_df: pd.DataFrame, val_col: str, ylabel: str,
                   title: str) -> plt.Figure:
    """Generic per-flow time-series with optional CI band."""
    fig, ax = plt.subplots(figsize=(10, 5))

    mean_col = f"{val_col}_mean"
    ci_col = f"{val_col}_ci95"

    pairs = agg_df.groupby(["src", "dst"])
    for i, ((src, dst), g) in enumerate(pairs):
        c = _COLORS[i % len(_COLORS)]
        g = g.sort_values("time_s")
        label = f"{src}\u2192{dst}"

        ax.plot(g["time_s"], g[mean_col], label=label, color=c,
                linewidth=1.0, alpha=0.8)

        if ci_col in g.columns:
            ci = g[ci_col]
            if ci.max() > 0:
                ax.fill_between(g["time_s"],
                                g[mean_col] - ci,
                                g[mean_col] + ci,
                                color=c, alpha=0.15)

    _style_timeseries(ax, ylabel=ylabel, title=title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public plot functions
# ---------------------------------------------------------------------------

def plot_sinr_timeseries(agg_df: pd.DataFrame,
                         title: str | None = None) -> plt.Figure:
    """SINR (dB) vs simulation time, one line per link."""
    # Filter placeholder values
    if "sinr_db_mean" in agg_df.columns:
        agg_df = agg_df[agg_df["sinr_db_mean"] > -900]
    return _plot_per_link(agg_df, "sinr_db", "SINR (dB)",
                          title or "SINR Time Series")


def plot_rx_power_timeseries(agg_df: pd.DataFrame,
                             title: str | None = None) -> plt.Figure:
    """Rx Power (dBm) vs simulation time, one line per link."""
    if "rx_power_dbm_mean" in agg_df.columns:
        agg_df = agg_df[agg_df["rx_power_dbm_mean"] > -900]
    return _plot_per_link(agg_df, "rx_power_dbm", "Rx Power (dBm)",
                          title or "Rx Power Time Series")


def plot_capacity_timeseries(agg_df: pd.DataFrame,
                             title: str | None = None) -> plt.Figure:
    """Capacity (Mbps) vs simulation time, one line per link."""
    return _plot_per_link(agg_df, "capacity_mbps", "Capacity (Mbps)",
                          title or "Link Capacity Time Series")


def plot_mcs_timeseries(agg_df: pd.DataFrame,
                        title: str | None = None) -> plt.Figure:
    """MCS index vs simulation time, step plot, one line per link."""
    fig = _plot_per_link(agg_df, "mcs_index", "MCS Index",
                         title or "MCS Index Time Series", step=True)
    fig.axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    return fig


def plot_throughput_timeseries(agg_df: pd.DataFrame,
                               title: str | None = None) -> plt.Figure:
    """Delivered throughput (Mbps) vs time. Works with link or flow data."""
    # Detect whether this is link data or flow data
    if "node_a" in agg_df.columns:
        return _plot_per_link(agg_df, "delivered_mbps",
                              "Delivered Throughput (Mbps)",
                              title or "Link Throughput Time Series")
    else:
        return _plot_per_flow(agg_df, "delivered_mbps",
                              "Delivered Throughput (Mbps)",
                              title or "Flow Throughput Time Series")


def plot_latency_timeseries(agg_df: pd.DataFrame,
                            title: str | None = None) -> plt.Figure:
    """Latency (ms) vs simulation time, one line per flow."""
    return _plot_per_flow(agg_df, "latency_ms", "Latency (ms)",
                          title or "Flow Latency Time Series")


def plot_derived_geometry(positions_df: pd.DataFrame,
                          node_pairs: list[tuple[int, int]] | None = None,
                          title: str | None = None) -> plt.Figure:
    """Elevation angle and azimuth vs time, derived from node positions.

    *positions_df* should be raw (non-aggregated) from a single seed's
    positions.csv, or concatenated across seeds (will be averaged).

    *node_pairs*: list of (node_a, node_b) to plot. If None, plots all pairs.
    """
    df = positions_df.copy()
    df["time_s"] = df["time_s"].round(3)

    # Average positions across seeds if multiple seeds present
    avg = df.groupby(["time_s", "node_id"], as_index=False)[["x", "y", "z"]].mean()

    # Build all pairs
    node_ids = sorted(avg["node_id"].unique())
    if node_pairs is None:
        node_pairs = [(a, b) for i, a in enumerate(node_ids)
                      for b in node_ids[i + 1:]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for idx, (na, nb) in enumerate(node_pairs):
        c = _COLORS[idx % len(_COLORS)]
        pa = avg[avg["node_id"] == na].set_index("time_s")
        pb = avg[avg["node_id"] == nb].set_index("time_s")
        common = pa.index.intersection(pb.index)
        if len(common) == 0:
            continue

        dx = pb.loc[common, "x"].values - pa.loc[common, "x"].values
        dy = pb.loc[common, "y"].values - pa.loc[common, "y"].values
        dz = pb.loc[common, "z"].values - pa.loc[common, "z"].values
        horiz = np.sqrt(dx**2 + dy**2)

        elevation_deg = np.degrees(np.arctan2(dz, horiz))
        azimuth_deg = np.degrees(np.arctan2(dx, dy)) % 360

        label = f"{na}-{nb}"
        ax1.plot(common, elevation_deg, label=label, color=c,
                 linewidth=1.0, alpha=0.8)
        ax2.plot(common, azimuth_deg, label=label, color=c,
                 linewidth=1.0, alpha=0.8)

    _style_timeseries(ax1, xlabel="", ylabel="Elevation Angle (deg)",
                      title=title or "Derived Geometry")
    _style_timeseries(ax2, ylabel="Azimuth (deg)", title="")

    fig.tight_layout()
    return fig
