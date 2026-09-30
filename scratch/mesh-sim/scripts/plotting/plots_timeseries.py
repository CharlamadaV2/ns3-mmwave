"""Time-series plot functions for mesh-sim outputs.

Each function takes a pre-aggregated DataFrame (output of aggregate_timeseries)
and returns a matplotlib Figure. No function calls savefig or plt.show.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


## @brief Apply axis labels, optional title, legend and grid to a time-series axis.
def _style_timeseries(ax, xlabel="Simulation Time (s)", ylabel="", title=""):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)


## @brief Line (or step) plot of `<val_col>_mean` per (node_a, node_b) link, with a CI band.
#
# @param agg_df   Output of aggregate_timeseries with `time_s`, `node_a`, `node_b`.
# @param val_col  Base value name; needs `<val_col>_mean`, optionally `<val_col>_ci95`.
# @param ylabel   Y-axis label.
# @param title    Plot title.
# @param step     True for a step ("post") line instead of a straight line.
# @return The Figure. The CI band is drawn only when some CI is > 0.
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


## @brief Same as _plot_per_link but grouped by (src, dst) flow, labelled "src->dst".
#
# @param agg_df   Output of aggregate_timeseries with `time_s`, `src`, `dst`.
# @param val_col  Base value name; needs `<val_col>_mean`, optionally `<val_col>_ci95`.
# @param ylabel   Y-axis label.
# @param title    Plot title.
# @return The Figure.
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

## @fn plot_sinr_timeseries
# @brief SINR (signal-to-interference-plus-noise ratio, dB) vs simulation time, one line per link.
#
# @param agg_df  Aggregated links data with `sinr_db_mean` (and `sinr_db_ci95`).
# @param title   Optional title; default "SINR Time Series".
# @return matplotlib Figure (not saved or shown).
#
# Rows with `sinr_db_mean` <= -900 (placeholder for "no link") are dropped first.
def plot_sinr_timeseries(agg_df: pd.DataFrame,
                         title: str | None = None) -> plt.Figure:
    """SINR (dB) vs simulation time, one line per link."""
    # Filter placeholder values
    if "sinr_db_mean" in agg_df.columns:
        agg_df = agg_df[agg_df["sinr_db_mean"] > -900]
    return _plot_per_link(agg_df, "sinr_db", "SINR (dB)",
                          title or "SINR Time Series")


## @fn plot_rx_power_timeseries
# @brief Received power (dBm) vs simulation time, one line per link.
#
# @param agg_df  Aggregated rx-power data with `rx_power_dbm_mean` (and `_ci95`).
# @param title   Optional title; default "Rx Power Time Series".
# @return matplotlib Figure (not saved or shown).
#
# Rows with `rx_power_dbm_mean` <= -900 (placeholder value) are dropped first.
def plot_rx_power_timeseries(agg_df: pd.DataFrame,
                             title: str | None = None) -> plt.Figure:
    """Rx Power (dBm) vs simulation time, one line per link."""
    if "rx_power_dbm_mean" in agg_df.columns:
        agg_df = agg_df[agg_df["rx_power_dbm_mean"] > -900]
    return _plot_per_link(agg_df, "rx_power_dbm", "Rx Power (dBm)",
                          title or "Rx Power Time Series")


## @fn plot_capacity_timeseries
# @brief Link capacity (Mbps) vs simulation time, one line per link.
#
# @param agg_df  Aggregated links data with `capacity_mbps_mean` (and `_ci95`).
# @param title   Optional title; default "Link Capacity Time Series".
# @return matplotlib Figure (not saved or shown).
def plot_capacity_timeseries(agg_df: pd.DataFrame,
                             title: str | None = None) -> plt.Figure:
    """Capacity (Mbps) vs simulation time, one line per link."""
    return _plot_per_link(agg_df, "capacity_mbps", "Capacity (Mbps)",
                          title or "Link Capacity Time Series")


## @fn plot_mcs_timeseries
# @brief MCS (modulation and coding scheme) index vs simulation time as a step plot, one line per link.
#
# @param agg_df  Aggregated mcs data with `mcs_index_mean` (and `_ci95`).
# @param title   Optional title; default "MCS Index Time Series".
# @return matplotlib Figure with integer-only y ticks (not saved or shown).
def plot_mcs_timeseries(agg_df: pd.DataFrame,
                        title: str | None = None) -> plt.Figure:
    """MCS index vs simulation time, step plot, one line per link."""
    fig = _plot_per_link(agg_df, "mcs_index", "MCS Index",
                         title or "MCS Index Time Series", step=True)
    fig.axes[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    return fig


## @fn plot_throughput_timeseries
# @brief Delivered throughput (Mbps) vs simulation time, per link or per flow.
#
# @param agg_df  Aggregated data with `delivered_mbps_mean`; either links data
#                (has `node_a`) or flows data (has `src`, `dst`).
# @param title   Optional title; default "Link Throughput Time Series" or "Flow Throughput Time Series".
# @return matplotlib Figure (not saved or shown).
#
# The presence of a `node_a` column selects link mode; otherwise flow mode.
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


## @fn plot_latency_timeseries
# @brief End-to-end latency (ms) vs simulation time, one line per flow.
#
# @param agg_df  Aggregated flows data with `latency_ms_mean` (and `_ci95`).
# @param title   Optional title; default "Flow Latency Time Series".
# @return matplotlib Figure (not saved or shown).
def plot_latency_timeseries(agg_df: pd.DataFrame,
                            title: str | None = None) -> plt.Figure:
    """Latency (ms) vs simulation time, one line per flow."""
    return _plot_per_flow(agg_df, "latency_ms", "Latency (ms)",
                          title or "Flow Latency Time Series")


## @fn plot_derived_geometry
# @brief Elevation and azimuth angles between node pairs vs time, derived from positions.
#
# @param positions_df  Table with `time_s`, `node_id`, `x`, `y`, `z` (from positions.csv);
#                      may be concatenated across seeds, in which case positions are averaged.
# @param node_pairs    List of (node_a, node_b) ids to plot; None means every unordered pair.
# @param title         Optional title for the top panel; default "Derived Geometry".
# @return matplotlib Figure with two stacked panels: elevation (deg) and azimuth (deg) (not saved or shown).
#
# For each pair, uses timesteps where both nodes exist, takes the vector a -> b,
# elevation = atan2(dz, horizontal distance), azimuth = atan2(dx, dy) mod 360
# (measured from the +y axis toward +x). The input is copied, not modified.
# Pairs with no common timesteps are skipped.
def plot_derived_geometry(positions_df: pd.DataFrame,
                          node_pairs: list[tuple[int, int]] | None = None,
                          title: str | None = None) -> plt.Figure:
    """Elevation angle and azimuth vs time, derived from node positions."""
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
