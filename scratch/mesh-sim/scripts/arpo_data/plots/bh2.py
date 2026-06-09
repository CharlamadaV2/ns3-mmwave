##@file bh2.py
# @brief Per-radio bh2 plots: one figure per (src_rab, peer_rab), one subplot per local MAC.
#
#
##

from __future__ import annotations


from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from ..topology import (
    mac_radio_label,
    netdev_orientation,
    radio_index,
    radio_label,
)
from .common import concat_trace, crashed_suffix, scenario_caption

_BH2_PairResult = tuple[str, str, plt.Figure, pd.DataFrame]
_MIN_SAMPLES_PER_PEER_MAC = 5

## @brief Plot raw SNR per (source radio, peer antenna) for a scenario.
#
# Produces one figure per (src_rab, peer_rab) pair found in @p df. Each
# figure has one subplot per source-side local MAC. MCS reference lines are
# drawn at SNR thresholds 5, 10, 15, and 20 dB.
#
# @param df             Loaded bh2 DataFrame from @ref load_bh2_scenario.
#                       Must contain ``field_snr``, ``tag_local_mac``, and
#                       ``tag_sta_mac`` columns.
# @param scenario_name  Human-readable scenario identifier for figure titles.
# @return               List of ``(src, peer, Figure, trace_df)`` tuples,
#                       one per (src_rab, peer_rab) pair. Empty if no
#                       usable data is found.
def plot_bh2_snr(df: pd.DataFrame, scenario_name: str = "") -> list[_BH2_PairResult]:
    return _per_radio_metric(
        df=df,
        scenario_name=scenario_name,
        value_col="field_snr",
        metric_label="SNR (dB)",
        title_metric="SNR",
        trace_col="snr_db",
        file_prefix="bh2_snr",
        add_mcs_reference_lines=True,
        use_antenna_legend=True,
    )

## @brief Plot raw RCPI (received signal power) per (source radio, peer antenna).
#
# Produces one figure per (src_rab, peer_rab) pair. RCPI values are in dBm
# as reported directly by the driver without calibration.
#
# @param df             Loaded bh2 DataFrame from @ref load_bh2_scenario.
#                       Must contain ``field_rcpi``, ``tag_local_mac``, and
#                       ``tag_sta_mac`` columns.
# @param scenario_name  Human-readable scenario identifier for figure titles.
# @return               List of ``(src, peer, Figure, trace_df)`` tuples.
#                       Empty if no usable data is found.
def plot_bh2_rcpi(df: pd.DataFrame, scenario_name: str = "") -> list[_BH2_PairResult]:
    return _per_radio_metric(
        df=df,
        scenario_name=scenario_name,
        value_col="field_rcpi",
        metric_label="RCPI (dBm)",
        title_metric="RCPI",
        trace_col="rcpi_dbm",
        file_prefix="bh2_rcpi",
        use_antenna_legend=True,
    )

## @brief Plot raw MCS-TX index per (source radio, peer antenna).
#
# Produces one figure per (src_rab, peer_rab) pair. Points are connected by
# lines to make discrete level transitions visible over time. The Y axis is
# constrained to integer ticks.
#
# @param df             Loaded bh2 DataFrame from @ref load_bh2_scenario.
#                       Must contain ``field_mcs_tx``, ``tag_local_mac``,
#                       and ``tag_sta_mac`` columns.
# @param scenario_name  Human-readable scenario identifier for figure titles.
# @return               List of ``(src, peer, Figure, trace_df)`` tuples.
#                       Empty if no usable data is found.
def plot_bh2_mcs(df: pd.DataFrame, scenario_name: str = "") -> list[_BH2_PairResult]:
    return _per_radio_metric(
        df=df,
        scenario_name=scenario_name,
        value_col="field_mcs_tx",
        metric_label="MCS",
        title_metric="MCS-TX",
        trace_col="mcs_tx",
        file_prefix="bh2_mcs",
        integer_yaxis=True,
        use_antenna_legend=True,
        connect_lines=True,
    )

## @brief Plot raw packet-error rate per (source radio, peer antenna).
#
# Produces one figure per (src_rab, peer_rab) pair. PER is the raw counter
# value reported by the driver; it is not normalised.
#
# @param df             Loaded bh2 DataFrame from @ref load_bh2_scenario.
#                       Must contain ``field_per``, ``tag_local_mac``, and
#                       ``tag_sta_mac`` columns.
# @param scenario_name  Human-readable scenario identifier for figure titles.
# @return               List of ``(src, peer, Figure, trace_df)`` tuples.
#                       Empty if no usable data is found.
def plot_bh2_per(df: pd.DataFrame, scenario_name: str = "") -> list[_BH2_PairResult]:
    return _per_radio_metric(
        df=df,
        scenario_name=scenario_name,
        value_col="field_per",
        metric_label="PER",
        title_metric="PER",
        trace_col="per",
        file_prefix="bh2_per",
        use_antenna_legend=True,
    )

## @brief Plot PHY throughput rate per (source radio, peer antenna).
#
# Derives Mbps from ``field_bytes_tx`` deltas rather than reading a rate
# column directly. Counter resets (negative deltas) are dropped. Pairs with
# fewer than @ref _MIN_SAMPLES_PER_PEER_MAC valid samples after delta
# computation are skipped.
#
# @param df             Loaded bh2 DataFrame from @ref load_bh2_scenario.
#                       Must contain ``field_bytes_tx``, ``tag_local_mac``,
#                       and ``tag_sta_mac`` columns.
# @param scenario_name  Human-readable scenario identifier for figure titles.
# @return               List of ``(src, peer, Figure, trace_df)`` tuples
#                       where ``trace_df`` also includes ``raw_bytes_tx``,
#                       ``dt_s``, and ``db_bytes`` columns. Empty if no
#                       usable data is found.
def plot_bh2_throughput(df: pd.DataFrame, scenario_name: str = "") -> list[_BH2_PairResult]:
    if df is None or "field_bytes_tx" not in df.columns:
        return []
    sub = df.dropna(subset=["field_bytes_tx", "tag_local_mac", "tag_sta_mac"])
    if sub.empty:
        return []

    results: list[_BH2_PairResult] = []
    for (src, peer), pair_df in sub.groupby(["__session__", "__peer__"]):
        rates = _per_pair_rate_mbps(pair_df)
        if rates.empty:
            continue
        peer_macs_here = sorted(rates["tag_sta_mac"].dropna()
                                .astype(str).unique())
        peer_label_map = _build_peer_label_map(peer_macs_here)
        result = _build_per_radio_figure(
            pair_df=rates,
            src=src,
            peer=peer,
            scenario_name=scenario_name,
            value_col="__mbps__",
            metric_label="PHY (Mbps)",
            title_metric="PHY rate",
            trace_col="mbps",
            file_prefix="bh2_throughput",
            extra_trace_cols={
                "raw_bytes_tx": "field_bytes_tx",
                "dt_s": "__dt__",
                "db_bytes": "__db__",
            },
            use_antenna_legend=True,
            peer_label_map=peer_label_map,
        )
        if result is not None:
            results.append(result)
    return results

## @brief Adapt a silvus RF DataFrame to the column layout expected by the bh2 plot functions.
#
# The bh2 plot pipeline expects ``__session__``, ``__peer__``, ``tag_local_mac``,
# and ``tag_sta_mac`` for grouping, plus ``field_*`` metric columns. This adapter
# maps the silvus column names to those conventions so the shared plot machinery
# can be reused without modification.
#
# @param df  DataFrame produced by @ref load_rf_scenario.
# @return    Copy of @p df with bh2-compatible columns added.
## @brief Adapt a silvus RF DataFrame to the column layout expected by the bh2 plot functions.
#
# Sets ``__peer__`` to a constant so all neighbors are plotted as coloured
# series within one figure per metric rather than one figure per neighbor.
# Each unique ``neighbor`` value becomes a separate series via ``tag_sta_mac``.
#
# @param df  DataFrame produced by @ref load_rf_scenario.
# @return    Copy of @p df with bh2-compatible columns added.
def _silvus_to_bh2_format(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["__session__"]   = out["__node__"]
    out["__peer__"]      = "neighbors"       # constant — one figure per node
    out["tag_local_mac"] = out["node_id"].astype(str)
    out["tag_sta_mac"]   = out["neighbor"].astype(str)
    col_map = {
        "snr":             "field_snr",
        "rcpi":            "field_rcpi",
        "mcs":             "field_mcs_tx",
        "per":             "field_per",
        "throughput_mbps": "field_throughput_mbps",
    }
    for src_col, dst_col in col_map.items():
        if src_col in out.columns:
            out[dst_col] = out[src_col]
    return out
 
 
## @brief Plot SNR per (node, neighbor) pair from silvus RF data.
#
# @param df             DataFrame produced by @ref load_rf_scenario.
# @param scenario_name  Human-readable scenario identifier for figure titles.
# @return               List of ``(src, peer, Figure, trace_df)`` tuples.
def plot_silvus_snr(df: pd.DataFrame, scenario_name: str = "") -> list[_BH2_PairResult]:
    return _per_radio_metric(
        df=_silvus_to_bh2_format(df),
        scenario_name=scenario_name,
        value_col="field_snr",
        metric_label="SNR (dB)",
        title_metric="SNR",
        trace_col="snr_db",
        file_prefix="IH_snr",
        add_mcs_reference_lines=True,
        use_antenna_legend=True,
    )
 
## @brief Plot RCPI per (node, neighbor) pair from silvus RF data.
#
# RCPI is the mean RSSI across active antennas computed by @ref load_rf_scenario.
#
# @param df             DataFrame produced by @ref load_rf_scenario.
# @param scenario_name  Human-readable scenario identifier for figure titles.
# @return               List of ``(src, peer, Figure, trace_df)`` tuples.
def plot_silvus_rcpi(df: pd.DataFrame, scenario_name: str = "") -> list[_BH2_PairResult]:
    return _per_radio_metric(
        df=_silvus_to_bh2_format(df),
        scenario_name=scenario_name,
        value_col="field_rcpi",
        metric_label="RCPI (dBm)",
        title_metric="RCPI",
        trace_col="rcpi_dbm",
        file_prefix="IH_rcpi",
        use_antenna_legend=True,
    )
 
 
## @brief Plot MCS index per (node, neighbor) pair from silvus RF data.
#
# @param df             DataFrame produced by @ref load_rf_scenario.
# @param scenario_name  Human-readable scenario identifier for figure titles.
# @return               List of ``(src, peer, Figure, trace_df)`` tuples.
def plot_silvus_mcs(df: pd.DataFrame, scenario_name: str = "") -> list[_BH2_PairResult]:
    return _per_radio_metric(
        df=_silvus_to_bh2_format(df),
        scenario_name=scenario_name,
        value_col="field_mcs_tx",
        metric_label="MCS",
        title_metric="MCS",
        trace_col="mcs_tx",
        file_prefix="IH_mcs",
        integer_yaxis=True,
        use_antenna_legend=True,
        connect_lines=True,
    )
 
 
## @brief Plot throughput per (node, neighbor) pair from silvus RF data.
#
# Unlike @ref plot_bh2_throughput which derives rate from byte counter deltas,
# this function uses the pre-computed ``throughput_mbps`` column from
# @ref load_rf_scenario directly.
#
# @param df             DataFrame produced by @ref load_rf_scenario.
# @param scenario_name  Human-readable scenario identifier for figure titles.
# @return               List of ``(src, peer, Figure, trace_df)`` tuples.
def plot_silvus_throughput(df: pd.DataFrame, scenario_name: str = "") -> list[_BH2_PairResult]:
    return _per_radio_metric(
        df=_silvus_to_bh2_format(df),
        scenario_name=scenario_name,
        value_col="field_throughput_mbps",
        metric_label="Throughput (Mbps)",
        title_metric="Throughput",
        trace_col="throughput_mbps",
        file_prefix="IH_throughput",
        use_antenna_legend=True,
    )
 
 
## @brief Plot PER per (node, neighbor) pair from silvus RF data.
#
# @param df             DataFrame produced by @ref load_rf_scenario.
# @param scenario_name  Human-readable scenario identifier for figure titles.
# @return               List of ``(src, peer, Figure, trace_df)`` tuples.
def plot_silvus_per(df: pd.DataFrame, scenario_name: str = "") -> list[_BH2_PairResult]:
    return _per_radio_metric(
        df=_silvus_to_bh2_format(df),
        scenario_name=scenario_name,
        value_col="field_per",
        metric_label="PER",
        title_metric="PER",
        trace_col="per",
        file_prefix="IH_per",
        use_antenna_legend=True,
    )

## @brief Shared plot body for SNR, RCPI, MCS, and PER metrics.
#
# Groups @p df by ``(__session__, __peer__)`` and calls
# @ref _build_per_radio_figure for each pair. Throughput uses a separate
# code path in @ref plot_bh2_throughput because it requires rate derivation
# before plotting.
#
# @param df                     Loaded bh2 DataFrame.
# @param scenario_name          Scenario identifier for figure titles.
# @param value_col              DataFrame column to plot on the Y axis.
# @param metric_label           Y axis label string (e.g. ``"SNR (dB)"``).
# @param title_metric           Metric name used in the figure title.
# @param trace_col              Column name for the metric in the output trace CSV.
# @param file_prefix            Prefix used when saving the figure to disk.
# @param add_mcs_reference_lines Draw horizontal SNR threshold lines if ``True``.
# @param integer_yaxis          Constrain Y ticks to integers if ``True`` (MCS).
# @param use_antenna_legend     Show a figure-level color legend if ``True``.
# @param connect_lines          Connect scatter points with lines if ``True`` (MCS).
# @return                       List of ``(src, peer, Figure, trace_df)`` tuples.
def _per_radio_metric(
        df: pd.DataFrame,
        scenario_name: str,
        value_col: str,
        metric_label: str,
        title_metric: str,
        trace_col: str,
        file_prefix: str,
        add_mcs_reference_lines: bool = False,
        integer_yaxis: bool = False,
        use_antenna_legend: bool = False,
        connect_lines: bool = False,
) -> list[_BH2_PairResult]:
    if df is None or value_col not in df.columns:
        return []
    needed = [value_col, "tag_local_mac", "tag_sta_mac"]
    sub = df.dropna(subset=needed)
    if sub.empty:
        return []

    results: list[_BH2_PairResult] = []
    for (src, peer), pair_df in sub.groupby(["__session__", "__peer__"]):
        peer_macs_here = sorted(pair_df["tag_sta_mac"].dropna()
                                .astype(str).unique())
        peer_label_map = _build_peer_label_map(peer_macs_here)
        result = _build_per_radio_figure(
            pair_df=pair_df,
            src=src,
            peer=peer,
            scenario_name=scenario_name,
            value_col=value_col,
            metric_label=metric_label,
            title_metric=title_metric,
            trace_col=trace_col,
            file_prefix=file_prefix,
            add_mcs_reference_lines=add_mcs_reference_lines,
            integer_yaxis=integer_yaxis,
            use_antenna_legend=use_antenna_legend,
            connect_lines=connect_lines,
            peer_label_map=peer_label_map,
        )
        if result is not None:
            results.append(result)
    return results

## @brief Resolve a list of peer MACs to their ``<device>.<index>`` radio labels.
#
# Calls @ref mac_radio_label for each MAC. MACs that resolve to a label
# containing ``".?"`` (unknown radio index) are excluded from the output so
# only fully resolved labels are returned.
#
# @param peer_macs  List of peer MAC address strings.
# @return           Mapping from MAC string to resolved label (e.g. ``"sky01-mw.1"``).
#                   MACs that cannot be fully resolved are absent from the map.
def _build_peer_label_map(peer_macs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for mac in peer_macs:
        label = mac_radio_label(mac)
        if label and ".?" not in label:
            out[str(mac)] = label
    return out

## @brief Inch-based heights for the title, stats table, and legend block above the subplots.
#
# All values are in figure inches so they can be used with
# ``fig.tight_layout`` and ``fig.add_axes`` without converting units.
@dataclass(frozen=True)
class _HeaderLayout:
    title_in: float   ##< Height reserved for the two-line centered title.
    table_in: float   ##< Height reserved for the per-radio stats table.
    legend_in: float  ##< Height reserved for the peer antenna color legend (0 if unused).
    total_in: float   ##< Sum of all three sections plus inter-section padding.

## @brief Compute the inch-based header layout for a figure with @p n_macs source radios.
#
# Table height grows linearly with the number of source MACs. The legend
# block is only allocated when @p use_antenna_legend is ``True``.
#
# @param n_macs             Number of source-side local MACs in the figure.
# @param use_antenna_legend Whether the figure will include a peer antenna legend.
# @return                   Populated @ref _HeaderLayout with all four fields set.
def _compute_header_layout(n_macs: int, use_antenna_legend: bool) -> _HeaderLayout:
    title_in = 0.65
    table_in = 0.18 * (n_macs + 1) + 0.10
    legend_in = 0.45 if use_antenna_legend else 0.0
    total_in = title_in + table_in + legend_in + 0.30
    return _HeaderLayout(title_in, table_in, legend_in, total_in)

## @brief Build the peer MAC to display-label mapping for the figure legend.
#
# Uses @ref _build_peer_label_map for fully resolved labels. Falls back to
# ``"<peer_rab> ...<last8 of MAC>"`` for MACs that could not be resolved to
# a ``<device>.<index>`` label.
#
# @param peer_macs      List of peer MAC address strings seen in this figure.
# @param peer_rab       Peer chassis hostname (used in fallback labels).
# @param peer_label_map Resolved label map from @ref _build_peer_label_map.
# @return               Mapping from peer MAC string to legend display label.
def _build_peer_legend_labels(peer_macs: list[str], peer_rab: str,
                              peer_label_map: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for mac in peer_macs:
        canonical = peer_label_map.get(mac)
        out[mac] = canonical if canonical else f"{peer_rab} ...{str(mac)[-8:]}"
    return out

## @brief Plot one source-MAC subplot and return its trace rows and drawn peer MACs.
#
# Filters @p pair_df to rows where ``tag_local_mac == local_mac``, then
# iterates peer MACs. Peer MACs with fewer than @ref _MIN_SAMPLES_PER_PEER_MAC
# points are silently skipped and counted as hidden. Builds a trace DataFrame
# for each drawn peer MAC.
#
# @param ax                  Axes to draw into.
# @param src                 Source rab hostname.
# @param peer                Peer rab hostname.
# @param local_mac           The source-side local MAC address for this subplot.
# @param pair_df             DataFrame for this (src, peer) pair.
# @param value_col           Column to plot on the Y axis.
# @param metric_label        Y axis label string.
# @param trace_col           Column name for the metric in the output trace DataFrame.
# @param color_map           Peer MAC to matplotlib colour mapping.
# @param peer_legend_labels  Peer MAC to display label mapping.
# @param add_mcs_reference_lines Draw horizontal SNR threshold lines if ``True``.
# @param integer_yaxis       Constrain Y ticks to integers if ``True``.
# @param y_max               Fixed Y axis maximum, or ``None`` for auto-scale.
# @param use_antenna_legend  Show a figure-level legend instead of per-subplot if ``True``.
# @param connect_lines       Connect scatter points with lines if ``True``.
# @param extra_trace_cols    Additional columns to include in the trace DataFrame,
#                            as ``{output_col: source_col}`` pairs.
# @return                    Tuple of ``(trace_rows, plotted_peer_macs)`` where
#                            ``trace_rows`` is a list of per-peer DataFrames and
#                            ``plotted_peer_macs`` is the set of MACs that were drawn.
def _plot_one_subplot(
        ax: plt.Axes,
        *,
        src: str,
        peer: str,
        local_mac,
        pair_df: pd.DataFrame,
        value_col: str,
        metric_label: str,
        trace_col: str,
        color_map: dict[str, tuple],
        peer_legend_labels: dict[str, str],
        add_mcs_reference_lines: bool,
        integer_yaxis: bool,
        y_max: float | None,
        use_antenna_legend: bool,
        connect_lines: bool,
        extra_trace_cols: dict[str, str] | None,
) -> tuple[list[pd.DataFrame], set[str]]:
    row_df = pair_df[pair_df["tag_local_mac"] == local_mac]
    netdev = _first_str(row_df.get("tag_interface"))
    subtitle = _radio_subtitle(src, netdev, local_mac)

    trace_rows: list[pd.DataFrame] = []
    plotted_peer_macs: set[str] = set()
    plotted = 0
    hidden = 0
    for pmac, g in row_df.groupby("tag_sta_mac"):
        if len(g) < _MIN_SAMPLES_PER_PEER_MAC:
            hidden += 1
            continue
        color = color_map.get(str(pmac), "0.4")
        if use_antenna_legend:
            label = peer_legend_labels.get(str(pmac), f"...{str(pmac)[-8:]}")
        else:
            label = f"...{str(pmac)[-8:]}"
        g_plot = g.sort_values("__sec__") if connect_lines else g
        ax.plot(
            g_plot["__sec__"], g_plot[value_col],
            ".-" if connect_lines else ".",
            markersize=2,
            linewidth=0.4 if connect_lines else 0,
            alpha=0.45, color=color, label=label,
        )
        plotted_peer_macs.add(str(pmac))
        trace = pd.DataFrame({
            "sec": g["__sec__"].values,
            "source": src,
            "peer": peer,
            "tag_local_mac": g["tag_local_mac"].values,
            "tag_sta_mac": g["tag_sta_mac"].values,
            "tag_interface": (g["tag_interface"].values
                              if "tag_interface" in g.columns else netdev),
            trace_col: g[value_col].values,
        })
        if extra_trace_cols:
            for out_name, src_col in extra_trace_cols.items():
                if src_col in g.columns:
                    trace[out_name] = g[src_col].values
        trace_rows.append(trace)
        plotted += 1

    _format_subplot_axis(
        ax, subtitle=subtitle, metric_label=metric_label,
        add_mcs_reference_lines=add_mcs_reference_lines,
        integer_yaxis=integer_yaxis, y_max=y_max,
        show_per_subplot_legend=bool(plotted) and not use_antenna_legend,
        hidden=hidden,
    )
    return trace_rows, plotted_peer_macs

## @brief Apply axis formatting, reference lines, and annotations to one subplot.
#
# Draws MCS SNR threshold lines (5, 10, 15, 20 dB) when requested, applies
# integer Y ticks and an optional fixed Y maximum for MCS plots, sets the
# title and grid, and adds a hidden-peer annotation when some peer MACs were
# suppressed due to insufficient samples.
#
# @param ax                     Axes to format.
# @param subtitle               Left-aligned subplot title (radio label string).
# @param metric_label           Y axis label string.
# @param add_mcs_reference_lines Draw horizontal SNR threshold lines if ``True``.
# @param integer_yaxis          Constrain Y ticks to integers if ``True``.
# @param y_max                  Fixed Y axis maximum, or ``None`` for auto-scale.
# @param show_per_subplot_legend Show a per-subplot legend when not using a
#                               figure-level antenna legend.
# @param hidden                 Number of peer MACs suppressed for low sample count.
def _format_subplot_axis(
        ax: plt.Axes,
        *,
        subtitle: str,
        metric_label: str,
        add_mcs_reference_lines: bool,
        integer_yaxis: bool,
        y_max: float | None,
        show_per_subplot_legend: bool,
        hidden: int,
) -> None:
    if add_mcs_reference_lines:
        for thr, lbl in [(5, "MCS 0"), (10, "MCS 4"),
                         (15, "MCS 8"), (20, "MCS 12")]:
            ax.axhline(thr, color="0.7", linestyle=":",
                       linewidth=0.6, zorder=0)
            ax.text(1.0, thr, f" {lbl}",
                    transform=ax.get_yaxis_transform(),
                    color="0.55", va="center", fontsize=7, alpha=0.8)
    if integer_yaxis:
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if y_max is not None:
            ax.set_ylim(-0.5, y_max)
    ax.set_title(subtitle, fontsize=9, fontweight="bold", loc="left")
    ax.set_ylabel(metric_label, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if show_per_subplot_legend:
        ax.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.85)
    if hidden:
        ax.text(0.99, 0.02,
                f"hidden: {hidden} peer MAC(s) <{_MIN_SAMPLES_PER_PEER_MAC} samples",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="0.4")

## @brief Build one complete (src_rab, peer_rab) figure with one subplot per source MAC.
#
# Orchestrates the full figure construction pipeline:
# -# Determines stable local MAC order via @ref _stable_local_mac_order.
# -# Assigns deterministic colours to peer MACs via @ref _peer_mac_colors.
# -# Computes the header layout via @ref _compute_header_layout.
# -# Creates the subplot grid via @ref _make_per_radio_axes.
# -# Calls @ref _plot_one_subplot for each source MAC.
# -# Applies the figure chrome (title, table, legend) via @ref _render_figure_chrome.
#
# @param pair_df               DataFrame for this (src, peer) pair.
# @param src                   Source rab hostname.
# @param peer                  Peer rab hostname.
# @param scenario_name         Scenario identifier for the figure title.
# @param value_col             DataFrame column to plot on the Y axis.
# @param metric_label          Y axis label string.
# @param title_metric          Metric name for the figure title.
# @param trace_col             Column name for the metric in the trace CSV.
# @param file_prefix           Prefix for the output filename.
# @param add_mcs_reference_lines Draw SNR threshold lines if ``True``.
# @param integer_yaxis         Constrain Y ticks to integers if ``True``.
# @param extra_trace_cols      Additional trace columns as ``{output: source}`` pairs.
# @param use_antenna_legend    Show figure-level peer antenna legend if ``True``.
# @param connect_lines         Connect scatter points with lines if ``True``.
# @param peer_label_map        Pre-resolved ``{mac: label}`` map from @ref _build_peer_label_map.
# @return                      ``(src, peer, Figure, trace_df)`` tuple, or ``None``
#                              if @p pair_df is empty or contains no source MACs.
def _build_per_radio_figure(
        pair_df: pd.DataFrame,
        src: str,
        peer: str,
        scenario_name: str,
        value_col: str,
        metric_label: str,
        title_metric: str,
        trace_col: str,
        file_prefix: str,
        add_mcs_reference_lines: bool = False,
        integer_yaxis: bool = False,
        extra_trace_cols: dict[str, str] | None = None,
        use_antenna_legend: bool = False,
        connect_lines: bool = False,
        peer_label_map: dict[str, str] | None = None,
) -> _BH2_PairResult | None:
    if pair_df.empty:
        return None
    local_macs = _stable_local_mac_order(pair_df, src)
    if not local_macs:
        return None

    peer_macs = sorted(pair_df["tag_sta_mac"].dropna().astype(str).unique())
    color_map = _peer_mac_colors(peer_macs)
    peer_legend_labels = _build_peer_legend_labels(
        peer_macs, peer, peer_label_map or {},
    )

    layout = _compute_header_layout(len(local_macs), use_antenna_legend)
    fig, axes = _make_per_radio_axes(
        local_macs, height_per_panel=2.2, header_inches=layout.total_in,
    )
    y_max = (max(13.0, float(pair_df[value_col].max()) + 0.5)
             if integer_yaxis else None)

    trace_rows: list[pd.DataFrame] = []
    plotted_peer_macs: set[str] = set()
    for ax, lm in zip(axes, local_macs):
        rows, drawn = _plot_one_subplot(
            ax,
            src=src, peer=peer, local_mac=lm, pair_df=pair_df,
            value_col=value_col, metric_label=metric_label,
            trace_col=trace_col, color_map=color_map,
            peer_legend_labels=peer_legend_labels,
            add_mcs_reference_lines=add_mcs_reference_lines,
            integer_yaxis=integer_yaxis, y_max=y_max,
            use_antenna_legend=use_antenna_legend,
            connect_lines=connect_lines,
            extra_trace_cols=extra_trace_cols,
        )
        trace_rows.extend(rows)
        plotted_peer_macs.update(drawn)
    axes[-1].set_xlabel("seconds since each session's start", fontweight="bold")

    _render_figure_chrome(
        fig, layout=layout, scenario_name=scenario_name,
        src=src, peer=peer, title_metric=title_metric,
        pair_df=pair_df, value_col=value_col, metric_label=metric_label,
        use_antenna_legend=use_antenna_legend,
        plotted_peer_macs=plotted_peer_macs,
        color_map=color_map, peer_legend_labels=peer_legend_labels,
    )
    return src, peer, fig, concat_trace(trace_rows)

## @brief Apply tight_layout and render the centered header, stats table, and legend.
#
# Called once per figure after all subplots are drawn. Computes the top
# margin from the header layout so ``tight_layout`` does not overlap the
# header block. Then delegates to three helper functions for the title,
# stats table, and optional antenna legend.
#
# @param fig                  Figure to annotate.
# @param layout               Header layout from @ref _compute_header_layout.
# @param scenario_name        Scenario identifier for the subtitle line.
# @param src                  Source rab hostname.
# @param peer                 Peer rab hostname.
# @param title_metric         Metric name for the main title line.
# @param pair_df              Full (src, peer) DataFrame used for the stats table.
# @param value_col            Column supplying values to the stats table.
# @param metric_label         Y axis label string passed to the stats table.
# @param use_antenna_legend   Render the peer antenna legend if ``True``.
# @param plotted_peer_macs    Set of peer MACs that were actually drawn.
# @param color_map            Peer MAC to colour mapping for the legend.
# @param peer_legend_labels   Peer MAC to display label mapping for the legend.
def _render_figure_chrome(
        fig: plt.Figure,
        *,
        layout: _HeaderLayout,
        scenario_name: str,
        src: str,
        peer: str,
        title_metric: str,
        pair_df: pd.DataFrame,
        value_col: str,
        metric_label: str,
        use_antenna_legend: bool,
        plotted_peer_macs: set[str],
        color_map: dict[str, tuple],
        peer_legend_labels: dict[str, str],
) -> None:
    fig_h = float(fig.get_size_inches()[1])
    top_margin = max(1.0 - layout.total_in / fig_h, 0.55)
    fig.tight_layout(rect=(0, 0, 1, top_margin))

    _figure_centered_header(
        fig, scenario_name=scenario_name,
        src=src, peer=peer, title_metric=title_metric, df=pair_df,
        y_main=1.0 - 0.25 / fig_h,
        y_scenario=1.0 - 0.50 / fig_h,
    )

    table_bottom_in = layout.title_in + layout.table_in
    _render_per_radio_stats_table(
        fig, pair_df, value_col=value_col, metric_label=metric_label,
        src_rab=src,
        y_top=1.0 - layout.title_in / fig_h,
        y_bottom=1.0 - table_bottom_in / fig_h,
    )

    if use_antenna_legend and plotted_peer_macs:
        legend_center_y = 1.0 - (table_bottom_in + layout.legend_in / 2) / fig_h
        _render_peer_antenna_legend(
            fig, peer_rab=peer, plotted_macs=plotted_peer_macs,
            color_map=color_map, peer_legend_labels=peer_legend_labels,
            y_center=legend_center_y,
        )

## @brief Create a column of subplots, one row per source-side local MAC.
#
# All subplots share the X axis. Figure height is computed from
# @p height_per_panel per subplot plus @p header_inches for the title block.
#
# @param local_macs      Ordered list of local MAC addresses (one subplot each).
# @param height_per_panel Inches of height per subplot row.
# @param header_inches   Inches reserved above the subplots for the header block.
# @return                ``(Figure, axes_list)`` where ``axes_list`` has one
#                        ``Axes`` per entry in @p local_macs.
def _make_per_radio_axes(local_macs: list[str], height_per_panel: float,
                         header_inches: float = 1.6):
    n = max(len(local_macs), 1)
    fig, axes = plt.subplots(n, 1,
                             figsize=(13, height_per_panel * n + header_inches),
                             sharex=True, squeeze=False)
    return fig, [axes[i][0] for i in range(n)]

## @brief Assign deterministic matplotlib colours to peer MACs for one figure.
#
# Sorts MACs alphabetically for stability and cycles through ``tab10`` (up to
# 10 MACs) or ``tab20`` (more than 10). Colours are stable within one figure
# but may differ across figures.
#
# @param peer_macs  List of peer MAC address strings.
# @return           Mapping from MAC string to matplotlib colour value.
#                   Empty dict if @p peer_macs is empty.
def _peer_mac_colors(peer_macs: list[str]) -> dict[str, str]:
    if not peer_macs:
        return {}
    cmap = plt.get_cmap("tab10" if len(peer_macs) <= 10 else "tab20")
    n = max(cmap.N, 1)
    return {mac: cmap(i % n) for i, mac in enumerate(sorted(peer_macs))}

## @brief Order source-side local MACs by their global radio index, unknowns last.
#
# Reads ``tag_interface`` for each local MAC to look up its 1-based radio
# index via @ref radio_index. MACs with an unknown index are sorted to the
# end with a synthetic key of 10,000.
#
# @param pair_df  DataFrame for this (src, peer) pair.
# @param src_rab  Source rab hostname used for radio index lookup.
# @return         Ordered list of local MAC values.
def _stable_local_mac_order(pair_df: pd.DataFrame, src_rab: str) -> list:
    macs = pair_df["tag_local_mac"].dropna().unique()
    keyed: list[tuple[int, str, object]] = []
    for lm in macs:
        rows = pair_df[pair_df["tag_local_mac"] == lm]
        netdev = _first_str(rows.get("tag_interface"))
        idx = radio_index(src_rab, netdev) if netdev else None
        keyed.append((idx if idx is not None else 10_000, str(lm), lm))
    keyed.sort()
    return [lm for _, _, lm in keyed]

## @brief Compute per-source-radio median and sample count in stable radio-index order.
#
# Groups by ``tag_local_mac``, looks up the netdev and radio index for each,
# computes the median of @p value_col, and sorts by radio index (unknowns last).
#
# @param pair_df   DataFrame for this (src, peer) pair.
# @param value_col Column to compute the median over.
# @param src_rab   Source rab hostname for radio index lookup.
# @return          List of ``(label, median, n)`` tuples in radio-index order.
def _per_radio_stats_rows(pair_df: pd.DataFrame, value_col: str,
                          src_rab: str) -> list[tuple[str, float, int]]:
    rows: list[tuple[int, str, float, int]] = []
    for lm, g in pair_df.groupby("tag_local_mac"):
        netdev = _first_str(g.get("tag_interface"))
        label = _radio_subtitle(src_rab, netdev, lm)
        idx = radio_index(src_rab, netdev) if netdev else None
        median = pd.to_numeric(g[value_col], errors="coerce").median()
        rows.append((
            idx if idx is not None else 10_000,
            label,
            float(median) if pd.notna(median) else float("nan"),
            int(g.shape[0]),
        ))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [(label, median, n) for _, label, median, n in rows]

## @brief Render a two-line centered figure header above the subplots.
#
# Line 1: ``"{src} per radio -> {peer}   ({title_metric})"`` plus a crash
# warning suffix if the scenario is known bad. Line 2: scenario name and
# caption, omitted when @p scenario_name is empty.
#
# @param fig            Figure to annotate.
# @param scenario_name  Scenario identifier; suppresses line 2 if empty.
# @param src            Source rab hostname.
# @param peer           Peer rab hostname.
# @param title_metric   Metric name for the main title line.
# @param df             DataFrame used to compute @ref scenario_caption.
# @param y_main         Normalised Y position for the main title line.
# @param y_scenario     Normalised Y position for the scenario subtitle line.
def _figure_centered_header(fig: plt.Figure, scenario_name: str,
                            src: str, peer: str, title_metric: str,
                            df: pd.DataFrame, y_main: float,
                            y_scenario: float) -> None:
    caption = scenario_caption(df)
    crashed = crashed_suffix(scenario_name)
    main = f"{src} per radio  ->  {peer}   ({title_metric})"
    fig.text(0.5, y_main, main + crashed,
             fontsize=14, ha="center", va="center", fontweight="bold")
    if scenario_name:
        fig.text(0.5, y_scenario, f"scenario: {scenario_name}  ({caption})",
                 fontsize=10, ha="center", va="center",
                 fontweight="bold", color="0.15")

## @brief Render a centered per-radio median/sample-count table above the subplots.
#
# Adds a floating ``Axes`` at the specified Y range, draws a ``matplotlib``
# table with one row per source radio, and applies alternating row shading.
# Does nothing if @ref _per_radio_stats_rows returns no rows.
#
# @param fig          Figure to annotate.
# @param pair_df      DataFrame for this (src, peer) pair.
# @param value_col    Column to compute medians from.
# @param metric_label Column header label for the median column.
# @param src_rab      Source rab hostname for radio ordering.
# @param y_top        Normalised Y coordinate of the table top edge.
# @param y_bottom     Normalised Y coordinate of the table bottom edge.
def _render_per_radio_stats_table(fig: plt.Figure, pair_df: pd.DataFrame,
                                  value_col: str, metric_label: str,
                                  src_rab: str, y_top: float,
                                  y_bottom: float) -> None:
    rows = _per_radio_stats_rows(pair_df, value_col, src_rab)
    if not rows:
        return

    cell_text = []
    for label, median, n in rows:
        med_txt = "n/a" if median != median else f"{median:.2f}"
        cell_text.append([label, med_txt, f"{n:,}"])
    headers = ["Radio", f"Median {metric_label}", "Samples"]

    height = max(y_top - y_bottom, 0.02)
    width = 0.40
    x_left = (1.0 - width) / 2.0
    ax = fig.add_axes([x_left, y_bottom, width, height])
    ax.set_axis_off()
    tbl = ax.table(cellText=cell_text, colLabels=headers,
                   cellLoc="center", loc="center",
                   colWidths=[0.50, 0.28, 0.22])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for col_idx, _h in enumerate(headers):
        cell = tbl[0, col_idx]
        cell.set_text_props(fontweight="bold", color="white", fontsize=8)
        cell.set_facecolor("0.35")
    for row_idx in range(1, len(cell_text) + 1):
        for col_idx in range(len(headers)):
            cell = tbl[row_idx, col_idx]
            cell.set_facecolor("0.97" if row_idx % 2 else "white")
            if col_idx == 0:
                cell.set_text_props(family="monospace", fontsize=8)

## @brief Render a figure-level horizontal legend mapping colour to peer radio label.
#
# Sorts handles by their display label for consistent ordering. Skips MACs
# that were not actually drawn (not in @p plotted_macs). Limited to 6 columns
# to prevent overflow on wide figures.
#
# @param fig                Figure to add the legend to.
# @param peer_rab           Peer rab hostname used in the legend title.
# @param plotted_macs       Set of peer MACs that were drawn in at least one subplot.
# @param color_map          Peer MAC to matplotlib colour mapping.
# @param peer_legend_labels Peer MAC to display label mapping.
# @param y_center           Normalised Y position for the legend center.
def _render_peer_antenna_legend(fig: plt.Figure, peer_rab: str,
                                plotted_macs: set[str],
                                color_map: dict[str, tuple],
                                peer_legend_labels: dict[str, str],
                                y_center: float) -> None:
    macs = [m for m in peer_legend_labels if m in plotted_macs]
    macs.sort(key=lambda m: peer_legend_labels.get(m, m))
    handles: list[Line2D] = []
    for mac in macs:
        handles.append(Line2D(
            [0], [0], marker="o", linestyle="", markersize=6,
            color=color_map.get(mac, "0.4"),
            label=peer_legend_labels[mac],
        ))
    if not handles:
        return
    ncol = min(len(handles), 6)
    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, y_center),
        ncol=ncol,
        fontsize=9,
        framealpha=0.9,
        title=f"Color = peer radio on {peer_rab}",
        title_fontsize=9,
    )

## @brief Build a composite subtitle for one source-radio subplot.
#
# Combines the resolved device radio label (e.g. ``"titan02-mw.3"``), the
# antenna face orientation when known (``"(rear)"``), the raw netdev name,
# and the last 8 characters of the local MAC. Parts are joined with two
# spaces for readability.
#
# @param rab        Source rab hostname.
# @param netdev     Linux netdev name (e.g. ``"wlP2p1s0f0"``).
# @param local_mac  Source-side local MAC address value.
# @return           Composite label string for the subplot title.
def _radio_subtitle(rab: str, netdev: str, local_mac) -> str:
    label = mac_radio_label(local_mac) or radio_label(rab, netdev)
    parts = [label]
    orient = netdev_orientation(rab, netdev) if netdev else None
    if orient:
        parts.append(f"({orient})")
    if netdev:
        parts.append(netdev)
    parts.append(f"...{str(local_mac)[-8:]}")
    return "  ".join(parts)

## @brief Compute per-beam PHY throughput in Mbps from ``field_bytes_tx`` deltas.
#
# Groups by ``(tag_local_mac, tag_sta_mac)``, sorts by ``__sec__``, and
# computes byte-count differences between consecutive samples. Negative deltas
# (counter resets) are dropped via ``where(db >= 0)``. Rate is computed as
# ``(delta_bytes / delta_seconds) * 8 / 1e6``. Groups with fewer than 2 samples
# are skipped.
#
# @param pair_df  DataFrame for one (src, peer) pair.
# @return         DataFrame with ``__mbps__``, ``__dt__``, and ``__db__`` columns
#                 appended, with NaN ``__mbps__`` rows dropped. Empty if no
#                 valid rate samples remain.
def _per_pair_rate_mbps(pair_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    grouped = pair_df.groupby(["tag_local_mac", "tag_sta_mac"], sort=False)
    for (lmac, smac), g in grouped:
        if len(g) < 2:
            continue
        g = g.sort_values("__sec__").copy()
        dt = g["__sec__"].diff()
        db = g["field_bytes_tx"].diff()
        rate_bps = (db / dt).where(db >= 0) * 8.0
        g["__dt__"] = dt
        g["__db__"] = db
        g["__mbps__"] = rate_bps / 1e6
        out.append(g)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True).dropna(subset=["__mbps__"])

## @brief Return the first non-null string value from a Series, or an empty string.
#
# Used to extract the ``tag_interface`` netdev name for a group of rows where
# all rows are expected to share the same value. Handles ``None`` input and
# empty Series without raising.
#
# @param series  A pandas Series or ``None``.
# @return        First non-null string value, or ``""`` if none exists.
def _first_str(series) -> str:
    if series is None:
        return ""
    try:
        s = series.dropna()
    except AttributeError:
        return ""
    if s.empty:
        return ""
    return str(s.iloc[0])