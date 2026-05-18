##@package docstring
# Per-radio bh2 plots: one figure per (src_rab, peer_rab), one subplot per local MAC.

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

## Documentation for a function.
#
#  More details.
def plot_bh2_snr(df: pd.DataFrame, scenario_name: str = "") -> list[_BH2_PairResult]:
    ##Raw SNR per (source radio, peer antenna).##
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

## Documentation for a function.
#
#  More details.
def plot_bh2_rcpi(df: pd.DataFrame, scenario_name: str = "") -> list[_BH2_PairResult]:
    ##Raw RCPI (received signal power, dBm) per (source radio, peer antenna).##
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

## Documentation for a function.
#
#  More details.
def plot_bh2_mcs(df: pd.DataFrame, scenario_name: str = "") -> list[_BH2_PairResult]:
    ##Raw MCS-TX index per (source radio, peer antenna).##
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

## Documentation for a function.
#
#  More details.
def plot_bh2_per(df: pd.DataFrame, scenario_name: str = "") -> list[_BH2_PairResult]:
    ##Raw packet-error counter per (source radio, peer antenna).##
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

## Documentation for a function.
#
#  More details.
def plot_bh2_throughput(df: pd.DataFrame, scenario_name: str = "") -> list[_BH2_PairResult]:
    ##PHY rate per beam-pair from ``field_bytes_tx`` deltas; drops counter resets.##
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

## @brief
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
    ##Shared body for SNR/RCPI/MCS/PER -- everything except throughput.##
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

## @brief
def _build_peer_label_map(peer_macs: list[str]) -> dict[str, str]:
    ##``{peer_mac: 'rab2.3'}`` resolved through the global topology cache.##
    out: dict[str, str] = {}
    for mac in peer_macs:
        label = mac_radio_label(mac)
        if label and ".?" not in label:
            out[str(mac)] = label
    return out

## Documentation for a class.
#
#  More details.
@dataclass(frozen=True)
class _HeaderLayout:
    ##Inch-based heights for the title/table/legend block above the subplots.##
    title_in: float
    table_in: float
    legend_in: float
    total_in: float

## @brief
def _compute_header_layout(n_macs: int, use_antenna_legend: bool) -> _HeaderLayout:
    title_in = 0.65
    table_in = 0.18 * (n_macs + 1) + 0.10
    legend_in = 0.45 if use_antenna_legend else 0.0
    total_in = title_in + table_in + legend_in + 0.30
    return _HeaderLayout(title_in, table_in, legend_in, total_in)

## @brief
def _build_peer_legend_labels(peer_macs: list[str], peer_rab: str,
                              peer_label_map: dict[str, str]) -> dict[str, str]:
    ##{peer_mac: 'rab2.3'} when known, else '{peer} ...MAC' fallback.##
    out: dict[str, str] = {}
    for mac in peer_macs:
        canonical = peer_label_map.get(mac)
        out[mac] = canonical if canonical else f"{peer_rab} ...{str(mac)[-8:]}"
    return out

## @brief
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
    ##Plot one source-MAC subplot. Returns (trace rows, peer MACs that were drawn).##
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

## @brief
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
    ##Apply MCS reference lines, axis limits, title/grid/legend annotations.##
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

## @brief
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
    ##One (src_rab, peer_rab) figure with one subplot per source local_mac.##
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

## @brief
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
    ##Apply tight_layout + render the centered header, stats table, and legend.##
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

## @brief
def _make_per_radio_axes(local_macs: list[str], height_per_panel: float,
                         header_inches: float = 1.6):
    ##One column of subplots, one row per source-side local MAC.##
    n = max(len(local_macs), 1)
    fig, axes = plt.subplots(n, 1,
                             figsize=(13, height_per_panel * n + header_inches),
                             sharex=True, squeeze=False)
    return fig, [axes[i][0] for i in range(n)]

## @brief
def _peer_mac_colors(peer_macs: list[str]) -> dict[str, str]:
    ##Deterministic peer-MAC -> matplotlib color, stable within one figure.##
    if not peer_macs:
        return {}
    cmap = plt.get_cmap("tab10" if len(peer_macs) <= 10 else "tab20")
    n = max(cmap.N, 1)
    return {mac: cmap(i % n) for i, mac in enumerate(sorted(peer_macs))}

## @brief
def _stable_local_mac_order(pair_df: pd.DataFrame, src_rab: str) -> list:
    ##Source-side local MACs ordered by global radio index; unknowns at the end.##
    macs = pair_df["tag_local_mac"].dropna().unique()
    keyed: list[tuple[int, str, object]] = []
    for lm in macs:
        rows = pair_df[pair_df["tag_local_mac"] == lm]
        netdev = _first_str(rows.get("tag_interface"))
        idx = radio_index(src_rab, netdev) if netdev else None
        keyed.append((idx if idx is not None else 10_000, str(lm), lm))
    keyed.sort()
    return [lm for _, _, lm in keyed]

## @brief
def _per_radio_stats_rows(pair_df: pd.DataFrame, value_col: str,
                          src_rab: str) -> list[tuple[str, float, int]]:
    ##Per-source-radio ``(label, median, n)`` in stable radio-index order.##
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

## @brief
def _figure_centered_header(fig: plt.Figure, scenario_name: str,
                            src: str, peer: str, title_metric: str,
                            df: pd.DataFrame, y_main: float,
                            y_scenario: float) -> None:
    ##Two-line centered header at caller-supplied y offsets.##
    caption = scenario_caption(df)
    crashed = crashed_suffix(scenario_name)
    main = f"{src} per radio  ->  {peer}   ({title_metric})"
    fig.text(0.5, y_main, main + crashed,
             fontsize=14, ha="center", va="center", fontweight="bold")
    if scenario_name:
        fig.text(0.5, y_scenario, f"scenario: {scenario_name}  ({caption})",
                 fontsize=10, ha="center", va="center",
                 fontweight="bold", color="0.15")

## @brief
def _render_per_radio_stats_table(fig: plt.Figure, pair_df: pd.DataFrame,
                                  value_col: str, metric_label: str,
                                  src_rab: str, y_top: float,
                                  y_bottom: float) -> None:
    ##Centered per-source-radio median/n table above the subplots.##
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

## @brief
def _render_peer_antenna_legend(fig: plt.Figure, peer_rab: str,
                                plotted_macs: set[str],
                                color_map: dict[str, tuple],
                                peer_legend_labels: dict[str, str],
                                y_center: float) -> None:
    ##Single figure-level legend mapping color -> peer radio.##
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

## @brief
def _radio_subtitle(rab: str, netdev: str, local_mac) -> str:
    ##``rab2.3 (rear)  wlP2p1s0f0  ...02:30`` style label.##
    parts = [radio_label(rab, netdev)]
    orient = netdev_orientation(rab, netdev) if netdev else None
    if orient:
        parts.append(f"({orient})")
    if netdev:
        parts.append(netdev)
    parts.append(f"...{str(local_mac)[-8:]}")
    return "  ".join(parts)

## @brief
def _per_pair_rate_mbps(pair_df: pd.DataFrame) -> pd.DataFrame:
    ##PHY rate per (tag_local_mac, tag_sta_mac); drops counter resets only.##
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

## @brief
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
