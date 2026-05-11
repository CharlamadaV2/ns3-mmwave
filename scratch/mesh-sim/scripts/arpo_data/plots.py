"""
Quicklook plot functions for one scenario.

Each function takes a loaded DataFrame and returns either a
``matplotlib.Figure`` or a ``(Figure, trace_df)`` tuple, or ``None`` if
there's nothing to plot. ``cli.py`` is the only place that saves and
closes figures; it also writes the trace CSV alongside the PNG.

Design conventions for these plots (see PLOTS.md for the reading guide):

- Stable peer color across plots, via ``topology.node_color``.
- bh2 plots use small multiples: one row per source rab. Within a row
  every drawn line goes to a distinct peer, so peer-keyed colors never
  collide on the same axes.
- Each plot title carries the scenario name + node count + duration so a
  loose PNG is self-describing.
- Noisy time series get a 30 s (or 5 s for throughput) rolling median
  drawn over a faint scatter -- the scatter shows variance, the line
  shows the trend.
- Filter dominant peers first (drop one-shot associations that would
  otherwise add 20 noise series to a legend).
- Each plot also returns a "trace" DataFrame that contains the exact
  (sec, source, peer, raw, smoothed) values it drew, so a reader can
  grep any visible point back to the underlying CSV.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .topology import node_color


def filter_dominant_peers(df: pd.DataFrame, peer_col: str = "tag_sta_mac",
                          min_frac: float = 0.05, top_k: int = 3) -> pd.DataFrame:
    """
    Drop peers that are < ``min_frac`` of a node's samples (and aren't in
    the top-k). Cuts noisy single-packet associations so plots stay readable.
    """
    keep: set[tuple[str, str]] = set()
    for node, g in df.groupby("__node__"):
        counts = g[peer_col].value_counts(dropna=True)
        if counts.empty:
            continue
        thresh = max(int(len(g) * min_frac), 1)
        for peer, n in counts.items():
            if n >= thresh:
                keep.add((str(node), str(peer)))
        for peer in counts.head(top_k).index:
            keep.add((str(node), str(peer)))

    keys = df["__node__"].astype(str) + "|" + df[peer_col].astype(str)
    keep_keys = {f"{n}|{p}" for n, p in keep}
    return df[keys.isin(keep_keys)].copy()


# ---------------------------------------------------------------------------
# bh2: SNR / MCS / throughput  (one subplot row per source rab)
# ---------------------------------------------------------------------------

def plot_bh2_snr(df: pd.DataFrame, scenario_name: str = "") -> tuple[plt.Figure, pd.DataFrame] | None:
    """
    SNR per peer for each source rab, stacked vertically.

    For each (source, peer) the raw multi-antenna samples are collapsed
    to *max SNR per 1 s bin* -- the simulator emits one SINR per link
    per tick, and the field-data analog is the antenna with the best
    signal. Faint scatter = the surviving max-per-bin samples; bold
    line = 30 s rolling median. Reference horizontal lines anchor SNR
    to typical MCS thresholds.
    """
    sub = filter_dominant_peers(df.dropna(subset=["field_snr"]))
    if sub.empty:
        return None

    sources = sorted(sub["__node__"].astype(str).unique())
    fig, axes = _make_per_source_axes(sources, height_per_panel=2.6)
    trace_rows: list[pd.DataFrame] = []

    for ax, src in zip(axes, sources):
        node_df = sub[sub["__node__"] == src]
        for peer, g in node_df.groupby("__peer__"):
            g = _max_across_macs_per_bin(g, "field_snr")
            if g.empty:
                continue
            color = node_color(peer)
            ax.plot(g["__sec__"], g["field_snr"], ".", markersize=1.6,
                    alpha=0.12, color=color)
            smooth = _rolling_median_seconds(g, "field_snr", "30s")
            ax.plot(g["__sec__"], smooth, "-", linewidth=1.6, alpha=0.9,
                    color=color, label=f"-> {peer}")
            trace_rows.append(pd.DataFrame({
                "sec": g["__sec__"].values,
                "source": src,
                "peer": peer,
                "tag_sta_mac": g["tag_sta_mac"].values,
                "raw_snr_db": g["field_snr"].values,
                "smoothed_snr_db_30s": smooth,
            }))

        for thr, lbl in [(5, "MCS 0"), (10, "MCS 4"), (15, "MCS 8"), (20, "MCS 12")]:
            ax.axhline(thr, color="0.7", linestyle=":", linewidth=0.6, zorder=0)
            ax.text(1.0, thr, f" {lbl}", transform=ax.get_yaxis_transform(),
                    color="0.55", va="center", fontsize=7, alpha=0.8)
        ax.set_ylabel(f"{src}\nSNR (dB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.85)

    axes[-1].set_xlabel("seconds since each node's session start")
    _figure_suptitle(
        fig, scenario_name, "bh2 SNR per source rab", sub,
        "rows = source rab; line color = peer (PLOTS.md table); "
        "max SNR across antennas per 1 s bin, then 30 s rolling median; "
        "dotted lines = approx. MCS SNR thresholds",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, _concat_trace(trace_rows)


def plot_bh2_mcs(df: pd.DataFrame, scenario_name: str = "") -> tuple[plt.Figure, pd.DataFrame] | None:
    """
    MCS index per peer for each source rab, stacked vertically.

    MCS is a discrete integer index; rate adaptation steps it up/down in
    response to SNR. Multi-antenna samples are collapsed to *max MCS
    per 1 s bin* per (source, peer). The 30 s rolling median is drawn
    as a step line.
    """
    if "field_mcs_tx" not in df.columns:
        return None
    sub = filter_dominant_peers(df.dropna(subset=["field_mcs_tx"]))
    if sub.empty:
        return None

    sources = sorted(sub["__node__"].astype(str).unique())
    fig, axes = _make_per_source_axes(sources, height_per_panel=2.6)
    trace_rows: list[pd.DataFrame] = []
    y_max = max(13, sub["field_mcs_tx"].max() + 0.5)

    for ax, src in zip(axes, sources):
        node_df = sub[sub["__node__"] == src]
        for peer, g in node_df.groupby("__peer__"):
            g = _max_across_macs_per_bin(g, "field_mcs_tx")
            if g.empty:
                continue
            color = node_color(peer)
            ax.plot(g["__sec__"], g["field_mcs_tx"], ".", markersize=1.4,
                    alpha=0.10, color=color)
            smooth = _rolling_median_seconds(g, "field_mcs_tx", "30s")
            ax.step(g["__sec__"], smooth, where="post", linewidth=1.4,
                    alpha=0.9, color=color, label=f"-> {peer}")
            trace_rows.append(pd.DataFrame({
                "sec": g["__sec__"].values,
                "source": src,
                "peer": peer,
                "tag_sta_mac": g["tag_sta_mac"].values,
                "raw_mcs": g["field_mcs_tx"].values,
                "smoothed_mcs_30s": smooth,
            }))

        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylim(-0.5, y_max)
        ax.set_ylabel(f"{src}\nMCS")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.85)

    axes[-1].set_xlabel("seconds since each node's session start")
    _figure_suptitle(
        fig, scenario_name, "bh2 MCS index per source rab", sub,
        "rows = source rab; max MCS across antennas per 1 s bin, then "
        "30 s rolling-median step (rate-adaptation behavior)",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, _concat_trace(trace_rows)


def plot_bh2_throughput(df: pd.DataFrame, scenario_name: str = "") -> tuple[plt.Figure, pd.DataFrame] | None:
    """
    Per-link PHY rate per source rab, derived from field_bytes_tx deltas.

    field_bytes_tx is a *cumulative* counter that resets at session
    boundaries. Each antenna has its own counter, so the rate is
    computed *per (source, peer, MAC)* before collapsing across
    antennas. Per-sample filters:

    - dt < 0.1 s   -> timestamp jitter, drop the sample
    - dt > 2.0 s   -> session gap, drop
    - db < 0       -> counter reset, drop
    - rate > 6 Gbps -> physically implausible for these radios, drop

    Multi-antenna samples are then collapsed to *max rate per 1 s bin*
    per (source, peer) -- traffic for one mmWave link rides the
    best-aimed antenna at any moment. Result smoothed with a 30 s
    rolling median.

    Note: idle links show 0 even when healthy; cross-check bh2_snr.png
    to distinguish "no traffic" from "broken link".
    """
    if "field_bytes_tx" not in df.columns:
        return None
    sub = filter_dominant_peers(df.dropna(subset=["field_bytes_tx"]))
    if sub.empty:
        return None

    sources = sorted(sub["__node__"].astype(str).unique())
    fig, axes = _make_per_source_axes(sources, height_per_panel=2.6)
    trace_rows: list[pd.DataFrame] = []

    for ax, src in zip(axes, sources):
        node_df = sub[sub["__node__"] == src]
        for peer, peer_g in node_df.groupby("__peer__"):
            per_mac = _per_mac_rate_mbps(peer_g)
            if per_mac.empty:
                continue
            collapsed = _max_across_macs_per_bin(per_mac, "__mbps__")
            if collapsed.empty:
                continue
            smooth = _rolling_median_seconds(collapsed, "__mbps__", "30s")
            color = node_color(peer)
            ax.plot(collapsed["__sec__"], smooth, "-", linewidth=1.4,
                    alpha=0.9, color=color, label=f"-> {peer}")
            trace_rows.append(pd.DataFrame({
                "sec": collapsed["__sec__"].values,
                "source": src,
                "peer": peer,
                "tag_sta_mac": collapsed["tag_sta_mac"].values,
                "raw_bytes_tx": collapsed["field_bytes_tx"].values,
                "dt_s": collapsed["__dt__"].values,
                "db_bytes": collapsed["__db__"].values,
                "raw_mbps": collapsed["__mbps__"].values,
                "smoothed_mbps_30s": smooth,
            }))

        ax.set_ylabel(f"{src}\nPHY (Mbps)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.85)

    axes[-1].set_xlabel("seconds since each node's session start")
    _figure_suptitle(
        fig, scenario_name, "bh2 PHY rate per source rab", sub,
        "rows = source rab; per-antenna bytes_tx deltas, then max rate "
        "across antennas per 1 s bin, then 30 s rolling median; "
        "counter resets and <100 ms / >2 s / >6 Gbps samples filtered out",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, _concat_trace(trace_rows)


# ---------------------------------------------------------------------------
# ping: small multiples per target
# ---------------------------------------------------------------------------

def plot_ping_latency(df: pd.DataFrame, scenario_name: str = "") -> tuple[plt.Figure, pd.DataFrame] | None:
    """
    Ping latency, one panel per target IP.

    Each panel overlays per-source rolling-median lines on faint raw
    samples. Log-scale y because latency spans 3 decades on these links.
    """
    if "field_average_response_ms" not in df.columns:
        return None
    sub = df.dropna(subset=["field_average_response_ms"])
    if sub.empty:
        return None

    if "tag_url" not in sub.columns:
        return None
    targets = sorted(sub["tag_url"].dropna().astype(str).unique())
    n = len(targets)
    if n == 0:
        return None
    cols = min(2, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 3.0 * rows + 0.6),
                             sharex=True, sharey=True, squeeze=False)

    sources = sorted(sub["__node__"].astype(str).unique())
    src_color = {s: node_color(s) for s in sources}
    trace_rows: list[pd.DataFrame] = []

    for i, target in enumerate(targets):
        ax = axes[i // cols][i % cols]
        for src in sources:
            g = sub[(sub["tag_url"] == target) & (sub["__node__"] == src)]
            if g.empty:
                continue
            g = g.sort_values("__sec__")
            color = src_color[src]
            ax.plot(g["__sec__"], g["field_average_response_ms"], ".",
                    markersize=1.2, alpha=0.10, color=color)
            smooth = _rolling_median_seconds(g, "field_average_response_ms", "30s")
            ax.plot(g["__sec__"], smooth, "-", linewidth=1.4, alpha=0.9,
                    color=color, label=src)
            trace_rows.append(pd.DataFrame({
                "sec": g["__sec__"].values,
                "source": src,
                "target": target,
                "raw_ms": g["field_average_response_ms"].values,
                "smoothed_ms_30s": smooth,
            }))
        ax.set_yscale("log")
        ax.set_title(f"target = {target}", fontsize=9)
        ax.grid(True, which="both", alpha=0.25)
        if i // cols == rows - 1:
            ax.set_xlabel("seconds since each node's session start")
        if i % cols == 0:
            ax.set_ylabel("avg response (ms)")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    _figure_suptitle(fig, scenario_name, "ping latency by target",
                     sub, "30 s rolling median per source rab; log y; "
                          "one panel per target IP")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, _concat_trace(trace_rows)


# ---------------------------------------------------------------------------
# GPS
# ---------------------------------------------------------------------------

def plot_gps_tracks(df: pd.DataFrame, scenario_name: str = "") -> tuple[plt.Figure, pd.DataFrame] | None:
    """
    Node positions in meters from the centroid of all observations.

    Lat/lon are reprojected via the equirectangular approximation (valid
    over <~10 km extents). Marker shape distinguishes nodes; color = time.
    A scale bar shows ground distance.
    """
    if df.empty:
        return None

    lat0 = df["__lat__"].mean()
    lon0 = df["__lon__"].mean()
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    df = df.copy()
    df["__y_m__"] = (df["__lat__"] - lat0) * m_per_deg_lat
    df["__x_m__"] = (df["__lon__"] - lon0) * m_per_deg_lon

    fig, ax = plt.subplots(figsize=(8, 8))
    t_origin = df["__t__"].min()
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    sc = None
    for i, (node, g) in enumerate(df.groupby("__node__")):
        t = (g["__t__"] - t_origin).dt.total_seconds()
        op_label = g["__label__"].iloc[0]
        legend_label = node if str(op_label) == str(node) else f"{node} ({op_label})"
        sc = ax.scatter(
            g["__x_m__"], g["__y_m__"], c=t, s=22, alpha=0.85,
            marker=markers[i % len(markers)], edgecolors="black", linewidths=0.3,
            label=legend_label,
        )

    ax.set_xlabel("east of centroid (m)")
    ax.set_ylabel("north of centroid (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="seconds since scenario start", shrink=0.6)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    _add_scale_bar(ax)
    _set_title(
        ax, scenario_name,
        f"GPS tracks  (origin {lat0:.5f}, {lon0:.5f})",
        df, "marker shape = device; color = seconds since scenario start; "
            "axes in meters from centroid",
    )
    fig.tight_layout()

    trace = pd.DataFrame({
        "label": df["__label__"].values,
        "node": df["__node__"].values,
        "t_utc": df["__t__"].values,
        "sec_since_origin": (df["__t__"] - t_origin).dt.total_seconds().values,
        "lat_deg": df["__lat__"].values,
        "lon_deg": df["__lon__"].values,
        "east_m": df["__x_m__"].values,
        "north_m": df["__y_m__"].values,
    })
    return fig, trace


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_per_source_axes(sources: list[str], height_per_panel: float):
    """One column of subplots, one row per source rab. Returns (fig, axes-list)."""
    n = max(len(sources), 1)
    fig, axes = plt.subplots(n, 1, figsize=(13, height_per_panel * n + 1.2),
                             sharex=True, squeeze=False)
    return fig, [axes[i][0] for i in range(n)]


def _max_across_macs_per_bin(g: pd.DataFrame, value_col: str,
                             bin_seconds: float = 1.0) -> pd.DataFrame:
    """
    For each ``bin_seconds`` window, keep the row with the max ``value_col``.

    A peer label can cover up to 4 MAC interfaces of one physical radio.
    This collapses them to a single "best antenna at this moment" series
    so downstream plotting/smoothing sees one sample per bin -- comparable
    to the simulator's per-tick per-link SINR/capacity. The trace caller
    keeps ``tag_sta_mac`` so the chosen antenna is recoverable.
    """
    if g.empty:
        return g
    valid = g.dropna(subset=[value_col])
    if valid.empty:
        return valid
    bin_idx = (valid["__sec__"] // bin_seconds).astype("int64")
    keep = valid.groupby(bin_idx)[value_col].idxmax()
    return valid.loc[keep].sort_values("__sec__")


def _per_mac_rate_mbps(peer_g: pd.DataFrame) -> pd.DataFrame:
    """
    Compute PHY rate (Mbps) per (peer, tag_sta_mac), one cumulative
    counter per antenna. Output keeps ``__sec__``, ``tag_sta_mac``,
    ``field_bytes_tx``, ``__dt__``, ``__db__``, ``__mbps__`` so the
    caller can collapse across antennas and emit a faithful trace.
    """
    out = []
    for mac, g in peer_g.groupby("tag_sta_mac"):
        if len(g) < 4:
            continue
        g = g.sort_values("__sec__").copy()
        dt = g["__sec__"].diff()
        db = g["field_bytes_tx"].diff()
        good = (dt >= 0.1) & (dt <= 2.0) & (db >= 0)
        rate_bps = (db / dt).where(good) * 8.0
        rate_bps = rate_bps.where(rate_bps <= 6e9)  # 6 Gbps cap
        g["__dt__"] = dt
        g["__db__"] = db
        g["__mbps__"] = rate_bps / 1e6
        out.append(g)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def _concat_trace(rows: list[pd.DataFrame]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _rolling_median_seconds(g: pd.DataFrame, value_col: str, window: str) -> np.ndarray:
    """
    Time-based rolling median of ``value_col`` per (already-grouped) frame.

    *g* must be sorted by ``__sec__``. Returns a numpy array aligned with
    g's row order (so callers can plot vs. ``g['__sec__']``).
    """
    if g.empty:
        return np.array([])
    idx = pd.to_timedelta(g["__sec__"].values, unit="s")
    s = pd.Series(g[value_col].values, index=idx)
    # Rolling requires a strictly monotonic index; jitter equal indices.
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    return s.rolling(window, min_periods=1).median().reindex(idx, method="nearest").values


def _scenario_caption(df: pd.DataFrame) -> str:
    """e.g. '3 rabs, 9.7 ks'."""
    n_nodes = df["__node__"].nunique() if "__node__" in df.columns else 0
    if "__sec__" in df.columns and not df["__sec__"].empty:
        duration_s = float(df["__sec__"].max())
    elif "__t__" in df.columns and not df["__t__"].empty:
        duration_s = (df["__t__"].max() - df["__t__"].min()).total_seconds()
    else:
        duration_s = 0.0
    if duration_s >= 1000:
        dur = f"{duration_s / 1000:.1f} ks"
    else:
        dur = f"{duration_s:.0f} s"
    label = "rabs" if n_nodes != 1 else "rab"
    return f"{n_nodes} {label}, {dur}"


def _set_title(ax: plt.Axes, scenario: str, title: str,
               df: pd.DataFrame, subtitle: str) -> None:
    head = f"{scenario}  --  {title}  ({_scenario_caption(df)})" if scenario else \
           f"{title}  ({_scenario_caption(df)})"
    ax.set_title(head, fontsize=11, loc="left", pad=18)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=8, color="0.4", va="bottom")


def _figure_suptitle(fig: plt.Figure, scenario: str, title: str,
                     df: pd.DataFrame, subtitle: str) -> None:
    head = f"{scenario}  --  {title}  ({_scenario_caption(df)})" if scenario else \
           f"{title}  ({_scenario_caption(df)})"
    fig.suptitle(head, fontsize=12, x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.965, subtitle, fontsize=8, color="0.4", ha="left")


def _add_scale_bar(ax: plt.Axes) -> None:
    """
    Draw a horizontal scale bar in the lower-left corner, length set to
    a tidy fraction of the axes width."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span = x1 - x0
    # Pick a "tidy" length: 1, 2, 5 * 10^k that's roughly 20% of the span.
    raw = span * 0.2
    exp = math.floor(math.log10(max(raw, 1e-6)))
    base = raw / (10 ** exp)
    nice = 1 if base < 2 else 2 if base < 5 else 5
    length = nice * (10 ** exp)
    bar_x = x0 + 0.05 * span
    bar_y = y0 + 0.05 * (y1 - y0)
    ax.plot([bar_x, bar_x + length], [bar_y, bar_y], color="black", linewidth=2.5)
    label = f"{length:g} m" if length >= 1 else f"{length * 100:g} cm"
    ax.text(bar_x + length / 2, bar_y + 0.015 * (y1 - y0), label,
            ha="center", va="bottom", fontsize=8)
