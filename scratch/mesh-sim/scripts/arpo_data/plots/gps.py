##@package docstring
# GPS track plot for one scenario.

##

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from .common import crashed_suffix, scenario_caption

## Documentation for a function.
#
#  More details.
def plot_gps_tracks(df: pd.DataFrame, scenario_name: str = "") -> tuple[plt.Figure, pd.DataFrame] | None:
    ##Node positions in metres from the centroid; color = time since start.##
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
    legend_handles: list[Line2D] = []
    for i, (node, g) in enumerate(df.groupby("__node__")):
        t = (g["__t__"] - t_origin).dt.total_seconds()
        marker = markers[i % len(markers)]
        op_label = g["__label__"].iloc[0]
        legend_label = node if str(op_label) == str(node) else f"{node} ({op_label})"
        sc = ax.scatter(
            g["__x_m__"], g["__y_m__"], c=t, s=10, alpha=0.85,
            marker=marker, edgecolors="black", linewidths=0.2,
            cmap="viridis",
        )
        legend_handles.append(Line2D(
            [0], [0], marker=marker, linestyle="", markersize=7,
            color="0.35", markeredgecolor="black", markeredgewidth=0.4,
            label=legend_label,
        ))

    ax.set_xlabel("east of centroid (m)", fontweight="bold")
    ax.set_ylabel("north of centroid (m)", fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="seconds since scenario start", shrink=0.6)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=8, framealpha=0.85, title="device",
                  title_fontsize=8)

    _add_scale_bar(ax)

    caption = scenario_caption(df)
    crashed = crashed_suffix(scenario_name)
    main = f"GPS tracks   (origin {lat0:.5f}, {lon0:.5f})"
    fig.text(0.5, 0.975, main + crashed,
             fontsize=14, ha="center", va="center", fontweight="bold")
    if scenario_name:
        fig.text(0.5, 0.945, f"scenario: {scenario_name}  ({caption})",
                 fontsize=10, ha="center", va="center",
                 fontweight="bold", color="0.15")
    fig.tight_layout(rect=(0, 0, 1, 0.92))

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

## @brief
def _add_scale_bar(ax: plt.Axes) -> None:
    ##Horizontal scale bar in the lower-left, length rounded to a tidy value.##
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span = x1 - x0
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
