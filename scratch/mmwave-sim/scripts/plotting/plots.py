"""Plot functions for mmwave-sim outputs.

Each function takes data and returns a matplotlib Figure.
No function calls plt.show() or fig.savefig() — the caller handles that.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _metric_label(metric: str) -> str:
    """Human-readable axis label for a metric key."""
    labels = {
        "mean_sinr_db": "Mean SINR (dB)",
        "min_sinr_db": "Min SINR (dB)",
        "max_sinr_db": "Max SINR (dB)",
        "corruption_rate": "Corruption Rate",
        "los_fraction": "LOS Fraction",
        "dl_throughput_mbps": "DL Throughput (Mbps)",
        "sum_dl_throughput_mbps": "Sum DL Throughput (Mbps)",
        "dl_delay_mean_ms": "DL Delay (ms)",
        "wall_elapsed_s": "Wall-Clock Time (s)",
    }
    return labels.get(metric, metric)


# ---------------------------------------------------------------------------
# 1. Per-UE bar chart (single seed)
# ---------------------------------------------------------------------------

def plot_per_ue_bars(summary: dict, metric: str,
                     title: str | None = None) -> plt.Figure:
    """Bar chart of one metric per UE from a single-seed summary."""
    per_ue = summary.get("per_ue", {})
    ue_ids = sorted(per_ue.keys())
    values = [per_ue[uid].get(metric) for uid in ue_ids]

    fig, ax = plt.subplots(figsize=(max(4, len(ue_ids) * 1.2), 4))
    x = np.arange(len(ue_ids))
    bars = ax.bar(x, values, color=_COLORS[0], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(ue_ids, rotation=45, ha="right")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(title or f"{_metric_label(metric)} per UE (seed={summary.get('seed')})")

    # Value labels on bars
    for bar, val in zip(bars, values):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Sweep bar chart with error bars (multi-seed)
# ---------------------------------------------------------------------------

def plot_sweep_bars(agg: dict, metric: str,
                    title: str | None = None) -> plt.Figure:
    """Bar chart with error bars (CI if available, else std) per UE."""
    per_ue = agg.get("per_ue", {})
    ue_ids = sorted(per_ue.keys())

    means = []
    errors = []
    for uid in ue_ids:
        stats = per_ue[uid].get(metric, {})
        means.append(stats.get("mean"))
        # Prefer CI if available, fall back to std
        if "ci95" in stats and stats["ci95"] is not None:
            errors.append(stats["ci95"])
        else:
            errors.append(stats.get("std"))

    fig, ax = plt.subplots(figsize=(max(4, len(ue_ids) * 1.2), 4))
    x = np.arange(len(ue_ids))

    ax.bar(x, means, yerr=errors, capsize=4,
           color=_COLORS[0], edgecolor="white", alpha=0.85,
           error_kw={"linewidth": 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(ue_ids, rotation=45, ha="right")
    ax.set_ylabel(_metric_label(metric))

    n_seeds = agg.get("num_seeds", "?")
    err_type = "95% CI" if any("ci95" in per_ue[uid].get(metric, {})
                               for uid in ue_ids) else "std"
    ax.set_title(title or f"{_metric_label(metric)} per UE "
                 f"({n_seeds} seeds, {err_type})")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. SINR time series from links.csv
# ---------------------------------------------------------------------------

def plot_sinr_timeseries(links_df: pd.DataFrame,
                         title: str | None = None) -> plt.Figure:
    """Line plot of mean SINR vs simulation time, averaged across seeds.

    One line per UE-eNB link. If multiple seeds are present, SINR is averaged
    at each time step.
    """
    # Filter out placeholder SINR values
    df = links_df[links_df["sinr_db"] > -900].copy()

    # Aggregate across seeds at each (time, link) point
    group_cols = ["time_s", "node_a", "node_b"]
    avg = df.groupby(group_cols, as_index=False)["sinr_db"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    pairs = avg.groupby(["node_a", "node_b"])
    for i, ((node_a, node_b), group) in enumerate(pairs):
        color = _COLORS[i % len(_COLORS)]
        group = group.sort_values("time_s")

        ax.plot(group["time_s"], group["sinr_db"],
                label=f"{node_a}-{node_b}",
                color=color, linewidth=1.0, alpha=0.8)

    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("SINR (dB)")
    ax.set_title(title or "SINR Time Series")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Throughput time series from DlPdcpStats
# ---------------------------------------------------------------------------

def plot_throughput_timeseries(pdcp_df: pd.DataFrame,
                               title: str | None = None) -> plt.Figure:
    """Line plot of DL throughput vs time, one line per UE (IMSI), averaged
    across seeds."""
    df = pdcp_df.copy()
    # Use midpoint of each time window
    df["time_mid"] = (df["start"] + df["end"]) / 2

    avg = df.groupby(["time_mid", "IMSI"], as_index=False)["throughput_mbps"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (imsi, group) in enumerate(avg.groupby("IMSI")):
        color = _COLORS[i % len(_COLORS)]
        group = group.sort_values("time_mid")
        ax.plot(group["time_mid"], group["throughput_mbps"],
                label=f"UE {imsi}", color=color, linewidth=1.0, alpha=0.8)

    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("DL Throughput (Mbps)")
    ax.set_title(title or "DL Throughput Time Series")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. MCS time series from RxPacketTrace
# ---------------------------------------------------------------------------

def plot_mcs_timeseries(rx_df: pd.DataFrame,
                        title: str | None = None) -> plt.Figure:
    """Line plot of MCS vs time, one line per UE (RNTI), averaged across seeds."""
    avg = rx_df.groupby(["time", "rnti"], as_index=False)["mcs"].mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    _LINE_STYLES = ["-", "--", ":", "-."]
    for i, (rnti, group) in enumerate(avg.groupby("rnti")):
        color = _COLORS[i % len(_COLORS)]
        ls = _LINE_STYLES[i % len(_LINE_STYLES)]
        group = group.sort_values("time")
        ax.plot(group["time"], group["mcs"],
                label=f"RNTI {rnti}", color=color, linewidth=1.2,
                alpha=0.8, linestyle=ls)

    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("MCS Index")
    ax.set_title(title or "MCS Time Series")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Simulation runtime per seed
# ---------------------------------------------------------------------------

def plot_sim_runtime(summaries: list[dict],
                     title: str | None = None) -> plt.Figure:
    """Bar chart of wall-clock runtime per seed with mean line."""
    seeds = []
    runtimes = []
    for s in summaries:
        if "wall_elapsed_s" in s:
            seeds.append(str(s.get("seed", "?")))
            runtimes.append(s["wall_elapsed_s"])

    if not runtimes:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "No timing data available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(max(4, len(seeds) * 0.8), 4))
    x = np.arange(len(seeds))

    ax.bar(x, runtimes, color=_COLORS[1], edgecolor="white", alpha=0.85)

    mean_rt = sum(runtimes) / len(runtimes)
    ax.axhline(mean_rt, color=_COLORS[3], linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_rt:.1f}s")

    # Pad y-axis so value labels don't overlap the mean line
    y_max = max(runtimes)
    ax.set_ylim(0, y_max * 1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds], rotation=45, ha="right")
    ax.set_ylabel("Wall-Clock Time (s)")
    ax.set_title(title or "Simulation Runtime per Seed")
    ax.legend(fontsize=8, loc="upper right")

    # Value labels above bars
    for xi, rt in zip(x, runtimes):
        ax.text(xi, rt + y_max * 0.02, f"{rt:.1f}s",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Network summary table
# ---------------------------------------------------------------------------

def plot_network_summary(data: dict,
                         title: str | None = None) -> plt.Figure:
    """Table figure showing network-level metrics."""
    net = data.get("network", {})
    is_agg = isinstance(next(iter(net.values()), None), dict)

    rows = []
    for key in ["mean_sinr_db", "min_sinr_db", "max_sinr_db",
                "corruption_rate", "los_fraction",
                "sum_dl_throughput_mbps"]:
        label = _metric_label(key)
        if is_agg:
            stats = net.get(key, {})
            mean = stats.get("mean")
            std = stats.get("std")
            ci = stats.get("ci95")
            n = stats.get("n", 0)
            if mean is not None:
                val_str = f"{mean:.3f}"
                if ci is not None:
                    val_str += f" +/- {ci:.3f} (CI95)"
                elif std is not None:
                    val_str += f" +/- {std:.3f} (std)"
                val_str += f"  [n={n}]"
            else:
                val_str = "N/A"
        else:
            val = net.get(key)
            val_str = f"{val:.3f}" if val is not None else "N/A"
        rows.append([label, val_str])

    # Add timing if available
    if "wall_elapsed_s" in data:
        rows.append(["Wall-Clock Time", f"{data['wall_elapsed_s']:.1f}s"])
    elif "per_seed" in data:
        rts = [s["wall_elapsed_s"] for s in data["per_seed"]
               if "wall_elapsed_s" in s]
        if rts:
            mean_rt = sum(rts) / len(rts)
            rows.append(["Mean Runtime", f"{mean_rt:.1f}s"])

    fig, ax = plt.subplots(figsize=(8, max(2, len(rows) * 0.5 + 1)))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Style header row
    for j in range(2):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", weight="bold")

    scenario = data.get("scenario", "")
    n_seeds = data.get("num_seeds", "")
    seed = data.get("seed", "")
    subtitle = f"Scenario: {scenario}" if scenario else ""
    if n_seeds:
        subtitle += f"  ({n_seeds} seeds)"
    elif seed:
        subtitle += f"  (seed={seed})"

    ax.set_title(title or "Network Summary", fontsize=12, weight="bold",
                 pad=20)
    if subtitle:
        ax.text(0.5, 0.95, subtitle, ha="center", va="top",
                transform=ax.transAxes, fontsize=9, color="gray")

    fig.tight_layout()
    return fig
