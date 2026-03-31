"""Summary plot functions for mesh-sim outputs (from summary.json).

Each function takes aggregated summary data and returns a matplotlib Figure.
No function calls savefig or plt.show.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

_METRIC_LABELS = {
    "mean_sinr_db": "Mean SINR (dB)",
    "min_sinr_db": "Min SINR (dB)",
    "max_sinr_db": "Max SINR (dB)",
    "num_links": "Connected Links",
    "num_los_links": "LOS Links",
    "tx_throughput_mbps": "TX Throughput (Mbps)",
    "rx_throughput_mbps": "RX Throughput (Mbps)",
    "demand_mbps": "Demand (Mbps)",
    "delivered_mbps": "Delivered (Mbps)",
    "latency_ms": "Latency (ms)",
    "hop_count": "Hop Count",
    "sum_throughput_mbps": "Total Throughput (Mbps)",
    "connectivity": "Connectivity",
    "mean_hop_count": "Mean Hop Count",
    "flows_routed": "Flows Routed",
    "flows_unroutable": "Flows Unroutable",
}


def _label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, metric)


def _bar_chart(ids: list[str], stats_by_id: dict[str, dict],
               metric: str, entity_label: str,
               title: str | None) -> plt.Figure:
    """Shared logic for per-node and per-flow bar charts."""
    means = []
    errors = []
    for eid in ids:
        s = stats_by_id.get(eid, {}).get(metric, {})
        means.append(s.get("mean"))
        errors.append(s.get("ci95") or s.get("std") or 0)

    fig, ax = plt.subplots(figsize=(max(4, len(ids) * 1.2), 4))
    x = np.arange(len(ids))

    has_err = any(e and e > 0 for e in errors)
    ax.bar(x, means, yerr=errors if has_err else None,
           capsize=4 if has_err else 0,
           color=_COLORS[0], edgecolor="white", alpha=0.85,
           error_kw={"linewidth": 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.set_ylabel(_label(metric))
    ax.set_title(title or f"{_label(metric)} per {entity_label}")

    fig.tight_layout()
    return fig


def plot_per_node_bars(agg: dict, metric: str,
                       title: str | None = None) -> plt.Figure:
    """Bar chart of one metric per node, with CI error bars if multi-seed."""
    per_node = agg.get("per_node", {})
    return _bar_chart(sorted(per_node.keys()), per_node, metric, "Node", title)


def plot_per_flow_bars(agg: dict, metric: str,
                       title: str | None = None) -> plt.Figure:
    """Bar chart of one metric per flow, with CI error bars if multi-seed."""
    per_flow = agg.get("per_flow", {})
    return _bar_chart(sorted(per_flow.keys()), per_flow, metric, "Flow", title)


def plot_network_summary(agg: dict,
                         title: str | None = None) -> plt.Figure:
    """Table figure showing network-level metrics with CI where available."""
    net = agg.get("network", {})

    rows = []
    for key in ["mean_sinr_db", "sum_throughput_mbps", "connectivity",
                "mean_hop_count", "flows_routed", "flows_unroutable"]:
        stats = net.get(key, {})
        mean = stats.get("mean")
        ci = stats.get("ci95")
        std = stats.get("std")
        n = stats.get("n", 0)

        if mean is not None:
            val_str = f"{mean:.3f}"
            if ci is not None:
                val_str += f" +/- {ci:.3f} (CI95)"
            elif std is not None and n > 1:
                val_str += f" +/- {std:.3f} (std)"
            val_str += f"  [n={n}]"
        else:
            val_str = "N/A"
        rows.append([_label(key), val_str])

    # Runtime
    per_seed = agg.get("per_seed", [])
    rts = [s["wall_elapsed_s"] for s in per_seed if "wall_elapsed_s" in s]
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

    for j in range(2):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", weight="bold")

    n_seeds = agg.get("num_seeds", "")
    ax.set_title(title or f"Network Summary ({n_seeds} seeds)",
                 fontsize=12, weight="bold", pad=20)

    fig.tight_layout()
    return fig


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

    y_max = max(runtimes)
    ax.set_ylim(0, y_max * 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds], rotation=45, ha="right")
    ax.set_ylabel("Wall-Clock Time (s)")
    ax.set_title(title or "Simulation Runtime per Seed")
    ax.legend(fontsize=8, loc="upper right")

    for xi, rt in zip(x, runtimes):
        ax.text(xi, rt + y_max * 0.02, f"{rt:.1f}s",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig
