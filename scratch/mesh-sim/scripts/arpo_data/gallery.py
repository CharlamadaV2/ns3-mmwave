"""
Four-style SNR comparison gallery + topology MAC card.

Useful when picking which view of per-link SNR best surfaces a phenomenon
for a paper or talk -- same underlying data, four different framings.
Each function returns a ``matplotlib.Figure`` (or ``None``).
"""

import matplotlib.pyplot as plt
import pandas as pd

from .topology import node_color


def plot_topology_card(topo: dict[str, str]) -> plt.Figure | None:
    """Reference legend: which MACs belong to which physical node."""
    if not topo:
        return None

    by_node: dict[str, list[str]] = {}
    for mac, node in sorted(topo.items()):
        by_node.setdefault(node, []).append(mac)
    nodes = sorted(by_node.keys(), key=lambda s: (s.startswith("ext"), s))

    line_h = 0.6
    total_lines = sum(len(v) + 1 for v in by_node.values())
    fig, ax = plt.subplots(figsize=(8, 0.5 * total_lines + 1.5))
    ax.set_axis_off()
    y = 0.0
    for node in nodes:
        ax.add_patch(plt.Rectangle((0, -y - line_h * 0.6), 0.15, line_h * 0.7,
                                   color=node_color(node), alpha=0.7))
        ax.text(0.18, -y - line_h * 0.3, node, fontsize=12, fontweight="bold", va="center")
        y += line_h
        for mac in by_node[node]:
            ax.text(0.25, -y - line_h * 0.3, mac, fontsize=10, family="monospace", va="center")
            y += line_h
        y += 0.2
    ax.set_xlim(0, 1)
    ax.set_ylim(-y - 0.2, 0.5)
    ax.set_title("Topology (MAC -> physical node)", loc="left")
    fig.tight_layout()
    return fig


def plot_snr_per_node(df: pd.DataFrame) -> plt.Figure | None:
    """Style A: stacked subplots, one per source node."""
    nodes = sorted(df["__node__"].unique())
    if not nodes:
        return None
    fig, axes = plt.subplots(len(nodes), 1, figsize=(11, 2.5 * len(nodes)), sharex=True)
    if len(nodes) == 1:
        axes = [axes]
    for ax, node in zip(axes, nodes):
        g_node = df[df["__node__"] == node]
        seen: set[str] = set()
        for peer in sorted(g_node["__peer__"].unique()):
            g = g_node[g_node["__peer__"] == peer].sort_values("__sec__")
            label = f"-> {peer}" if peer not in seen else None
            seen.add(peer)
            ax.plot(g["__sec__"], g["field_snr"], ".", markersize=2, alpha=0.55,
                    color=node_color(peer), label=label)
        ax.set_ylabel(f"{node}\nSNR (dB)")
        ax.legend(loc="upper right", fontsize=8, ncol=3)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("seconds since each node's session start")
    fig.suptitle("Style A: per-node SNR (color = peer node)")
    fig.tight_layout()
    return fig


def plot_snr_per_link(df: pd.DataFrame) -> plt.Figure | None:
    """Style B: small-multiples grid, one panel per (node -> peer)."""
    pairs = sorted(df.groupby(["__node__", "__peer__"]).size().index.tolist())
    if not pairs:
        return None
    cols = min(3, len(pairs))
    rows = (len(pairs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.2 * rows),
                             sharex=True, sharey=True, squeeze=False)
    for i, (node, peer) in enumerate(pairs):
        ax = axes[i // cols][i % cols]
        g = df[(df["__node__"] == node) & (df["__peer__"] == peer)].sort_values("__sec__")
        ax.plot(g["__sec__"], g["field_snr"], ".", markersize=1.5, alpha=0.55,
                color=node_color(peer))
        ax.set_title(f"{node} -> {peer}  (n={len(g)})", fontsize=9)
        ax.grid(True, alpha=0.3)
    for j in range(len(pairs), rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    for r in range(rows):
        axes[r][0].set_ylabel("SNR (dB)")
    for c in range(cols):
        axes[-1][c].set_xlabel("seconds")
    fig.suptitle("Style B: per-link SNR (color = peer node)")
    fig.tight_layout()
    return fig


def plot_snr_heatmap(df: pd.DataFrame, bin_seconds: float = 5.0) -> plt.Figure | None:
    """Style C: link x time heatmap of median SNR."""
    if df.empty:
        return None
    df = df.copy()
    df["__bin__"] = (df["__sec__"] // bin_seconds).astype("Int64")
    df["__pair__"] = df["__node__"] + " -> " + df["__peer__"].astype(str)
    pivot = df.pivot_table(index="__pair__", columns="__bin__",
                           values="field_snr", aggfunc="median")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, 0.4 * len(pivot) + 2))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                   cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    n_bins = pivot.shape[1]
    tick_step = max(1, n_bins // 10)
    ax.set_xticks(range(0, n_bins, tick_step))
    ax.set_xticklabels([f"{int(b * bin_seconds)}" for b in pivot.columns[::tick_step]],
                       rotation=0, fontsize=8)
    ax.set_xlabel(f"seconds since session start (bin = {int(bin_seconds)}s)")
    ax.set_title("Style C: link x time heatmap of median SNR")
    fig.colorbar(im, ax=ax, label="median SNR (dB)")
    fig.tight_layout()
    return fig


def plot_snr_box(df: pd.DataFrame) -> plt.Figure | None:
    """Style D: per-link SNR distribution as horizontal boxplots."""
    if df.empty:
        return None
    df = df.copy()
    df["__pair__"] = df["__node__"] + " -> " + df["__peer__"].astype(str)
    pairs = df.groupby("__pair__")["field_snr"].median().sort_values().index.tolist()
    data = [df[df["__pair__"] == p]["field_snr"].values for p in pairs]
    colors = [node_color(p.split(" -> ")[1]) for p in pairs]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(pairs) + 2)))
    bp = ax.boxplot(data, vert=False, tick_labels=pairs,
                    showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_xlabel("SNR (dB)")
    ax.set_title("Style D: per-link SNR distribution (color = peer node)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
