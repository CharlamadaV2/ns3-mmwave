"""
Cross-scenario comparison: median SNR per (scenario, node) on each
node's primary peer link, written as a heatmap PNG and a CSV.
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd

from .loaders import load_bh2_scenario
from .paths import CSV_ROOT, PLOTS_DIR
from .plots import filter_dominant_peers


def compare() -> int:
    if not CSV_ROOT.exists():
        print(f"ERROR: {CSV_ROOT} not found -- run `extract` first", file=sys.stderr)
        return 1

    out_dir = PLOTS_DIR / "_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for scen in sorted(p for p in CSV_ROOT.iterdir() if p.is_dir()):
        df = load_bh2_scenario(scen)
        if df is None:
            continue
        snr = filter_dominant_peers(df.dropna(subset=["field_snr"]))
        if snr.empty:
            continue
        primary = (snr.groupby(["__node__", "tag_sta_mac"]).size()
                      .reset_index(name="n")
                      .sort_values(["__node__", "n"], ascending=[True, False])
                      .drop_duplicates("__node__", keep="first"))
        for _, r in primary.iterrows():
            sub = snr[(snr["__node__"] == r["__node__"]) &
                      (snr["tag_sta_mac"] == r["tag_sta_mac"])]
            rows.append({
                "scenario": scen.name,
                "node": r["__node__"],
                "median_snr": float(sub["field_snr"].median()),
                "n_samples": int(r["n"]),
            })

    if not rows:
        print("No bh2 SNR data across scenarios", file=sys.stderr)
        return 1

    summary = pd.DataFrame(rows)
    csv_path = out_dir / "snr_per_scenario.csv"
    png_path = out_dir / "snr_per_scenario.png"
    summary.to_csv(csv_path, index=False)

    fig = _heatmap(summary)
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png_path}")
    print(f"  wrote {csv_path}")
    return 0


def _heatmap(summary: pd.DataFrame) -> plt.Figure:
    pivot = summary.pivot_table(index="scenario", columns="node",
                                values="median_snr", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(pivot) + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_xlabel("node")
    ax.set_title("Median SNR (dB) on each node's primary peer link, per scenario")

    valid = pivot.values[~pd.isna(pivot.values)]
    midpoint = valid.mean() if valid.size else 0.0
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v < midpoint else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="median SNR (dB)")
    fig.tight_layout()
    return fig
