'''compare_baseline.py'''
## @file compare_baseline.py
# @brief Per-node comparison of each scenario's sim-vs-field result against a baseline scenario.
#
# All scenarios are run through the SAME sim config; each produces a
# ``validation_summary.csv`` of sim-vs-field deltas per source node. This tool
# takes one scenario as the baseline (the run where sim tracks field well) and,
# for every other scenario, differences that scenario's per-node value against
# the baseline's — exposing how the same config diverges from field under
# changing conditions (e.g. interference the sim can't model).
#
# The comparison is RAW (signed), not absolute: a cell of +18 dB vs -18 dB
# tells you the DIRECTION the accuracy fails in, not just the magnitude.
#
# Output (one per metric):
#   heatmap_<metric>_<value>.png   rows = nodes, cols = scenarios,
#                                  cell = scenario value - baseline value.
#   baseline_comparison.csv        the same grid in long form.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_UNIT_BY_SHORT = {"snr": "dB", "rcpi": "dB", "mcs": "", "per": "", "throughput": "Mbps"}

## @brief Node key column and the per-node value columns that can be differenced.
_NODE_KEY = "src_rab"
_VALUE_COLS = {
    "sim_median":       "sim median",
    "field_median":     "field median",
    "diff_medians":     "Δmedian (sim-field)",
    "sim_mean":         "sim mean",
    "field_mean":       "field mean",
    "diff_means":       "Δmean (sim-field)",
    "ks_statistic":     "K-S",
}


## @brief Load one scenario's validation_summary.csv, or None if unusable.
#
# Resolves @p path to its ``validation_summary.csv`` (accepts either the file
# itself or the directory containing it) and tags every row with the scenario
# label (the summary's own ``scenario`` column if present, else the dir name).
def _load_summary(path: Path) -> pd.DataFrame | None:
    csv = path if path.is_file() else path / "validation_summary.csv"
    if not csv.is_file():
        print(f"  skipping {path} (no validation_summary.csv)", file=sys.stderr)
        return None
    df = pd.read_csv(csv)
    if _NODE_KEY not in df.columns or "metric" not in df.columns:
        print(f"  skipping {csv} (missing {_NODE_KEY!r} or 'metric')", file=sys.stderr)
        return None

    # Derive signed sim-field diffs; the stored abs_diff_* are unsigned.
    if {"sim_median", "field_median"}.issubset(df.columns):
        df["diff_medians"] = df["sim_median"] - df["field_median"]
    if {"sim_mean", "field_mean"}.issubset(df.columns):
        df["diff_means"] = df["sim_mean"] - df["field_mean"]

    label = path.stem if path.is_file() else path.name
    scen_col = df["scenario"] if "scenario" in df.columns else None
    df["_scenario"] = str(scen_col.iloc[0]) if scen_col is not None and not scen_col.empty else label
    return df


## @brief Reduce a summary to a node x metric Series for one value column.
def _node_metric_series(df: pd.DataFrame, value_col: str) -> pd.Series:
    if value_col not in df.columns:
        return pd.Series(dtype=float)
    s = df[[_NODE_KEY, "metric", value_col]].dropna(subset=[value_col])
    # one value per (node, metric); mean collapses any duplicate peer rows
    return s.groupby([_NODE_KEY, "metric"])[value_col].mean()


## @brief Render the per-node scenario-vs-baseline heatmap for one metric.
#
# Rows are nodes, columns are comparison scenarios; each cell is
# ``scenario[node] - baseline[node]`` for @p value_col at @p metric. A
# diverging colormap centred on 0 shows sign (direction of failure).
#
# @return True if a figure was written.
def _heatmap_delta(per_scen: dict[str, pd.Series], baseline_name: str,
                   metric: str, value_col: str, value_label: str,
                   out_path: Path) -> bool:
    base = per_scen[baseline_name]
    base_m = base.xs(metric, level="metric") if metric in base.index.get_level_values("metric") \
        else pd.Series(dtype=float)
    if base_m.empty:
        return False

    scen_names = [s for s in per_scen if s != baseline_name]
    cols, node_set = {}, set()
    for s in scen_names:
        ser = per_scen[s]
        if metric not in ser.index.get_level_values("metric"):
            continue
        m = ser.xs(metric, level="metric")
        delta = m.subtract(base_m, fill_value=np.nan).dropna()
        if not delta.empty:
            cols[s] = delta
            node_set.update(delta.index.tolist())
    if not cols:
        return False

    nodes = sorted(node_set)
    scen_order = [s for s in scen_names if s in cols]
    grid = np.full((len(nodes), len(scen_order)), np.nan)
    for j, s in enumerate(scen_order):
        for i, n in enumerate(nodes):
            if n in cols[s].index:
                grid[i, j] = cols[s][n]

    absmax = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
    absmax = absmax if absmax > 0 else 1.0

    fig_w = max(5.0, 1.8 + 1.7 * len(scen_order))
    fig_h = max(3.0, 1.2 + 0.5 * len(nodes))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-absmax, vmax=absmax)

    for i in range(len(nodes)):
        for j in range(len(scen_order)):
            v = grid[i, j]
            if np.isfinite(v):
                shade = abs(v) > 0.6 * absmax
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white" if shade else "black")

    ax.set_xticks(range(len(scen_order)))
    ax.set_xticklabels(scen_order, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(nodes, fontsize=9)
    ax.set_xlabel(f"scenario  (cell = scenario − baseline '{baseline_name}')",
                  fontweight="bold")
    ax.set_ylabel("node", fontweight="bold")

    unit = _UNIT_BY_SHORT.get(metric, "") if value_col != "ks_statistic" else ""
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(f"Δ {value_label}{f' [{unit}]' if unit else ''}  vs baseline")

    ax.set_title(f"{value_label} vs baseline — {metric}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


## @brief Long-form CSV of every node x scenario x metric delta vs baseline.
def _write_csv(per_scen: dict[str, pd.Series], baseline_name: str,
               value_col: str, metrics: list[str], out_path: Path) -> int:
    base = per_scen[baseline_name]
    rows = []
    for s, ser in per_scen.items():
        if s == baseline_name:
            continue
        for metric in metrics:
            if metric not in ser.index.get_level_values("metric"):
                continue
            m = ser.xs(metric, level="metric")
            bm = base.xs(metric, level="metric") if metric in base.index.get_level_values("metric") \
                else pd.Series(dtype=float)
            for node, val in m.items():
                if node in bm.index:
                    rows.append({
                        "scenario": s, "baseline": baseline_name, "node": node,
                        "metric": metric, "value_col": value_col,
                        "scenario_value": round(float(val), 3),
                        "baseline_value": round(float(bm[node]), 3),
                        "delta": round(float(val - bm[node]), 3),
                    })
    if not rows:
        return 0
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return len(rows)


## @brief CLI entry point.
#
# @param argv Argument list; defaults to ``sys.argv[1:]`` when ``None``.
# @return 0 on success, 1 on error.
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Per-node comparison of each scenario's sim-vs-field result "
                    "against a baseline scenario (raw/signed).")
    p.add_argument("scenarios", nargs="+", type=Path,
                   help="scenario dirs (or validation_summary.csv paths) to compare")
    p.add_argument("--baseline", type=Path, required=True,
                   help="baseline scenario dir (or its validation_summary.csv)")
    p.add_argument("--value", default="diff_medians", choices=sorted(_VALUE_COLS),
                   help="per-node value to difference vs baseline "
                        "(default: diff_medians = signed sim-field median gap)")
    p.add_argument("--metrics", default="snr,rcpi,mcs",
                   help="comma-separated metric shorts (default: snr,rcpi,mcs)")
    p.add_argument("--out", default=None, type=Path,
                   help="output dir (default: <baseline>/baseline_comparison)")
    args = p.parse_args(argv)

    baseline_df = _load_summary(args.baseline)
    if baseline_df is None:
        print(f"ERROR: could not load baseline summary at {args.baseline}",
              file=sys.stderr)
        return 1
    baseline_name = baseline_df["_scenario"].iloc[0]

    per_scen: dict[str, pd.Series] = {baseline_name: _node_metric_series(baseline_df, args.value)}
    for sc in args.scenarios:
        df = _load_summary(sc)
        if df is None:
            continue
        name = df["_scenario"].iloc[0]
        if name == baseline_name:
            print(f"  note: {sc} is the baseline scenario itself; skipping",
                  file=sys.stderr)
            continue
        per_scen[name] = _node_metric_series(df, args.value)

    if len(per_scen) < 2:
        print("ERROR: need at least one comparison scenario distinct from baseline",
              file=sys.stderr)
        return 1
    if per_scen[baseline_name].empty:
        print(f"ERROR: baseline has no data for value '{args.value}'", file=sys.stderr)
        return 1

    out_dir = args.out.resolve() if args.out else (
        (args.baseline if args.baseline.is_dir() else args.baseline.parent)
        / "baseline_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [s.strip() for s in args.metrics.split(",") if s.strip()]
    label = _VALUE_COLS[args.value]

    n_written = 0
    for metric in metrics:
        out = out_dir / f"heatmap_{metric}_{args.value}.png"
        if _heatmap_delta(per_scen, baseline_name, metric, args.value, label, out):
            print(f"wrote {out}")
            n_written += 1

    n_rows = _write_csv(per_scen, baseline_name, args.value, metrics,
                        out_dir / "baseline_comparison.csv")
    if n_rows:
        print(f"wrote {out_dir / 'baseline_comparison.csv'}  ({n_rows} rows)")

    if not n_written:
        print("no heatmaps written (no overlapping node+metric rows)", file=sys.stderr)
        return 1
    print(f"\nbaseline: {baseline_name}   scenarios compared: "
          f"{', '.join(s for s in per_scen if s != baseline_name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())