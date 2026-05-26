## @file compare_runs.py
# @brief Cross-batch summary table: one row per validation batch.
#
# Reads the ``validation_summary.csv`` produced by @ref compare for each batch
# and prints ranked comparison tables useful for tuning simulation parameters
# (channel model, TX power, antenna gain).
#
# **Tables printed**
# | Table                  | Contents                                                   |
# |------------------------|------------------------------------------------------------|
# | Batch summary          | Per-run mean |Δmed| and mean K-S across all links.         |
# | Link-class breakdown   | Separate |Δmed| for "close" (rab1↔rab2) vs "far" (↔rab3). |
# | Per-scenario |Δmed|    | Rows=scenario, cols=run — which scenario drives error.     |
# | Per-scenario K-S       | Same layout with K-S statistic.                            |

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT    = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
_UNIT_BY_SHORT = {"snr": "dB", "rcpi": "dB", "mcs": "", "per": "", "throughput": "Mbps"}

_BATCH_PATH_RE = re.compile(r"(\d{4}-\d{2})/(\d{2})/(\d{2}-\d{2}-\d{2})")


def _batch_label(batch_path: str) -> str:
    """`outputs/2026-05/18/09-25-05-validation` -> `09:25:05\n2026-05-18`."""
    m = _BATCH_PATH_RE.search(batch_path.replace("\\", "/"))
    if not m:
        return Path(batch_path).name
    ym, dd, hms = m.groups()
    return f"{hms.replace('-', ':')}\n{ym}-{dd}"


def _scenario_label(name: str) -> str:
    """`1-1_static_04172026` -> `1-1 static\n04/17/2026`."""
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) == 8 and parts[1].isdigit():
        body = parts[0].replace("_", " ")
        d = parts[1]
        return f"{body}\n{d[0:2]}/{d[2:4]}/{d[4:8]}"
    return name


## @brief Recursively discover all batch directories under a root.
#
# A directory qualifies if it contains a ``validation_summary.csv``.
#
# @param root Directory to search.
# @return Sorted list of batch directory paths.
def _discover_batches(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("validation_summary.csv"))


def _heatmap_png(df: pd.DataFrame, metric: str, value_col: str,
                 value_label: str, out_path: Path) -> bool:
    """rows = scenarios, cols = batches, cells = mean of ``value_col`` across links."""
    sub = df[(df["metric"] == metric) & df[value_col].notna()].copy()
    if sub.empty:
        return False
    pivot = sub.groupby(["field_scenario", "_batch"])[value_col].mean().unstack("_batch")
    if pivot.empty:
        return False

    scenarios = sorted(pivot.index.tolist())
    batches = sorted(pivot.columns.tolist())
    grid = pivot.loc[scenarios, batches].to_numpy(dtype=float)

    fig_w = max(5.0, 1.5 + 1.6 * len(batches))
    fig_h = max(3.0, 1.2 + 0.55 * len(scenarios))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = float(np.nanmax(grid))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)

    for si in range(len(scenarios)):
        for bi in range(len(batches)):
            v = grid[si, bi]
            if np.isfinite(v):
                # viridis is dark at low values — flip text color for legibility
                txt_color = "white" if v < 0.5 * vmax else "black"
                ax.text(bi, si, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=txt_color)

    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels([_batch_label(b) for b in batches], fontsize=9)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([_scenario_label(s) for s in scenarios], fontsize=9)
    ax.set_xlabel("batch  (cell = mean across rab links)", fontweight="bold")

    unit = _UNIT_BY_SHORT.get(metric, "") if value_col != "ks_statistic" else ""
    unit_suffix = f" [{unit}]" if unit else ""
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"{value_label}{unit_suffix}")

    ax.set_title(f"{value_label}  —  {metric}  (cross-batch)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


## @brief CLI entry point for the cross-batch comparison tool.
#
# Auto-discovers batches under @ref OUTPUTS_ROOT when no explicit paths are
# given. Prints several comparison tables and optionally writes CSV exports.
#
# @param argv Argument list; defaults to ``sys.argv[1:]`` when ``None``.
# @return 0 on success, 1 on error.
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Cross-batch validation heatmaps (sim vs ARPO field).")
    p.add_argument("batches", nargs="*",
                   help="batch dirs; if omitted, auto-discovers under outputs/")
    p.add_argument("--metrics", default="snr,rcpi,mcs",
                   help="comma-separated metric shorts (default: snr,rcpi,mcs)")
    p.add_argument("--out", default=None,
                   help="output dir for PNGs (default: outputs/cross_batch_summary)")
    args = p.parse_args(argv)

    if args.batches:
        batch_dirs = [Path(b).resolve() for b in args.batches]
    else:
        if not OUTPUTS_ROOT.is_dir():
            print(f"no outputs root at {OUTPUTS_ROOT}", file=sys.stderr)
            return 1
        batch_dirs = _discover_batches(OUTPUTS_ROOT)
    if not batch_dirs:
        print("no batches with validation_summary.csv found", file=sys.stderr)
        return 1

    frames: list[pd.DataFrame] = []
    for bd in batch_dirs:
        summary_csv = bd / "validation_summary.csv"
        if not summary_csv.is_file():
            print(f"  skipping {bd} (no validation_summary.csv)", file=sys.stderr)
            continue
        df = pd.read_csv(summary_csv)
        try:
            df["_batch"] = str(bd.resolve().relative_to(REPO_ROOT))
        except ValueError:
            df["_batch"] = str(bd.resolve())
        frames.append(df)
    if not frames:
        return 1
    big = pd.concat(frames, ignore_index=True)

    out_dir = (Path(args.out).resolve() if args.out
               else (OUTPUTS_ROOT / "cross_batch_summary"))
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [s.strip() for s in args.metrics.split(",") if s.strip()]
    scores = [("abs_diff_medians", "abs-dmed", "|Δmedian|"),
              ("abs_diff_means",   "abs-dmean", "|Δmean|"),
              ("ks_statistic",     "ks", "K-S")]
    n_written = 0
    for metric in metrics:
        for col, tag, label in scores:
            if col not in big.columns:
                continue
            out = out_dir / f"heatmap_{metric}_{tag}.png"
            if _heatmap_png(big, metric, col, label, out):
                print(f"wrote {out}")
                n_written += 1
    if not n_written:
        print("no heatmaps written (no matching rows)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())