"""Histogram + KDE comparison of pooled sim seeds vs ARPO field traces."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# TODO: Would be cool to have plots configurable (e.g., x/y limits)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIELD_TRACES_ROOT = REPO_ROOT / "data" / "arpo_extracted" / "_plots" / "per_day"

_TRACE_FILE_RE = re.compile(
    r"^bh2_(?P<metric>[a-z]+)__(?P<src>[a-z0-9]+)_to_(?P<peer>[a-z0-9]+)_trace\.csv$"
)
_SIM_NAME_RE = re.compile(r"^arpo-(\d+)-([\dxX])-(.+?)-(\d{8})$")
_MCS_FIELD_CAP = 12


@dataclass(frozen=True)
class _MetricSpec:
    short: str
    column: str
    label: str
    integer_cap: int | None = None


_METRICS: tuple[_MetricSpec, ...] = (
    _MetricSpec("snr",  "snr_db",   "SNR (dB)"),
    _MetricSpec("rcpi", "rcpi_dbm", "RCPI (dBm)"),
    _MetricSpec("mcs",  "mcs_tx",   "MCS",       integer_cap=_MCS_FIELD_CAP),
)
_METRICS_BY_SHORT = {m.short: m for m in _METRICS}


def sim_to_field_scenario(sim_name: str) -> str | None:
    """`arpo-1-1-static-04172026` -> `1-1_static_04172026`."""
    m = _SIM_NAME_RE.match(sim_name)
    if not m:
        return None
    major, minor, mid, day = m.groups()
    mid_joined = mid.replace("-", "_")
    return f"{major}-{minor.upper()}_{mid_joined}_{day}"


def _scenario_label(name: str) -> str:
    """`arpo-1-1-static-04172026` -> `1-1 static\n04/17/2026`."""
    m = _SIM_NAME_RE.match(name)
    if not m:
        return name
    major, minor, mid, day = m.groups()
    mid = mid.replace("-", " ")
    date = f"{day[0:2]}/{day[2:4]}/{day[4:8]}"
    return f"{major}-{minor.upper()} {mid}\n{date}"


def _read_metric_column(csv_path: Path, column: str) -> np.ndarray:
    if not csv_path.is_file():
        return np.array([], dtype=np.float64)
    try:
        df = pd.read_csv(csv_path, usecols=lambda c: c == column)
    except (ValueError, pd.errors.EmptyDataError):
        return np.array([], dtype=np.float64)
    if column not in df.columns:
        return np.array([], dtype=np.float64)
    vals = pd.to_numeric(df[column], errors="coerce").dropna()
    return vals.to_numpy(dtype=np.float64)


def _discover_pairs(seed_traces_root: Path) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    csvs = seed_traces_root / "csvs"
    if not csvs.is_dir():
        return out
    for src_dir in csvs.iterdir():
        if not src_dir.is_dir():
            continue
        for f in src_dir.iterdir():
            m = _TRACE_FILE_RE.match(f.name)
            if not m:
                continue
            a, b = m.group("src"), m.group("peer")
            out.add(tuple(sorted([a, b])))
    return out


def _pool_sim(scenario_dir: Path, src: str, peer: str,
              spec: _MetricSpec) -> np.ndarray:
    sim_traces = scenario_dir / "sim_traces"
    if not sim_traces.is_dir():
        return np.array([])
    arrs: list[np.ndarray] = []
    for seed_dir in sorted(sim_traces.iterdir()):
        if not seed_dir.is_dir():
            continue
        for a, b in [(src, peer), (peer, src)]:
            p = seed_dir / "csvs" / a / f"bh2_{spec.short}__{a}_to_{b}_trace.csv"
            arrs.append(_read_metric_column(p, spec.column))
    return np.concatenate(arrs) if arrs else np.array([])


def _pool_field(field_scen_dir: Path, src: str, peer: str,
                spec: _MetricSpec) -> np.ndarray:
    if not field_scen_dir.is_dir():
        return np.array([])
    arrs: list[np.ndarray] = []
    for a, b in [(src, peer), (peer, src)]:
        p = field_scen_dir / "csvs" / a / f"bh2_{spec.short}__{a}_to_{b}_trace.csv"
        arrs.append(_read_metric_column(p, spec.column))
    return np.concatenate(arrs) if arrs else np.array([])


def _ks_2samp(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample K-S D-statistic. P-value intentionally dropped: with N in the
    tens of thousands, p ≈ 0 for operationally-trivial differences."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_s = np.sort(a)
    b_s = np.sort(b)
    joint = np.sort(np.concatenate([a_s, b_s]))
    cdf_a = np.searchsorted(a_s, joint, side="right") / a_s.size
    cdf_b = np.searchsorted(b_s, joint, side="right") / b_s.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _fd_bins(values: np.ndarray, max_bins: int = 80, min_bins: int = 20) -> np.ndarray:
    """Freedman-Diaconis bin edges across pooled values."""
    if values.size < 2:
        lo = float(np.min(values)) if values.size else 0.0
        return np.linspace(lo, lo + 1.0, min_bins + 1)
    q25, q75 = np.percentile(values, [25, 75])
    iqr = float(q75 - q25)
    lo, hi = float(np.min(values)), float(np.max(values))
    if iqr <= 0 or hi <= lo:
        return np.linspace(lo, hi + 1e-9, min_bins + 1)
    width = 2.0 * iqr / (values.size ** (1.0 / 3.0))
    n_bins = int(np.clip(round((hi - lo) / width), min_bins, max_bins))
    return np.linspace(lo, hi, n_bins + 1)


def _kde_gaussian(values: np.ndarray, x_grid: np.ndarray,
                  max_samples: int = 20000) -> np.ndarray:
    """Gaussian KDE on ``x_grid``, Silverman bandwidth, numpy-only."""
    n = values.size
    if n == 0 or x_grid.size == 0:
        return np.full_like(x_grid, np.nan, dtype=np.float64)
    if n > max_samples:
        idx = np.random.default_rng(0).choice(n, size=max_samples, replace=False)
        values = values[idx]
        n = max_samples
    sigma = float(np.std(values))
    if sigma <= 0:
        sigma = 1.0
    h = 1.06 * sigma * n ** (-1.0 / 5.0)
    if h <= 0:
        h = 1e-3
    diff = (x_grid[:, None] - values[None, :]) / h
    return np.sum(np.exp(-0.5 * diff * diff), axis=1) / (n * h * np.sqrt(2.0 * np.pi))


def _summary(values: np.ndarray) -> dict[str, float]:
    """Median, mean, IQR for one distribution."""
    if values.size == 0:
        nan = float("nan")
        return {"median": nan, "mean": nan, "iqr": nan}
    q25, med, q75 = np.percentile(values, [25, 50, 75])
    return {
        "median": float(med),
        "mean":   float(np.mean(values)),
        "iqr":    float(q75 - q25),
    }


_SIM_COLOR = "#ff7f0e"
_FIELD_COLOR = "#1f77b4"
_UNIT_BY_SHORT = {"snr": "dB", "rcpi": "dB", "mcs": "", "per": "", "throughput": "Mbps"}


def _plot_one(sim_vals: np.ndarray, field_vals: np.ndarray,
              spec: _MetricSpec, scenario: str, src: str, peer: str,
              ks_d: float) -> plt.Figure:
    """Normalized-histogram (filled bars + KDE overlay) sim vs field."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    sim_s = _summary(sim_vals)
    field_s = _summary(field_vals)

    if sim_vals.size == 0 and field_vals.size == 0:
        ax.text(0.5, 0.5, "no samples", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        return fig

    pooled = np.concatenate([v for v in (sim_vals, field_vals) if v.size])
    bins = _fd_bins(pooled)
    lo, hi = float(bins[0]), float(bins[-1])
    pad = max(0.05 * (hi - lo), 1e-6)
    x_grid = np.linspace(lo - pad, hi + pad, 400)
    unit = _UNIT_BY_SHORT.get(spec.short, "")
    unit_suffix = f" {unit}" if unit else ""

    if sim_vals.size:
        sim_den, _ = np.histogram(sim_vals, bins=bins, density=True)
        ax.stairs(sim_den, bins, fill=True,
                  color=_SIM_COLOR, alpha=0.18, edgecolor=_SIM_COLOR, linewidth=1.2,
                  label=f"sim  n={sim_vals.size:,}  "
                        f"med={sim_s['median']:.2f}  mean={sim_s['mean']:.2f}{unit_suffix}")
        sim_kde = _kde_gaussian(sim_vals, x_grid)
        ax.plot(x_grid, sim_kde, color=_SIM_COLOR, linewidth=2.0, alpha=0.95)
        ax.axvline(sim_s["median"], color=_SIM_COLOR,
                   linestyle=":", linewidth=0.8, alpha=0.6)
    if field_vals.size:
        field_den, _ = np.histogram(field_vals, bins=bins, density=True)
        ax.stairs(field_den, bins, fill=True,
                  color=_FIELD_COLOR, alpha=0.18, edgecolor=_FIELD_COLOR, linewidth=1.2,
                  label=f"field  n={field_vals.size:,}  "
                        f"med={field_s['median']:.2f}  mean={field_s['mean']:.2f}{unit_suffix}")
        field_kde = _kde_gaussian(field_vals, x_grid)
        ax.plot(x_grid, field_kde, color=_FIELD_COLOR, linewidth=2.0, alpha=0.95)
        ax.axvline(field_s["median"], color=_FIELD_COLOR,
                   linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(spec.label, fontweight="bold")
    ax.set_ylabel("density  (area = 1, comparable across N)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    main = f"{scenario}:  {src} ↔ {peer}   ({spec.label})"
    bits = []
    if np.isfinite(ks_d):
        bits.append(f"K–S = {ks_d:.3f}")
    if np.isfinite(sim_s["median"]) and np.isfinite(field_s["median"]):
        bits.append(f"|Δmed| = {abs(sim_s['median'] - field_s['median']):.2f}{unit_suffix}")
    if np.isfinite(sim_s["mean"]) and np.isfinite(field_s["mean"]):
        bits.append(f"|Δmean| = {abs(sim_s['mean'] - field_s['mean']):.2f}{unit_suffix}")
    if spec.integer_cap is not None:
        bits.append(f"MCS capped at {spec.integer_cap} (firmware ceiling)")
    sub = "   ·   ".join(bits) if bits else ""
    fig.suptitle(main, fontsize=13, fontweight="bold", y=0.98)
    if sub:
        fig.text(0.5, 0.905, sub, fontsize=10, ha="center", color="0.2")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def _apply_mcs_cap(values: np.ndarray, cap: int | None) -> np.ndarray:
    if cap is None or values.size == 0:
        return values
    return np.clip(values, None, cap)


def _heatmap_png(rows: list[dict], spec: _MetricSpec, value_col: str,
                 value_label: str, out_path: Path) -> bool:
    """One heatmap: rows = scenarios (with dates), cols = rab links,
    cells = ``value_col`` for ``spec.short``."""
    rel = [r for r in rows
           if r["metric"] == spec.short and np.isfinite(r.get(value_col, np.nan))]
    if not rel:
        return False

    scenarios = sorted({r["scenario"] for r in rel})
    links: list[tuple[str, str]] = sorted(
        {tuple(sorted([r["src_rab"], r["peer_rab"]])) for r in rel}
    )
    grid = np.full((len(scenarios), len(links)), np.nan)
    for r in rel:
        si = scenarios.index(r["scenario"])
        li = links.index(tuple(sorted([r["src_rab"], r["peer_rab"]])))
        grid[si, li] = float(r[value_col])

    fig_w = max(5.0, 1.5 + 1.8 * len(links))
    fig_h = max(3.0, 1.2 + 0.55 * len(scenarios))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = float(np.nanmax(grid)) or 1.0
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)

    for si in range(len(scenarios)):
        for li in range(len(links)):
            v = grid[si, li]
            if np.isfinite(v):
                txt_color = "white" if v < 0.5 * vmax else "black"
                ax.text(li, si, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=txt_color)

    ax.set_xticks(range(len(links)))
    ax.set_xticklabels([f"{a}↔{b}" for (a, b) in links], fontsize=10)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([_scenario_label(s) for s in scenarios], fontsize=9)
    ax.set_xlabel("link", fontweight="bold")

    unit = _UNIT_BY_SHORT.get(spec.short, "") if value_col != "ks_statistic" else ""
    unit_suffix = f" [{unit}]" if unit else ""
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"{value_label}{unit_suffix}")

    ax.set_title(f"{value_label}  —  {spec.label}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _write_batch_heatmaps(rows: list[dict], metrics: list[_MetricSpec],
                          batch_root: Path) -> None:
    """One PNG per (metric, score) for the batch."""
    out_dir = batch_root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = [("abs_diff_medians", "|Δmedian|"),
              ("abs_diff_means",   "|Δmean|"),
              ("ks_statistic",     "K-S")]
    for spec in metrics:
        for col, label in scores:
            tag = label.replace("|", "").replace("Δ", "d").replace("-", "")
            out = out_dir / f"heatmap_{spec.short}_{tag}.png"
            if _heatmap_png(rows, spec, col, label, out):
                print(f"  wrote {out.relative_to(batch_root)}")


def _process_scenario(scenario_dir: Path, field_root: Path, metrics: list[_MetricSpec],
                      out_dir: Path) -> list[dict]:
    field_name = sim_to_field_scenario(scenario_dir.name)
    field_dir = (field_root / field_name) if field_name else None
    if field_dir is None or not field_dir.is_dir():
        print(f"  field traces not found (expected {field_dir}); skipping")
        return []

    sim_traces_root = scenario_dir / "sim_traces"
    seed_dirs = sorted(d for d in sim_traces_root.iterdir()
                       if d.is_dir() and d.name.startswith("seed-"))
    if not seed_dirs:
        print(f"  no seed-* under {sim_traces_root}; skipping")
        return []
    pairs = sorted(_discover_pairs(seed_dirs[0]))
    rows: list[dict] = []
    for src, peer in pairs:
        for spec in metrics:
            sim_vals = _pool_sim(scenario_dir, src, peer, spec)
            field_vals = _pool_field(field_dir, src, peer, spec)
            sim_vals = _apply_mcs_cap(sim_vals, spec.integer_cap)
            field_vals = _apply_mcs_cap(field_vals, spec.integer_cap)
            if sim_vals.size == 0 and field_vals.size == 0:
                continue
            ks_d = _ks_2samp(sim_vals, field_vals)
            sim_s = _summary(sim_vals)
            field_s = _summary(field_vals)
            both_med  = np.isfinite(sim_s["median"]) and np.isfinite(field_s["median"])
            both_mean = np.isfinite(sim_s["mean"])   and np.isfinite(field_s["mean"])
            rows.append({
                "scenario":         scenario_dir.name,
                "field_scenario":   field_name,
                "src_rab":          src,
                "peer_rab":         peer,
                "metric":           spec.short,
                "n_sim":            int(sim_vals.size),
                "n_field":          int(field_vals.size),
                "sim_median":       sim_s["median"],
                "sim_mean":         sim_s["mean"],
                "sim_iqr":          sim_s["iqr"],
                "field_median":     field_s["median"],
                "field_mean":       field_s["mean"],
                "field_iqr":        field_s["iqr"],
                "abs_diff_medians": (abs(field_s["median"] - sim_s["median"]) if both_med  else float("nan")),
                "abs_diff_means":   (abs(field_s["mean"]   - sim_s["mean"])   if both_mean else float("nan")),
                "ks_statistic":     ks_d,
            })
            fig = _plot_one(sim_vals, field_vals, spec, scenario_dir.name,
                            src, peer, ks_d)
            png_dir = out_dir / "pngs" / src
            png_dir.mkdir(parents=True, exist_ok=True)
            png_path = png_dir / f"hist_{spec.short}__{src}_to_{peer}.png"
            fig.savefig(png_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"    wrote {png_path.relative_to(scenario_dir)}")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Histogram + KDE comparison of sim vs ARPO field.")
    p.add_argument("batch_root", help="batch output dir (parent of per-scenario dirs)")
    p.add_argument("--field-root", default=str(FIELD_TRACES_ROOT),
                   help="root holding <field_scenario>/csvs/<src>/*.csv trace files")
    p.add_argument("--only", default=None,
                   help="restrict to one scenario name (basename of a dir in batch_root)")
    p.add_argument("--metrics", default="snr,rcpi,mcs",
                   help="comma-separated metric shorts to compare")
    args = p.parse_args(argv)

    batch_root = Path(args.batch_root).resolve()
    field_root = Path(args.field_root).resolve()
    if not batch_root.is_dir():
        print(f"batch root not found: {batch_root}", file=sys.stderr)
        return 1

    metrics: list[_MetricSpec] = []
    for short in [s.strip() for s in args.metrics.split(",") if s.strip()]:
        spec = _METRICS_BY_SHORT.get(short)
        if spec is None:
            print(f"unknown metric: {short} (known: {list(_METRICS_BY_SHORT)})",
                  file=sys.stderr)
            return 1
        metrics.append(spec)

    scenarios = sorted(p for p in batch_root.iterdir()
                       if p.is_dir() and (p / "sim_traces").is_dir())
    if args.only:
        scenarios = [s for s in scenarios if s.name == args.only]
    if not scenarios:
        print(f"no scenarios with sim_traces/ under {batch_root}", file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    for scen in scenarios:
        print(f"[{scen.name}]")
        out_dir = scen / "validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = _process_scenario(scen, field_root, metrics, out_dir)
        if rows:
            scen_csv = out_dir / "metrics.csv"
            pd.DataFrame(rows).to_csv(scen_csv, index=False)
            print(f"    wrote {scen_csv.relative_to(scen)}")
        all_rows.extend(rows)

    if all_rows:
        summary_csv = batch_root / "validation_summary.csv"
        pd.DataFrame(all_rows).to_csv(summary_csv, index=False)
        print(f"\nsummary: {summary_csv}")
        print("\nbatch heatmaps:")
        _write_batch_heatmaps(all_rows, metrics, batch_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
