"""ECDF + bootstrap-CI comparison of pooled sim seeds vs ARPO field traces."""

from __future__ import annotations

import argparse
import math
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


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    a_s = np.sort(a)
    b_s = np.sort(b)
    joint = np.sort(np.concatenate([a_s, b_s]))
    cdf_a = np.searchsorted(a_s, joint, side="right") / a_s.size
    cdf_b = np.searchsorted(b_s, joint, side="right") / b_s.size
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    n_e = (a_s.size * b_s.size) / (a_s.size + b_s.size)
    p = float(min(1.0, 2.0 * math.exp(-2.0 * n_e * d * d)))
    return d, p


def _bootstrap_ecdf_band(values: np.ndarray, x_grid: np.ndarray,
                         n_boot: int, ci_pct: float,
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = values.size
    if n == 0 or x_grid.size == 0:
        empty = np.full_like(x_grid, np.nan, dtype=np.float64)
        return empty, empty
    sorted_vals = np.sort(values)
    boot = np.empty((n_boot, x_grid.size), dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = np.sort(sorted_vals[idx])
        boot[i] = np.searchsorted(sample, x_grid, side="right") / n
    half = (100.0 - ci_pct) / 2.0
    return np.percentile(boot, half, axis=0), np.percentile(boot, 100.0 - half, axis=0)


def _stats(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    q25, med, q75 = np.percentile(values, [25, 50, 75])
    return float(med), float(q75 - q25)


def _plot_one(sim_vals: np.ndarray, field_vals: np.ndarray,
              spec: _MetricSpec, scenario: str, src: str, peer: str,
              ks_d: float, ks_p: float,
              n_boot: int, ci_pct: float,
              rng: np.random.Generator) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    sim_med, sim_iqr = _stats(sim_vals)
    field_med, field_iqr = _stats(field_vals)

    lo_x = np.nanmin([sim_vals.min() if sim_vals.size else np.nan,
                       field_vals.min() if field_vals.size else np.nan])
    hi_x = np.nanmax([sim_vals.max() if sim_vals.size else np.nan,
                       field_vals.max() if field_vals.size else np.nan])
    if not (np.isfinite(lo_x) and np.isfinite(hi_x)):
        ax.text(0.5, 0.5, "no samples", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        return fig
    pad = max(0.05 * (hi_x - lo_x), 1e-6)
    x_grid = np.linspace(lo_x - pad, hi_x + pad, 256)

    if sim_vals.size:
        lo_band, hi_band = _bootstrap_ecdf_band(sim_vals, x_grid, n_boot, ci_pct, rng)
        ax.fill_between(x_grid, lo_band, hi_band, color="#ff7f0e", alpha=0.25,
                        step="post", label=f"sim {int(ci_pct)}% CI (bootstrap, B={n_boot})")
        sx, sy = _ecdf(sim_vals)
        ax.step(sx, sy, where="post", color="#ff7f0e", linewidth=1.8,
                label=f"sim ECDF   n={sim_vals.size:,}   med={sim_med:.2f}  IQR={sim_iqr:.2f}")
        ax.axvline(sim_med, color="#ff7f0e", linestyle=":", linewidth=0.8, alpha=0.6)
    if field_vals.size:
        fx, fy = _ecdf(field_vals)
        ax.step(fx, fy, where="post", color="#1f77b4", linewidth=1.8,
                label=f"field ECDF n={field_vals.size:,}   med={field_med:.2f}  IQR={field_iqr:.2f}")
        ax.axvline(field_med, color="#1f77b4", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(spec.label, fontweight="bold")
    ax.set_ylabel("ECDF  (fraction of samples ≤ x)", fontweight="bold")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    main = f"{scenario}:  {src} ↔ {peer}   ({spec.label})"
    bits = []
    if np.isfinite(ks_d):
        bits.append(f"K–S D={ks_d:.3f}   p≈{ks_p:.2g}")
    if np.isfinite(sim_med) and np.isfinite(field_med):
        bits.append(f"|Δmedians|={abs(sim_med - field_med):.2f}")
    if spec.integer_cap is not None:
        bits.append(f"sim MCS post-capped at {spec.integer_cap}")
    sub = "   ·   ".join(bits) if bits else ""
    fig.suptitle(main, fontsize=13, fontweight="bold")
    if sub:
        fig.text(0.5, 0.92, sub, fontsize=10, ha="center", color="0.2")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def _apply_mcs_cap(values: np.ndarray, cap: int | None) -> np.ndarray:
    if cap is None or values.size == 0:
        return values
    return np.clip(values, None, cap)


def _process_scenario(scenario_dir: Path, field_root: Path, metrics: list[_MetricSpec],
                      out_dir: Path, n_boot: int, ci_pct: float,
                      rng: np.random.Generator) -> list[dict]:
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
            ks_d, ks_p = _ks_2samp(sim_vals, field_vals)
            sim_med, sim_iqr = _stats(sim_vals)
            field_med, field_iqr = _stats(field_vals)
            rows.append({
                "scenario":         scenario_dir.name,
                "field_scenario":   field_name,
                "src_rab":          src,
                "peer_rab":         peer,
                "metric":           spec.short,
                "n_sim":            int(sim_vals.size),
                "n_field":          int(field_vals.size),
                "sim_median":       sim_med,
                "sim_iqr":          sim_iqr,
                "field_median":     field_med,
                "field_iqr":        field_iqr,
                "abs_diff_medians": (abs(sim_med - field_med)
                                     if np.isfinite(sim_med) and np.isfinite(field_med)
                                     else float("nan")),
                "ks_statistic":     ks_d,
                "ks_pvalue":        ks_p,
            })
            fig = _plot_one(sim_vals, field_vals, spec, scenario_dir.name,
                            src, peer, ks_d, ks_p, n_boot, ci_pct, rng)
            png_dir = out_dir / "pngs" / src
            png_dir.mkdir(parents=True, exist_ok=True)
            png_path = png_dir / f"ecdf_{spec.short}__{src}_to_{peer}.png"
            fig.savefig(png_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"    wrote {png_path.relative_to(scenario_dir)}")
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ECDF + bootstrap-CI comparison of sim vs ARPO field.")
    p.add_argument("batch_root", help="batch output dir (parent of per-scenario dirs)")
    p.add_argument("--field-root", default=str(FIELD_TRACES_ROOT),
                   help="root holding <field_scenario>/csvs/<src>/*.csv trace files")
    p.add_argument("--only", default=None,
                   help="restrict to one scenario name (basename of a dir in batch_root)")
    p.add_argument("--metrics", default="snr,rcpi,mcs",
                   help="comma-separated metric shorts to compare")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="bootstrap iterations for the sim ECDF CI band")
    p.add_argument("--ci", type=float, default=90.0,
                   help="confidence level (percent) for the sim ECDF band")
    p.add_argument("--seed", type=int, default=42,
                   help="rng seed for the bootstrap")
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

    rng = np.random.default_rng(args.seed)
    all_rows: list[dict] = []
    for scen in scenarios:
        print(f"[{scen.name}]")
        out_dir = scen / "validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = _process_scenario(scen, field_root, metrics, out_dir,
                                 args.n_boot, args.ci, rng)
        if rows:
            scen_csv = out_dir / "metrics.csv"
            pd.DataFrame(rows).to_csv(scen_csv, index=False)
            print(f"    wrote {scen_csv.relative_to(scen)}")
        all_rows.extend(rows)

    if all_rows:
        summary_csv = batch_root / "validation_summary.csv"
        pd.DataFrame(all_rows).to_csv(summary_csv, index=False)
        print(f"\nsummary: {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
