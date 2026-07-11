## @file compare.py
# @brief ECDF + bootstrap-CI comparison of pooled sim seeds vs ARPO field traces.
#
# For each (scenario, link, metric) triple this module:
# -# Pools sim samples across all seed runs.
# -# Loads the matching field trace.
# -# Computes a two-sample K-S statistic and p-value.
# -# Renders an ECDF plot with a bootstrap confidence band around the sim curve.
# -# Writes per-scenario and aggregate ``metrics.csv`` / ``validation_summary.csv``.
#
# Also exports @ref sim_to_field_scenario, the canonical name-mapping function
# used throughout the pipeline to translate sim scenario names to field names.

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

REPO_ROOT         = Path(__file__).resolve().parents[2]
FIELD_TRACES_ROOT = REPO_ROOT / "data" / "arpo_extracted" / "_plots" / "per_day"

## @brief Regex for matching a per-link trace CSV filename.
#
# Expected form: ``bh2_<metric>__<src>_to_<peer>_trace.csv``
_TRACE_FILE_RE = re.compile(
    r"^bh2_(?P<metric>[a-z]+)__(?P<src>[a-z0-9]+)_to_(?P<peer>[a-z0-9]+)_trace\.csv$"
)

## @brief Regex for parsing a sim scenario directory name.
#
# Expected form: ``arpo-<major>-<minor>-<description>-<MMDDYYYY>``
# Example: ``arpo-1-1-static-04172026``
_SIM_NAME_RE = re.compile(r"^arpo-(\d+)-([\dxX])-(.+?)-(\d{8})$")

## @brief MCS values above this threshold are clipped before comparison.
#
# The field hardware reports MCS indices up to 12; the sim can produce higher
# values. Capping both distributions at 12 avoids artefacts from the mismatch.
_MCS_FIELD_CAP = 12


## @brief Specification for one radio metric.
@dataclass(frozen=True)
class _MetricSpec:
    short:       str       ##< Short name used in filenames (e.g. ``"snr"``).
    column:      str       ##< DataFrame column holding the metric values.
    label:       str       ##< Human-readable axis label (e.g. ``"SNR (dB)"``).
    integer_cap: int | None = None  ##< Optional upper clip applied to both sim and field.


## @brief All metrics supported by the comparison pipeline.
_METRICS: tuple[_MetricSpec, ...] = (
    _MetricSpec("snr",  "snr_db",   "SNR (dB)"),
    _MetricSpec("rcpi", "rcpi_dbm", "RCPI (dBm)"),
    _MetricSpec("mcs",  "mcs_tx",   "MCS", integer_cap=_MCS_FIELD_CAP),
)
_METRICS_BY_SHORT = {m.short: m for m in _METRICS}  ##< Lookup by short name.


## @brief Convert a sim scenario name to its corresponding field scenario name.
#
# Sim names follow ``arpo-1-1-static-04172026``.
# Field names follow ``1-1_static_04172026``.
#
# @param sim_name  Sim-form scenario directory name.
# @return          Field-form scenario name, or ``None`` if the name does not
#                  match the expected pattern.
def sim_to_field_scenario(sim_name: str) -> str | None:
    m = _SIM_NAME_RE.match(sim_name)
    if not m:
        return None
    major, minor, mid, day = m.groups()
    mid_joined = mid.replace("-", "_")
    return f"{major}-{minor.upper()}_{mid_joined}_{day}"


## @brief Format a sim scenario name as a two-line plot label.
#
# Converts ``arpo-1-1-static-04172026`` to ``1-1 static\n04/17/2026``.
# Returns the name unchanged if it does not match the expected pattern.
#
# @param name  Sim scenario directory name.
# @return      Two-line label string.
def _scenario_label(name: str) -> str:
    m = _SIM_NAME_RE.match(name)
    if not m:
        return name
    major, minor, mid, day = m.groups()
    mid  = mid.replace("-", " ")
    date = f"{day[0:2]}/{day[2:4]}/{day[4:8]}"
    return f"{major}-{minor.upper()} {mid}\n{date}"


## @brief Candidate time-column names used for the comparison window.
#
# The first one present in a trace is used to filter rows to the requested
# window. Sim traces use ``sec``; field IH traces also use ``sec``.
_TIME_CANDIDATES = ("sec", "time_s", "sec_since_origin", "t")


## @brief Read one numeric column from a trace CSV, optionally time-windowed.
#
# Returns an empty float64 array if the file is absent, unreadable, or does
# not contain the requested column. Non-numeric values are coerced to NaN
# and dropped before returning. When @p window_s is given, only rows whose
# time column (see @ref _TIME_CANDIDATES) is ``<= window_s`` are kept, so the
# same [0, window_s] period is used for both sim and field.
#
# @param csv_path  Path to the trace CSV file.
# @param column    Column name to extract.
# @param window_s  If set, keep only rows with time ``<= window_s`` seconds.
# @return          1-D float64 array of valid (non-NaN) values.
def _read_metric_column(csv_path: Path, column: str,
                        window_s: float | None = None) -> np.ndarray:
    if not csv_path.is_file():
        return np.array([], dtype=np.float64)
    try:
        if window_s is None:
            usecols = lambda c: c == column
        else:
            usecols = lambda c: c == column or c in _TIME_CANDIDATES
        df = pd.read_csv(csv_path, usecols=usecols)
    except (ValueError, pd.errors.EmptyDataError):
        return np.array([], dtype=np.float64)
    if column not in df.columns:
        return np.array([], dtype=np.float64)
    if window_s is not None:
        tcol = next((c for c in _TIME_CANDIDATES if c in df.columns), None)
        if tcol is not None:
            t  = pd.to_numeric(df[tcol], errors="coerce")
            t0 = t.min()                       
            if pd.notna(t0):                  
                df = df[t <= t0 + window_s] #< Keeps any data within window max
    vals = pd.to_numeric(df[column], errors="coerce").dropna()
    return vals.to_numpy(dtype=np.float64)


## @brief Discover all (src, peer) pairs present in a seed's trace directory.
#
# In scenario mode, parses bh2 trace filenames to find ``(rab_a, rab_b)``
# pairs. In node mode, returns ``(IH_node, "neighbors")`` for every IH
# directory that also exists in the field traces. Gateway and non-IH
# directories are skipped.
#
# @param seed_traces_root  Root directory of one seed's trace output.
# @param mode              ``"scenario"`` or ``"node"``.
# @param field_dir         Field traces directory used to filter node mode
#                          results to nodes present in both sim and field.
# @return                  Set of ``(src, peer)`` string tuples.
def _discover_pairs(seed_traces_root: Path, mode: str,
                    field_dir: Path | None = None) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    csvs = seed_traces_root / "csvs"
    if not csvs.is_dir():
        return out

    if mode == "scenario":
        for src_dir in csvs.iterdir():
            if not src_dir.is_dir():
                continue
            for f in src_dir.iterdir():
                m = _TRACE_FILE_RE.match(f.name)
                if not m:
                    continue
                a, b = m.group("src"), m.group("peer")
                out.add(tuple(sorted([a, b])))

    elif mode == "node":
        for node_dir in csvs.iterdir():
            if not node_dir.is_dir():
                continue
            if field_dir is None or (field_dir / node_dir.name / "csvs" / node_dir.name).is_dir():
                out.add((node_dir.name, "neighbors"))
    return out


## @brief Pool metric samples for one link across all sim seeds.
#
# In scenario mode reads both directions (src→peer and peer→src) using the
# ``bh2_<metric>__<src>_to_<peer>_trace.csv`` filename pattern.
# In node mode (``peer == "neighbors"``) reads the combined IH trace file:
# ``IH_<metric>__<src>_to_neighbors_trace.csv``.
#
# @param scenario_dir  Scenario directory containing ``sim_traces/``.
# @param src           Source node label.
# @param peer          Peer node label, or ``"neighbors"`` for node mode.
# @param spec          Metric specification.
# @return              Concatenated float64 array of all valid samples.
def _pool_sim(scenario_dir: Path, src: str, peer: str,
              spec: _MetricSpec, window_s: float | None = None) -> np.ndarray:
    sim_traces = scenario_dir / "sim_traces"
    if not sim_traces.is_dir():
        return np.array([])
    
    arrs: list[np.ndarray] = []
    for seed_dir in sorted(sim_traces.iterdir()):
        if not seed_dir.is_dir():
            continue
        if peer == "neighbors":
            p = seed_dir / "csvs" / src / f"IH_{spec.short}__{src}_to_neighbors_trace.csv"
            arrs.append(_read_metric_column(p, spec.column, window_s))
        else:
            for a, b in [(src, peer), (peer, src)]:
                p = seed_dir / "csvs" / a / f"bh2_{spec.short}__{a}_to_{b}_trace.csv"
                arrs.append(_read_metric_column(p, spec.column, window_s))

    if not arrs:
        return np.array([])
    
    lengths = [a.size for a in arrs]
    if len(set(lengths)) == 1 and lengths[0] > 0:
        stacked = np.stack(arrs, axis=0)  # shape: (n_seeds, n_ticks)
        return np.mean(stacked, axis=0)   # shape: (n_ticks,) — per-tick mean across seeds
    else:
        return np.concatenate(arrs)  


## @brief Pool metric samples for one link from the field traces.
#
# In scenario mode reads both directions using the bh2 filename pattern.
# In node mode (``peer == "neighbors"``) reads the IH trace file using the
# field layout:
# ``per_day/<node>/csvs/<node>/IH_<metric>__<node>_to_neighbors_trace.csv``
#
# @param field_scen_dir  Field traces root directory.
# @param src             Source node label.
# @param peer            Peer node label, or ``"neighbors"`` for node mode.
# @param spec            Metric specification.
# @return                Concatenated float64 array of all valid samples.
def _pool_field(field_scen_dir: Path, src: str, peer: str,
                spec: _MetricSpec, window_s: float | None = None) -> np.ndarray:
    if not field_scen_dir.is_dir():
        return np.array([])

    arrs: list[np.ndarray] = []
    if peer == "neighbors":
        p = field_scen_dir / src / "csvs" / src / f"IH_{spec.short}__{src}_to_neighbors_trace.csv"
        arrs.append(_read_metric_column(p, spec.column, window_s))
    else:
        for a, b in [(src, peer), (peer, src)]:
            p = field_scen_dir / "csvs" / a / f"bh2_{spec.short}__{a}_to_{b}_trace.csv"
            arrs.append(_read_metric_column(p, spec.column, window_s))
    return np.concatenate(arrs) if arrs else np.array([])


## @brief Compute the two-sample K-S D-statistic between two distributions.
#
# Returns the maximum absolute difference between the two empirical CDFs.
# The p-value is intentionally not returned — with N in the tens of thousands
# it is always near zero even for operationally trivial differences.
#
# @param a  First sample array.
# @param b  Second sample array.
# @return   K-S D statistic in [0, 1], or NaN if either array is empty.
def _ks_2samp(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_s = np.sort(a)
    b_s = np.sort(b)
    joint = np.sort(np.concatenate([a_s, b_s]))
    cdf_a = np.searchsorted(a_s, joint, side="right") / a_s.size
    cdf_b = np.searchsorted(b_s, joint, side="right") / b_s.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


## @brief Compute Freedman-Diaconis bin edges for a pooled value array.
#
# Bin width is ``2 * IQR * n^(-1/3)``. Falls back to a linear grid when
# the IQR is zero or all values are identical. Bin count is clamped to
# [@p min_bins, @p max_bins].
#
# @param values    1-D array of pooled sim and field values.
# @param max_bins  Maximum number of bins (default: 80).
# @param min_bins  Minimum number of bins (default: 20).
# @return          1-D array of bin edge values.
def _fd_bins(values: np.ndarray, max_bins: int = 80, min_bins: int = 20) -> np.ndarray:
    if values.size < 2:
        lo = float(np.min(values)) if values.size else 0.0
        return np.linspace(lo, lo + 1.0, min_bins + 1)
    q25, q75 = np.percentile(values, [25, 75])
    iqr = float(q75 - q25)
    lo, hi = float(np.min(values)), float(np.max(values))
    if iqr <= 0 or hi <= lo:
        return np.linspace(lo, hi + 1e-9, min_bins + 1)
    width  = 2.0 * iqr / (values.size ** (1.0 / 3.0))
    n_bins = int(np.clip(round((hi - lo) / width), min_bins, max_bins))
    return np.linspace(lo, hi, n_bins + 1)


## @brief Evaluate a Gaussian KDE on @p x_grid using Silverman's bandwidth rule.
#
# Subsamples to @p max_samples with a fixed RNG seed (0) for reproducibility
# when the input is large. Returns an array of NaN when either input is empty.
#
# @param values       1-D array of observed values.
# @param x_grid       1-D array of evaluation points.
# @param max_samples  Maximum samples before subsampling (default: 20000).
# @return             1-D array of density estimates at each point in @p x_grid.
def _kde_gaussian(values: np.ndarray, x_grid: np.ndarray,
                  max_samples: int = 20000) -> np.ndarray:
    n = values.size
    if n == 0 or x_grid.size == 0:
        return np.full_like(x_grid, np.nan, dtype=np.float64)
    if n > max_samples:
        idx    = np.random.default_rng(0).choice(n, size=max_samples, replace=False)
        values = values[idx]
        n      = max_samples
    sigma = float(np.std(values))
    if sigma <= 0:
        sigma = 1.0
    h = 1.06 * sigma * n ** (-1.0 / 5.0)
    if h <= 0:
        h = 1e-3
    diff = (x_grid[:, None] - values[None, :]) / h
    return np.sum(np.exp(-0.5 * diff * diff), axis=1) / (n * h * np.sqrt(2.0 * np.pi))


## @brief Return median, mean, and IQR for a sample array.
#
# Returns NaN for all three statistics when @p values is empty.
#
# @param values  1-D float64 array of samples.
# @return        Dict with keys ``"median"``, ``"mean"``, and ``"iqr"``.
def _summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        nan = float("nan")
        return {"median": nan, "mean": nan, "iqr": nan}
    q25, med, q75 = np.percentile(values, [25, 50, 75])
    return {
        "median": float(med),
        "mean":   float(np.mean(values)),
        "iqr":    float(q75 - q25),
    }


_SIM_COLOR   = "#ff7f0e"
_FIELD_COLOR = "#1f77b4"
_UNIT_BY_SHORT = {"snr": "dB", "rcpi": "dB", "mcs": "", "per": "", "throughput": "Mbps"}


## @brief Render a normalized histogram overlay of sim vs field for one (link, metric).
#
# Sim is shown in orange, field in blue. Vertical dotted lines mark each
# median. The subtitle reports K-S distance, |Δmedian|, and |Δmean|.
# The caller is responsible for saving and closing the returned figure.
#
# @param sim_vals    Pooled sim samples.
# @param field_vals  Field samples.
# @param spec        Metric specification.
# @param scenario    Scenario name for the figure title.
# @param src         Source node label.
# @param peer        Peer node label.
# @param ks_d        Pre-computed K-S D statistic.
# @return            Matplotlib Figure.
def _plot_one(sim_vals: np.ndarray, field_vals: np.ndarray,
              spec: _MetricSpec, scenario: str, src: str, peer: str,
              ks_d: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sim_s   = _summary(sim_vals)
    field_s = _summary(field_vals)

    if sim_vals.size == 0 and field_vals.size == 0:
        ax.text(0.5, 0.5, "no samples", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        return fig

    pooled      = np.concatenate([v for v in (sim_vals, field_vals) if v.size])
    bins        = _fd_bins(pooled)
    lo, hi      = float(bins[0]), float(bins[-1])
    pad         = max(0.05 * (hi - lo), 1e-6)
    x_grid      = np.linspace(lo - pad, hi + pad, 400)
    unit        = _UNIT_BY_SHORT.get(spec.short, "")
    unit_suffix = f" {unit}" if unit else ""

    if sim_vals.size:
        sim_den, _ = np.histogram(sim_vals, bins=bins, density=True)
        ax.stairs(sim_den, bins, fill=True,
                  color=_SIM_COLOR, alpha=0.18, edgecolor=_SIM_COLOR, linewidth=1.2,
                  label=f"sim  n={sim_vals.size:,}  "
                        f"med={sim_s['median']:.2f}  mean={sim_s['mean']:.2f}{unit_suffix}")
        ax.plot(x_grid, _kde_gaussian(sim_vals, x_grid),
                color=_SIM_COLOR, linewidth=2.0, alpha=0.95)
        ax.axvline(sim_s["median"], color=_SIM_COLOR,
                   linestyle=":", linewidth=0.8, alpha=0.6)
    if field_vals.size:
        field_den, _ = np.histogram(field_vals, bins=bins, density=True)
        ax.stairs(field_den, bins, fill=True,
                  color=_FIELD_COLOR, alpha=0.18, edgecolor=_FIELD_COLOR, linewidth=1.2,
                  label=f"field  n={field_vals.size:,}  "
                        f"med={field_s['median']:.2f}  mean={field_s['mean']:.2f}{unit_suffix}")
        ax.plot(x_grid, _kde_gaussian(field_vals, x_grid),
                color=_FIELD_COLOR, linewidth=2.0, alpha=0.95)
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


## @brief Clip values to the metric's integer cap.
#
# A no-op when @p cap is ``None`` or @p values is empty.
#
# @param values  Sample array to clip.
# @param cap     Upper bound, or ``None`` to skip.
# @return        Clipped array, or the original array if no cap applies.
def _apply_mcs_cap(values: np.ndarray, cap: int | None) -> np.ndarray:
    if cap is None or values.size == 0:
        return values
    return np.clip(values, None, cap)


## @brief Write one heatmap PNG for a single (metric, score) combination.
#
# Rows are scenarios, columns are (src, peer) links. Each cell shows
# @p value_col for @p spec.short. Color scale is viridis from 0 to the
# maximum finite value. Returns ``False`` if no matching rows are found.
#
# @param rows         All result dicts from the batch.
# @param spec         Metric specification to filter rows by.
# @param value_col    Column key to read from each row (e.g. ``"ks_statistic"``).
# @param value_label  Human-readable label for the color bar.
# @param out_path     Destination path for the PNG file.
# @return             ``True`` if the figure was written, ``False`` otherwise.
def _heatmap_png(rows: list[dict], spec: _MetricSpec, value_col: str,
                 value_label: str, out_path: Path) -> bool:
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
    im   = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax)

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

    unit        = _UNIT_BY_SHORT.get(spec.short, "") if value_col != "ks_statistic" else ""
    unit_suffix = f" [{unit}]" if unit else ""
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"{value_label}{unit_suffix}")
    ax.set_title(f"{value_label}  —  {spec.label}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


## @brief Write one heatmap PNG per (metric, score) combination for the batch.
#
# Writes to ``<batch_root>/summary/``. Score types are ``|Δmedian|``,
# ``|Δmean|``, and K-S. Skips combinations with no finite data.
#
# @param rows        All result dicts from the batch.
# @param metrics     List of @ref _MetricSpec instances to iterate over.
# @param batch_root  Root directory of the batch; ``summary/`` is created here.
def _write_batch_heatmaps(rows: list[dict], metrics: list[_MetricSpec],
                          batch_root: Path) -> None:
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


## @brief Run the sim-vs-field comparison for one scenario directory.
#
# In scenario mode derives the field directory from the scenario name via
# @ref sim_to_field_scenario. In node mode uses @p field_dir directly.
# Discovers (src, peer) pairs, pools sim and field samples, computes K-S
# and summary statistics, writes one histogram PNG per (pair, metric), and
# returns a list of result dicts for the summary CSV.
#
# @param scenario_dir  Scenario directory containing ``sim_traces/``.
# @param field_root    Field traces root (used in scenario mode only).
# @param metrics       List of @ref _MetricSpec instances to evaluate.
# @param out_dir       Output directory for PNGs and per-scenario metrics CSV.
# @param field_dir     Explicit field traces directory; when provided bypasses
#                      name derivation (node mode).
# @return              List of result dicts, one per (pair, metric) combination.
def _process_scenario(scenario_dir: Path, field_root: Path, metrics: list[_MetricSpec],
                      out_dir: Path, field_dir: Path | None = None,
                      window_s: float | None = None) -> list[dict]:
    if field_dir is None:
        field_name     = sim_to_field_scenario(scenario_dir.name)
        resolved_field = (field_root / field_name) if field_name else None
        print(f"Processing {field_name}")
        mode = "scenario"
    else:
        field_name     = field_dir.name
        resolved_field = field_dir
        print("Processing node scenario...")
        mode = "node"

    if resolved_field is None or not resolved_field.is_dir():
        print(f"Error: field traces not found (expected {resolved_field}); skipping")
        return []

    sim_traces_root = scenario_dir / "sim_traces"
    if not sim_traces_root.is_dir():
        print(f"Error: no sim_traces/ under {scenario_dir}; skipping")
        return []

    seed_dirs = sorted(d for d in sim_traces_root.iterdir()
                       if d.is_dir() and d.name.startswith("seed-"))
    if not seed_dirs:
        print(f"Error: no seed-* under {sim_traces_root}; skipping")
        return []

    pairs = sorted(_discover_pairs(seed_dirs[0], mode, resolved_field))
    if not pairs:
        print("Error: Pairs not found, validation cannot be made")
        return []
    
    rows: list[dict] = []
    for src, peer in pairs:
        for spec in metrics:
            sim_vals   = _apply_mcs_cap(_pool_sim(scenario_dir, src, peer, spec, window_s),
                                        spec.integer_cap)
            field_vals = _apply_mcs_cap(_pool_field(resolved_field, src, peer, spec, window_s),
                                        spec.integer_cap)
            if sim_vals.size == 0 and field_vals.size == 0:
                continue
            ks_d      = _ks_2samp(sim_vals, field_vals)
            sim_s     = _summary(sim_vals)
            field_s   = _summary(field_vals)
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
                "abs_diff_medians": abs(field_s["median"] - sim_s["median"]) if both_med  else float("nan"),
                "abs_diff_means":   abs(field_s["mean"]   - sim_s["mean"])   if both_mean else float("nan"),
                "ks_statistic":     ks_d,
            })
            fig      = _plot_one(sim_vals, field_vals, spec, scenario_dir.name,
                                 src, peer, ks_d)
            png_dir  = out_dir / "pngs" / src
            png_dir.mkdir(parents=True, exist_ok=True)
            png_path = png_dir / f"hist_{spec.short}__{src}_to_{peer}.png"
            fig.savefig(png_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {png_path.relative_to(scenario_dir)}")
    return rows


## @brief CLI entry point for the sim-vs-field comparison tool.
#
# In node mode @p batch_root is the scenario directory itself and
# ``--field-root`` points at the per-day field traces root.
# In scenario mode @p batch_root contains named scenario subdirectories
# each with their own ``sim_traces/``.
#
# @param argv  Argument list; defaults to ``sys.argv[1:]`` when ``None``.
# @return      0 on success, 1 on error.
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Histogram + KDE comparison of sim vs ARPO field.")
    p.add_argument("--batch_root", required=True,
                   help="output root directory of simulation")
    p.add_argument("--mode", "-m", choices=["scenario", "node"], required=True,
                   help="format of the provided dataset")
    p.add_argument("--field-root", default=str(FIELD_TRACES_ROOT),
                   help="file path root directory of arpo_data field traces")
    p.add_argument("--only", default=None,
                   help="restrict to one scenario name (scenario mode only)")
    p.add_argument("--metrics", default="snr,rcpi,mcs",
                   help="comma-separated metric shorts to compare")
    p.add_argument("--window", type=float, default=None,
                   help="compare only the first N seconds of sim AND field "
                        "(by the trace 'sec' column); default: use all rows")
    args = p.parse_args(argv)

    window_s = args.window
    if window_s is not None:
        print(f"comparison window: first {window_s:g} s")

    batch_root = Path(args.batch_root).resolve()
    if not batch_root.is_dir():
        print(f"Error: batch root not found: {batch_root}", file=sys.stderr)
        return 1

    metrics: list[_MetricSpec] = []
    for short in [s.strip() for s in args.metrics.split(",") if s.strip()]:
        spec = _METRICS_BY_SHORT.get(short)
        if spec is None:
            print(f"Error: unknown metric: {short} (known: {list(_METRICS_BY_SHORT)})",
                  file=sys.stderr)
            return 1
        metrics.append(spec)

    all_rows: list[dict] = []

    if args.mode == "node": #< Used for current datasets
        field_dir = Path(args.field_root).resolve()
        if not field_dir.is_dir():  
            print(f"Error: field-root not found: {field_dir}", file=sys.stderr)
            return 1
        print(f"[{batch_root.name}]")
        out_dir = batch_root / "validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = _process_scenario(batch_root, Path(), metrics, out_dir,
                                 field_dir=field_dir, window_s=window_s)
        if rows:
            scen_csv = out_dir / "metrics.csv"
            pd.DataFrame(rows).to_csv(scen_csv, index=False)
            print(f"    wrote {scen_csv.relative_to(batch_root)}")
        all_rows.extend(rows)

    elif args.mode == "scenario": #< Used for scenario based datasets
        field_root = Path(args.field_root).resolve()
        if not field_root.is_dir():
            print(f"Error: field root not found: {field_root}", file=sys.stderr)
            return 1
        scenarios = sorted(s for s in batch_root.iterdir()
                           if s.is_dir() and (s / "sim_traces").is_dir())
        if args.only:
            scenarios = [s for s in scenarios if s.name == args.only]
        if not scenarios:
            print(f"Error: no scenarios with sim_traces/ under {batch_root}",
                  file=sys.stderr)
            return 1
        for scen in scenarios:
            print(f"[{scen.name}]")
            out_dir = scen / "validation"
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = _process_scenario(scen, field_root, metrics, out_dir,
                                     window_s=window_s)
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