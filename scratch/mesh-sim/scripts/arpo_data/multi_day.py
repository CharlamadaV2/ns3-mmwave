## @file multi_day.py
# @brief Day-vs-day comparison for the same scenario family.
#
# Pools per-day trace CSVs at the (link, metric) level, then renders ECDF
# overlays and computes two-sample K-S distances for every day pair.
# Antenna identity is dropped here — drill into the per-day plots if you
# need per-radio breakdown.
#
# **Output files written to multi_day_root/**
# | File                        | Contents                                          |
# |-----------------------------|---------------------------------------------------|
# | ``<family>/pngs/<src>/``    | ECDF overlay PNGs per (link, metric).             |
# | ``_per_day_stats.csv``      | Quantile summary per (family, day, link, metric). |
# | ``_pairwise_ks.csv``        | K-S statistic + median delta for every day pair.  |

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import KNOWN_BAD_SCENARIOS, MULTI_DAY_DIR, PER_DAY_DIR

## @brief Regex matching the date suffix of a scenario directory name.
#
# Handles two naming conventions:
# - ``<family>_<MMDDYYYY>``
# - ``<family>_baseline_<N>_<MMDDYYYY>``
_SUFFIX_RE = re.compile(r"_(?:baseline_\d+_)?(\d{8})$")

## @brief Regex matching a per-source trace CSV filename.
#
# Expected form: ``bh2_<metric>__<src>_to_<peer>_trace.csv``
_TRACE_RE = re.compile(r"^bh2_([a-z]+)__([a-z0-9]+)_to_([a-z0-9]+)_trace\.csv$")


## @brief Specification for loading and rendering one radio metric.
#
# Instances are collected in @ref _METRICS and looked up by prefix or short name.
@dataclass(frozen=True)
class _MetricSpec:
    prefix: str  ##< CSV filename prefix (e.g. ``"bh2_snr"``).
    column: str  ##< DataFrame column name holding the numeric values.
    label:  str  ##< Axis label shown on plots (e.g. ``"SNR (dB)"``).
    short:  str  ##< Short key used in filenames and CSV outputs (e.g. ``"snr"``).


## @brief All supported backhaul metrics, in display order.
_METRICS: tuple[_MetricSpec, ...] = (
    _MetricSpec("bh2_snr",        "snr_db",   "SNR (dB)",   "snr"),
    _MetricSpec("bh2_rcpi",       "rcpi_dbm", "RCPI (dBm)", "rcpi"),
    _MetricSpec("bh2_mcs",        "mcs_tx",   "MCS",        "mcs"),
    _MetricSpec("bh2_per",        "per",      "PER",        "per"),
    _MetricSpec("bh2_throughput", "mbps",     "PHY (Mbps)", "throughput"),
)
_METRICS_BY_PREFIX = {m.prefix: m for m in _METRICS}  ##< Lookup by CSV prefix.
_METRICS_BY_SHORT  = {m.short:  m for m in _METRICS}  ##< Lookup by short name.

## @brief Minimum samples per day for a bag to be included in ECDF/K-S analysis.
#
# Bags below this threshold produce unreliable ECDFs and noisy K-S statistics.
_MIN_SAMPLES_PER_DAY = 100


## @brief Parse a scenario name into its family and date components.
#
# @param name Scenario directory name, e.g. ``"1-1_static_baseline_2_04162026"``.
# @return ``(family, date_str)`` tuple such as ``("1-1_static", "04162026")``,
#         or ``None`` if the name does not match the expected pattern.
def _parse_scenario(name: str) -> tuple[str, str] | None:
    m = _SUFFIX_RE.search(name)
    if not m:
        return None
    return name[:m.start()], m.group(1)


## @brief Group scenario directories under a per-day root by family and date.
#
# Scenarios listed in @ref KNOWN_BAD_SCENARIOS are silently excluded.
#
# @param per_day_root Root directory produced by the ``plot`` subcommand.
# @return Nested mapping ``{family: {date_str: [scenario_dir, ...]}}``
def _scenarios_by_family(per_day_root: Path) -> dict[str, dict[str, list[Path]]]:
    out: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for scen in sorted(per_day_root.iterdir()):
        if not scen.is_dir() or scen.name in KNOWN_BAD_SCENARIOS:
            continue
        parsed = _parse_scenario(scen.name)
        if parsed is None:
            continue
        family, day = parsed
        out[family][day].append(scen)
    return out


## @brief All pooled samples for one (family, day, link, metric) combination.
@dataclass
class _DayBag:
    values:      np.ndarray            ##< 1-D array of all observed metric values.
    beam_pairs:  set[tuple[str, str]]  ##< Unique (local_mac, sta_mac) pairs seen.
    n_scenarios: int                   ##< Number of scenario directories contributed.


## @brief Load and pool trace CSVs across all scenarios for one family.
#
# Iterates every per-day directory, finds trace CSVs matching @ref _TRACE_RE,
# and accumulates samples into @ref _DayBag objects keyed by
# ``(day, src, peer, metric_short)``.
#
# @param days Mapping ``{date_str: [scenario_dir, ...]}`` for one family.
# @return Dict keyed by ``(day, src_rab, peer_rab, metric_short)`` → @ref _DayBag.
def _load_traces_for_family(
    days: dict[str, list[Path]],
    *,
    audit: bool = False,
) -> dict[tuple[str, str, str, str], _DayBag]:
    bags: dict[tuple[str, str, str, str], _DayBag] = {}
    for day, scen_dirs in days.items():
        for scen_dir in scen_dirs:
            csvs_root = scen_dir / "csvs"
            if not csvs_root.is_dir():
                continue
            for src_dir in sorted(csvs_root.iterdir()):
                if not src_dir.is_dir():
                    continue
                for csv_path in sorted(src_dir.iterdir()):
                    parsed = _TRACE_RE.match(csv_path.name)
                    if parsed is None:
                        continue
                    metric_short, src, peer = parsed.groups()
                    prefix = f"bh2_{metric_short}"
                    spec = _METRICS_BY_PREFIX.get(prefix)
                    if spec is None:
                        continue
                    df = pd.read_csv(csv_path, usecols=lambda c: c in
                                     (spec.column, "tag_local_mac", "tag_sta_mac"))
                    if spec.column not in df.columns or df.empty:
                        continue
                    raw_col = df[spec.column]
                    vals = pd.to_numeric(raw_col, errors="coerce").dropna()
                    if audit:
                        _audit_csv(csv_path, raw_col, vals, day, src, peer, spec.short)
                    if vals.empty:
                        continue
                    key = (day, src, peer, spec.short)
                    bag = bags.get(key)
                    if bag is None:
                        bag = _DayBag(values=vals.to_numpy(),
                                      beam_pairs=set(), n_scenarios=0)
                        bags[key] = bag
                    else:
                        bag.values = np.concatenate([bag.values, vals.to_numpy()])
                    if "tag_local_mac" in df.columns and "tag_sta_mac" in df.columns:
                        pairs = df[["tag_local_mac", "tag_sta_mac"]].dropna()
                        for lm, sm in pairs.itertuples(index=False):
                            bag.beam_pairs.add((str(lm), str(sm)))
                    bag.n_scenarios += 1
    return bags


def _audit_csv(csv_path: Path, raw_col: pd.Series, parsed: pd.Series,
               day: str, src: str, peer: str, metric: str) -> None:
    """Compare raw vs parsed counts/max — surfaces NaN-coerce drops + tail-loss."""
    raw_n = int(raw_col.size)
    parsed_n = int(parsed.size)
    raw_max = pd.to_numeric(raw_col, errors="coerce").max()
    parsed_max = float(parsed.max()) if parsed_n else float("nan")
    dropped = raw_n - parsed_n
    flag = ""
    if dropped > 0:
        flag += f"  DROPPED={dropped}"
    if parsed_n and pd.notna(raw_max) and float(raw_max) > parsed_max + 1e-9:
        flag += f"  MAX_MISMATCH(raw={float(raw_max):.3f} > parsed={parsed_max:.3f})"
    print(f"  [audit] {day} {src}->{peer} {metric:>10s}: "
          f"raw_n={raw_n} parsed_n={parsed_n} "
          f"raw_max={float(raw_max) if pd.notna(raw_max) else 'NaN'} "
          f"parsed_max={parsed_max}{flag}")

## @brief Compute quantile summary statistics for an array of values.
#
# @param values 1-D numeric array.
# @return Dict with keys ``n_samples``, ``mean``, ``std``, ``min``,
#         ``q05``, ``q25``, ``median``, ``q75``, ``q95``, ``max``.
def _summary(values: np.ndarray) -> dict:
    q = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "n_samples": int(values.size),
        "mean":      float(np.mean(values)),
        "std":       float(np.std(values)),
        "min":       float(np.min(values)),
        "q05":       float(q[0]),
        "q25":       float(q[1]),
        "median":    float(q[2]),
        "q75":       float(q[3]),
        "q95":       float(q[4]),
        "max":       float(np.max(values)),
    }


## @brief Compute the two-sample Kolmogorov-Smirnov statistic without scipy.
#
# Evaluates ``max |F_a(x) − F_b(x)|`` over the joint support by merging
# both sorted arrays and using binary search to evaluate each empirical CDF.
#
# @param a First sample array.
# @param b Second sample array.
# @return K-S statistic in [0, 1]; larger values indicate greater distributional
#         distance between the two days.
def _ks_2samp(a: np.ndarray, b: np.ndarray) -> float:
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    joint    = np.sort(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, joint, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, joint, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


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


_UNIT_BY_SHORT = {"snr": "dB", "rcpi": "dB", "mcs": "", "per": "", "throughput": "Mbps"}


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


def _plot_histograms(
    family: str,
    src: str,
    peer: str,
    metric: _MetricSpec,
    bags_by_day: dict[str, _DayBag],
    ks_value:    float | None,
) -> plt.Figure:
    """One normalized histogram per day, overlaid."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    days = sorted(bags_by_day.keys())
    pooled = np.concatenate([bags_by_day[d].values for d in days])
    bins = _fd_bins(pooled)
    lo, hi = float(bins[0]), float(bins[-1])
    pad = max(0.05 * (hi - lo), 1e-6)
    x_grid = np.linspace(lo - pad, hi + pad, 400)

    medians = {d: float(np.median(bags_by_day[d].values)) for d in days}
    means = {d: float(np.mean(bags_by_day[d].values)) for d in days}
    unit = _UNIT_BY_SHORT.get(metric.short, "")
    unit_suffix = f" {unit}" if unit else ""

    cmap = plt.get_cmap("tab10")
    for i, day in enumerate(days):
        bag = bags_by_day[day]
        color = cmap(i % 10)

        density, edges = np.histogram(bag.values, bins=bins, density=True)
        ax.stairs(
            density, edges, fill=True,
            color=color, alpha=0.18, edgecolor=color, linewidth=1.2,
            label=f"{_fmt_day(day)}   n={bag.values.size:,}   "
                  f"med={medians[day]:.2f}  mean={means[day]:.2f}{unit_suffix}",
        )
        kde = _kde_gaussian(bag.values, x_grid)
        ax.plot(x_grid, kde, color=color, linewidth=2.0, alpha=0.95)
        ax.axvline(medians[day], color=color,
                   linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(metric.label, fontweight="bold")
    ax.set_ylabel("density  (area = 1, comparable across N)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    # Build the subtitle: median delta between the two most-different days.
    if len(days) == 2:
        d_a, d_b = days
        delta     = medians[d_b] - medians[d_a]
        delta_str = (f"Δmed ({_fmt_day(d_b)} − {_fmt_day(d_a)}) "
                     f"= {delta:+.2f}{unit_suffix}")
    else:
        max_pair = max(
            ((a, b) for i, a in enumerate(days) for b in days[i + 1:]),
            key=lambda ab: abs(medians[ab[1]] - medians[ab[0]]),
            default=None,
        )
        delta_str = None
        if max_pair is not None:
            a, b  = max_pair
            delta = medians[b] - medians[a]
            delta_str = (f"max |Δmed| = {abs(delta):.2f}{unit_suffix}   "
                         f"({_fmt_day(a)} vs {_fmt_day(b)})")

    main     = f"{family}:  {src} → {peer}   ({metric.label})"
    sub_bits = [f"{len(days)} days"]
    if delta_str:
        sub_bits.append(delta_str)
    if ks_value is not None:
        sub_bits.append(f"K–S = {ks_value:.3f}")
    sub = "   ·   ".join(sub_bits)
    fig.suptitle(main, fontsize=13, fontweight="bold", y=0.98)
    fig.text(0.5, 0.905, sub, fontsize=10, ha="center", color="0.2")

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


## @brief Format an 8-digit date string as ``MM/DD/YYYY`` for display.
#
# Falls back to the raw string if it does not match the expected format.
#
# @param day 8-character string such as ``"04162026"``.
# @return Formatted string such as ``"04/16/2026"``.
def _fmt_day(day: str) -> str:
    if len(day) == 8 and day.isdigit():
        return f"{day[0:2]}/{day[2:4]}/{day[4:8]}"
    return day


## @brief Run the full multi-day analysis and write all output files.
#
# For each scenario family that has at least two days of data:
# -# Pool trace CSVs into @ref _DayBag objects via @ref _load_traces_for_family.
# -# Compute quantile summaries for ``_per_day_stats.csv``.
# -# Compute K-S statistics for every day pair for ``_pairwise_ks.csv``.
# -# Render and save an ECDF overlay PNG per (link, metric).
#
# @param per_day_root  Directory of per-day plot output (produced by ``plot``).
# @param multi_day_root Output directory for multi-day results.
# @return Tuple ``(per_day_rows, pairwise_rows)`` of the raw dicts written to CSV,
#         useful for testing or downstream processing.
# @throws FileNotFoundError if ``per_day_root`` does not exist.
def run_multi_day(
    per_day_root:   Path,
    multi_day_root: Path,
    *,
    audit: bool = False,
    family_filter: str | None = None,
) -> tuple[list[dict], list[dict]]:
    if not per_day_root.is_dir():
        raise FileNotFoundError(f"per-day plots dir not found: {per_day_root}")

    families = _scenarios_by_family(per_day_root)
    if family_filter is not None:
        families = {f: d for f, d in families.items() if f == family_filter}
        if not families:
            print(f"No family matches --family={family_filter!r}.")
            return [], []
    multi_families = {fam: days for fam, days in families.items() if len(days) >= 2}

    print(f"Found {len(families)} scenario families "
          f"({len(multi_families)} with >=2 days of data).")
    for fam, days in sorted(families.items()):
        marker      = "[OK]" if len(days) >= 2 else "[--]"
        day_summary = ", ".join(
            f"{_fmt_day(d)} ({len(scens)} scn)"
            for d, scens in sorted(days.items())
        )
        print(f"  {marker} {fam}: {day_summary}")
    if not multi_families:
        print("Nothing to compare; need >=2 days for at least one family.")
        return [], []

    per_day_rows:  list[dict] = []
    pairwise_rows: list[dict] = []

    multi_day_root.mkdir(parents=True, exist_ok=True)

    for family, days in sorted(multi_families.items()):
        print(f"\n[{family}]")
        bags = _load_traces_for_family(days, audit=audit)
        by_link_metric: dict[tuple[str, str, str], dict[str, _DayBag]] = defaultdict(dict)
        for (day, src, peer, metric_short), bag in bags.items():
            if bag.values.size < _MIN_SAMPLES_PER_DAY:
                if audit:
                    print(f"  [audit] dropped (n<{_MIN_SAMPLES_PER_DAY}): "
                          f"{day} {src}->{peer} {metric_short} n={bag.values.size}")
                continue
            by_link_metric[(src, peer, metric_short)][day] = bag

        fam_dir = multi_day_root / family
        for (src, peer, metric_short), bags_by_day in sorted(by_link_metric.items()):
            if len(bags_by_day) < 2:
                continue  # Only one day with enough samples — nothing to compare.

            metric = _METRICS_BY_SHORT[metric_short]

            for day, bag in bags_by_day.items():
                stats = _summary(bag.values)
                per_day_rows.append({
                    "family":       family,
                    "day":          day,
                    "src_rab":      src,
                    "peer_rab":     peer,
                    "metric":       metric_short,
                    "n_scenarios":  bag.n_scenarios,
                    "n_beam_pairs": len(bag.beam_pairs),
                    **stats,
                })

            days_sorted = sorted(bags_by_day.keys())
            ks_value: float | None = None
            for i, day_a in enumerate(days_sorted):
                for day_b in days_sorted[i + 1:]:
                    bag_a = bags_by_day[day_a]
                    bag_b = bags_by_day[day_b]
                    a_vals, b_vals = bag_a.values, bag_b.values
                    ks = _ks_2samp(a_vals, b_vals)
                    med_a = float(np.median(a_vals))
                    med_b = float(np.median(b_vals))
                    mean_a = float(np.mean(a_vals))
                    mean_b = float(np.mean(b_vals))
                    pairwise_rows.append({
                        "family":       family,
                        "src_rab":      src,
                        "peer_rab":     peer,
                        "metric":       metric_short,
                        "day_a":        day_a,
                        "day_b":        day_b,
                        "n_a":          int(a_vals.size),
                        "n_b":          int(b_vals.size),
                        "ks_statistic": ks,
                        "median_a":     med_a,
                        "median_b":     med_b,
                        "median_delta": med_b - med_a,
                        "mean_a":       mean_a,
                        "mean_b":       mean_b,
                        "mean_delta":   mean_b - mean_a,
                    })
                    # With exactly 2 days this is the only pair; surface it on the plot.
                    ks_value = ks

            fig = _plot_histograms(family, src, peer, metric, bags_by_day, ks_value)
            png_dir = fam_dir / "pngs" / src
            png_dir.mkdir(parents=True, exist_ok=True)
            png_path = png_dir / f"{metric.prefix}__{src}_to_{peer}.png"
            fig.savefig(png_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"    wrote {png_path}")

    per_day_csv  = multi_day_root / "_per_day_stats.csv"
    pairwise_csv = multi_day_root / "_pairwise_ks.csv"
    pd.DataFrame(per_day_rows).to_csv(per_day_csv,  index=False)
    pd.DataFrame(pairwise_rows).to_csv(pairwise_csv, index=False)
    print(f"\nwrote {per_day_csv}")
    print(f"wrote {pairwise_csv}")

    return per_day_rows, pairwise_rows


## @brief CLI entry point for the ``multi-day`` subcommand.
#
# Delegates to @ref run_multi_day using the default @ref PER_DAY_DIR and
# @ref MULTI_DAY_DIR paths.
#
# @return 0 on success.
def multi_day(audit: bool = False, family_filter: str | None = None) -> int:
    # CLI entrypoint.
    run_multi_day(PER_DAY_DIR, MULTI_DAY_DIR,
                  audit=audit, family_filter=family_filter)
    return 0