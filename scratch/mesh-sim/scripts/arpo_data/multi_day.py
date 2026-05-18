##@package docstring
# Day-vs-day comparison for the same scenario family.

# Pools per-day trace CSVs at the (link, metric) level, then renders ECDF
# overlays + K-S distances. Antenna identity is dropped here -- drill into
# the per-day plots if you need it.
##

#TODO: Finish Documentation for this page

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import KNOWN_BAD_SCENARIOS, MULTI_DAY_DIR, PER_DAY_DIR

# ``<family>_<MMDDYYYY>`` or ``<family>_baseline_<N>_<MMDDYYYY>``.
_SUFFIX_RE = re.compile(r"_(?:baseline_\d+_)?(\d{8})$")
_TRACE_RE = re.compile(r"^bh2_([a-z]+)__([a-z0-9]+)_to_([a-z0-9]+)_trace\.csv$")

## Documentation for a class.
#
#  More details.
@dataclass(frozen=True)
class _MetricSpec:
    ##How to load + render one metric's trace CSVs.##
    prefix: str
    column: str
    label: str
    short: str


_METRICS: tuple[_MetricSpec, ...] = (
    _MetricSpec("bh2_snr",        "snr_db",   "SNR (dB)",     "snr"),
    _MetricSpec("bh2_rcpi",       "rcpi_dbm", "RCPI (dBm)",   "rcpi"),
    _MetricSpec("bh2_mcs",        "mcs_tx",   "MCS",          "mcs"),
    _MetricSpec("bh2_per",        "per",      "PER",          "per"),
    _MetricSpec("bh2_throughput", "mbps",     "PHY (Mbps)",   "throughput"),
)
_METRICS_BY_PREFIX = {m.prefix: m for m in _METRICS}
_METRICS_BY_SHORT = {m.short: m for m in _METRICS}

# Bags below this many samples make ECDFs meaningless and K-S noisy.
_MIN_SAMPLES_PER_DAY = 100

## @brief
def _parse_scenario(name: str) -> tuple[str, str] | None:
    ##``("1-1_static", "04162026")`` from ``1-1_static_baseline_2_04162026``.##
    m = _SUFFIX_RE.search(name)
    if not m:
        return None
    return name[:m.start()], m.group(1)

## @brief
def _scenarios_by_family(per_day_root: Path) -> dict[str, dict[str, list[Path]]]:
    ##``{family: {day: [scenario_dir, ...]}}`` skipping crashed scenarios.##
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

## Documentation for a class.
#
#  More details.
@dataclass
class _DayBag:
    ##All samples pooled for one (family, day, link, metric).##
    values: np.ndarray
    beam_pairs: set[tuple[str, str]]
    n_scenarios: int

## @brief
def _load_traces_for_family(
    days: dict[str, list[Path]],
) -> dict[tuple[str, str, str, str], _DayBag]:
    ##Returns ``{(day, src, peer, metric_short): _DayBag}`` across all scenarios.##
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
                    vals = pd.to_numeric(df[spec.column], errors="coerce").dropna()
                    if vals.empty:
                        continue
                    key = (day, src, peer, spec.short)
                    bag = bags.get(key)
                    if bag is None:
                        bag = _DayBag(
                            values=vals.to_numpy(),
                            beam_pairs=set(),
                            n_scenarios=0,
                        )
                        bags[key] = bag
                    else:
                        bag.values = np.concatenate([bag.values, vals.to_numpy()])
                    if "tag_local_mac" in df.columns and "tag_sta_mac" in df.columns:
                        pairs = df[["tag_local_mac", "tag_sta_mac"]].dropna()
                        for lm, sm in pairs.itertuples(index=False):
                            bag.beam_pairs.add((str(lm), str(sm)))
                    bag.n_scenarios += 1
    return bags

## @brief
def _summary(values: np.ndarray) -> dict:
    ##Quantile summary used for the per-day-stats CSV.##
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

## @brief
def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ##Step ECDF: x = sorted values, y = (i+1)/n at each step.##
    x = np.sort(values)
    y = np.arange(1, x.size + 1) / x.size
    return x, y

## @brief
def _ks_2samp(a: np.ndarray, b: np.ndarray) -> float:
    ##Two-sample K-S statistic ``max |F_a(x) - F_b(x)|`` (no scipy).##
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    joint = np.sort(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, joint, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, joint, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


_UNIT_BY_SHORT = {"snr": "dB", "rcpi": "dB", "mcs": "", "per": "", "throughput": "Mbps"}

## @brief
def _plot_ecdfs(
    family: str,
    src: str,
    peer: str,
    metric: _MetricSpec,
    bags_by_day: dict[str, _DayBag],
    ks_value: float | None,
) -> plt.Figure:
    ##ECDF overlay, one line per day, with stats table beneath the title.##
    fig, ax = plt.subplots(figsize=(9, 5.5))

    days = sorted(bags_by_day.keys())
    medians = {d: float(np.median(bags_by_day[d].values)) for d in days}
    unit = _UNIT_BY_SHORT.get(metric.short, "")
    unit_suffix = f" {unit}" if unit else ""

    cmap = plt.get_cmap("tab10")
    for i, day in enumerate(days):
        bag = bags_by_day[day]
        x, y = _ecdf(bag.values)
        ax.step(
            x, y, where="post",
            color=cmap(i % 10), linewidth=1.6, alpha=0.9,
            label=f"{_fmt_day(day)}   n={bag.values.size:,}   "
                  f"pairs={len(bag.beam_pairs)}   scn={bag.n_scenarios}   "
                  f"med={medians[day]:.2f}{unit_suffix}",
        )
        ax.axvline(medians[day], color=cmap(i % 10),
                   linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel(metric.label, fontweight="bold")
    ax.set_ylabel("ECDF  (fraction of samples ≤ x)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9,
              title="day              samples / beam-pairs / scenarios / median",
              title_fontsize=8)

    if len(days) == 2:
        d_a, d_b = days
        delta = medians[d_b] - medians[d_a]
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
            a, b = max_pair
            delta = medians[b] - medians[a]
            delta_str = (f"max |Δmed| = {abs(delta):.2f}{unit_suffix}   "
                         f"({_fmt_day(a)} vs {_fmt_day(b)})")

    main = f"{family}:  {src} → {peer}   ({metric.label})"
    sub_bits = [f"{len(days)} days"]
    if delta_str:
        sub_bits.append(delta_str)
    if ks_value is not None:
        sub_bits.append(f"K–S = {ks_value:.3f}")
    sub = "   ·   ".join(sub_bits)
    fig.suptitle(main, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.92, sub, fontsize=10, ha="center", color="0.2")

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig

## @brief
def _fmt_day(day: str) -> str:
    ##``"04162026" -> "04/16/2026"`` for legend labels.##
    if len(day) == 8 and day.isdigit():
        return f"{day[0:2]}/{day[2:4]}/{day[4:8]}"
    return day

## Documentation for a function.
#
#  More details.
def run_multi_day(
    per_day_root: Path,
    multi_day_root: Path,
) -> tuple[list[dict], list[dict]]:
    ##Write PNGs + the two summary CSVs; return ``(per_day_rows, pairwise_rows)``.##
    if not per_day_root.is_dir():
        raise FileNotFoundError(f"per-day plots dir not found: {per_day_root}")

    families = _scenarios_by_family(per_day_root)
    multi_families = {fam: days for fam, days in families.items() if len(days) >= 2}

    print(f"Found {len(families)} scenario families "
          f"({len(multi_families)} with >=2 days of data).")
    for fam, days in sorted(families.items()):
        marker = "[OK]" if len(days) >= 2 else "[--]"
        day_summary = ", ".join(
            f"{_fmt_day(d)} ({len(scens)} scn)"
            for d, scens in sorted(days.items())
        )
        print(f"  {marker} {fam}: {day_summary}")
    if not multi_families:
        print("Nothing to compare; need >=2 days for at least one family.")
        return [], []

    per_day_rows: list[dict] = []
    pairwise_rows: list[dict] = []

    multi_day_root.mkdir(parents=True, exist_ok=True)

    for family, days in sorted(multi_families.items()):
        print(f"\n[{family}]")
        bags = _load_traces_for_family(days)
        by_link_metric: dict[tuple[str, str, str], dict[str, _DayBag]] = defaultdict(dict)
        for (day, src, peer, metric_short), bag in bags.items():
            if bag.values.size < _MIN_SAMPLES_PER_DAY:
                continue
            by_link_metric[(src, peer, metric_short)][day] = bag

        fam_dir = multi_day_root / family
        for (src, peer, metric_short), bags_by_day in sorted(by_link_metric.items()):
            if len(bags_by_day) < 2:
                continue  # only one day with enough samples

            metric = _METRICS_BY_SHORT[metric_short]

            for day, bag in bags_by_day.items():
                stats = _summary(bag.values)
                per_day_rows.append({
                    "family":      family,
                    "day":         day,
                    "src_rab":     src,
                    "peer_rab":    peer,
                    "metric":      metric_short,
                    "n_scenarios": bag.n_scenarios,
                    "n_beam_pairs": len(bag.beam_pairs),
                    **stats,
                })

            days_sorted = sorted(bags_by_day.keys())
            ks_value: float | None = None
            for i, day_a in enumerate(days_sorted):
                for day_b in days_sorted[i + 1:]:
                    bag_a = bags_by_day[day_a]
                    bag_b = bags_by_day[day_b]
                    ks = _ks_2samp(bag_a.values, bag_b.values)
                    median_delta = float(np.median(bag_b.values)
                                         - np.median(bag_a.values))
                    pairwise_rows.append({
                        "family":          family,
                        "src_rab":         src,
                        "peer_rab":        peer,
                        "metric":          metric_short,
                        "day_a":           day_a,
                        "day_b":           day_b,
                        "n_a":             int(bag_a.values.size),
                        "n_b":             int(bag_b.values.size),
                        "ks_statistic":    ks,
                        "median_a":        float(np.median(bag_a.values)),
                        "median_b":        float(np.median(bag_b.values)),
                        "median_delta":    median_delta,
                    })
                    # With 2 days this is the only pair; surface it on the plot.
                    ks_value = ks

            fig = _plot_ecdfs(family, src, peer, metric, bags_by_day, ks_value)
            png_dir = fam_dir / "pngs" / src
            png_dir.mkdir(parents=True, exist_ok=True)
            png_path = png_dir / f"{metric.prefix}__{src}_to_{peer}.png"
            fig.savefig(png_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"    wrote {png_path}")

    per_day_csv = multi_day_root / "_per_day_stats.csv"
    pairwise_csv = multi_day_root / "_pairwise_ks.csv"
    pd.DataFrame(per_day_rows).to_csv(per_day_csv, index=False)
    pd.DataFrame(pairwise_rows).to_csv(pairwise_csv, index=False)
    print(f"\nwrote {per_day_csv}")
    print(f"wrote {pairwise_csv}")

    return per_day_rows, pairwise_rows

## Documentation for a function.
#
#  More details.
def multi_day() -> int:
    ##CLI entrypoint.##
    run_multi_day(PER_DAY_DIR, MULTI_DAY_DIR)
    return 0
