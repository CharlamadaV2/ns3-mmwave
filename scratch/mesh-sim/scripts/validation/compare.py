"""
Validation: mesh-sim vs ARPO field bh2.csv per rab pair, time-series style.

Default invocation (from scratch/mesh-sim, after a sim run):

    python -m scripts.validation.compare 1-1_static_04172026

Picks the latest sim run under outputs/, auto-resolves the field root
(env > in-repo > Desktop), and emits one figure + one metrics CSV per
rab pair under <sim-run>/validation/sim-vs-arpo/<scenario>/.

Each figure has three rows of paired panels (sim left, field right) for
SNR, MCS, RX power, plus a small stats strip with median + IQR + |diff|.
Time is shown explicitly: sim is ~60 s; field is ~3000 s. We are *not*
aligning timestamps -- both sides are steady-state for static collects,
so we're checking that the sim's flat level matches the field's central
tendency, with the field's natural drift visible.

Throughput is intentionally omitted: sim is driven with synthetic
`demand_mbps` (so it always shows non-zero); field is idle telemetry
(so it shows ~zero). The comparison is meaningless without a matching
load profile.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.validation.loaders import (
    filter_sim_csv_for_pair,
    filter_sim_links_for_pair,
    load_field_bh2,
    load_links_csv,
    load_mcs_csv,
    load_rx_power_csv,
)
from scripts.validation.mac_map import (
    KNOWN_RABS,
    build_mac_to_rab,
    filter_field_for_pair,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_field_root() -> Path:
    """
    First of: $MESH_SIM_FIELD_ROOT, in-repo data/arpo_extracted/csv.
    """
    env = os.environ.get("MESH_SIM_FIELD_ROOT")
    if env:
        return Path(env)
    in_repo = REPO_ROOT / "data" / "arpo_extracted" / "csv"
    if in_repo.is_dir():
        return in_repo
    return Path.home() / "Desktop" / "arpo-data" / "arpo_extracted" / "csv"


def latest_sim_run() -> Optional[Path]:
    """
    Most recently modified seed-* dir under outputs/ that contains
    links.csv (the canonical signal of a completed sim). Skips aborted
    runs whose seed dir was created but never populated."""
    out = REPO_ROOT / "outputs"
    if not out.is_dir():
        return None
    candidates = [p.parent for p in out.glob("*/*/*/seed-*/links.csv")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


MCS_FIELD_CAP = 12  # titan caps OLSR routing cost at MCS 12; sim emits 0..14
RCPI_TO_DBM = lambda x: x  # see NOTES.md §5.2 -- field_rcpi treated as dBm

ALL_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("rab1", "rab2"),
    ("rab1", "rab3"),
    ("rab2", "rab3"),
)


# ---------------------------------------------------------------------------
# Field-side smoothing (mirrors scripts/arpo_data/plots.py conventions)
# ---------------------------------------------------------------------------

def _anchor_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``__sec__`` = seconds since this DataFrame's first sample.

    Caller must pass a single-rab slice -- field rabs have skewed clocks
    (NOTES.md §2), so anchoring after concatenating both directions of a
    link gives bogus 50,000 s gaps. Each rab anchors to its own t0.
    """
    if "timestamp_ns" not in df.columns or df.empty:
        return df.assign(__sec__=np.array([]))
    t0 = df["timestamp_ns"].min()
    return df.assign(__sec__=(df["timestamp_ns"] - t0) / 1e9)


def _max_per_bin(df: pd.DataFrame, value_col: str, bin_s: float = 1.0) -> pd.DataFrame:
    """
    Per ``bin_s`` window, keep the row with the max ``value_col``.

    Mirrors scripts.arpo_data.plots._max_across_macs_per_bin: each rab has 4
    Hydra antennas (4 local MACs), so the field reports up to 4 samples for
    the same instant. The traffic for a directional mmWave link uses the
    best-aimed antenna, so "max across MACs" is the correct collapse.
    """
    if df.empty or value_col not in df.columns:
        return df.iloc[0:0]
    valid = df.dropna(subset=[value_col, "__sec__"])
    if valid.empty:
        return valid
    bin_idx = (valid["__sec__"] // bin_s).astype("int64")
    keep = valid.groupby(bin_idx)[value_col].idxmax()
    return valid.loc[keep].sort_values("__sec__")


def _rolling_median_seconds(df: pd.DataFrame, value_col: str, window: str = "30s") -> np.ndarray:
    """
    Time-based rolling median of ``value_col``, returned as a numpy array
    aligned with ``df`` row order. ``df`` must already be sorted by __sec__."""
    if df.empty:
        return np.array([])
    idx = pd.to_timedelta(df["__sec__"].values, unit="s")
    s = pd.Series(df[value_col].values, index=idx)
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    return s.rolling(window, min_periods=1).median().reindex(idx, method="nearest").values


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"n": 0, "median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def _abs_diff(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return abs(a - b)


def compute_metrics(
    snr_field: np.ndarray, snr_sim: np.ndarray,
    mcs_field: np.ndarray, mcs_sim: np.ndarray,
    rxp_field: np.ndarray, rxp_sim: np.ndarray,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for label, f, s, note in [
        ("snr_db", snr_field, snr_sim, ""),
        ("mcs_capped_at_12", mcs_field, mcs_sim, "sim has no HARQ/BLER -- expect overshoot"),
        ("rx_power_dbm", rxp_field, rxp_sim, "field rcpi treated as dBm (semantics unconfirmed)"),
    ]:
        sf = _stats(f)
        ss = _stats(s)
        rows.append({
            "metric": label,
            "field_median": sf["median"],
            "field_iqr": sf["q75"] - sf["q25"] if sf["n"] else float("nan"),
            "sim_median": ss["median"],
            "sim_iqr": ss["q75"] - ss["q25"] if ss["n"] else float("nan"),
            "abs_diff_medians": _abs_diff(sf["median"], ss["median"]),
            "n_field": sf["n"],
            "n_sim": ss["n"],
            "note": note,
        })
    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

SIM_COLOR = "#ff7f0e"
FIELD_COLOR = "#1f77b4"


def _plot_sim_timeseries(ax, t_s: np.ndarray, y: np.ndarray, label: str,
                         ylabel: str, ylim: Optional[Tuple[float, float]] = None) -> None:
    ax.plot(t_s, y, "-", linewidth=1.4, color=SIM_COLOR, alpha=0.95, label=label)
    ax.set_xlabel("sim time (s)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _plot_field_timeseries(ax, df: pd.DataFrame, value_col: str, ylabel: str,
                           ylim: Optional[Tuple[float, float]] = None,
                           rolling: str = "30s") -> None:
    if df.empty:
        ax.text(0.5, 0.5, "no field samples", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="0.5")
        ax.set_xlabel("field time (s)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        return
    sec = df["__sec__"].values
    raw = df[value_col].values
    smooth = _rolling_median_seconds(df, value_col, rolling)
    ax.plot(sec, raw, ".", markersize=1.2, alpha=0.15, color=FIELD_COLOR)
    ax.plot(sec, smooth, "-", linewidth=1.4, alpha=0.9, color=FIELD_COLOR,
            label=f"{rolling} rolling median")
    ax.set_xlabel("field time (s)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _stats_strip(ax, metrics: List[Dict[str, object]]) -> None:
    ax.axis("off")
    headers = ["metric", "sim median (IQR)", "field median (IQR)",
               "|diff medians|", "n sim", "n field", "notes"]
    body = []
    for r in metrics:
        body.append([
            str(r["metric"]),
            _fmt_med_iqr(r["sim_median"], r["sim_iqr"]),
            _fmt_med_iqr(r["field_median"], r["field_iqr"]),
            _fmt(r["abs_diff_medians"]),
            str(r["n_sim"]),
            str(r["n_field"]),
            str(r["note"]),
        ])
    tbl = ax.table(cellText=body, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    # Per-column width tuned so 'metric' and 'notes' don't truncate at 1080px.
    col_widths = [0.13, 0.14, 0.14, 0.10, 0.07, 0.07, 0.35]
    for i, w in enumerate(col_widths):
        for r in range(len(body) + 1):
            cell = tbl[(r, i)]
            cell.set_width(w)


def _fmt_med_iqr(med: float, iqr: float) -> str:
    if not np.isfinite(med):
        return "n/a"
    if not np.isfinite(iqr) or iqr == 0:
        return f"{med:.2f}"
    return f"{med:.2f} (IQR {iqr:.2f})"


def _fmt(v) -> str:
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:.2f}" if isinstance(v, float) else str(v)
    except Exception:
        return "n/a"


def build_figure(
    field_snr: pd.DataFrame, sim_snr_t: np.ndarray, sim_snr_y: np.ndarray,
    field_mcs: pd.DataFrame, sim_mcs_t: np.ndarray, sim_mcs_y: np.ndarray,
    field_rxp: pd.DataFrame, sim_rxp_t: np.ndarray, sim_rxp_y: np.ndarray,
    metrics: List[Dict[str, object]],
    field_scenario: str,
    pair: Tuple[str, str],
    radio_param_note: str,
) -> plt.Figure:
    """Three rows of (sim, field) timeseries panels + a stats strip."""
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(
        4, 2, height_ratios=[2, 2, 2, 1.2],
        hspace=0.55, wspace=0.18,
        top=0.94, bottom=0.08, left=0.06, right=0.98,
    )

    ax_snr_sim = fig.add_subplot(gs[0, 0])
    ax_snr_field = fig.add_subplot(gs[0, 1])
    ax_mcs_sim = fig.add_subplot(gs[1, 0])
    ax_mcs_field = fig.add_subplot(gs[1, 1])
    ax_rxp_sim = fig.add_subplot(gs[2, 0])
    ax_rxp_field = fig.add_subplot(gs[2, 1])
    ax_table = fig.add_subplot(gs[3, :])

    snr_ylim = _shared_ylim([sim_snr_y, field_snr["field_snr"].values if not field_snr.empty else np.array([])], pad=2.0)
    mcs_ylim = (-0.5, MCS_FIELD_CAP + 0.5)
    rxp_ylim = _shared_ylim([sim_rxp_y, RCPI_TO_DBM(field_rxp["field_rcpi"]).values if not field_rxp.empty else np.array([])], pad=3.0)

    _plot_sim_timeseries(ax_snr_sim, sim_snr_t, sim_snr_y,
                         "sim SINR", "SINR (dB)", ylim=snr_ylim)
    ax_snr_sim.set_title(f"SNR  —  sim", fontsize=10, loc="left")

    _plot_field_timeseries(ax_snr_field, field_snr, "field_snr",
                           "SNR (dB)", ylim=snr_ylim)
    ax_snr_field.set_title("SNR  —  field", fontsize=10, loc="left")

    _plot_sim_timeseries(ax_mcs_sim, sim_mcs_t, np.clip(sim_mcs_y, None, MCS_FIELD_CAP),
                         "sim MCS (capped at 12)", "MCS index", ylim=mcs_ylim)
    ax_mcs_sim.set_title("MCS  —  sim", fontsize=10, loc="left")

    _plot_field_timeseries(ax_mcs_field, field_mcs, "field_mcs_tx",
                           "MCS index", ylim=mcs_ylim)
    ax_mcs_field.set_title("MCS  —  field (TX)", fontsize=10, loc="left")

    _plot_sim_timeseries(ax_rxp_sim, sim_rxp_t, sim_rxp_y,
                         "sim RX power", "dBm", ylim=rxp_ylim)
    ax_rxp_sim.set_title("RX power  —  sim", fontsize=10, loc="left")

    _plot_field_timeseries(ax_rxp_field, field_rxp, "field_rcpi",
                           "dBm (assumed)", ylim=rxp_ylim)
    ax_rxp_field.set_title("RX power  —  field (rcpi)", fontsize=10, loc="left")

    _stats_strip(ax_table, metrics)

    field_dur = (field_snr["__sec__"].max() if not field_snr.empty else 0)
    sim_dur = float(sim_snr_t.max()) if sim_snr_t.size else 0.0
    fig.suptitle(
        f"mesh-sim vs ARPO  |  scenario={field_scenario}  |  link={pair[0]}↔{pair[1]}  "
        f"|  sim={sim_dur:.0f} s, field≈{field_dur:.0f} s",
        fontsize=12, x=0.02, ha="left", y=0.995,
    )
    fig.text(
        0.02, 0.02,
        f"{radio_param_note}.  "
        "field side: 1 s bin (max across the 4 Hydra antenna MACs) → 30 s rolling median.  "
        "sim MCS post-capped at 12 to match field firmware limit; sim has no HARQ/BLER so MCS overshoots.  "
        "field 'field_rcpi' interpreted as dBm directly (vendor unconfirmed; see NOTES.md §5.2).",
        fontsize=8, color="0.35", ha="left", va="bottom", wrap=True,
    )
    return fig


def _shared_ylim(arrays: List[np.ndarray], pad: float = 1.0) -> Optional[Tuple[float, float]]:
    parts = [a for a in arrays if a is not None and len(a)]
    if not parts:
        return None
    lo = min(np.nanmin(a) for a in parts)
    hi = max(np.nanmax(a) for a in parts)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return (float(lo) - pad, float(hi) + pad)


# ---------------------------------------------------------------------------
# Per-pair work
# ---------------------------------------------------------------------------

def _resolve_sim_pair_ids(pair: Tuple[str, str], sim_id_map: Dict[str, int]) -> Tuple[object, object]:
    if pair[0] in sim_id_map and pair[1] in sim_id_map:
        return sim_id_map[pair[0]], sim_id_map[pair[1]]
    return pair[0], pair[1]


def _field_for_pair(field_dir: Path, pair: Tuple[str, str],
                    mac_to_rab: Dict[str, str]) -> pd.DataFrame:
    """
    Concat both directions of bh2 for the rab pair, with seconds column.

    Each rab is anchored to its own first sample before concat -- field
    clocks are skewed across rabs by tens of thousands of seconds.
    """
    chunks: List[pd.DataFrame] = []
    for own, peer in [(pair[0], pair[1]), (pair[1], pair[0])]:
        df = load_field_bh2(str(field_dir), own)
        if df is None:
            continue
        df = filter_field_for_pair(df, own, peer, mac_to_rab)
        if df.empty:
            continue
        chunks.append(_anchor_seconds(df))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).sort_values("__sec__")


def run_one_pair(
    pair: Tuple[str, str],
    field_dir: Path,
    sim_run: Path,
    sim_links_all: pd.DataFrame,
    sim_mcs_all: Optional[pd.DataFrame],
    sim_rxp_all: Optional[pd.DataFrame],
    sim_id_map: Dict[str, int],
    mac_to_rab: Dict[str, str],
    field_scenario: str,
    out_dir: Path,
    radio_param_note: str,
) -> Optional[Tuple[Path, Path]]:
    """
    Run the comparison for one (rab_a, rab_b) pair. Returns (png, csv) paths
    or None if there's nothing to compare.
    """
    field = _field_for_pair(field_dir, pair, mac_to_rab)
    if field.empty:
        print(f"[validation] {pair[0]}↔{pair[1]}: no field samples, skipping")
        return None

    sim_a, sim_b = _resolve_sim_pair_ids(pair, sim_id_map)
    sim_links = filter_sim_links_for_pair(sim_links_all, sim_a, sim_b)
    if sim_links.empty:
        print(f"[validation] {pair[0]}↔{pair[1]}: no sim rows for this pair, skipping")
        return None
    sim_mcs = filter_sim_csv_for_pair(sim_mcs_all, sim_a, sim_b) if sim_mcs_all is not None else pd.DataFrame()
    sim_rxp = filter_sim_csv_for_pair(sim_rxp_all, sim_a, sim_b) if sim_rxp_all is not None else pd.DataFrame()

    # Field per-metric: max-per-1s-bin across the 4 antenna MACs.
    field_snr = _max_per_bin(field, "field_snr")
    field_mcs = _max_per_bin(field, "field_mcs_tx")
    field_rxp = _max_per_bin(field, "field_rcpi")
    if not field_rxp.empty:
        field_rxp = field_rxp.assign(field_rcpi=RCPI_TO_DBM(field_rxp["field_rcpi"]))

    # Sim time-series (already one row per tick per pair).
    sim_snr_t = sim_links["time_s"].to_numpy()
    sim_snr_y = sim_links["sinr_db"].to_numpy()
    sim_mcs_t = sim_mcs["time_s"].to_numpy() if not sim_mcs.empty else np.array([])
    sim_mcs_y = sim_mcs["mcs_index"].to_numpy() if not sim_mcs.empty else np.array([])
    sim_rxp_t = sim_rxp["time_s"].to_numpy() if not sim_rxp.empty else np.array([])
    sim_rxp_y = sim_rxp["rx_power_dbm"].to_numpy() if not sim_rxp.empty else np.array([])

    # Metrics: arrays for medians/IQR (post-cap for MCS, post-RCPI-conv for rxp).
    metrics = compute_metrics(
        snr_field=field_snr["field_snr"].to_numpy() if not field_snr.empty else np.array([]),
        snr_sim=sim_snr_y,
        mcs_field=field_mcs["field_mcs_tx"].clip(upper=MCS_FIELD_CAP).to_numpy() if not field_mcs.empty else np.array([]),
        mcs_sim=np.clip(sim_mcs_y, None, MCS_FIELD_CAP) if sim_mcs_y.size else np.array([]),
        rxp_field=field_rxp["field_rcpi"].to_numpy() if not field_rxp.empty else np.array([]),
        rxp_sim=sim_rxp_y,
    )

    pair_tag = f"{pair[0]}-{pair[1]}"
    metrics_path = out_dir / f"metrics_{field_scenario}_{pair_tag}.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    fig = build_figure(
        field_snr, sim_snr_t, sim_snr_y,
        field_mcs, sim_mcs_t, sim_mcs_y,
        field_rxp, sim_rxp_t, sim_rxp_y,
        metrics, field_scenario, pair, radio_param_note,
    )
    png_path = out_dir / f"validation_{field_scenario}_{pair_tag}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[validation] {pair[0]}↔{pair[1]}: wrote {png_path.name} + {metrics_path.name}")
    return png_path, metrics_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mesh-sim vs ARPO field comparison")
    p.add_argument("field_scenario", help="dirname under <field-root>/")
    p.add_argument("--sim-run", default=None,
                   help="path to a sim seed-N output dir (default: latest under outputs/)")
    p.add_argument("--rab-pair", default=None,
                   help="restrict to one pair, e.g. rab1,rab3 (default: all 3)")
    p.add_argument("--out", default=None, help="override output directory")
    p.add_argument(
        "--radio-param-note",
        default="radio params = sim defaults (28 GHz / 400 MHz / 30 dBm)",
        help="annotation under the sim SNR panel",
    )
    return p.parse_args(argv)


def load_sim_node_id_map(sim_run: Path) -> Dict[str, int]:
    """Map nodes.json string ids to the zero-indexed ints sim writes to CSVs.

    The sim archives the scenario's nodes.json at <run>/inputs/nodes.json
    (sibling of seed-N/). Order of entries in that file determines the
    integer id used in links.csv / mcs.csv / rx-power.csv.
    """
    nodes_path = sim_run.parent / "inputs" / "nodes.json"
    if not nodes_path.is_file():
        return {}
    with open(nodes_path) as f:
        nodes = json.load(f)
    return {str(n["id"]): i for i, n in enumerate(nodes)}


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    field_root = resolve_field_root()
    field_dir = field_root / args.field_scenario
    if not field_dir.is_dir():
        raise SystemExit(f"field scenario dir not found: {field_dir}\n"
                         f"  (field root: {field_root}; override with $MESH_SIM_FIELD_ROOT)")
    print(f"[validation] field root: {field_root}")

    if args.sim_run:
        sim_run = Path(args.sim_run).resolve()
    else:
        latest = latest_sim_run()
        if latest is None:
            raise SystemExit(f"no sim runs under {REPO_ROOT}/outputs/ -- "
                             f"run the sim first or pass --sim-run")
        sim_run = latest.resolve()
        print(f"[validation] sim run (auto): {sim_run}")
    if not sim_run.is_dir():
        raise SystemExit(f"sim run dir not found: {sim_run}")

    if args.rab_pair:
        a, b = (s.strip() for s in args.rab_pair.split(","))
        pairs: List[Tuple[str, str]] = [tuple(sorted([a, b]))]  # type: ignore[list-item]
    else:
        pairs = list(ALL_PAIRS)

    out_dir = (
        Path(args.out)
        if args.out
        else sim_run / "validation" / "sim-vs-arpo" / args.field_scenario
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    mac_to_rab = build_mac_to_rab(str(field_dir))
    sim_id_map = load_sim_node_id_map(sim_run)

    sim_links_all = load_links_csv(str(sim_run / "links.csv"))
    if sim_links_all is None:
        raise SystemExit(f"sim run missing links.csv at {sim_run}")
    sim_mcs_all = load_mcs_csv(str(sim_run / "mcs.csv"))
    sim_rxp_all = load_rx_power_csv(str(sim_run / "rx-power.csv"))

    n_done = 0
    for pair in pairs:
        if run_one_pair(
            pair, field_dir, sim_run, sim_links_all, sim_mcs_all, sim_rxp_all,
            sim_id_map, mac_to_rab, args.field_scenario, out_dir,
            args.radio_param_note,
        ) is not None:
            n_done += 1

    print(f"[validation] done -- {n_done}/{len(pairs)} pairs written to {out_dir}")
    return 0 if n_done > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
