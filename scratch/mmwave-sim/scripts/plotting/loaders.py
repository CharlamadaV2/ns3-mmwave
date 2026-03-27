"""Data loading and multi-seed aggregation for mmwave-sim outputs."""

import json
import math
import os
import re
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# t-distribution critical values for 95% CI (two-tailed, df = n-1).
# For df >= 30, 1.96 is a good approximation.
# ---------------------------------------------------------------------------
_T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
}


def _t_critical(n: int) -> float:
    """Return t critical value for 95% CI with n observations."""
    df = n - 1
    if df <= 0:
        return 0.0
    if df in _T_TABLE_95:
        return _T_TABLE_95[df]
    # Interpolate or fall back
    for k in sorted(_T_TABLE_95.keys()):
        if k >= df:
            return _T_TABLE_95[k]
    return 1.96


# ---------------------------------------------------------------------------
# Key metrics that get 95% CI computed
# ---------------------------------------------------------------------------
_CI_METRICS = {"mean_sinr_db", "dl_throughput_mbps", "sum_dl_throughput_mbps"}


# ---------------------------------------------------------------------------
# Single-file loaders
# ---------------------------------------------------------------------------

def load_summary(path: str) -> dict:
    """Load a single seed's summary.json."""
    with open(path) as f:
        return json.load(f)


def load_links_csv(path: str) -> pd.DataFrame:
    """Load links.csv into a DataFrame.

    Skips comment lines (starting with #). Columns:
    time_s, node_a, node_b, dist_m, sinr_db, condition
    """
    return pd.read_csv(path, comment="#")


def load_dl_pdcp_stats(path: str) -> pd.DataFrame:
    """Load DlPdcpStats.txt into a DataFrame.

    Returns columns: start, end, IMSI, RxBytes, throughput_mbps.
    Filters to LCID=3 (data bearer) rows only.
    """
    df = pd.read_csv(path, sep="\t", comment="%", header=None)
    # Trailing tab produces an extra NaN column — drop it
    df = df.dropna(axis=1, how="all")
    df.columns = ["start", "end", "CellId", "IMSI", "RNTI", "LCID",
                   "nTxPDUs", "TxBytes", "nRxPDUs", "RxBytes",
                   "delay", "stdDev", "min", "max",
                   "PduSize", "PduStdDev", "PduMin", "PduMax"]
    # Filter to data bearer
    df = df[df["LCID"] == 3].copy()
    duration = df["end"] - df["start"]
    df["throughput_mbps"] = (df["RxBytes"] * 8) / (duration * 1e6)
    return df[["start", "end", "IMSI", "RxBytes", "throughput_mbps"]]


def load_rx_packet_trace(path: str) -> pd.DataFrame:
    """Load RxPacketTrace.txt into a DataFrame.

    Returns DL rows only with columns: time, rnti, mcs, sinr_db, tbSize.
    """
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    df = df[df["DL/UL"] == "DL"].copy()
    df.rename(columns={"SINR(dB)": "sinr_db"}, inplace=True)
    return df[["time", "rnti", "mcs", "sinr_db", "tbSize"]]


# ---------------------------------------------------------------------------
# Seed discovery
# ---------------------------------------------------------------------------

_SEED_DIR_RE = re.compile(r"^seed-(\d+)$")


def discover_seed_dirs(data_dir: str) -> list[str]:
    """Find all seed-N/ subdirs in data_dir, return sorted by seed number."""
    results = []
    if not os.path.isdir(data_dir):
        return results
    for name in os.listdir(data_dir):
        m = _SEED_DIR_RE.match(name)
        if m and os.path.isdir(os.path.join(data_dir, name)):
            results.append((int(m.group(1)), os.path.join(data_dir, name)))
    results.sort(key=lambda x: x[0])
    return [path for _, path in results]


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def _stats(values: list[float], use_ci: bool = False) -> dict[str, Any]:
    """Compute mean, sample std, min, max, n, and optionally 95% CI."""
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)

    result = {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "n": n,
    }

    if use_ci and n > 1:
        t = _t_critical(n)
        result["ci95"] = t * std / math.sqrt(n)

    return result


def aggregate_summaries(summaries: list[dict]) -> dict[str, Any]:
    """Aggregate multiple seed summary dicts into mean/std/min/max/ci95.

    CI (95%) is computed for key metrics: mean_sinr_db, dl_throughput_mbps,
    sum_dl_throughput_mbps.

    Returns a dict with:
        num_seeds, seeds, network, per_ue, per_seed
    """
    if not summaries:
        return {"num_seeds": 0, "seeds": [], "network": {}, "per_ue": {},
                "per_seed": []}

    # Network-level aggregation
    network_keys = ["mean_sinr_db", "min_sinr_db", "max_sinr_db",
                    "corruption_rate", "los_fraction",
                    "sum_dl_throughput_mbps"]
    net_agg = {}
    for key in network_keys:
        vals = [s["network"][key] for s in summaries
                if "network" in s and s["network"].get(key) is not None]
        net_agg[key] = _stats(vals, use_ci=(key in _CI_METRICS))

    # Per-UE aggregation
    all_ue_ids: set[str] = set()
    for s in summaries:
        all_ue_ids.update(s.get("per_ue", {}).keys())

    ue_keys = ["mean_sinr_db", "min_sinr_db", "max_sinr_db",
               "corruption_rate", "los_fraction",
               "dl_throughput_mbps", "dl_delay_mean_ms"]
    per_ue_agg: dict[str, dict] = {}
    for uid in sorted(all_ue_ids):
        ue_agg = {}
        for key in ue_keys:
            vals = [s["per_ue"][uid][key] for s in summaries
                    if uid in s.get("per_ue", {})
                    and s["per_ue"][uid].get(key) is not None]
            ue_agg[key] = _stats(vals, use_ci=(key in _CI_METRICS))
        per_ue_agg[uid] = ue_agg

    # Per-seed metadata (for runtime plotting)
    per_seed = []
    for s in summaries:
        entry = {"seed": s.get("seed")}
        if "wall_elapsed_s" in s:
            entry["wall_elapsed_s"] = s["wall_elapsed_s"]
        per_seed.append(entry)

    return {
        "num_seeds": len(summaries),
        "seeds": [s.get("seed") for s in summaries],
        "network": net_agg,
        "per_ue": per_ue_agg,
        "per_seed": per_seed,
    }
