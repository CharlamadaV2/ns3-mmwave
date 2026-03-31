"""Multi-seed aggregation with 95% confidence intervals.

Two strategies:
  1. aggregate_timeseries — for CSV time-series data (mean + CI per timestep)
  2. aggregate_summaries  — for summary.json scalar metrics
"""

import math
import os
import warnings
from typing import Any, Callable

import pandas as pd

# ---------------------------------------------------------------------------
# t-distribution critical values for 95% CI (two-tailed, df = n-1).
# For df >= 30, 1.96 is a good approximation (normal distribution).
# ---------------------------------------------------------------------------
_T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447,  7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
}


def _t_critical(n: int) -> float:
    """Return t critical value for 95% CI with *n* observations."""
    df = n - 1
    if df <= 0:
        return 0.0
    if df in _T_TABLE_95:
        return _T_TABLE_95[df]
    for k in sorted(_T_TABLE_95.keys()):
        if k >= df:
            return _T_TABLE_95[k]
    return 1.96


# ---------------------------------------------------------------------------
# Time-series aggregation (CSV data)
# ---------------------------------------------------------------------------

def aggregate_timeseries(
    seed_dirs: list[str],
    loader: Callable[[str], pd.DataFrame | None],
    filename: str,
    time_col: str,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame | None:
    """Load *filename* from each seed dir, then compute per-group mean and CI.

    For each (time_col, *group_cols) combination, computes mean and 95% CI
    of each value_col across seeds.

    Returns a DataFrame with columns:
        time_col, *group_cols, {val}_mean, {val}_ci95 for each val in value_cols.
    Returns None if no data is found.
    """
    frames = []
    for sd in seed_dirs:
        path = os.path.join(sd, filename)
        df = loader(path)
        if df is None:
            continue
        # Check that expected columns exist
        missing = [c for c in [time_col] + group_cols + value_cols
                   if c not in df.columns]
        if missing:
            warnings.warn(f"{path}: missing columns {missing}, skipping")
            continue
        frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)

    # Round time to nearest ms to handle floating-point drift across seeds
    combined[time_col] = combined[time_col].round(3)

    keys = [time_col] + group_cols
    grouped = combined.groupby(keys, as_index=False)

    agg_dict = {}
    for vc in value_cols:
        agg_dict[f"{vc}_mean"] = (vc, "mean")
        agg_dict[f"{vc}_std"] = (vc, "std")
        agg_dict[f"{vc}_n"] = (vc, "count")

    result = grouped.agg(**agg_dict)

    # Compute CI95 for each value column
    for vc in value_cols:
        n_col = f"{vc}_n"
        std_col = f"{vc}_std"
        ci_col = f"{vc}_ci95"

        def _ci(row):
            n = row[n_col]
            if n <= 1:
                return 0.0
            return _t_critical(int(n)) * row[std_col] / math.sqrt(n)

        result[ci_col] = result.apply(_ci, axis=1)
        # Drop intermediate columns
        result.drop(columns=[std_col, n_col], inplace=True)

    return result


# ---------------------------------------------------------------------------
# Summary aggregation (JSON data)
# ---------------------------------------------------------------------------

_CI_METRICS = {"mean_sinr_db", "sum_throughput_mbps", "connectivity",
               "tx_throughput_mbps", "rx_throughput_mbps",
               "delivered_mbps", "latency_ms"}


def _stats(values: list[float], use_ci: bool = False) -> dict[str, Any]:
    """Compute mean, sample std, min, max, n, and optionally 95% CI."""
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)

    result: dict[str, Any] = {
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
        "n": n,
    }
    if use_ci and n > 1:
        result["ci95"] = _t_critical(n) * std / math.sqrt(n)
    return result


def aggregate_summaries(summaries: list[dict]) -> dict[str, Any]:
    """Aggregate multiple seed summary dicts into mean/std/ci95.

    Returns a dict with: num_seeds, seeds, network, per_node, per_flow, per_seed.
    """
    if not summaries:
        return {"num_seeds": 0, "seeds": [], "network": {}, "per_node": {},
                "per_flow": {}, "per_seed": []}

    # --- Network-level ---
    net_keys = ["mean_sinr_db", "sum_throughput_mbps", "connectivity",
                "mean_hop_count", "flows_routed", "flows_unroutable"]
    net_agg = {}
    for key in net_keys:
        vals = [s["network"][key] for s in summaries
                if "network" in s and s["network"].get(key) is not None]
        net_agg[key] = _stats(vals, use_ci=(key in _CI_METRICS))

    # --- Per-node ---
    all_node_ids: set[str] = set()
    for s in summaries:
        all_node_ids.update(s.get("per_node", {}).keys())

    node_keys = ["mean_sinr_db", "min_sinr_db", "max_sinr_db",
                 "num_links", "num_los_links",
                 "tx_throughput_mbps", "rx_throughput_mbps"]
    per_node_agg: dict[str, dict] = {}
    for nid in sorted(all_node_ids):
        nagg = {}
        for key in node_keys:
            vals = [s["per_node"][nid][key] for s in summaries
                    if nid in s.get("per_node", {})
                    and s["per_node"][nid].get(key) is not None]
            nagg[key] = _stats(vals, use_ci=(key in _CI_METRICS))
        per_node_agg[nid] = nagg

    # --- Per-flow ---
    all_flow_ids: set[str] = set()
    for s in summaries:
        all_flow_ids.update(s.get("per_flow", {}).keys())

    flow_keys = ["demand_mbps", "delivered_mbps", "latency_ms", "hop_count"]
    per_flow_agg: dict[str, dict] = {}
    for fid in sorted(all_flow_ids):
        fagg = {}
        for key in flow_keys:
            vals = [s["per_flow"][fid][key] for s in summaries
                    if fid in s.get("per_flow", {})
                    and s["per_flow"][fid].get(key) is not None]
            fagg[key] = _stats(vals, use_ci=(key in _CI_METRICS))
        per_flow_agg[fid] = fagg

    # --- Per-seed metadata ---
    per_seed = []
    for s in summaries:
        entry: dict[str, Any] = {"seed": s.get("seed")}
        if "wall_elapsed_s" in s:
            entry["wall_elapsed_s"] = s["wall_elapsed_s"]
        per_seed.append(entry)

    return {
        "num_seeds": len(summaries),
        "seeds": [s.get("seed") for s in summaries],
        "network": net_agg,
        "per_node": per_node_agg,
        "per_flow": per_flow_agg,
        "per_seed": per_seed,
    }
