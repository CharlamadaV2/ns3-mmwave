"""
Loaders for the validation cross-comparison.

Sim side: thin reuse of `scripts.plotting.loaders` (its module CLAUDE.md
declares it sim-only). Field side: read bh2.csv directly with pandas.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

# Reuse sim-side loaders without redefining them.
from scripts.plotting.loaders import (  # noqa: F401
    load_links_csv,
    load_mcs_csv,
    load_rx_power_csv,
    load_summary,
)

FIELD_BH2_COLUMNS = [
    "timestamp",
    "tag_local_mac",
    "tag_sta_mac",
    "field_snr",
    "field_snr_ga64",
    "field_mcs_tx",
    "field_mcs_rx",
    "field_rcpi",
    "field_per",
    "field_bytes_tx",
    "field_packets_tx",
]


def load_field_bh2(scenario_dir: str, rab: str) -> Optional[pd.DataFrame]:
    """
    Load `<scenario_dir>/<rab>/bh2.csv` keeping only validation-relevant cols.

    Column order differs between rabs in the field data, so we read by name.
    Returns None if the file is missing.
    """
    path = os.path.join(scenario_dir, rab, "bh2.csv")
    if not os.path.isfile(path):
        return None
    head = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in FIELD_BH2_COLUMNS if c in head]
    df = pd.read_csv(path, usecols=cols)
    if "timestamp" in df.columns:
        df["timestamp_ns"] = pd.to_numeric(df["timestamp"], errors="coerce")
    return df


def filter_sim_links_for_pair(
    links: pd.DataFrame, node_a, node_b
) -> pd.DataFrame:
    """
    Return rows of links.csv where {node_a, node_b} == {a, b}, either order.

    node_a/node_b may be int (zero-indexed sim ids) or str (scenario id);
    both sides of the comparison are coerced to str.
    """
    a = links["node_a"].astype(str)
    b = links["node_b"].astype(str)
    sa, sb = str(node_a), str(node_b)
    mask = ((a == sa) & (b == sb)) | ((a == sb) & (b == sa))
    return links[mask]


def filter_sim_csv_for_pair(
    df: pd.DataFrame, node_a: str, node_b: str
) -> pd.DataFrame:
    """Same as filter_sim_links_for_pair but for any per-link CSV (mcs/rx-power)."""
    return filter_sim_links_for_pair(df, node_a, node_b)
