"""
Resolve peer-MAC <-> rab-ID mapping for an ARPO field scenario.

Each rab has multiple Hydra interfaces (4 per titan), so a rab owns several
`tag_local_mac` values. Peer ownership is determined by intersecting an
observed `tag_sta_mac` against every other rab's local-mac set.

For 1-1_static_baseline_3_04162026 the resolved mapping is:
    rab1 owns 6e/6f/70/71  rab2 owns 2e/2f/30/31  rab3 owns 68/69/6a/6b
plus extra mesh peers (e.g. 9c, 99) that are not rab1/2/3 -- these are
returned as "ext".
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd

KNOWN_RABS: Tuple[str, ...] = ("rab1", "rab2", "rab3")
EXT_LABEL = "ext"


def _bh2_path(scenario_dir: str, rab: str) -> str:
    return os.path.join(scenario_dir, rab, "bh2.csv")


def collect_local_macs(scenario_dir: str) -> Dict[str, List[str]]:
    """Return {rab_id: [local_macs]} read from each rab's bh2.csv."""
    out: Dict[str, List[str]] = {}
    for rab in KNOWN_RABS:
        path = _bh2_path(scenario_dir, rab)
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path, usecols=["tag_local_mac"])
        out[rab] = sorted(df["tag_local_mac"].dropna().unique().tolist())
    return out


def build_mac_to_rab(scenario_dir: str) -> Dict[str, str]:
    """Map every observed local MAC to its owning rab. Unknown MACs are absent."""
    locals_by_rab = collect_local_macs(scenario_dir)
    mac_to_rab: Dict[str, str] = {}
    for rab, macs in locals_by_rab.items():
        for m in macs:
            mac_to_rab[m] = rab
    return mac_to_rab


def resolve_rab(peer_mac: str, mac_to_rab: Dict[str, str]) -> str:
    """Return the rab id that owns *peer_mac*, or 'ext' if unknown."""
    return mac_to_rab.get(peer_mac, EXT_LABEL)


def link_sample_counts(
    scenario_dir: str, mac_to_rab: Dict[str, str]
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Per-link stats keyed by sorted (rab_a, rab_b).

    Returns {(rab_a, rab_b): {n, snr_mean, snr_std, snr_median}} aggregated
    across both directions of the link. Links involving 'ext' are kept under
    keys like ('rab1', 'ext').
    """
    rows: List[Tuple[str, str, float]] = []
    for rab in KNOWN_RABS:
        path = _bh2_path(scenario_dir, rab)
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path, usecols=["tag_sta_mac", "field_snr"]).dropna(
            subset=["field_snr"]
        )
        for _, row in df.iterrows():
            peer = resolve_rab(row["tag_sta_mac"], mac_to_rab)
            a, b = sorted([rab, peer])
            rows.append((a, b, float(row["field_snr"])))
    if not rows:
        return {}
    big = pd.DataFrame(rows, columns=["a", "b", "snr"])
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for (a, b), grp in big.groupby(["a", "b"]):
        out[(a, b)] = {
            "n": int(len(grp)),
            "snr_mean": float(grp["snr"].mean()),
            "snr_std": float(grp["snr"].std()),
            "snr_median": float(grp["snr"].median()),
        }
    return out


def pick_dominant_pair(
    stats: Dict[Tuple[str, str], Dict[str, float]],
    require_known: bool = True,
) -> Tuple[str, str] | None:
    """
    Pick (rab_a, rab_b) with the highest sample count and lowest SNR std.

    With *require_known* (default), excludes links that touch 'ext' so the
    pick is always between known rabs whose surveyed positions exist.
    """
    candidates = [
        (k, v)
        for k, v in stats.items()
        if not require_known or (k[0] in KNOWN_RABS and k[1] in KNOWN_RABS)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda kv: (-kv[1]["n"], kv[1]["snr_std"]))
    return candidates[0][0]


def filter_field_for_pair(
    df: pd.DataFrame,
    own_rab: str,
    peer_rab: str,
    mac_to_rab: Dict[str, str],
) -> pd.DataFrame:
    """
    Return rows of *df* whose peer MAC resolves to *peer_rab*.

    Caller passes a bh2 DataFrame already filtered to rows from *own_rab*.
    """
    return df[df["tag_sta_mac"].map(lambda m: mac_to_rab.get(m, EXT_LABEL)) == peer_rab]
