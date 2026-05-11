"""
Map MAC addresses to physical-node labels and stable plot colors.

Known nodes (rab1/rab2/rab3) come from each rab's own ``tag_local_mac``
column. Each radio carries four sequential MACs on its interfaces, so
unknown peer MACs get clustered into 4-MAC blocks and labeled
``ext1``, ``ext2``, ...
"""

from pathlib import Path

import pandas as pd

FIXED_COLORS = {
    "rab1": "tab:blue",
    "rab2": "tab:orange",
    "rab3": "tab:green",
    "rab4": "tab:purple",
    "ext1": "tab:red",
    "ext2": "tab:brown",
    "ext3": "tab:pink",
    "ext4": "tab:olive",
    "ext5": "tab:cyan",
    "?": "0.6",
}


def node_color(label: str) -> str:
    return FIXED_COLORS.get(label, "0.4")


def build_topology(scen_dir: Path) -> dict[str, str]:
    """Return ``{mac: node_label}`` for every MAC observed in *scen_dir*."""
    topo: dict[str, str] = {}
    all_peers: set[str] = set()
    for node in sorted(scen_dir.iterdir()):
        if not node.is_dir() or node.name == "sdwan":
            continue
        fp = node / "bh2.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, low_memory=False, usecols=["tag_local_mac", "tag_sta_mac"])
        for mac in df["tag_local_mac"].dropna().unique():
            topo[str(mac)] = node.name
        all_peers.update(str(m) for m in df["tag_sta_mac"].dropna().unique())

    unknown = sorted(m for m in all_peers if m not in topo)
    for i, group in enumerate(_cluster_macs(unknown), start=1):
        for mac in group:
            topo[mac] = f"ext{i}"
    return topo


def resolve_peer(mac, topo: dict[str, str]) -> str:
    if pd.isna(mac):
        return "?"
    return topo.get(str(mac), f"?:{str(mac)[-5:]}")


def _cluster_macs(macs: list[str]) -> list[list[str]]:
    """Group MACs whose last byte is within 3 of an existing group member."""
    groups: list[list[str]] = []
    for mac in macs:
        last = _last_byte(mac)
        for g in groups:
            if any(abs(last - _last_byte(m)) <= 3 for m in g):
                g.append(mac)
                break
        else:
            groups.append([mac])
    return groups


def _last_byte(mac: str) -> int:
    return int(mac.split(":")[-1], 16)
