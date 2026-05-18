##@package docstring
# MAC -> rab label and stable plot color.

# Vocab: a *rab* is a chassis hostname (``rab1`` ...). Each rab carries
# several radios; each radio is one Linux *netdev* (``wlP1p1s0f0`` ...)
# with its own MAC. ``rabN.M`` = rab N's M-th netdev, alphabetical, 1-based.

# Labels come from the global union of ``tag_local_mac`` across every
# scenario, so a peer MAC still resolves when its owner's bh2.csv is
# missing from this scenario. Peer MACs no rab ever claims become
# ``ext1``, ``ext2``, ....

# Hostname is the finest label -- no chassis sub-splitting.
##

#TODO: Finish Documentation for this page

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

# rab2's netdevs -> antenna face (from data/titan01-mw.json). rab1 and rab3
# carry extra lowercase ``wlp*`` netdevs this layout can't describe, so we
# don't guess for them.
RAB2_NETDEV_ORIENTATION: dict[str, str] = {
    "wlP1p1s0f0": "front",
    "wlP1p1s0f1": "right",
    "wlP2p1s0f0": "rear",
    "wlP2p1s0f1": "left",
}

## Documentation for a function.
#
#  More details.
def netdev_orientation(rab: str, netdev: str) -> str | None:
    if rab != "rab2":
        return None
    return RAB2_NETDEV_ORIENTATION.get(netdev)


# {rab: sorted netdevs}, global union across scenarios so ``rab1.5`` is stable.
_RAB_NETDEVS_CACHE: dict[Path, dict[str, list[str]]] = {}

## @brief
def _rab_netdevs(csv_root: Path) -> dict[str, list[str]]:
    if csv_root in _RAB_NETDEVS_CACHE:
        return _RAB_NETDEVS_CACHE[csv_root]
    out: dict[str, set[str]] = {}
    if csv_root.is_dir():
        for scen in sorted(csv_root.iterdir()):
            if not scen.is_dir():
                continue
            for node in sorted(scen.iterdir()):
                if not node.is_dir() or node.name == "sdwan":
                    continue
                fp = node / "bh2.csv"
                if not fp.exists():
                    continue
                try:
                    df = pd.read_csv(fp, low_memory=False, usecols=["tag_interface"])
                except (ValueError, KeyError):
                    continue
                for d in df["tag_interface"].dropna().unique():
                    out.setdefault(node.name, set()).add(str(d))
    ordered = {rab: sorted(netdevs) for rab, netdevs in out.items()}
    _RAB_NETDEVS_CACHE[csv_root] = ordered
    return ordered

## Documentation for a function.
#
#  More details.
def radio_index(rab: str, netdev: str, csv_root: Path | None = None) -> int | None:
    ##1-based position of ``netdev`` in ``rab``'s sorted netdev list, or None.##
    if not netdev:
        return None
    if csv_root is None:
        from .paths import CSV_ROOT
        csv_root = CSV_ROOT
    netdevs = _rab_netdevs(csv_root).get(rab, [])
    try:
        return netdevs.index(netdev) + 1
    except ValueError:
        return None

## Documentation for a function.
#
#  More details.
def radio_label(rab: str, netdev: str, csv_root: Path | None = None) -> str:
    ##``rabN.M`` label for a (rab, netdev) pair; ``{rab}.?`` if unknown.##
    idx = radio_index(rab, netdev, csv_root=csv_root)
    return f"{rab}.{idx}" if idx is not None else f"{rab}.?"


# {mac: netdev} from every (local_mac, tag_interface) pair seen anywhere --
# lets a peer MAC resolve when its owner's bh2.csv isn't in this scenario.
_GLOBAL_MAC_NETDEVS_CACHE: dict[Path, dict[str, str]] = {}

## @brief
def _global_mac_netdevs(csv_root: Path) -> dict[str, str]:
    if csv_root in _GLOBAL_MAC_NETDEVS_CACHE:
        return _GLOBAL_MAC_NETDEVS_CACHE[csv_root]
    out: dict[str, str] = {}
    if csv_root.is_dir():
        for scen in sorted(csv_root.iterdir()):
            if not scen.is_dir():
                continue
            for node in sorted(scen.iterdir()):
                if not node.is_dir() or node.name == "sdwan":
                    continue
                fp = node / "bh2.csv"
                if not fp.exists():
                    continue
                try:
                    df = pd.read_csv(fp, low_memory=False,
                                     usecols=["tag_local_mac", "tag_interface"])
                except (ValueError, KeyError):
                    continue
                pairs = df.dropna(subset=["tag_local_mac", "tag_interface"]) \
                    .drop_duplicates()
                for mac, netdev in zip(pairs["tag_local_mac"], pairs["tag_interface"]):
                    out.setdefault(str(mac), str(netdev))
    _GLOBAL_MAC_NETDEVS_CACHE[csv_root] = out
    return out

## Documentation for a function.
#
#  More details.
def mac_radio_label(mac: str, csv_root: Path | None = None) -> str | None:
    ##MAC -> ``rabN.M``; None if no rab ever claimed this MAC as a local MAC.##
    if mac is None:
        return None
    if csv_root is None:
        from .paths import CSV_ROOT
        csv_root = CSV_ROOT
    owners = _global_local_macs(csv_root)
    rab = owners.get(str(mac))
    if rab is None:
        return None
    netdev = _global_mac_netdevs(csv_root).get(str(mac))
    if not netdev:
        return None
    return radio_label(rab, netdev, csv_root=csv_root)


# Plot pipelines call build_topology many times per scenario -- cache the scan.
_GLOBAL_OWNERS_CACHE: dict[Path, dict[str, str]] = {}

## Documentation for a function.
#
#  More details.
def node_color(label: str) -> str:
    return FIXED_COLORS.get(label, "0.4")

## @brief
def _global_local_macs(csv_root: Path) -> dict[str, str]:
    ##{mac: hostname} for every ``tag_local_mac`` ever claimed by any node.##
    if csv_root in _GLOBAL_OWNERS_CACHE:
        return _GLOBAL_OWNERS_CACHE[csv_root]
    out: dict[str, str] = {}
    if csv_root.is_dir():
        for scen in sorted(csv_root.iterdir()):
            if not scen.is_dir():
                continue
            for node in sorted(scen.iterdir()):
                if not node.is_dir() or node.name == "sdwan":
                    continue
                fp = node / "bh2.csv"
                if not fp.exists():
                    continue
                try:
                    df = pd.read_csv(fp, low_memory=False, usecols=["tag_local_mac"])
                except (ValueError, KeyError):
                    continue
                for m in df["tag_local_mac"].dropna().unique():
                    out.setdefault(str(m), node.name)
    _GLOBAL_OWNERS_CACHE[csv_root] = out
    return out

## Documentation for a function.
#
#  More details.
def build_topology(scen_dir: Path) -> dict[str, str]:
    ##{mac: label} for every MAC in *scen_dir*. Unknowns cluster into ``extN``.##
    topo = dict(_global_local_macs(scen_dir.parent))

    all_peers: set[str] = set()
    for node in sorted(scen_dir.iterdir()):
        if not node.is_dir() or node.name == "sdwan":
            continue
        fp = node / "bh2.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, low_memory=False, usecols=["tag_sta_mac"])
        all_peers.update(str(m) for m in df["tag_sta_mac"].dropna().unique())

    unknown = sorted(m for m in all_peers if m not in topo)
    for i, group in enumerate(_cluster_macs(unknown), start=1):
        for mac in group:
            topo[mac] = f"ext{i}"
    return topo

## Documentation for a function.
#
#  More details.
def resolve_peer(mac, topo: dict[str, str]) -> str:
    if pd.isna(mac):
        return "?"
    return topo.get(str(mac), f"?:{str(mac)[-5:]}")

## @brief
def _cluster_macs(macs: list[str]) -> list[list[str]]:
    ##Group MACs whose last byte is within 3 of an existing group member.##
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

## @brief
def _last_byte(mac: str) -> int:
    return int(mac.split(":")[-1], 16)
