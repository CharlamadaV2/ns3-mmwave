## @file topology.py
# @brief MAC-address-to-rab-label resolution and stable per-node plot colours.
#
# **Vocabulary**
# - A *rab* is a chassis hostname (``rab1``, ``rab2``, …).
# - Each rab carries several radios; each radio is one Linux netdev
#   (e.g. ``wlP1p1s0f0``) with its own MAC address.
# - ``rabN.M`` denotes rab N's M-th netdev in alphabetical order, 1-based.
#
# **Label resolution strategy**
# Labels are built from the global union of ``tag_local_mac`` values across
# every scenario, so a peer MAC still resolves even when its owner's
# ``bh2.csv`` is absent from the current scenario. Peer MACs that no rab
# ever claims are grouped by last-byte proximity and labelled ``ext1``,
# ``ext2``, …
#
# **Granularity**
# Resolution stops at the hostname level — no chassis sub-splitting.

from pathlib import Path

import pandas as pd

## @brief Fixed matplotlib colour for each well-known node label.
#
# Unmapped labels fall back to ``"0.4"`` (medium grey). ``"?"`` is the
# sentinel used when a MAC cannot be resolved at all.
FIXED_COLORS: dict[str, str] = {
    "rab1": "tab:blue",
    "rab2": "tab:orange",
    "rab3": "tab:green",
    "rab4": "tab:purple",
    "ext1": "tab:red",
    "ext2": "tab:brown",
    "ext3": "tab:pink",
    "ext4": "tab:olive",
    "ext5": "tab:cyan",
    "?":    "0.6",
}

## @brief Netdev-to-antenna-face mapping for rab2 only.
#
# rab1 and rab3 carry extra lowercase ``wlp*`` netdevs that this layout
# cannot describe, so orientation is only defined for rab2.
# Source: ``data/titan01-mw.json``.
RAB2_NETDEV_ORIENTATION: dict[str, str] = {
    "wlP1p1s0f0": "front",
    "wlP1p1s0f1": "right",
    "wlP2p1s0f0": "rear",
    "wlP2p1s0f1": "left",
}


## @brief Return the antenna-face orientation for a (rab, netdev) pair.
#
# Only rab2 has a known orientation table; all other rabs return None.
#
# @param rab    Chassis hostname (e.g. ``"rab2"``).
# @param netdev Linux netdev name (e.g. ``"wlP1p1s0f0"``).
# @return Orientation string (``"front"``, ``"rear"``, ``"left"``, ``"right"``)
#         or ``None`` if unknown.
def netdev_orientation(rab: str, netdev: str) -> str | None:
    if rab != "rab2":
        return None
    return RAB2_NETDEV_ORIENTATION.get(netdev)


# {rab: sorted netdevs} — global union across all scenarios so rabN.M indices
# are stable regardless of which scenario is being processed.
_RAB_NETDEVS_CACHE: dict[Path, dict[str, list[str]]] = {}


## @brief Build the global {rab: sorted_netdevs} map from a CSV root, with caching.
#
# Scans every ``bh2.csv`` under ``csv_root`` for ``tag_interface`` values and
# groups them by node directory name. Results are cached keyed by ``csv_root``
# so repeated calls within one process pay the I/O cost only once.
#
# @param csv_root Root of the per-scenario CSV tree (see @ref DatasetPaths.csv_root).
# @return Mapping from rab hostname to its sorted list of observed netdev names.
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


## @brief Return the 1-based position of a netdev in its rab's sorted list.
#
# @param rab      Chassis hostname.
# @param netdev   Linux netdev name.
# @param csv_root CSV root to scan; defaults to @ref CSV_ROOT when ``None``.
# @return 1-based index, or ``None`` if the netdev is not found.
def radio_index(rab: str, netdev: str, csv_root: Path | None = None) -> int | None:
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


## @brief Return the human-readable ``rabN.M`` label for a (rab, netdev) pair.
#
# Falls back to ``"{rab}.?"`` when the netdev index cannot be determined.
#
# @param rab      Chassis hostname.
# @param netdev   Linux netdev name.
# @param csv_root CSV root to scan; defaults to @ref CSV_ROOT when ``None``.
# @return Label string such as ``"rab2.3"`` or ``"rab1.?"``.
def radio_label(rab: str, netdev: str, csv_root: Path | None = None) -> str:
    idx = radio_index(rab, netdev, csv_root=csv_root)
    return f"{rab}.{idx}" if idx is not None else f"{rab}.?"


# {mac: netdev} from every (tag_local_mac, tag_interface) pair seen anywhere.
# Allows a peer MAC to resolve even when its owner's bh2.csv isn't present.
_GLOBAL_MAC_NETDEVS_CACHE: dict[Path, dict[str, str]] = {}


## @brief Build the global {mac: netdev} map across all scenarios, with caching.
#
# @param csv_root Root of the per-scenario CSV tree.
# @return Mapping from MAC address string to the netdev name that owns it.
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


_MAC_DEVICE_CACHE: dict[Path, dict[str, str]] = {}
_DEVICE_NETDEVS_CACHE: dict[Path, dict[str, list[str]]] = {}


## @brief Build the global {mac: tag_device_name} map across all scenarios, with caching.
#
# ``tag_device_name`` is the full chassis identifier (e.g. ``"sky01-mw"``),
# which is distinct from the node directory name used as the rab hostname.
# The first device name seen for a MAC wins (setdefault).
#
# @param csv_root Root of the per-scenario CSV tree.
# @return Mapping from MAC address string to chassis device name.
def _global_mac_devices(csv_root: Path) -> dict[str, str]:
    if csv_root in _MAC_DEVICE_CACHE:
        return _MAC_DEVICE_CACHE[csv_root]
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
                                     usecols=["tag_local_mac", "tag_device_name"])
                except (ValueError, KeyError):
                    continue
                pairs = df.dropna(subset=["tag_local_mac", "tag_device_name"]) \
                    .drop_duplicates()
                for mac, dev in zip(pairs["tag_local_mac"], pairs["tag_device_name"]):
                    out.setdefault(str(mac), str(dev))
    _MAC_DEVICE_CACHE[csv_root] = out
    return out


## @brief Build the global {device_name: sorted_netdevs} map, with caching.
#
# Derives netdev lists per device by joining @ref _global_mac_devices and
# @ref _global_mac_netdevs. Used to compute the 1-based radio index for
# @ref mac_radio_label without knowing the rab hostname.
#
# @param csv_root Root of the per-scenario CSV tree.
# @return Mapping from chassis device name to its sorted list of observed netdev names.
def _device_netdevs(csv_root: Path) -> dict[str, list[str]]:
    if csv_root in _DEVICE_NETDEVS_CACHE:
        return _DEVICE_NETDEVS_CACHE[csv_root]
    out: dict[str, set[str]] = {}
    devs = _global_mac_devices(csv_root)
    nets = _global_mac_netdevs(csv_root)
    for mac, dev in devs.items():
        nd = nets.get(mac)
        if nd:
            out.setdefault(dev, set()).add(nd)
    ordered = {dev: sorted(netdevs) for dev, netdevs in out.items()}
    _DEVICE_NETDEVS_CACHE[csv_root] = ordered
    return ordered


## @brief Resolve a MAC address to a ``<device_name>.<radio_index>`` label.
#
# Uses @ref _global_mac_devices for the chassis name and @ref _device_netdevs
# for the 1-based radio index. Returns ``None`` when the MAC has no known
# device, and ``"<device>.?"`` when the netdev index cannot be determined.
#
# @param mac      MAC address string to resolve (may be ``None``).
# @param csv_root CSV root to scan; defaults to @ref CSV_ROOT when ``None``.
# @return Label string such as ``"sky01-mw.1"``, or ``None`` if unresolvable.
def mac_radio_label(mac: str, csv_root: Path | None = None) -> str | None:
    if mac is None:
        return None
    if csv_root is None:
        from .paths import CSV_ROOT
        csv_root = CSV_ROOT
    dev = _global_mac_devices(csv_root).get(str(mac))
    netdev = _global_mac_netdevs(csv_root).get(str(mac))
    if not dev or not netdev:
        return None
    netdevs = _device_netdevs(csv_root).get(dev, [])
    try:
        idx = netdevs.index(netdev) + 1
    except ValueError:
        return f"{dev}.?"
    return f"{dev}.{idx}"


# Plot pipelines call build_topology many times per scenario — cache the scan.
_GLOBAL_OWNERS_CACHE: dict[Path, dict[str, str]] = {}


## @brief Return the matplotlib colour string for a node label.
#
# Unmapped labels (not in @ref FIXED_COLORS) return ``"0.4"`` (medium grey).
#
# @param label Node label such as ``"rab1"`` or ``"ext2"``.
# @return Matplotlib colour string.
def node_color(label: str) -> str:
    return FIXED_COLORS.get(label, "0.4")


## @brief Build the global {mac: hostname} map for every local MAC, with caching.
#
# A MAC is "local" if it ever appeared in ``tag_local_mac`` for a node's
# ``bh2.csv``. The first hostname that claims a MAC wins (setdefault).
#
# @param csv_root Root of the per-scenario CSV tree.
# @return Mapping from MAC address string to chassis hostname.
def _global_local_macs(csv_root: Path) -> dict[str, str]:
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


## @brief Build the {mac: label} topology map for a single scenario directory.
#
# Labels known MACs from the global local-MAC table, then groups any remaining
# peer-only MACs by last-byte proximity and assigns them ``ext1``, ``ext2``, …
#
# @param scen_dir Path to one scenario directory (parent is assumed to be csv_root).
# @return Mapping from MAC address string to human-readable node label.
def build_topology(scen_dir: Path) -> dict[str, str]:
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


## @brief Resolve a peer MAC to its label using a pre-built topology map.
#
# Returns ``"?"`` for NaN/None MACs and ``"?:<last5>"`` for MACs absent from
# the topology, providing enough context to identify the device in logs.
#
# @param mac  MAC address value (may be ``float("nan")`` from pandas).
# @param topo Topology map produced by @ref build_topology.
# @return Human-readable label string.
def resolve_peer(mac, topo: dict[str, str]) -> str:
    if pd.isna(mac):
        return "?"
    return topo.get(str(mac), f"?:{str(mac)[-5:]}")


## @brief Group MAC addresses whose last byte is within 3 of any group member.
#
# This heuristic clusters MACs that belong to the same physical device, which
# often increments its last byte across radios (e.g. ``aa:bb:01``, ``aa:bb:02``).
#
# @param macs List of MAC address strings to cluster.
# @return List of groups, each group being a list of related MAC strings.
def _cluster_macs(macs: list[str]) -> list[list[str]]:
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


## @brief Parse the last octet of a colon-separated MAC address as an integer.
#
# @param mac MAC address string (e.g. ``"aa:bb:cc:dd:ee:ff"``).
# @return Integer value of the last byte.
def _last_byte(mac: str) -> int:
    return int(mac.split(":")[-1], 16)