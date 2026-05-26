## @file topology_audit.py
# @brief Surface label collisions in the (rab, netdev) -> rabN.M scheme.
#


from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .paths import CSV_ROOT


## @brief Scan all bh2.csv files and return one row per unique (rab, mac, netdev, device_name, host).
#
# Iterates every scenario and node directory under @ref CSV_ROOT, reads
# ``tag_local_mac``, ``tag_interface``, ``tag_device_name``, and ``tag_host``
# from each ``bh2.csv``, and deduplicates the result. Files missing any of
# the required columns are silently skipped.
#
# @return DataFrame with columns ``["rab", "mac", "netdev", "device_name", "host"]``,
#         deduplicated across all scenarios. Empty if no bh2.csv files are found.
def _scan() -> pd.DataFrame:
    rows = []
    if not CSV_ROOT.is_dir():
        return pd.DataFrame(rows)
    for scen in sorted(CSV_ROOT.iterdir()):
        if not scen.is_dir():
            continue
        for node in sorted(scen.iterdir()):
            if not node.is_dir() or node.name == "sdwan":
                continue
            fp = node / "bh2.csv"
            if not fp.exists():
                continue
            try:
                df = pd.read_csv(
                    fp, low_memory=False,
                    usecols=["tag_local_mac", "tag_interface",
                             "tag_device_name", "tag_host"],
                )
            except (ValueError, KeyError):
                continue
            df = df.dropna(subset=["tag_local_mac", "tag_interface"]).drop_duplicates()
            for _, r in df.iterrows():
                rows.append({
                    "rab":         node.name,
                    "mac":         str(r["tag_local_mac"]),
                    "netdev":      str(r["tag_interface"]),
                    "device_name": str(r.get("tag_device_name", "")),
                    "host":        str(r.get("tag_host", "")),
                })
    return pd.DataFrame(rows).drop_duplicates()


## @brief Count the total number of peer-MAC bh2 rows seen for each local MAC.
#
# Aggregates row counts from ``tag_sta_mac`` across every scenario, grouped by
# ``tag_local_mac``. Used by @ref main to annotate which MACs in a collision
# group were actively communicating and which were passive.
#
# @return Mapping from local MAC address string to total number of peer-facing
#         bh2 rows observed across all scenarios.
def _peer_counts() -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for scen in sorted(CSV_ROOT.iterdir()):
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
                                 usecols=["tag_local_mac", "tag_sta_mac"])
            except (ValueError, KeyError):
                continue
            df = df.dropna(subset=["tag_local_mac", "tag_sta_mac"])
            for mac, n in df.groupby("tag_local_mac").size().items():
                out[str(mac)] += int(n)
    return dict(out)


## @brief Report (rab, netdev) keys whose MAC address is claimed by more than one device.
#
# Calls @ref _scan to build the full (rab, mac, netdev, device_name, host)
# inventory and @ref _peer_counts to annotate activity levels. For each
# collision it prints whether the duplicate MACs share the same chassis
# (same-chassis card swap) or represent two physically distinct boxes merged
# under one hostname.
#
# @return 0 if no collisions are found or all collisions are reported
#         successfully; 1 if no bh2.csv files exist under @ref CSV_ROOT.
def main() -> int:
    df = _scan()
    if df.empty:
        print(f"no bh2.csv files under {CSV_ROOT}")
        return 1

    peers = _peer_counts()
    grouped = df.groupby(["rab", "netdev"])["mac"].nunique()
    collisions = grouped[grouped > 1]

    if collisions.empty:
        print(f"no (rab, netdev) collisions across {len(df)} unique rows.")
        return 0

    print(f"Found {len(collisions)} (rab, netdev) keys with >1 MAC:\n")
    for (rab, netdev), n_macs in collisions.items():
        block = df[(df["rab"] == rab) & (df["netdev"] == netdev)]
        unique_dn   = block["device_name"].nunique()
        unique_host = block["host"].nunique()
        verdict = ("same-chassis (one box, two card sets)"
                   if unique_dn == 1 and unique_host == 1
                   else "MERGED-HOSTNAME (two physical boxes under one name)")
        print(f"  {rab}/{netdev}  ({n_macs} MACs)  ->  {verdict}")
        for _, r in block.iterrows():
            n_peer = peers.get(r["mac"], 0)
            print(f"    mac={r['mac']}  device_name={r['device_name']}  "
                  f"host={r['host']}  peer_rows={n_peer:,}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())