"""Surface label collisions in the (rab, netdev) -> rabN.M scheme."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .paths import CSV_ROOT


def _scan() -> pd.DataFrame:
    """One row per unique (rab, mac, netdev, device_name, host)."""
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


def _peer_counts() -> dict[str, int]:
    """{local_mac: number of peer-MAC bh2 rows} across all scenarios."""
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
