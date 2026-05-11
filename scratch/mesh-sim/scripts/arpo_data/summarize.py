"""
Inspection report for the ARPO dataset.

Per scenario: file presence, row counts, time range. A trailing
``ISSUES`` section flags missing files, clock skew, low GPS lock rate,
and out-of-range SNR -- the things you actually want to know about
before trusting the data, rather than a wall of column statistics.
"""

import sys
from pathlib import Path

import pandas as pd

from .paths import CSV_ROOT, NODE_DIRS, PRIORITY_FILES

# Plausible-range bounds. Outside these, the data is suspect.
_SNR_MIN_DB, _SNR_MAX_DB = -40.0, 60.0
_CLOCK_SKEW_WARN_S = 3600.0       # 1 hour
_GPS_LOCK_WARN_FRAC = 0.5
_CHRONY_OFFSET_WARN_S = 0.1       # 100 ms median clock offset


def summarize() -> int:
    if not CSV_ROOT.exists():
        print(f"ERROR: {CSV_ROOT} not found -- run `extract` first", file=sys.stderr)
        return 1

    scenarios = sorted(p for p in CSV_ROOT.iterdir() if p.is_dir())
    print(f"ARPO Spring Lake summary  ({len(scenarios)} scenarios)")
    print("=" * 60)

    issues: list[str] = []
    for scen in scenarios:
        print(f"\n[{scen.name}]")
        issues.extend(_summarize_scenario(scen))

    print("\n" + "=" * 60)
    if issues:
        print(f"ISSUES ({len(issues)})")
        for msg in issues:
            print(f"  ! {msg}")
    else:
        print("No issues flagged.")
    return 0


def _summarize_scenario(scen: Path) -> list[str]:
    nodes = sorted(p for p in scen.iterdir() if p.is_dir() and p.name in NODE_DIRS)
    if not nodes:
        print("  (no node subdirectories)")
        return [f"{scen.name}: no node subdirectories"]

    issues: list[str] = []

    presence = []
    for node in nodes:
        present = sum(1 for f in PRIORITY_FILES if (node / f).exists())
        presence.append(f"{node.name}={present}/{len(PRIORITY_FILES)}")
        for f in PRIORITY_FILES:
            if not (node / f).exists():
                issues.append(f"{scen.name}/{node.name}: {f} missing")
    print(f"  files:   {'  '.join(presence)}")

    rows, snr, peers = _bh2_stats(nodes)
    if rows:
        if not snr.empty:
            print(f"  bh2:     {rows:,} rows  SNR {snr.min():.0f} to {snr.max():.0f} dB  "
                  f"({len(peers)} peer MACs)")
            if snr.min() < _SNR_MIN_DB or snr.max() > _SNR_MAX_DB:
                issues.append(
                    f"{scen.name}: SNR out of plausible range "
                    f"[{snr.min():.0f}, {snr.max():.0f}] dB"
                )
        else:
            print(f"  bh2:     {rows:,} rows  ({len(peers)} peer MACs, no SNR column)")

    for node in nodes:
        fp = node / "gps.csv"
        if not fp.exists():
            continue
        locked, total = _gps_lock_rate(fp)
        if total == 0:
            continue
        frac = locked / total
        if frac < _GPS_LOCK_WARN_FRAC:
            issues.append(
                f"{scen.name}/{node.name}: GPS lock rate {frac:.0%} "
                f"({locked}/{total} rows have a fix)"
            )

    starts = _node_session_starts(nodes)
    if starts:
        starts_str = "  ".join(f"{n}={t.date()}" for n, t in starts.items())
        print(f"  clocks:  {starts_str}")
        skew_s = (max(starts.values()) - min(starts.values())).total_seconds()
        if skew_s > _CLOCK_SKEW_WARN_S:
            issues.append(f"{scen.name}: clock skew {skew_s / 3600:.1f}h between nodes")

    chrony = _chrony_stats(nodes)
    if chrony:
        parts = [f"{chrony['n_nodes']} nodes"]
        offset = chrony["offset"]
        if not offset.empty:
            parts.append(
                f"|offset| median {offset.median() * 1000:.1f}ms  "
                f"max {offset.max() * 1000:.0f}ms"
            )
        if chrony["strata"]:
            parts.append(f"stratum {','.join(sorted(chrony['strata']))}")
        print(f"  chrony:  {'  '.join(parts)}")
        if not offset.empty and offset.median() > _CHRONY_OFFSET_WARN_S:
            issues.append(
                f"{scen.name}: chrony median clock offset "
                f"{offset.median() * 1000:.0f} ms (poorly disciplined)"
            )

    mcm = _mcm_stats(nodes)
    if mcm:
        parts = [f"{mcm['n_nodes']} nodes"]
        if mcm["ssids"]:
            parts.append(f"{len(mcm['ssids'])} SSID")
        if mcm["bssids"]:
            parts.append(f"{len(mcm['bssids'])} BSSIDs")
        if mcm["channels"]:
            parts.append(f"ch {','.join(sorted(mcm['channels']))}")
        if not mcm["n_connected"].empty:
            parts.append(
                f"peers {int(mcm['n_connected'].min())}-{int(mcm['n_connected'].max())}"
            )
        if not mcm["rssi"].empty:
            parts.append(
                f"beacon RSSI {mcm['rssi'].min():.0f} to {mcm['rssi'].max():.0f} dBm"
            )
        print(f"  mcm:     {'  '.join(parts)}")

    return issues


def _bh2_stats(nodes: list[Path]) -> tuple[int, pd.Series, set[str]]:
    rows = 0
    snrs: list[pd.Series] = []
    peers: set[str] = set()
    for node in nodes:
        fp = node / "bh2.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, low_memory=False,
                         usecols=lambda c: c in ("field_snr", "tag_sta_mac"))
        rows += len(df)
        if "field_snr" in df.columns:
            snrs.append(pd.to_numeric(df["field_snr"], errors="coerce").dropna())
        if "tag_sta_mac" in df.columns:
            peers.update(df["tag_sta_mac"].dropna().astype(str).unique())
    snr_all = pd.concat(snrs) if snrs else pd.Series(dtype=float)
    return rows, snr_all, peers


def _gps_lock_rate(fp: Path) -> tuple[int, int]:
    df = pd.read_csv(fp, low_memory=False,
                     usecols=lambda c: c in ("field_lat", "field_lon"))
    if "field_lat" not in df.columns or "field_lon" not in df.columns:
        return 0, 0
    lat = pd.to_numeric(df["field_lat"], errors="coerce")
    lon = pd.to_numeric(df["field_lon"], errors="coerce")
    locked = int(((lat != 0) | (lon != 0)).sum())
    return locked, len(df)


def _node_session_starts(nodes: list[Path]) -> dict[str, pd.Timestamp]:
    starts: dict[str, pd.Timestamp] = {}
    for node in nodes:
        fp = node / "bh2.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, low_memory=False, usecols=["timestamp"])
        ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
        if ts.empty:
            continue
        starts[node.name] = pd.to_datetime(int(ts.min()), unit="ns", utc=True)
    return starts


def _chrony_stats(nodes: list[Path]) -> dict | None:
    """Aggregate chrony clock-discipline stats across nodes.

    Cross-checks the bh2-timestamp clock-skew flag: chrony tells us whether
    a skew is real ongoing drift or just a node booting on a different day.
    """
    offsets: list[pd.Series] = []
    strata: set[str] = set()
    n_nodes = 0
    for node in nodes:
        fp = node / "chrony.csv"
        if not fp.exists():
            continue
        n_nodes += 1
        df = pd.read_csv(fp, low_memory=False,
                         usecols=lambda c: c in ("field_last_offset", "tag_stratum"))
        if "field_last_offset" in df.columns:
            offsets.append(pd.to_numeric(df["field_last_offset"], errors="coerce")
                           .dropna().abs())
        if "tag_stratum" in df.columns:
            strata.update(df["tag_stratum"].dropna().astype(str).unique())
    if n_nodes == 0:
        return None
    return {
        "n_nodes": n_nodes,
        "offset": pd.concat(offsets) if offsets else pd.Series(dtype=float),
        "strata": strata,
    }


def _mcm_stats(nodes: list[Path]) -> dict | None:
    """Aggregate mesh-client-manager radio-association stats across nodes."""
    ssids: set[str] = set()
    bssids: set[str] = set()
    channels: set[str] = set()
    n_connected: list[pd.Series] = []
    rssi: list[pd.Series] = []
    n_nodes = 0
    for node in nodes:
        fp = node / "mcm.csv"
        if not fp.exists():
            continue
        n_nodes += 1
        df = pd.read_csv(
            fp, low_memory=False,
            usecols=lambda c: c in (
                "field_ssid", "field_bssid", "field_channel",
                "field_num_connected", "field_beacon_rssi",
            ),
        )
        if "field_ssid" in df.columns:
            ssids.update(df["field_ssid"].dropna().astype(str).unique())
        if "field_bssid" in df.columns:
            bssids.update(df["field_bssid"].dropna().astype(str).unique())
        if "field_channel" in df.columns:
            channels.update(df["field_channel"].dropna().astype(str).unique())
        if "field_num_connected" in df.columns:
            n_connected.append(
                pd.to_numeric(df["field_num_connected"], errors="coerce").dropna()
            )
        if "field_beacon_rssi" in df.columns:
            rssi.append(pd.to_numeric(df["field_beacon_rssi"], errors="coerce").dropna())
    if n_nodes == 0:
        return None
    return {
        "n_nodes": n_nodes,
        "ssids": ssids,
        "bssids": bssids,
        "channels": channels,
        "n_connected": pd.concat(n_connected) if n_connected else pd.Series(dtype=float),
        "rssi": pd.concat(rssi) if rssi else pd.Series(dtype=float),
    }
