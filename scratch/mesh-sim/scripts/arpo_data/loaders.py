"""
Per-scenario CSV loaders.

Node clocks are skewed, so loaders add ``__sec__`` (seconds since each
session's first sample) for cross-session comparison. ``__session__``
is the hostname. Loaders return ``DataFrame | None``; None means no
usable data.
"""

from pathlib import Path

import pandas as pd

from .topology import build_topology, resolve_peer


def to_datetime(df: pd.DataFrame) -> pd.Series:
    """``timestamp`` (ns since epoch) -> UTC datetime."""
    return pd.to_datetime(
        pd.to_numeric(df["timestamp"], errors="coerce"),
        unit="ns", utc=True,
    )


def session_relative_seconds(df: pd.DataFrame) -> pd.Series:
    """Seconds since each session's earliest ``__t__``."""
    key = "__session__" if "__session__" in df.columns else "__node__"
    t0 = df.groupby(key)["__t__"].transform("min")
    return (df["__t__"] - t0).dt.total_seconds()


def load_bh2_scenario(scen_dir: Path) -> pd.DataFrame | None:
    """Every node's bh2.csv concatenated, with topology and ``__sec__``."""
    topo = build_topology(scen_dir)
    frames = []
    for node in sorted(scen_dir.iterdir()):
        if not node.is_dir() or node.name == "sdwan":
            continue
        fp = node / "bh2.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, low_memory=False)
        df["__node__"] = node.name
        df["__session__"] = node.name
        df["__t__"] = to_datetime(df)
        for col in ("field_snr", "field_snr_ga64",
                    "field_mcs_tx", "field_mcs_rx",
                    "field_bytes_tx", "field_bytes_rx",
                    "field_per", "field_rcpi",
                    "field_packets_tx", "field_agc", "field_bh2_temperature",
                    "field_num_mpdu_rx_fcs_error",
                    "field_num_mpdu_tx_failed",
                    "field_num_mpdu_tx_no_ack"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True).dropna(subset=["__t__"])
    if out.empty:
        return None
    out["__sec__"] = session_relative_seconds(out)
    out["__peer__"] = out["tag_sta_mac"].map(lambda m: resolve_peer(m, topo))
    out.attrs["topology"] = topo
    return out


def load_mcm_scenario(scen_dir: Path) -> pd.DataFrame | None:
    """Every node's mcm.csv concatenated -- per-radio mesh-client-manager state."""
    frames = []
    for node in sorted(scen_dir.iterdir()):
        if not node.is_dir() or node.name == "sdwan":
            continue
        fp = node / "mcm.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, low_memory=False)
        if "field_mac" not in df.columns:
            continue
        df["__node__"] = node.name
        df["__t__"] = to_datetime(df)
        df["__session__"] = node.name
        for col in ("field_channel", "field_beacon_rssi",
                    "field_num_connected", "field_sample_rate"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).dropna(subset=["__t__"])
    return out if not out.empty else None


def load_gps_scenario(scen_dir: Path) -> pd.DataFrame | None:
    """GPS per node (prefers ``geotak_gps.csv``); drops lat=lon=0 (no fix)."""
    frames = []
    for node in sorted(scen_dir.iterdir()):
        if not node.is_dir() or node.name == "sdwan":
            continue
        frame = _load_node_gps(node)
        if frame is not None:
            frames.append(frame)
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True).dropna(subset=["__lat__", "__lon__", "__t__"])
    df = df[(df["__lat__"] != 0) | (df["__lon__"] != 0)]
    if df.empty:
        return None
    df["__sec__"] = session_relative_seconds(df)
    return df


def _load_node_gps(node: Path) -> pd.DataFrame | None:
    geotak = node / "geotak_gps.csv"
    if geotak.exists():
        df = pd.read_csv(geotak, low_memory=False)
        if {"lat", "lon", "time"}.issubset(df.columns):
            df = df.rename(columns={"lat": "__lat__", "lon": "__lon__"})
            df["__t__"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
            df["__node__"] = node.name
            df["__label__"] = df["uid"].astype(str) if "uid" in df.columns else node.name
            return df[["__node__", "__label__", "__t__", "__lat__", "__lon__"]]

    gpsd = node / "gps.csv"
    if gpsd.exists():
        df = pd.read_csv(gpsd, low_memory=False)
        if {"field_lat", "field_lon", "timestamp"}.issubset(df.columns):
            df = df.assign(
                __lat__=pd.to_numeric(df["field_lat"], errors="coerce"),
                __lon__=pd.to_numeric(df["field_lon"], errors="coerce"),
                __t__=to_datetime(df),
                __node__=node.name,
                __label__=node.name,
            )
            return df[["__node__", "__label__", "__t__", "__lat__", "__lon__"]]

    return None
