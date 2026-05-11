"""
Per-scenario CSV loaders.

Field-node clocks are skewed across nodes, so absolute time is not
comparable. Every loader adds ``__sec__`` (seconds since each node's
first timestamp) for cross-node comparisons.

Loaders return ``pd.DataFrame | None``: ``None`` means "no usable data
for this scenario." Callers must handle that case.
"""

from pathlib import Path

import pandas as pd

from .topology import build_topology, resolve_peer


def to_datetime(df: pd.DataFrame) -> pd.Series:
    """Parse the ``timestamp`` column (ns since epoch) to UTC datetime."""
    return pd.to_datetime(
        pd.to_numeric(df["timestamp"], errors="coerce"),
        unit="ns", utc=True,
    )


def session_relative_seconds(df: pd.DataFrame) -> pd.Series:
    """Seconds since each node's earliest ``__t__`` value."""
    t0 = df.groupby("__node__")["__t__"].transform("min")
    return (df["__t__"] - t0).dt.total_seconds()


def load_bh2_scenario(scen_dir: Path) -> pd.DataFrame | None:
    """Concatenate every node's ``bh2.csv`` with topology and relative time."""
    frames = []
    for node in sorted(scen_dir.iterdir()):
        if not node.is_dir() or node.name == "sdwan":
            continue
        fp = node / "bh2.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, low_memory=False)
        df["__node__"] = node.name
        df["__t__"] = to_datetime(df)
        for col in ("field_snr", "field_mcs_tx", "field_mcs_rx",
                    "field_bytes_tx", "field_per", "field_rcpi", "field_packets_tx"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True).dropna(subset=["__t__"])
    if out.empty:
        return None
    out["__sec__"] = session_relative_seconds(out)
    topo = build_topology(scen_dir)
    out["__peer__"] = out["tag_sta_mac"].map(lambda m: resolve_peer(m, topo))
    out.attrs["topology"] = topo
    return out


def load_ping_scenario(scen_dir: Path) -> pd.DataFrame | None:
    """Concatenate every node's ``ping.csv``."""
    frames = []
    for node in sorted(scen_dir.iterdir()):
        if not node.is_dir() or node.name == "sdwan":
            continue
        fp = node / "ping.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, low_memory=False)
        df["__node__"] = node.name
        df["__t__"] = to_datetime(df)
        frames.append(df)
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True).dropna(subset=["__t__"])
    if df.empty:
        return None
    if "field_average_response_ms" in df.columns:
        df["field_average_response_ms"] = pd.to_numeric(
            df["field_average_response_ms"], errors="coerce")
    df["__sec__"] = session_relative_seconds(df)
    return df


def load_gps_scenario(scen_dir: Path) -> pd.DataFrame | None:
    """
    Load each node's GPS, preferring ``geotak_gps.csv`` over ``gps.csv``.

    Output columns: ``__node__``, ``__label__``, ``__t__``, ``__lat__``, ``__lon__``.
    Rows with lat=lon=0 (no GPS fix) are dropped.
    """
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
    return df if not df.empty else None


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
