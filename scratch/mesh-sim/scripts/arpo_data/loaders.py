## @file loaders.py
# @brief Per-scenario CSV loaders for backhaul-radio and GPS data.
#
# **Clock skew handling**
# Node clocks drift relative to each other, so raw timestamps are not
# directly comparable across nodes. All loaders add a ``__sec__`` column
# containing seconds since each *session's* first sample, making
# cross-session time axes directly comparable.
#
# **Column conventions**
# | Column        | Type             | Description                                 |
# |---------------|------------------|---------------------------------------------|
# | ``__node__``  | str              | Node directory name (chassis hostname).     |
# | ``__session__``| str             | Session identifier; same as ``__node__``.   |
# | ``__t__``     | datetime (UTC)   | Parsed timestamp.                           |
# | ``__sec__``   | float            | Seconds since session start.                |
# | ``__peer__``  | str              | Peer label resolved via topology (bh2 only).|
# | ``__lat__``   | float            | Latitude in decimal degrees (GPS only).     |
# | ``__lon__``   | float            | Longitude in decimal degrees (GPS only).    |

from pathlib import Path

import pandas as pd

from .topology import build_topology, resolve_peer


## @brief Convert a ``timestamp`` column (nanoseconds since epoch) to UTC datetimes.
#
# Non-numeric values are coerced to NaT rather than raising.
#
# @param df DataFrame containing a ``timestamp`` column.
# @return Series of timezone-aware UTC datetimes.
def to_datetime(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        pd.to_numeric(df["timestamp"], errors="coerce"),
        unit="ns", utc=True,
    )


## @brief Compute seconds elapsed since each session's earliest ``__t__`` value.
#
# Groups by ``__session__`` if present, otherwise by ``__node__``. This
# normalises clock skew so that t=0 is the first sample for each node.
#
# @param df DataFrame with ``__t__`` (datetime) and either ``__session__``
#           or ``__node__`` columns.
# @return Float Series of relative seconds, aligned to the input index.
def session_relative_seconds(df: pd.DataFrame) -> pd.Series:
    key = "__session__" if "__session__" in df.columns else "__node__"
    t0 = df.groupby(key)["__t__"].transform("min")
    return (df["__t__"] - t0).dt.total_seconds()


## @brief Load and concatenate every node's ``bh2.csv`` for a scenario.
#
# Backhaul-2 (bh2) logs contain per-link radio metrics sampled at the
# driver level: SNR, RCPI, MCS (TX/RX), byte counters, PER, and AGC.
#
#
# @param scen_dir Path to the scenario directory (contains one subdir per node).
# @return Concatenated DataFrame, or ``None`` if no ``bh2.csv`` files were found.
def load_bh2_scenario(scen_dir: Path) -> pd.DataFrame | None:
    topo = build_topology(scen_dir)
    frames = []
    for node in sorted(scen_dir.iterdir()):
        if not node.is_dir() or node.name == "sdwan":
            continue
        fp = node / "bh2.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, low_memory=False)
        df["__node__"]    = node.name
        df["__session__"] = node.name
        df["__t__"]       = to_datetime(df)
        # Coerce all known numeric metric columns; unknown columns are ignored.
        for col in (
            "field_snr", "field_snr_ga64",
            "field_mcs_tx", "field_mcs_rx",
            "field_bytes_tx", "field_bytes_rx",
            "field_per", "field_rcpi",
            "field_packets_tx", "field_agc", "field_bh2_temperature",
            "field_num_mpdu_rx_fcs_error",
            "field_num_mpdu_tx_failed",
            "field_num_mpdu_tx_no_ack",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        frames.append(df)
    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True).dropna(subset=["__t__"])
    if out.empty:
        return None
    out["__sec__"]  = session_relative_seconds(out)
    out["__peer__"] = out["tag_sta_mac"].map(lambda m: resolve_peer(m, topo))
    out.attrs["topology"] = topo
    return out


## @brief Load and concatenate every node's ``mcm.csv`` for a scenario.
#
# Mesh-Client-Manager (MCM) logs contain per-radio state: channel, beacon
# RSSI, connected-client count, and sample rate. ``field_mac`` is required;
# files missing that column are skipped.
#
# @param scen_dir Path to the scenario directory.
# @return Concatenated DataFrame, or ``None`` if no usable ``mcm.csv`` files exist.
def load_mcm_scenario(scen_dir: Path) -> pd.DataFrame | None:
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
        df["__node__"]    = node.name
        df["__t__"]       = to_datetime(df)
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


## @brief Load and concatenate GPS fixes for every node in a scenario.
#
# Two GPS source formats are supported, in priority order:
# -# ``geotak_gps.csv`` — columns ``lat``, ``lon``, ``time`` (ISO-8601).
# -# ``gps.csv`` — columns ``field_lat``, ``field_lon``, ``timestamp``
#    (nanoseconds since epoch).
#
# Rows with latitude=0 **and** longitude=0 are dropped as "no fix" sentinels.
#
# @param scen_dir Path to the scenario directory.
# @return Concatenated DataFrame with ``__lat__``, ``__lon__``, ``__sec__``
#         columns, or ``None`` if no GPS data is available.
def load_gps_scenario(scen_dir: Path) -> pd.DataFrame | None:
    frames = []
    for node in sorted(scen_dir.iterdir()):
        if not node.is_dir() or node.name == "sdwan":
            continue
        frame = _load_node_gps(node)
        if frame is not None:
            frames.append(frame)
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True).dropna(
        subset=["__lat__", "__lon__", "__t__"])
    df = df[(df["__lat__"] != 0) | (df["__lon__"] != 0)]
    if df.empty:
        return None
    df["__sec__"] = session_relative_seconds(df)
    return df


## @brief Load GPS data for a single node directory, trying both source formats.
#
# Prefers ``geotak_gps.csv`` over ``gps.csv``. Returns only the canonical
# columns needed by @ref load_gps_scenario.
#
# @param node Path to one node subdirectory.
# @return Normalised DataFrame with columns
#         ``[__node__, __label__, __t__, __lat__, __lon__]``,
#         or ``None`` if neither GPS file is present or usable.
def _load_node_gps(node: Path) -> pd.DataFrame | None:
    # Prefer the higher-quality GeoTAK source.
    geotak = node / "geotak_gps.csv"
    if geotak.exists():
        df = pd.read_csv(geotak, low_memory=False)
        if {"lat", "lon", "time"}.issubset(df.columns):
            df = df.rename(columns={"lat": "__lat__", "lon": "__lon__"})
            df["__t__"]     = pd.to_datetime(df["time"], errors="coerce", utc=True)
            df["__node__"]  = node.name
            # Use the TAK UID as the display label when available; fall back to hostname.
            df["__label__"] = df["uid"].astype(str) if "uid" in df.columns else node.name
            return df[["__node__", "__label__", "__t__", "__lat__", "__lon__"]]

    # Fall back to the gpsd-derived CSV.
    gpsd = node / "gps.csv"
    if gpsd.exists():
        df = pd.read_csv(gpsd, low_memory=False)
        if {"field_lat", "field_lon", "timestamp"}.issubset(df.columns):
            df = df.assign(
                __lat__   = pd.to_numeric(df["field_lat"],  errors="coerce"),
                __lon__   = pd.to_numeric(df["field_lon"],  errors="coerce"),
                __t__     = to_datetime(df),
                __node__  = node.name,
                __label__ = node.name,
            )
            return df[["__node__", "__label__", "__t__", "__lat__", "__lon__"]]

    return None