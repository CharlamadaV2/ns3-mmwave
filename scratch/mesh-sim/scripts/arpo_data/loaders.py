"""loaders script"""
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
# Supported source formats in priority order:
# -# ``gps/gps_position.csv`` or ``gps_position.csv`` — GeoTAK dedicated receiver.
# -# ``geotak_gps.csv`` — legacy GeoTAK format.
# -# ``gps.csv`` (legacy) — columns ``field_lat``, ``field_lon``, ``timestamp``.
#
# Silvus radio GPS (``silvus/gps.csv``) is intentionally excluded — it reports
# stale coordinates and contaminates the position data.
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


## @brief Load GPS data for a single node directory, trying all known source formats.
#
# Tries GPS source formats in priority order. Silvus radio GPS
# (``silvus/gps.csv``) is intentionally skipped — it reports stale
# coordinates that contaminate the position data.
#
# Priority order:
# -# ``<node>/gps/gps_position.csv`` or ``<node>/gps_position.csv``
#    — dedicated GPS receiver, ``latitude``/``longitude``/``time`` (ns).
# -# ``<node>/geotak_gps.csv`` — legacy GeoTAK, ``lat``/``lon``/``time`` (ISO-8601).
# -# ``<node>/gps.csv`` (legacy) — ``field_lat``/``field_lon``/``timestamp`` (ns).
#
# @param node Path to one node subdirectory.
# @return Normalised DataFrame with columns
#         ``[__node__, __label__, __t__, __lat__, __lon__]``,
#         or ``None`` if no GPS file is present or usable.
def _load_node_gps(node: Path) -> pd.DataFrame | None:

    # Priority 1: gps_position.csv — check gps/ subfolder first, then flat.
    for candidate in [node / "gps" / "gps_position.csv",
                      node / "gps_position.csv"]:
        if candidate.exists():
            df = pd.read_csv(candidate, low_memory=False)
            if {"latitude", "longitude", "time"}.issubset(df.columns):
                df = df.rename(columns={"latitude": "__lat__", "longitude": "__lon__"})
                df["__t__"]    = pd.to_datetime(
                    pd.to_numeric(df["time"], errors="coerce"), unit="ns", utc=True)
                df["__node__"] = node.name
                if "callsign" in df.columns:
                    df["__label__"] = df["callsign"].astype(str)
                elif "uid" in df.columns:
                    df["__label__"] = df["uid"].astype(str)
                else:
                    df["__label__"] = node.name
                return df[["__node__", "__label__", "__t__", "__lat__", "__lon__"]]

    # Priority 2: geotak_gps.csv — legacy GeoTAK format.
    geotak = node / "geotak_gps.csv"
    if geotak.exists():
        df = pd.read_csv(geotak, low_memory=False)
        if {"lat", "lon", "time"}.issubset(df.columns):
            df = df.rename(columns={"lat": "__lat__", "lon": "__lon__"})
            df["__t__"]     = pd.to_datetime(df["time"], errors="coerce", utc=True)
            df["__node__"]  = node.name
            df["__label__"] = df["uid"].astype(str) if "uid" in df.columns else node.name
            return df[["__node__", "__label__", "__t__", "__lat__", "__lon__"]]

    # Priority 3: legacy gps.csv — field_lat/field_lon/timestamp format only.
    # NOTE: silvus/gps.csv (lat/long/time format) is intentionally NOT loaded here.
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


## @brief Load GPS data from a directory of per-node subdirectories, returning
#  all nodes in one DataFrame ready for @ref plot_gps_tracks.
#
# Iterates every subdirectory under @p directory (skipping ``sdwan``) and
# calls @ref _load_node_gps on each. Silvus GPS is excluded. Static nodes
# (all coordinates identical — stale GPS fix) are also filtered out.
#
# @param directory  Path to the parent directory containing per-node subdirs.
# @return           Concatenated DataFrame with canonical GPS columns
#                   (``__node__``, ``__label__``, ``__t__``, ``__lat__``,
#                   ``__lon__``, ``__sec__``), or ``None`` if no usable
#                   GPS data is found in any subdirectory.
def load_gps_flat(directory: Path) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []

    for node_dir in sorted(directory.iterdir()):
        if not node_dir.is_dir() or node_dir.name == "sdwan":
            continue
        frame = _load_node_gps(node_dir)
        if frame is not None:
            frames.append(frame)

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True).dropna(
        subset=["__lat__", "__lon__", "__t__"])
    out = out[(out["__lat__"] != 0) | (out["__lon__"] != 0)]
    if out.empty:
        return None

    # Drop nodes where all coordinates are identical — stale GPS fix.
    out = out.groupby("__node__").filter(
        lambda g: g["__lat__"].nunique() > 1 or g["__lon__"].nunique() > 1
    )
    if out.empty:
        return None

    out["__sec__"] = session_relative_seconds(out)
    return out


## @brief Parse through node CSV files and format data for the simulator.
#
# Reads CSV files from per-node subdirectories under @p config_path and
# produces ``run.ini`` and ``nodes.json`` files for use by the ns-3 simulator.
#
# @param config_path  Path to the directory containing per-node subdirs.
# @return             ``None``; output files are written to disk as a side effect.
def load_config(config_path: Path) -> None:
    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        return None

    # TODO: iterate node subdirectories and read config.csv
    # TODO: write run.ini
    # TODO: write nodes.json

    return None