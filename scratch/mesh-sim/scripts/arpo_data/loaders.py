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


## @brief Filter a loaded DataFrame down to rows from a single calendar day.
#
# Works on any DataFrame produced by the loaders in this module, since they
# all add a ``__t__`` UTC datetime column. Returns ``None`` instead of an
# empty DataFrame so callers can use the same "skip if None" pattern they
# already use for missing data.
#
# @param df    DataFrame containing a ``__t__`` UTC datetime column.
# @param day   Calendar date to keep, e.g. ``date(2026, 5, 14)``.
# @return      Filtered DataFrame with the original index reset, or ``None``
#             if @p df is ``None``, lacks ``__t__``, or has no rows on @p day.
def filter_by_day(df: pd.DataFrame | None, day) -> pd.DataFrame | None:
    if df is None or "__t__" not in df.columns:
        return None
    out = df[df["__t__"].dt.date == day]
    if out.empty:
        return None
    return out.reset_index(drop=True)


## @brief List all calendar days present in a loaded DataFrame's ``__t__`` column.
#
# @param df  DataFrame containing a ``__t__`` UTC datetime column.
# @return    Sorted list of ``date`` objects, or an empty list if @p df is
#           ``None`` or lacks ``__t__``.
def days_present(df: pd.DataFrame | None) -> list:
    if df is None or "__t__" not in df.columns:
        return []
    return sorted(df["__t__"].dt.date.dropna().unique())


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


## @brief Load GPS fixes for a single node directory.
#
# Wraps @ref _load_node_gps and applies the standard post-processing:
# drops NaN coordinates, drops zero-coordinate no-fix sentinels, and
# adds the ``__sec__`` session-relative seconds column.
#
# @param node_dir Path to one node subdirectory (e.g. ``calfex_csv/IH01``).
# @return DataFrame with ``__lat__``, ``__lon__``, ``__sec__`` columns,
#         or ``None`` if no GPS data is available or all fixes are invalid.
def load_gps_scenario(node_dir: Path) -> pd.DataFrame | None:
    df = _load_node_gps(node_dir)
    if df is None:
        return None

    df = df.dropna(subset=["__lat__", "__lon__", "__t__"])
    df = df[(df["__lat__"] != 0) | (df["__lon__"] != 0)]
    if df.empty:
        return None

    df["__sec__"] = session_relative_seconds(df)
    return df


## @brief Load GPS data for a single node directory, trying all known source formats.
#
# Tries GPS source formats in priority order. Silvus radio GPS
# (``silvus/gps.csv``) is skipped.
#
# @param node Path to one node subdirectory.
# @return Normalised DataFrame with columns
#         ``[__node__, __label__, __t__, __lat__, __lon__]``,
#         or ``None`` if no GPS file is present or usable.
def _load_node_gps(node: Path) -> pd.DataFrame | None:

    # Reads data from gps_position
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

    return None


## @brief Load GPS data from a directory of per-node subdirectories, returning
#  all nodes in one DataFrame ready for @ref plot_gps_tracks.
#
# Iterates every subdirectory under @p directory (skipping ``sdwan``) and
# calls @ref _load_node_gps on each. Silvus GPS is excluded.
#
# @param directory  Path to the parent directory containing per-node subdirs.
# @return           Concatenated DataFrame with canonical GPS columns
#                   (``__node__``, ``__label__``, ``__t__``, ``__lat__``,
#                   ``__lon__``, ``__sec__``), or ``None`` if no usable
#                   GPS data is found in any subdirectory.
def load_gps_all(directory: Path) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []

    for node_dir in sorted(directory.iterdir()):
        # Check if directory exists
        if not node_dir.is_dir() or node_dir.name == "sdwan":
            continue
        # Check if node contains both gps and silvus directories
        if not (node_dir / "gps").is_dir() or not (node_dir / "silvus").is_dir():
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

## @brief Reads the node ID and name from a node's silvus config file.
#
# @param node_dir  Path to one node subdirectory.
# @return          Tuple of ``(node_name, node_id)`` or ``None`` if
#                  ``config.csv`` is missing or has no ``node_id`` column.
def load_node_id(node_dir: Path) -> dict[str, str] | None:
    cfg_fp = node_dir / "silvus" / "config.csv"
    if not cfg_fp.exists():
        return None
    cfg = pd.read_csv(cfg_fp, usecols=["node_id"], nrows=1, low_memory=False)
    if cfg.empty:
        return None
    return {str(cfg["node_id"].iloc[0]): node_dir.name}

## @brief Parses through the network status file for IH Node
#
# Reads ``network_status.csv`` from @p net_stat_dir and returns a normalised
# DataFrame with per-link RF quality columns. Computes ``rcpi`` as the mean
# RSSI across active antennas (antennas at −110 dBm sentinel are excluded).
# Rows whose ``neighbor`` is not in @p valid_nodes are dropped.
#
# @param net_stat_dir  Path to the ``silvus/`` directory containing ``network_status.csv``.
# @param valid_nodes   List of node ID strings to keep in the ``neighbor`` column.
# @return              DataFrame with columns ``__t__``, ``node_id``, ``virtual_ip``,
#                      ``neighbor``, ``snr``, ``mcs``, ``mcs_rx``,
#                      ``rssi_ant1..4``, ``rcpi``, or ``None`` if the file is
#                      absent or empty.
def _load_net_stat(net_stat_dir: Path, valid_nodes: list[str]) -> pd.DataFrame | None:
    fp = net_stat_dir / "network_status.csv"
    if not fp.exists():
        return None

    df = pd.read_csv(fp, low_memory=False)
    if df.empty:
        return None

    #Converts time to ns
    df["__t__"] = pd.to_datetime(
        pd.to_numeric(df["time"], errors="coerce"), unit="ns", utc=True)

    #Changes data type to int
    for col in ("mcs", "mcs_rx", "snr",
                "rssi_ant1", "rssi_ant2", "rssi_ant3", "rssi_ant4"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only rows where neighbor is a valid IH node.
    if valid_nodes and "neighbor" in df.columns:
        df = df[df["neighbor"].astype(str).isin(valid_nodes)]

    if df.empty:
        return None

    rssi_cols = [c for c in ("rssi_ant1", "rssi_ant2", "rssi_ant3", "rssi_ant4")
                 if c in df.columns]
    if rssi_cols:
        rssi = df[rssi_cols].copy()
        rssi[rssi <= -110] = float("nan")
        df["rcpi"] = rssi.mean(axis=1)

    return df.dropna(subset=["__t__"]).reset_index(drop=True)


## @brief Parses through local stats file for IH Node
#
# Reads ``local_stats.csv`` from @p local_stat_dir and returns a normalised
# DataFrame with per-node noise, interference, throughput, and PER columns.
# Rows whose ``node_id`` is not in @p valid_nodes are dropped.
#
# @param local_stat_dir  Path to the ``silvus/`` directory containing ``local_stats.csv``.
# @param valid_nodes     List of node ID strings to keep.
# @return                DataFrame with columns ``__t__``, ``node_id``, ``virtual_ip``,
#                        ``noise_level``, ``interference``, ``throughput_mbps``,
#                        ``per``, or ``None`` if the file is absent or empty.
def _load_local_stat(local_stat_dir: Path, valid_nodes: list[str]) -> pd.DataFrame | None:
    fp = local_stat_dir / "local_stats.csv"
    if not fp.exists():
        return None

    df = pd.read_csv(fp, low_memory=False)
    if df.empty:
        return None

    df["__t__"] = pd.to_datetime(
        pd.to_numeric(df["time"], errors="coerce"), unit="ns", utc=True)

    #Converts data type to int
    for col in ("noise_level", "interference"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    #Converts data type to string
    if "fw_uc" in df.columns:
        df["throughput_mbps"] = pd.to_numeric(
            df["fw_uc"].astype(str).str.strip('"'), errors="coerce")

    #Converts data type to string and calculates PER
    if {"input_dropped", "input_uc"}.issubset(df.columns):
        dropped = pd.to_numeric(
            df["input_dropped"].astype(str).str.strip('"'), errors="coerce")
        total   = pd.to_numeric(
            df["input_uc"].astype(str).str.strip('"'), errors="coerce")
        df["per"] = (dropped / total.replace(0, float("nan"))).clip(0, 1)

    # Keep only rows belonging to valid IH nodes.
    if valid_nodes and "node_id" in df.columns:
        df = df[df["node_id"].astype(str).isin(valid_nodes)]

    if df.empty:
        return None

    keep = ["__t__", "node_id", "virtual_ip", "noise_level", "interference",
            "throughput_mbps", "per"]
    return df[[c for c in keep if c in df.columns]].dropna(
        subset=["__t__"]).reset_index(drop=True)


## @brief Parses through config file for IH Node
#
# Reads ``config.csv`` from @p config_dir and returns a normalised DataFrame
# with per-node radio configuration columns. Rows whose ``node_id`` is not
# in @p valid_nodes are dropped.
#
# @param config_dir   Path to the ``silvus/`` directory containing ``config.csv``.
# @param valid_nodes  List of node ID strings to keep.
# @return             DataFrame with columns ``__t__``, ``node_id``, ``virtual_ip``,
#                     ``freq``, ``bw``, ``power_dBm``, or ``None`` if the file
#                     is absent or empty.
def _load_config_stat(config_dir: Path, valid_nodes: list[str]) -> pd.DataFrame | None:
    fp = config_dir / "config.csv"
    if not fp.exists():
        return None

    df = pd.read_csv(fp, low_memory=False)
    if df.empty:
        return None

    df["__t__"] = pd.to_datetime(
        pd.to_numeric(df["time"], errors="coerce"), unit="ns", utc=True)

    for col in ("freq", "bw", "power_dBm"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only rows belonging to valid IH nodes.
    if valid_nodes and "node_id" in df.columns:
        df = df[df["node_id"].astype(str).isin(valid_nodes)]

    if df.empty:
        return None

    keep = ["__t__", "node_id", "virtual_ip", "freq", "bw", "power_dBm"]
    return df[[c for c in keep if c in df.columns]].dropna(
        subset=["__t__"]).reset_index(drop=True)


## @brief Parse through silvus data from each node and produce an RF quality DataFrame.
#
# Merges network status, local stats, and config data for one node. Only
# rows belonging to valid IH nodes are kept throughout.
#
# @param node_dir     Path to the directory that contains the node's data.
# @param valid_nodes  List of node ID strings to keep across all three sources.
# @return             Concatenated DataFrame with RF Signal Quality columns,
#                     or ``None`` if no network status data is found.
def load_rf_scenario(node_dir: Path, valid_nodes: list[str], 
                    node_id_to_name: dict[str, str]) -> pd.DataFrame | None:
    silvus = node_dir / "silvus"

    if not silvus.is_dir() or not any(silvus.iterdir()):
        return None

    #Network Status Parser
    net = _load_net_stat(silvus, valid_nodes)
    if net is None:
        return None

    #Local Status + Network Status
    local = _load_local_stat(silvus, valid_nodes)
    if local is not None:
        net = net.merge(local, on=["__t__", "node_id"], how="left",
                        suffixes=("", "_local"))

    #Config Status + Local Status + Network Status
    cfg = _load_config_stat(silvus, valid_nodes)
    if cfg is not None:
        net = net.merge(cfg, on=["__t__", "node_id"], how="left",
                        suffixes=("", "_cfg"))

    # Replace node_id and neighbor with IH names.
    if "node_id" in net.columns:
        net["node_id"] = net["node_id"].astype(str).map(
            lambda nid: node_id_to_name.get(nid, nid))
    if "neighbor" in net.columns:
        net["neighbor"] = net["neighbor"].astype(str).map(
            lambda nid: node_id_to_name.get(nid, nid))

    net["__node__"] = node_dir.name
    net["__sec__"]  = (net["__t__"] - net["__t__"].min()).dt.total_seconds()

    front = ["__node__", "__t__", "__sec__", "node_id", "virtual_ip", "neighbor",
             "snr", "rcpi", "mcs", "mcs_rx",
             "rssi_ant1", "rssi_ant2", "rssi_ant3", "rssi_ant4",
             "throughput_mbps", "per",
             "noise_level", "interference",
             "freq", "bw", "power_dBm"]
    ordered = [c for c in front if c in net.columns]
    rest    = [c for c in net.columns if c not in ordered]
    return net[ordered + rest].reset_index(drop=True)