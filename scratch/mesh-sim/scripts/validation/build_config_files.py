'''build_config_files.py'''
## @file build_config_files.py
# @brief Creates per-day config files (run.ini and nodes.json) for the simulator.
#
# Channel parameters (freq / bw / power) are read ONCE from the per-node Silvus
# config CSVs under ``--csv-dir``. Node GPS start positions are read PER DAY from
# each day's ``gps_all_nodes_trace.csv`` under ``--input``, so every day gets its
# own ``nodes.json`` reflecting that day's starting geometry — no shared base
# copy. ``build_waypoints.py`` fills in each day's trajectories afterwards.


import argparse
import json
import math
import sys
import configparser
from pathlib import Path

import pandas as pd


_NODE_HEIGHT_M  = 1.5    ##< Default z height for ground-level IH nodes (metres).
_GATEWAY_HEIGHT = 30.0   ##< Default z height for the elevated gateway node (metres).
_DURATION_S     = 60.0   ##< Default simulation duration in seconds.
_WARMUP_S       = 0.0    ##< Default warmup period in seconds.
_TICK_S         = 0.1    ##< Default simulation tick interval in seconds.
_NOISE_FIGURE   = 5.0    ##< Default receiver noise figure in dB.
_DEMAND_MBPS    = 0.25   ##< Constant offered load written to [traffic] demand_mbps.

# Column-name candidates for gps_all_nodes_trace.csv. If auto-detection fails,
# the loader prints the actual columns — adjust these tuples to match.
# ENU (east_m/north_m) is preferred over lat/lon when present.
_TRACE_NODE_COLS  = ("node", "node_id", "name", "id", "device", "deviceName")
_TRACE_EAST_COLS  = ("east_m", "east", "x")
_TRACE_NORTH_COLS = ("north_m", "north", "y")
_TRACE_LAT_COLS   = ("lat_deg", "latitude", "lat")
_TRACE_LON_COLS   = ("lon_deg", "longitude", "lon")
_TRACE_TIME_COLS  = ("sec_since_origin", "t_utc", "t", "time", "timestamp", "utc", "datetime")


## @brief Return the first candidate column that exists in @p df, else None.
def _first_col(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


## @brief Convert WGS-84 lat/lon to local ENU metres relative to an origin.
#
# Uses the flat-earth approximation — accurate to within ~1 m for areas
# smaller than 10 km across.
def _gps_to_enu(lat: float, lon: float,
                lat0: float, lon0: float) -> tuple[float, float]:
    R     = 6_378_137.0
    east  = R * math.cos(math.radians(lat0)) * math.radians(lon - lon0)
    north = R * math.radians(lat - lat0)
    return round(east, 2), round(north, 2)


## @brief Read median channel params (freq / bw / power) from per-node Silvus configs.
#
# Reads the first row of ``silvus/config.csv`` under each node subdirectory of
# @p csv_dir. GPS is deliberately NOT read here — start positions come per-day
# from @ref _load_day_gps.
#
# @param csv_dir  Directory containing per-node subdirs (from --csv-dir).
# @return         Dict with ``freq_mhz`` / ``bw_mhz`` / ``power_dbm``, or None.
def _load_channel_config(csv_dir: Path) -> dict | None:
    if not csv_dir.is_dir():
        print(f"ERROR: csv dir not found: {csv_dir}", file=sys.stderr)
        return None

    freqs, bws, powers = [], [], []
    n_nodes = 0
    for node_dir in sorted(csv_dir.iterdir()):
        if not node_dir.is_dir() or node_dir.name == "sdwan":
            continue
        cfg_fp = node_dir / "silvus" / "config.csv"
        if not cfg_fp.exists():
            print(f"  WARNING: {node_dir.name} has no silvus/config.csv, skipping")
            continue
        cfg = pd.read_csv(cfg_fp, nrows=1, low_memory=False)
        for col, lst in (("freq", freqs), ("bw", bws), ("power_dBm", powers)):
            if col in cfg.columns:
                val = pd.to_numeric(cfg[col].iloc[0], errors="coerce")
                if pd.notna(val):
                    lst.append(float(val))
        n_nodes += 1

    if n_nodes == 0:
        print(f"ERROR: no node configs found in {csv_dir}", file=sys.stderr)
        return None

    return {
        "freq_mhz":  float(pd.Series(freqs).median())  if freqs  else 2400.0,
        "bw_mhz":    float(pd.Series(bws).median())    if bws    else 20.0,
        "power_dbm": float(pd.Series(powers).median()) if powers else 30.0,
    }


## @brief Read each node's earliest position from one day's trace.
#
# Returns each node's first (earliest) fix as local ENU metres. Prefers the
# trace's own ``east_m``/``north_m`` columns (a fixed origin shared with
# ``build_waypoints.py``); falls back to converting ``lat_deg``/``lon_deg``
# about the day's centroid if the ENU columns are absent. Rows are sorted by
# the time column first so "first" means earliest in time.
#
# @param trace_fp  Path to one day's ``gps_all_nodes_trace.csv``.
# @return          list of ``{"name","x","y"}`` dicts (ENU metres), or None.
def _load_day_gps(trace_fp: Path) -> list[dict] | None:
    if not trace_fp.is_file():
        print(f"  WARNING: no trace file {trace_fp}", file=sys.stderr)
        return None

    df = pd.read_csv(trace_fp, low_memory=False)
    name_col  = _first_col(df, _TRACE_NODE_COLS)
    time_col  = _first_col(df, _TRACE_TIME_COLS)
    east_col  = _first_col(df, _TRACE_EAST_COLS)
    north_col = _first_col(df, _TRACE_NORTH_COLS)
    lat_col   = _first_col(df, _TRACE_LAT_COLS)
    lon_col   = _first_col(df, _TRACE_LON_COLS)

    if name_col is None:
        print(f"  ERROR: {trace_fp.name} has no node column "
              f"(have {list(df.columns)})", file=sys.stderr)
        return None

    if time_col:
        df = df.sort_values(time_col)

    # Preferred path: trace already provides ENU metres in a fixed origin.
    if east_col and north_col:
        df[east_col]  = pd.to_numeric(df[east_col], errors="coerce")
        df[north_col] = pd.to_numeric(df[north_col], errors="coerce")
        df = df.dropna(subset=[east_col, north_col])
        nodes = []
        for name, grp in df.groupby(name_col, sort=True):
            first = grp.iloc[0]
            nodes.append({
                "name": str(name),
                "x":    round(float(first[east_col]), 2),
                "y":    round(float(first[north_col]), 2),
            })
        return nodes or None

    # Fallback: convert lat/lon about this day's centroid.
    if lat_col and lon_col:
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        df = df.dropna(subset=[lat_col, lon_col])
        df = df[(df[lat_col] != 0) | (df[lon_col] != 0)]
        firsts = []
        for name, grp in df.groupby(name_col, sort=True):
            first = grp.iloc[0]
            firsts.append({"name": str(name),
                           "lat": float(first[lat_col]),
                           "lon": float(first[lon_col])})
        if not firsts:
            return None
        lat0 = sum(f["lat"] for f in firsts) / len(firsts)
        lon0 = sum(f["lon"] for f in firsts) / len(firsts)
        nodes = []
        for f in firsts:
            east, north = _gps_to_enu(f["lat"], f["lon"], lat0, lon0)
            nodes.append({"name": f["name"], "x": east, "y": north})
        return nodes or None

    print(f"  ERROR: {trace_fp.name} has no east_m/north_m or lat/lon columns "
          f"(have {list(df.columns)})", file=sys.stderr)
    return None


## @brief Write ``run.ini`` to @p output_path using the provided parameters.
def _create_ini(output_path: Path, scenario_name: str,
                freq_ghz: float, bw_mhz: float, power_dbm: float,
                demand_mbps: float, gateway_id: str | None) -> None:
    cfg = configparser.ConfigParser()

    cfg["scenario"] = {
        "name":           scenario_name,
        "seed":           "1",
        "run_id":         "1",
        "duration_s":     str(_DURATION_S),
        "warmup_s":       str(_WARMUP_S),
        "tick_s":         str(_TICK_S),
        "nodes_file":     "nodes.json",
        "buildings_file": "",
    }
    cfg["channel"] = {
        "frequency_ghz":     str(freq_ghz),
        "tx_power_dbm":      str(power_dbm),
        "scenario":          "RMa",
        "channel_model":     "3gpp",
        "blockage_enabled":  "false",
        "bandwidth_mhz":     str(bw_mhz),
        "noise_figure_db":   str(_NOISE_FIGURE),
        "condition_model":   "static_los",
        "amc_model":         "silvus",
        "beamforming_model": "table",
        "tx_array_gain_dbi": "6",
        "rx_array_gain_dbi": "6",
    }
    if gateway_id is not None:
        cfg["traffic"] = {
            "model":           "constant",
            "demand_mbps":     str(demand_mbps),
            "flow_topology":   "gateway",
            "gateway_node_id": gateway_id,
        }
    else:
        cfg["traffic"] = {
            "model":           "constant",
            "demand_mbps":     str(demand_mbps),
            "flow_topology":   "all_pairs",
            "gateway_node_id": "",
        }
    cfg["routing"] = {
        "algorithm": "shortest_path",
        "max_hops":  "5",
    }
    cfg["output"] = {
        "viz_tick_ms": "100",
    }

    ini_path = output_path / "run.ini"
    with open(ini_path, "w") as f:
        cfg.write(f)
    print(f"  wrote {ini_path}")


## @brief Write ``nodes.json`` to @p output_path for a single day.
#
# Node positions are the ENU metres returned by @ref _load_day_gps (same origin
# as the trace, and therefore as ``build_waypoints.py``). The gateway (if any)
# is placed at the centroid of that day's nodes, elevated and fixed; its ``id``
# matches the ``gateway_node_id`` written into ``run.ini``. IH nodes use
# waypoint mobility with empty waypoints for ``build_waypoints.py`` to fill.
def _create_json(output_path: Path, node_data: list[dict],
                 gateway_name: str | None) -> None:
    nodes = []
    if gateway_name is not None:
        gx = round(sum(n["x"] for n in node_data) / len(node_data), 2)
        gy = round(sum(n["y"] for n in node_data) / len(node_data), 2)
        nodes.append({
            "id":       gateway_name,
            "role":     "peer",
            "mobility": "fixed",
            "position": {"x": gx, "y": gy, "z": _GATEWAY_HEIGHT},
        })

    for node in node_data:
        nodes.append({
            "id":       node["name"],
            "role":     "peer",
            "mobility": "waypoint",
            "position": {"x": node["x"], "y": node["y"], "z": _NODE_HEIGHT_M},
            "waypoints": [],
        })

    json_path = output_path / "nodes.json"
    with open(json_path, "w") as f:
        json.dump(nodes, f, indent=2)
    print(f"  wrote {json_path}  ({len(nodes)} nodes)")


## @brief True if @p name is in YYYY-MM-DD format.
def _is_date_dir(name: str) -> bool:
    parts = name.split("-")
    return (
        len(parts) == 3
        and all(p.isdigit() for p in parts)
        and len(parts[0]) == 4
        and len(parts[1]) == 2
        and len(parts[2]) == 2
    )


## @brief Return the sorted YYYY-MM-DD day names discoverable under @p per_day_dir.
def _discover_days(per_day_dir: Path) -> list[str]:
    if not per_day_dir.is_dir():
        return []

    # Single day: per_day_dir itself is a YYYY-MM-DD directory.
    if _is_date_dir(per_day_dir.name) and (per_day_dir / "gps_all_nodes_trace.csv").is_file():
        return [per_day_dir.name]

    # Multiple days: per_day_dir contains YYYY-MM-DD subdirectories.
    days: set[str] = set()
    for d in per_day_dir.iterdir():
        if d.is_dir() and _is_date_dir(d.name) and (d / "gps_all_nodes_trace.csv").is_file():
            days.add(d.name)

    return sorted(days)


## @brief Generate one ``nodes.json`` + ``run.ini`` per day.
#
# Channel params are read once from @p csv_dir (day-independent). For each day,
# that day's GPS start positions are read from
# ``<per_day_dir>/<day>/gps_all_nodes_trace.csv`` and written to
# ``<output_path>/<day>/``.
#
# @param csv_dir        Per-node config dir (from --csv-dir).
# @param per_day_dir    Dir containing ``<YYYY-MM-DD>/gps_all_nodes_trace.csv``.
# @param output_path    Output root; one subdirectory per day is created.
# @param band           Radio band string (``"sub-6"`` or ``"mmwave"``).
# @param days           List of ``"YYYY-MM-DD"`` strings to produce configs for.
# @param gateway_enable Whether to add a gateway node + gateway traffic topology.
# @return               0 on success, 1 on error.
def load_calfex_data_per_day(csv_dir: Path, per_day_dir: Path, output_path: Path,
                             band: str, days: list[str], gateway_enable: bool) -> int:
    if not days:
        print("ERROR: no days to process", file=sys.stderr)
        return 1

    # Channel config is day-independent — read it once.
    chan = _load_channel_config(csv_dir)
    if chan is None:
        return 1
    freq_ghz  = round(chan["freq_mhz"] / 1000, 4)
    bw_mhz    = chan["bw_mhz"]
    power_dbm = chan["power_dbm"]
    gateway_name = "gateway" if gateway_enable else None

    print(f"Loading calfex node data ...")
    print(f"  csv dir: {csv_dir}")
    print(f"  channel: {freq_ghz} GHz  bw={bw_mhz} MHz  tx={power_dbm} dBm  [{band}]")

    n_ok = 0
    for day in days:
        trace_fp = per_day_dir / day / "gps_all_nodes_trace.csv"
        node_data = _load_day_gps(trace_fp)
        if not node_data:
            print(f"  WARNING: no GPS fixes for {day}, skipping")
            continue

        day_dir = output_path / day
        day_dir.mkdir(parents=True, exist_ok=True)
        _create_json(day_dir, node_data, gateway_name)
        _create_ini(day_dir,
                    scenario_name="calfex",
                    freq_ghz=freq_ghz,
                    bw_mhz=bw_mhz,
                    power_dbm=power_dbm,
                    demand_mbps=_DEMAND_MBPS,
                    gateway_id=gateway_name)
        print(f"  {day}: {len(node_data)} nodes")
        n_ok += 1

    if n_ok == 0:
        print("ERROR: no days produced a config", file=sys.stderr)
        return 1
    return 0


## @brief CLI entry point.
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Generates per-day config files for the simulation config loader")
    p.add_argument("--input", "-i",
                   default=Path("data/arpo_extracted/_plots/per_day"), type=Path,
                   help="Per-day dir containing <YYYY-MM-DD>/gps_all_nodes_trace.csv")
    p.add_argument("--output", "-o",
                   type=Path, default=Path("inputs/calfex"),
                   help="Output directory to store per-day config files")
    p.add_argument("--csv-dir", dest="csv", type=Path, required=True,
                   help="Per-node config dir containing <node>/silvus/config.csv")
    p.add_argument("--band", "-b",
                   type=str, choices=["mmwave", "sub-6"], required=True,
                   help="Radio frequency spectrum used by the node")
    p.add_argument("--mode", "-m",
                   type=str, choices=["node", "scenario"], required=True,
                   help="The format of the dataset")
    p.add_argument("--gateway", "-g", action="store_true",
                   help="Enables gateway node + gateway traffic topology")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--day", default=None,
                   help="Generate config for this single day (YYYY-MM-DD), "
                        "written to <output>/<day>/")
    g.add_argument("--all-days", action="store_true",
                   help="Generate config for every day found under --input, "
                        "each written to its own <output>/<day>/ subdirectory")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input path not found: {args.input}", file=sys.stderr)
        return 1

    # Resolve which day(s) to build. --day names a day; join it with --input.
    if args.day:
        if not _is_date_dir(args.day):
            print(f"ERROR: --day must be YYYY-MM-DD, got '{args.day}'", file=sys.stderr)
            return 1
        days = _discover_days(args.input / args.day)
    else:
        days = _discover_days(args.input)
    if not days:
        print(f"ERROR: no per-day trace files found under {args.input}", file=sys.stderr)
        return 1

    if not args.csv.is_dir():
        print(f"ERROR: --csv-dir not found or not a directory: {args.csv}", file=sys.stderr)
        return 1
    csv_dir = args.csv

    match args.mode, args.band:
        case "node", "sub-6":
            return load_calfex_data_per_day(
                csv_dir, args.input, args.output, args.band, days, args.gateway)
        case _:
            print("per-day generation only supports node mode + sub-6 band",
                  file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())