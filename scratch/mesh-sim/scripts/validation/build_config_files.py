'''build_config_files.py'''
## @file build_config_files.py
# @brief Creates config files (run.ini and nodes.json) to be used for the simulator.
#
# Reads per-node silvus and GPS CSVs from a calfex-style node directory and
# produces the two files the ns-3 mesh-sim config loader expects:
# ``nodes.json`` — one entry per IH node plus one gateway entry, and
# ``run.ini``    — channel, traffic, routing, and scenario parameters.


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


## @brief Return the first valid (lat, lon) fix from a node's GPS file.
#
# Reads ``gps/gps_position.csv`` and returns the first row whose latitude
# and longitude are both non-zero and non-NaN.
#
# @param node_dir  Path to one node subdirectory.
# @return          ``(lat, lon)`` tuple, or ``None`` if no valid fix is found.
def _gps_first_fix(node_dir: Path) -> tuple[float, float] | None:
    fp = node_dir / "gps" / "gps_position.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp, low_memory=False)
    if {"latitude", "longitude"}.issubset(df.columns):
        df = df.rename(columns={"latitude": "__lat__", "longitude": "__lon__"})
    elif {"lat", "lon"}.issubset(df.columns):
        df = df.rename(columns={"lat": "__lat__", "lon": "__lon__"})
    else:
        return None
    df["__lat__"] = pd.to_numeric(df["__lat__"], errors="coerce")
    df["__lon__"] = pd.to_numeric(df["__lon__"], errors="coerce")
    df = df.dropna(subset=["__lat__", "__lon__"])
    df = df[(df["__lat__"] != 0) | (df["__lon__"] != 0)]
    if df.empty:
        return None
    return float(df["__lat__"].iloc[0]), float(df["__lon__"].iloc[0])


## @brief Convert WGS-84 lat/lon to local ENU metres relative to an origin.
#
# Uses the flat-earth approximation — accurate to within ~1 m for areas
# smaller than 10 km across.
#
# @param lat   Target latitude in decimal degrees.
# @param lon   Target longitude in decimal degrees.
# @param lat0  Origin latitude in decimal degrees.
# @param lon0  Origin longitude in decimal degrees.
# @return      ``(east_m, north_m)`` tuple rounded to 2 decimal places.
def _gps_to_enu(lat: float, lon: float,
                lat0: float, lon0: float) -> tuple[float, float]:
    R     = 6_378_137.0
    east  = R * math.cos(math.radians(lat0)) * math.radians(lon - lon0)
    north = R * math.radians(lat - lat0)
    return round(east, 2), round(north, 2)


## @brief Load channel parameters and GPS fixes from a calfex node directory.
#
# Iterates every subdirectory under @p csv_dir that has both a ``gps/`` and
# a ``silvus/`` subdirectory. For each valid node reads the first row of
# ``silvus/config.csv`` to get ``freq``, ``bw``, and ``power_dBm``, and
# calls @ref _gps_first_fix to get the starting position.
#
# @param csv_dir  Parent directory containing per-node subdirs.
# @return         Dict with keys:
#                 - ``nodes`` — list of ``{"name": str, "lat": float, "lon": float}``
#                 - ``freq_mhz``   — median frequency across all nodes (MHz)
#                 - ``bw_mhz``     — median bandwidth across all nodes (MHz)
#                 - ``power_dbm``  — median TX power across all nodes (dBm)
#                 Returns ``None`` if no valid nodes are found.
def _load_csv(csv_dir: Path) -> dict | None:
    nodes  = []
    freqs, bws, powers = [], [], []

    for node_dir in sorted(csv_dir.iterdir()):
        if not node_dir.is_dir() or node_dir.name == "sdwan":
            continue
        if not (node_dir / "gps").is_dir() or not (node_dir / "silvus").is_dir():
            print(f"  WARNING: {node_dir.name} missing gps or silvus, skipping")
            continue

        # Channel params from silvus config.
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

        # GPS first fix.
        fix = _gps_first_fix(node_dir)
        if fix is None:
            print(f"  WARNING: {node_dir.name} has no GPS fix, skipping")
            continue

        nodes.append({"name": node_dir.name, "lat": fix[0], "lon": fix[1]})

    if not nodes:
        print(f"ERROR: no valid nodes found in {csv_dir}", file=sys.stderr)
        return None

    return {
        "nodes":     nodes,
        "freq_mhz":  float(pd.Series(freqs).median()) if freqs else 2400.0,
        "bw_mhz":    float(pd.Series(bws).median())   if bws   else 20.0,
        "power_dbm": float(pd.Series(powers).median()) if powers else 30.0,
    }


## @brief Load gateway name and mean traffic demand from the sdwan JSONL data.
#
# Searches for ``versa_tunnels`` and ``versa_vni_rx_bps`` anywhere under
# @p sdwan_dir, so the intermediate directory structure (e.g. ``dl/``) is
# not assumed.
#
# @param sdwan_dir  Path to the ``calfex_sdwan/`` directory.
# @return           ``(gateway_name, demand_mbps)`` tuple, or
#                   ``("gateway", 0.05)`` if the directory is not found.
def _load_sdwan(sdwan_dir: Path) -> tuple[str, float]:
    import json
    from collections import Counter

    if not sdwan_dir.is_dir():
        print("  WARNING: calfex_sdwan not found — using defaults for gateway and demand")
        return "gateway", 0.05

    # Find versa_tunnels and versa_vni_rx_bps wherever they sit under sdwan_dir.
    tunnels_dir = next(sdwan_dir.rglob("versa_tunnels"), None)
    vni_dir     = next(sdwan_dir.rglob("versa_vni_rx_bps"), None)

    # Find hub device from tunnel connections.
    gateway_name = "SpringLakeLab-sdwan"
    if tunnels_dir and tunnels_dir.is_dir():
        local_to_calfex: Counter = Counter()
        for f in sorted(tunnels_dir.rglob("*.jsonl"))[:30]:
            with open(f) as fh:
                for line in fh:
                    try:
                        rec    = json.loads(line)
                        tags   = rec["tags"]
                        local  = tags.get("deviceName", "")
                        remote = tags.get("remoteSiteName", "")
                        if "calfex" in remote.lower() and "calfex" not in local.lower():
                            local_to_calfex[local] += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        if local_to_calfex:
            gateway_name = local_to_calfex.most_common(1)[0][0]

    # Compute mean demand from gateway vni_rx_bps.
    demand_mbps = 0.05
    if vni_dir and vni_dir.is_dir():
        vals = []
        for f in sorted(vni_dir.rglob("*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        if rec["tags"]["deviceName"] == gateway_name:
                            bps = rec["fields"]["gauge"]
                            if bps > 0:
                                vals.append(bps)
                    except (json.JSONDecodeError, KeyError):
                        continue
        if vals:
            demand_mbps = round(sum(vals) / len(vals) / 1e6, 3)

    print(f"  gateway: {gateway_name}")
    print(f"  demand:  {demand_mbps} Mbps")
    return gateway_name, demand_mbps


## @brief Write ``run.ini`` to @p output_path using the provided parameters.
#
# @param output_path   Directory to write ``run.ini`` into.
# @param scenario_name Scenario name written to ``[scenario] name``.
# @param freq_ghz      Carrier frequency in GHz.
# @param bw_mhz        Channel bandwidth in MHz.
# @param power_dbm     TX power in dBm.
# @param demand_mbps   Mean traffic demand in Mbps.
# @param gateway_id    Node ID of the gateway (must match an entry in nodes.json).
def _create_ini(output_path: Path, scenario_name: str,
                freq_ghz: float, bw_mhz: float, power_dbm: float,
                demand_mbps: float, gateway_id: str) -> None:
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
        "frequency_ghz":    str(freq_ghz),
        "tx_power_dbm":     str(power_dbm),
        "scenario":         "UMa",
        "channel_model":    "3gpp",
        "blockage_enabled": "false",
        "bandwidth_mhz":    str(bw_mhz),
        "noise_figure_db":  str(_NOISE_FIGURE),
    }
    cfg["traffic"] = {
        "model":           "constant",
        "demand_mbps":     str(demand_mbps),
        "flow_topology":   "gateway",
        "gateway_node_id": gateway_id,
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


## @brief Write ``nodes.json`` to @p output_path.
#
# Places the gateway node at the ENU origin (0, 0, _GATEWAY_HEIGHT).
# Each IH node is written with ``"mobility": "waypoint"`` and empty
# waypoints so that ``build_waypoints.py`` can fill them in later.
# Positions are computed relative to the centroid of all node GPS fixes.
#
# @param output_path  Directory to write ``nodes.json`` into.
# @param node_data    List of ``{"name": str, "lat": float, "lon": float}`` dicts.
# @param gateway_name Display name written as the gateway ``"id"``.
def _create_json(output_path: Path, node_data: list[dict],
                 gateway_name: str) -> None:
    # ENU origin = centroid of all node first fixes.
    lat0 = sum(n["lat"] for n in node_data) / len(node_data)
    lon0 = sum(n["lon"] for n in node_data) / len(node_data)

    nodes = []

    # Gateway at origin, elevated and fixed.
    nodes.append({
        "id":       "gateway",
        "role":     "peer",
        "mobility": "fixed",
        "position": {"x": 0.0, "y": 0.0, "z": _GATEWAY_HEIGHT},
        "_sdwan_device": gateway_name,
    })

    # IH nodes — waypoint mobility; build_waypoints.py fills trajectories.
    for node in node_data:
        east, north = _gps_to_enu(node["lat"], node["lon"], lat0, lon0)
        nodes.append({
            "id":       node["name"],
            "role":     "peer",
            "mobility": "waypoint",
            "position": {"x": east, "y": north, "z": _NODE_HEIGHT_M},
            "waypoints": [],
        })

    json_path = output_path / "nodes.json"
    with open(json_path, "w") as f:
        json.dump(nodes, f, indent=2)
    print(f"  wrote {json_path}  ({len(nodes)} nodes)")


## @brief Load calfex field data and write ``nodes.json`` and ``run.ini``.
#
# Calls @ref _load_csv to read GPS and channel parameters from the node
# directories, @ref _load_sdwan to identify the gateway and traffic demand,
# then @ref _create_json and @ref _create_ini to write the output files.
#
# @param csv_dir     Path to the ``calfex_csv`` directory.
# @param output_path Directory to write output files into.
# @param band        Radio band string (``"sub-6"`` or ``"mmwave"``).
# @return            0 on success, 1 on error.
## @brief Discover available calendar days from a by-day GPS trace directory.
#
# Looks for files named ``<prefix>_<YYYY-MM-DD>.csv`` (the output of
# ``arpo_data.cli split-day``) and returns the sorted list of date strings.
#
# @param by_day_dir  Directory containing per-day split GPS trace CSVs.
# @return            Sorted list of ``"YYYY-MM-DD"`` strings, or an empty
#                    list if @p by_day_dir does not exist or has no matches.
def _discover_days(by_day_dir: Path) -> list[str]:
    if not by_day_dir.is_dir():
        return []
    days = []
    for f in by_day_dir.glob("*.csv"):
        stem = f.stem
        # Expect a trailing "_YYYY-MM-DD" suffix.
        if len(stem) >= 10 and stem[-10] == "_":
            candidate = stem[-10:]
        else:
            continue
        date_part = candidate[1:]
        parts = date_part.split("-")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            days.append(date_part)
    return sorted(set(days))


## @brief Load calfex field data and write per-day ``nodes.json``/``run.ini`` copies.
#
# Builds the base config once via @ref load_calfex_data, then for each day
# in @p days copies the result into ``output_path/<day>/`` so waypoints can
# be patched into each day's copy independently without disturbing the
# others.
#
# @param extracted_dir Path to the extracted calfex dataset root.
# @param output_path   Root output directory; one subdirectory per day is
#                      created inside it.
# @param band          Radio band string (``"sub-6"`` or ``"mmwave"``).
# @param days          List of ``"YYYY-MM-DD"`` strings to produce configs for.
# @return              0 on success, 1 on error.
def load_calfex_data_per_day(extracted_dir: Path, output_path: Path,
                             band: str, days: list[str]) -> int:
    if not days:
        print("ERROR: no days to process", file=sys.stderr)
        return 1

    # Build the base config once into a scratch location, then copy per day.
    base_dir = output_path / "_base"
    rc = load_calfex_data(extracted_dir, base_dir, band)
    if rc != 0:
        return rc

    import shutil
    for day in days:
        day_dir = output_path / day
        day_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_dir / "nodes.json", day_dir / "nodes.json")
        shutil.copy2(base_dir / "run.ini",    day_dir / "run.ini")
        print(f"  wrote {day_dir}/nodes.json and run.ini")

    shutil.rmtree(base_dir)
    return 0


def load_calfex_data(extracted_dir: Path, output_path: Path, band: str) -> int:
    print(f"Loading calfex node data from {extracted_dir} ...")
    csv_dir = extracted_dir / "calfex_csv"
    sdwan_dir = extracted_dir / "calfex_sdwan"
    
    data = _load_csv(csv_dir)
    if data is None:
        return 1

    gateway_name, demand_mbps = _load_sdwan(sdwan_dir)

    freq_ghz = round(data["freq_mhz"] / 1000, 4)
    bw_mhz   = data["bw_mhz"]
    power_dbm = data["power_dbm"]

    print(f"  nodes:   {len(data['nodes'])} valid IH nodes")
    print(f"  channel: {freq_ghz} GHz  bw={bw_mhz} MHz  tx={power_dbm} dBm  [{band}]")

    output_path.mkdir(parents=True, exist_ok=True)

    _create_json(output_path, data["nodes"], gateway_name)
    _create_ini(output_path,
                scenario_name="calfex",
                freq_ghz=freq_ghz,
                bw_mhz=bw_mhz,
                power_dbm=power_dbm,
                demand_mbps=demand_mbps,
                gateway_id="gateway")
    return 0

## @brief CLI entry point.
#
# @param argv  Argument list; defaults to ``sys.argv[1:]`` when ``None``.
# @return      0 if no errors occurred, 1 otherwise.
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Generates config files to be used by simulation config loader")
    p.add_argument("--input", "-i", required=True,
                   type=Path,
                   help="Input file path directory of node config data")
    p.add_argument("--output", "-o",
                   type=Path, default=Path("inputs/calfex"),
                   help="Output file path directory to store config files")
    p.add_argument("--band", "-b",
                   type=str, choices=["mmwave", "sub-6"], required=True,
                   help="Radio frequency spectrum used by the node")
    p.add_argument("--mode", "-m",
                   type=str, choices=["node", "scenario"], required=True,
                   help="The format of the dataset")
    p.add_argument("--by-day-dir", type=Path, default=None,
                help="Directory of per-day split GPS traces (output of "
                    "`arpo_data.cli split-day`); required with --day/--all-days. "
                    "Only used to discover which days exist.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--day", default=None,
                   help="Generate config for this single day (YYYY-MM-DD), "
                        "written to <output>/<day>/")
    g.add_argument("--all-days", action="store_true",
                   help="Generate config for every day found in --by-day-dir, "
                        "each written to its own <output>/<day>/ subdirectory")
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input path not found: {args.input}", file=sys.stderr)
        return 1

    if args.day or args.all_days:
        if args.by_day_dir is None:
            print("ERROR: --by-day-dir is required with --day or --all-days",
                  file=sys.stderr)
            return 1
        if args.day:
            days = [args.day]
        else:
            days = _discover_days(args.by_day_dir)
            if not days:
                print(f"ERROR: no per-day trace files found under {args.by_day_dir}",
                      file=sys.stderr)
                return 1
        match args.mode, args.band:
            case "node", "sub-6":
                return load_calfex_data_per_day(args.input, args.output, args.band, days)
            case _:
                print("per-day generation only supports node mode + sub-6 band",
                      file=sys.stderr)
                return 1

    match args.mode:
        case "node":
            match args.band:
                case "sub-6":
                    return load_calfex_data(args.input, args.output, args.band)
                case "mmwave":
                    # TODO: mmwave node dataset support
                    print("mmwave node mode not yet implemented", file=sys.stderr)
                    return 1
        case "scenario":
            # TODO: scenario-based dataset support
            print("scenario mode not yet implemented", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())