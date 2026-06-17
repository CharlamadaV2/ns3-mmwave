'''sim_to_traces.py'''
## @file sim_to_traces.py
# @brief Convert sim seed CSVs into arpo_data-style per-link trace CSVs.
#
# Two output formats are supported depending on the dataset mode:
#
# **scenario mode** (spring_lake) — one trace CSV per (src, peer) pair:
# ``<out>/csvs/<src>/bh2_<metric>__<src>_to_<peer>_trace.csv``
#
# **node mode** (calfex) — one trace CSV per source node with all neighbors
# combined, matching the IH field trace format:
# ``<out>/csvs/<src>/IH_<metric>__<src>_to_neighbors_trace.csv``
#
# Column layout in node mode:
# | Column          | Value                                         |
# |-----------------|-----------------------------------------------|
# | ``sec``         | seconds since session start (warmup removed)  |
# | ``source``      | source IH node name (e.g. ``IH01``)           |
# | ``peer``        | always ``"neighbors"``                        |
# | ``tag_local_mac``| source IH node name                          |
# | ``tag_sta_mac`` | actual neighbor IH node name (e.g. ``IH04``)  |
# | ``tag_interface``| empty                                        |
# | metric column   | ``snr_db``, ``rcpi_dbm``, or ``mcs_tx``       |

from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path

import pandas as pd

## @brief Metric specs for scenario mode — one trace per (src, peer) pair.
_METRIC_SPECS = (
    ("links",    "sinr_db",      "snr",  "snr_db"),
    ("mcs",      "mcs_index",    "mcs",  "mcs_tx"),
    ("rx-power", "rx_power_dbm", "rcpi", "rcpi_dbm"),
)

## @brief Metric specs for node mode — column names match the IH field trace format.
_METRIC_SPECS_NODE = (
    ("links",    "sinr_db",      "snr",  "snr_db"),
    ("mcs",      "mcs_index",    "mcs",  "mcs_tx"),
    ("rx-power", "rx_power_dbm", "rcpi", "rcpi_dbm"),
)


## @brief Build the integer-node-index → node-label map from ``nodes.json``.
#
# The sim assigns node IDs by position in the ``nodes.json`` array, so
# index 0 maps to the first node spec's ``id`` field.
#
# @param seed_dir One seed output directory; ``nodes.json`` is read from
#                 its parent's ``inputs/`` subdirectory.
# @return Dict ``{node_index: node_label_string}``.
# @throws FileNotFoundError if ``inputs/nodes.json`` is absent.
def _load_node_id_map(seed_dir: Path) -> dict[int, str]:
    nodes_json = seed_dir.parent / "inputs" / "nodes.json"
    if not nodes_json.is_file():
        raise FileNotFoundError(f"nodes.json not found: {nodes_json}")
    with open(nodes_json) as f:
        nodes = json.load(f)
    return {i: str(n["id"]) for i, n in enumerate(nodes)}


## @brief Read the ``warmup_s`` setting from a snapshotted ``run.ini``.
#
# @param seed_dir One seed output directory.
# @return Warmup duration in seconds, or 0.0 if not found.
def _read_warmup_s(seed_dir: Path) -> float:
    run_ini = seed_dir.parent / "inputs" / "run.ini"
    if not run_ini.is_file():
        return 0.0
    cfg = configparser.ConfigParser()
    cfg.read(run_ini)
    return cfg.getfloat("scenario", "warmup_s", fallback=0.0)


## @brief Load and normalise one sim metric CSV into a per-link DataFrame.
#
# Maps integer node IDs to labels, drops warmup rows and NaN values, and
# returns a DataFrame with canonical columns. Returns empty if the file is
# missing or unusable.
#
# @param csv_path  Path to the sim metric CSV.
# @param value_col Column holding the numeric metric value.
# @param trace_col Column name to use in the output trace DataFrame.
# @param node_map  Dict mapping integer node IDs to node labels.
# @param warmup_s  Seconds to discard from the start.
# @return Normalised DataFrame or empty DataFrame.
def _convert_one(csv_path: Path, value_col: str, trace_col: str,
                 node_map: dict[int, str], warmup_s: float) -> pd.DataFrame:
    if not csv_path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    needed = {"time_s", "node_a", "node_b", value_col}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    if warmup_s > 0:
        df = df[df["time_s"] >= warmup_s]
    df = df.dropna(subset=[value_col])
    if df.empty:
        return pd.DataFrame()

    src  = df["node_a"].map(node_map)
    peer = df["node_b"].map(node_map)
    keep = src.notna() & peer.notna()
    df   = df[keep]
    if df.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        "sec":           df["time_s"].to_numpy() - warmup_s,
        "source":        src[keep].to_numpy(),
        "peer":          peer[keep].to_numpy(),
        "tag_local_mac": src[keep].to_numpy(),
        "tag_sta_mac":   peer[keep].to_numpy(),
        "tag_interface": "",
        trace_col:       df[value_col].to_numpy(),
    })


## @brief Convert all metric CSVs for one seed (scenario mode).
#
# Writes one file per (source, peer) pair:
# ``<out_root>/csvs/<src>/bh2_<metric>__<src>_to_<peer>_trace.csv``
#
# @param seed_dir  Seed output directory.
# @param out_root  Destination root.
# @return Number of files written.
def convert_seed(seed_dir: Path, out_root: Path) -> int:
    node_map = _load_node_id_map(seed_dir)
    warmup_s = _read_warmup_s(seed_dir)
    n_written = 0
    for stem, value_col, metric_short, trace_col in _METRIC_SPECS:
        df = _convert_one(seed_dir / f"{stem}.csv",
                          value_col, trace_col, node_map, warmup_s)
        if df.empty:
            continue
        for (src, peer), pair_df in df.groupby(["source", "peer"]):
            out_dir  = out_root / "csvs" / str(src)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"bh2_{metric_short}__{src}_to_{peer}_trace.csv"
            pair_df.to_csv(out_path, index=False)
            n_written += 1
    return n_written


## @brief Convert all metric CSVs for one seed (node/calfex mode).
#
# Writes one file per source node per metric with all neighbors combined,
# matching the IH field trace format:
# ``<out_root>/csvs/<src>/IH_<metric>__<src>_to_neighbors_trace.csv``
#
# The ``peer`` column is always ``"neighbors"``; the actual neighbor ID
# is preserved in ``tag_sta_mac``.
#
# @param seed_dir  Seed output directory.
# @param out_root  Destination root.
# @return Number of files written.
def convert_seed_node(seed_dir: Path, out_root: Path) -> int:
    node_map = _load_node_id_map(seed_dir)
    warmup_s = _read_warmup_s(seed_dir)
    n_written = 0
    for stem, value_col, metric_short, trace_col in _METRIC_SPECS_NODE:
        df = _convert_one(seed_dir / f"{stem}.csv",
                          value_col, trace_col, node_map, warmup_s)
        if df.empty:
            continue

        # Group by source node only — all neighbors go into one file.
        for src, src_df in df.groupby("source"):
            out_df = src_df.copy()
            out_df["peer"] = "neighbors"  # matches IH field trace format
            # tag_sta_mac already holds the actual neighbor label

            out_dir  = out_root / "csvs" / str(src)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"IH_{metric_short}__{src}_to_neighbors_trace.csv"
            out_df.to_csv(out_path, index=False)
            n_written += 1
    return n_written


## @brief Convert all seed directories for one scenario.
#
# @param scenario_dir Top-level scenario output directory.
# @param mode         ``"scenario"`` or ``"node"``.
# @return Total trace CSV files written across all seeds.
def convert_scenario(scenario_dir: Path, mode: str = "scenario") -> int:
    seed_dirs = sorted(p for p in scenario_dir.iterdir()
                       if p.is_dir() and p.name.startswith("seed-"))
    if not seed_dirs:
        return 0
    sim_traces_root = scenario_dir / "sim_traces"
    total = 0
    for seed_dir in seed_dirs:
        out_root = sim_traces_root / seed_dir.name
        n = convert_seed_node(seed_dir, out_root) if mode == "node" \
            else convert_seed(seed_dir, out_root)
        total += n
        print(f"  {seed_dir.name}: {n} trace csvs")
    return total


## @brief Convert all scenarios in a batch output directory.
#
# - ``scenario`` mode — scans for named scenario subdirs with ``inputs/``.
# - ``node``     mode — treats ``batch_root`` as the scenario dir directly.
#
# @param batch_root Top-level batch output directory.
# @param mode       ``"scenario"`` or ``"node"``.
def convert_batch(batch_root: Path, mode: str) -> None:
    if mode == "node":
        if not (batch_root / "inputs").is_dir():
            print(f"Error: no inputs/ directory under {batch_root}", file=sys.stderr)
            return
        print(f"[{batch_root.name}]")
        n = convert_scenario(batch_root, mode="node")
        print(f"  total: {n} csvs")
        return

    scenarios = sorted(p for p in batch_root.iterdir()
                       if p.is_dir() and (p / "inputs").is_dir())
    if not scenarios:
        print(f"No scenario output dirs under {batch_root}", file=sys.stderr)
        return
    for scen in scenarios:
        print(f"[{scen.name}]")
        n = convert_scenario(scen, mode="scenario")
        print(f"  total: {n} csvs")


## @brief CLI entry point for the sim-to-traces converter.
#
# @param argv Argument list; defaults to ``sys.argv[1:]`` when ``None``.
# @return Always 0.
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert sim seed CSVs to arpo_data-style trace CSVs.")
    p.add_argument("batch_root",
                   help="batch output root; for node mode this is the scenario dir itself")
    p.add_argument("--mode", "-m",
                   choices=["node", "scenario"], required=True,
                   help="node=calfex (IH_<metric>__<src>_to_neighbors), "
                        "scenario=spring_lake (bh2_<metric>__<src>_to_<peer>)")
    args = p.parse_args(argv)
    convert_batch(Path(args.batch_root).resolve(), args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())