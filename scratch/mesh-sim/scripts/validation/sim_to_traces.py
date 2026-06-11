'''sim_to_traces.py'''
## @file sim_to_traces.py
# @brief Convert sim seed CSVs into arpo_data-style per-link trace CSVs.
#
# The ns-3 sim writes one CSV per metric (e.g. ``links.csv``, ``mcs.csv``)
# containing rows for every link sample across all nodes. This module
# reshapes those wide outputs into the narrow per-link trace format expected
# by the field-data pipeline:
#
#
# Each trace CSV has the same column layout as the field trace CSVs produced
# by ``arpo_data.cli plot``, enabling @ref compare to treat sim and field
# data identically.
#
# **Warmup trimming**
# Samples before ``warmup_s`` (read from ``run.ini``) are dropped so that
# the transient period before the network reaches steady state is excluded
# from validation metrics.

from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path

import pandas as pd

## @brief Mapping from sim CSV stem/column names to trace file names/column names.
#
# Each entry is a tuple ``(sim_csv_stem, sim_value_column, metric_short, trace_column)``:
# - ``sim_csv_stem``     — basename of the sim output CSV (without ``.csv``).
# - ``sim_value_column`` — column in that CSV holding the metric value.
# - ``metric_short``     — short name used in the trace filename (e.g. ``"snr"``).
# - ``trace_column``     — column name in the output trace CSV.
_METRIC_SPECS = (
    ("links",    "sinr_db",      "snr",  "snr_db"),
    ("mcs",      "mcs_index",    "mcs",  "mcs_tx"),
    ("rx-power", "rx_power_dbm", "rcpi", "rcpi_dbm"),
)


## @brief Build the integer-node-ID → rab-label map from the snapshotted ``nodes.json``.
#
# The sim assigns node IDs by position in the ``nodes.json`` array, so index 0
# maps to the first node spec's ``id`` field.
#
# @param seed_dir One seed output directory; ``nodes.json`` is read from
#                 its parent's ``inputs/`` subdirectory.
# @return Dict ``{node_id: rab_label_string}``.
# @throws FileNotFoundError if ``inputs/nodes.json`` is absent.
def _load_node_id_map(seed_dir: Path) -> dict[int, str]:
    nodes_json = seed_dir.parent / "inputs" / "nodes.json"
    if not nodes_json.is_file():
        raise FileNotFoundError(f"nodes.json archive not found: {nodes_json}")
    with open(nodes_json) as f:
        nodes = json.load(f)
    return {i: str(n["id"]) for i, n in enumerate(nodes)}


## @brief Read the ``warmup_s`` setting from a snapshotted ``run.ini``.
#
# Returns 0.0 if the file is absent or the key is not in the ``[scenario]``
# section, so callers can unconditionally subtract it.
#
# @param seed_dir One seed output directory; ``run.ini`` is read from its
#                 parent's ``inputs/`` subdirectory.
# @return Warmup duration in seconds.
def _read_warmup_s(seed_dir: Path) -> float:
    run_ini = seed_dir.parent / "inputs" / "run.ini"
    if not run_ini.is_file():
        return 0.0
    cfg = configparser.ConfigParser()
    cfg.read(run_ini)
    return cfg.getfloat("scenario", "warmup_s", fallback=0.0)


## @brief Convert one sim metric CSV into a normalised per-link DataFrame.
#
# Steps performed:
# -# Guard against missing files or columns.
# -# Drop rows before ``warmup_s``.
# -# Map integer node IDs to rab labels; drop rows with unknown IDs.
# -# Reindex columns to match the trace CSV schema.
#
# The output DataFrame always has columns:
# ``sec``, ``source``, ``peer``, ``tag_local_mac``, ``tag_sta_mac``,
# ``tag_interface``, ``<trace_column>``.
# The MAC/interface columns are empty strings because sim output has no
# hardware-level identifiers; downstream tools handle their absence gracefully.
#
# @param csv_path     Path to the sim metric CSV.
# @param value_col    Column in the CSV holding the numeric metric value.
# @param trace_col    Column name to use in the output trace DataFrame.
# @param node_map     Dict mapping integer node IDs to rab labels.
# @param warmup_s     Seconds to discard from the start of the trace.
# @return Normalised DataFrame, or an empty DataFrame if the input is unusable.
def _convert_one(csv_path: Path, value_col: str, trace_col: str,
                 node_map: dict[int, str], warmup_s: float) -> pd.DataFrame:
    if not csv_path.is_file():
        return pd.DataFrame()
    df     = pd.read_csv(csv_path)
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
        "tag_local_mac": "",   # No hardware MACs in sim output.
        "tag_sta_mac":   "",
        "tag_interface": "",
        trace_col:       df[value_col].to_numpy(),
    })


## @brief Convert all metric CSVs for one seed directory into per-link trace CSVs.
#
# Iterates @ref _METRIC_SPECS, converts each metric, splits the result by
# ``(source, peer)`` pair, and writes one trace CSV per pair under
# ``<out_root>/csvs/<src>/bh2_<metric>__<src>_to_<peer>_trace.csv``.
#
# @param seed_dir  One seed output directory containing the sim metric CSVs.
# @param out_root  Destination root for the trace CSV tree.
# @return Total number of trace CSV files written.
def convert_seed(seed_dir: Path, out_root: Path) -> int:
    node_map  = _load_node_id_map(seed_dir)
    warmup_s  = _read_warmup_s(seed_dir)
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


## @brief Convert all seed directories for one scenario into trace CSVs.
#
# Discovers ``seed-*`` subdirectories, converts each via @ref convert_seed,
# and writes results under ``<scenario_dir>/sim_traces/<seed>/``.
#
# @param scenario_dir Top-level scenario output directory.
# @return Total number of trace CSV files written across all seeds.
def convert_scenario(scenario_dir: Path) -> int:
    seed_dirs = sorted(p for p in scenario_dir.iterdir()
                       if p.is_dir() and p.name.startswith("seed-"))
    if not seed_dirs:
        return 0
    sim_traces_root = scenario_dir / "sim_traces"
    total = 0
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        out_root  = sim_traces_root / seed_name
        n         = convert_seed(seed_dir, out_root)
        total    += n
        print(f"  {seed_name}: {n} trace csvs")
    return total


## @brief Convert all scenarios in a batch output directory.
#
# A directory qualifies as a scenario if it contains an ``inputs/``
# subdirectory (written by @ref run_batch).
#
# @param batch_root Top-level batch output directory.
def convert_batch(batch_root: Path) -> None:
    scenarios = sorted(p for p in batch_root.iterdir()
                       if p.is_dir() and (p / "inputs").is_dir())
    if not scenarios:
        print(f"No scenario output dirs under {batch_root}", file=sys.stderr)
        return
    for scen in scenarios:
        print(f"[{scen.name}]")
        n = convert_scenario(scen)
        print(f"  total: {n} csvs")


## @brief CLI entry point for the sim-to-traces converter.
#
# Converts all scenarios under the given batch root and prints a per-seed
# and per-scenario count of CSV files written.
#
# @param argv Argument list; defaults to ``sys.argv[1:]`` when ``None``.
# @return Always 0 (errors are printed to stderr but do not abort).
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert sim seed CSVs to arpo_data-style trace CSVs.")
    p.add_argument("batch_root", help="batch output root from run_batch.py")
    args = p.parse_args(argv)
    convert_batch(Path(args.batch_root).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())