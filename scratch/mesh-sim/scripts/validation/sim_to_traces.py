"""Convert sim seed CSVs into arpo_data-style trace CSVs."""

from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path

import pandas as pd

_METRIC_SPECS = (
    ("links",    "sinr_db",      "snr",  "snr_db"),
    ("mcs",      "mcs_index",    "mcs",  "mcs_tx"),
    ("rx-power", "rx_power_dbm", "rcpi", "rcpi_dbm"),
)


def _load_node_id_map(seed_dir: Path) -> dict[int, str]:
    nodes_json = seed_dir.parent / "inputs" / "nodes.json"
    if not nodes_json.is_file():
        raise FileNotFoundError(f"nodes.json archive not found: {nodes_json}")
    with open(nodes_json) as f:
        nodes = json.load(f)
    return {i: str(n["id"]) for i, n in enumerate(nodes)}


def _read_warmup_s(seed_dir: Path) -> float:
    run_ini = seed_dir.parent / "inputs" / "run.ini"
    if not run_ini.is_file():
        return 0.0
    cfg = configparser.ConfigParser()
    cfg.read(run_ini)
    return cfg.getfloat("scenario", "warmup_s", fallback=0.0)


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
    src = df["node_a"].map(node_map)
    peer = df["node_b"].map(node_map)
    keep = src.notna() & peer.notna()
    df = df[keep]
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "sec":            df["time_s"].to_numpy() - warmup_s,
        "source":         src[keep].to_numpy(),
        "peer":           peer[keep].to_numpy(),
        "tag_local_mac":  "",
        "tag_sta_mac":    "",
        "tag_interface":  "",
        trace_col:        df[value_col].to_numpy(),
    })


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
            out_dir = out_root / "csvs" / str(src)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"bh2_{metric_short}__{src}_to_{peer}_trace.csv"
            pair_df.to_csv(out_path, index=False)
            n_written += 1
    return n_written


def convert_scenario(scenario_dir: Path) -> int:
    seed_dirs = sorted(p for p in scenario_dir.iterdir()
                       if p.is_dir() and p.name.startswith("seed-"))
    if not seed_dirs:
        return 0
    sim_traces_root = scenario_dir / "sim_traces"
    total = 0
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        out_root = sim_traces_root / seed_name
        n = convert_seed(seed_dir, out_root)
        total += n
        print(f"  {seed_name}: {n} trace csvs")
    return total


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert sim seed CSVs to arpo_data-style trace CSVs.")
    p.add_argument("batch_root", help="batch output root from run_batch.py")
    args = p.parse_args(argv)
    convert_batch(Path(args.batch_root).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
