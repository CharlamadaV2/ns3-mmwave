'''scenario_fidelity.py'''
## @file scenario_fidelity.py
# @brief Compare each scenario's sim layout and motion against the field collect.
#
# For each scenario in a batch output directory this module checks two things:
# -# **Mobility fidelity** — does each node's mobile/static classification in
#    the sim match the field GPS trace?
# -# **Geometry fidelity** — are the initial pairwise inter-node distances in
#    the sim within tolerance of the field GPS positions?
#
# Two modes are supported:
# - ``scenario`` — spring_lake style; batch root contains named scenario subdirs.
# - ``node``     — calfex style; batch root IS the scenario dir; field GPS is
#                  supplied via ``--field-gps``.
#
# Results are printed as human-readable tables with ``✓``/``✗`` markers.

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .compare import sim_to_field_scenario

REPO_ROOT          = Path(__file__).resolve().parents[2]
FIELD_PER_DAY_ROOT = REPO_ROOT / "data" / "arpo_extracted" / "_plots" / "per_day"

## @brief Bounding-box max-dimension threshold (metres) for classifying a node as mobile.
_MOBILE_BBOX_M = 20.0


## @brief Motion summary for one node derived from a position time series.
@dataclass(frozen=True)
class NodeMotion:
    node:       str
    n_samples:  int
    bbox_w:     float
    bbox_h:     float
    path_m:     float
    start_xy:   tuple[float, float]
    duration_s: float

    @property
    def is_mobile(self) -> bool:
        return max(self.bbox_w, self.bbox_h) > _MOBILE_BBOX_M

    @property
    def label(self) -> str:
        return "mobile" if self.is_mobile else "static"


## @brief Compute a @ref NodeMotion summary from raw position arrays.
#
# @param node Node identifier string.
# @param t    1-D time array (seconds).
# @param x    1-D east position array (metres).
# @param y    1-D north position array (metres).
# @return Populated @ref NodeMotion.
def _motion_from_xy(node: str, t: np.ndarray, x: np.ndarray,
                    y: np.ndarray) -> NodeMotion:
    n = x.size
    if n == 0:
        return NodeMotion(node, 0, 0.0, 0.0, 0.0, (float("nan"), float("nan")), 0.0)
    bbox_w = float(x.max() - x.min())
    bbox_h = float(y.max() - y.min())
    if n >= 2:
        path     = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        duration = float(t[-1] - t[0]) if t.size == n else 0.0
    else:
        path = duration = 0.0
    return NodeMotion(
        node=node, n_samples=n,
        bbox_w=bbox_w, bbox_h=bbox_h, path_m=path,
        start_xy=(float(x[0]), float(y[0])),
        duration_s=duration,
    )


## @brief Load per-node motion from a field scenario's GPS track trace (scenario mode).
#
# Reads ``csvs/gps  track_trace.csv`` from @p field_scenario_dir.
#
# @param field_scenario_dir Per-day output directory for the field scenario.
# @return Dict mapping node name to @ref NodeMotion, or empty dict if absent.
def _load_field_motion(field_scenario_dir: Path,
                       window_s: float | None = None) -> dict[str, NodeMotion]:
    trace = field_scenario_dir / "csvs" / "gps_track_trace.csv"
    if not trace.is_file():
        return {}
    return _load_field_motion_from_file(trace, window_s)


## @brief Load per-node motion directly from a GPS trace CSV file (node mode).
#
# Accepts any CSV with columns ``node``, ``sec_since_origin``,
# ``east_m``, ``north_m`` — including ``gps_all_nodes_trace.csv``.
#
# @param trace_path Direct path to the GPS trace CSV.
# @return Dict mapping node name to @ref NodeMotion, or empty dict if absent.
def _load_field_motion_from_file(trace_path: Path,
                                 window_s: float | None = None) -> dict[str, NodeMotion]:
    if not trace_path.is_file():
        return {}
    df  = pd.read_csv(trace_path,
                      usecols=["node", "sec_since_origin", "east_m", "north_m"])
    if window_s is not None: 
        t  = pd.to_numeric(df["sec_since_origin"], errors="coerce")
        t0 = t.min()                           # relative to trace start
        if pd.notna(t0):
            df = df[t <= t0 + window_s]
    out: dict[str, NodeMotion] = {}
    for node, g in df.groupby("node"):
        g = g.sort_values("sec_since_origin")
        out[str(node)] = _motion_from_xy(
            str(node),
            g["sec_since_origin"].to_numpy(dtype=np.float64),
            g["east_m"].to_numpy(dtype=np.float64),
            g["north_m"].to_numpy(dtype=np.float64),
        )
    return out


## @brief Build the integer-node-ID → node-label map from a snapshotted ``nodes.json``.
#
# @param scenario_out_dir Scenario output directory containing ``inputs/nodes.json``.
# @return Dict ``{node_id: label}``, or empty dict if the file is missing.
def _node_id_to_label(scenario_out_dir: Path) -> dict[int, str]:
    nodes_json = scenario_out_dir / "inputs" / "nodes.json"
    if not nodes_json.is_file():
        return {}
    try:
        specs = json.loads(nodes_json.read_text())
    except json.JSONDecodeError:
        return {}
    return {i: str(s.get("id", f"node{i}")) for i, s in enumerate(specs)}


## @brief Read the mobility mode string for each node from a snapshotted ``nodes.json``.
#
# @param scenario_out_dir Scenario output directory.
# @return Dict ``{node_label: mobility_mode_string}``.
def _sim_mobility_modes(scenario_out_dir: Path) -> dict[str, str]:
    nodes_json = scenario_out_dir / "inputs" / "nodes.json"
    if not nodes_json.is_file():
        return {}
    try:
        specs = json.loads(nodes_json.read_text())
    except json.JSONDecodeError:
        return {}
    return {str(s.get("id", "?")): str(s.get("mobility", "?")) for s in specs}


## @brief Load per-node motion from a sim scenario's ``positions.csv``.
#
# Reads ``<seed_name>/positions.csv`` (has ``#`` comment header lines).
# Maps integer node IDs to labels via ``inputs/nodes.json``.
#
# @param scenario_out_dir Scenario output directory (contains ``seed-*/``).
# @param seed_name        Seed subdirectory to read (default: ``"seed-1"``).
# @return Tuple ``(motion_dict, label_map)``.
def _load_sim_motion(scenario_out_dir: Path,
                     seed_name: str = "seed-1",
                     window_s: float | None = None) -> tuple[dict[str, NodeMotion], dict[int, str]]:
    pos = scenario_out_dir / seed_name / "positions.csv"
    if not pos.is_file():
        return {}, {}
    df        = pd.read_csv(pos, comment="#")
    if window_s is not None and "time_s" in df.columns:
        t  = pd.to_numeric(df["time_s"], errors="coerce")
        t0 = t.min()
        if pd.notna(t0):
            df = df[t <= t0 + window_s]
    label_map = _node_id_to_label(scenario_out_dir)
    out: dict[str, NodeMotion] = {}
    for node_id, g in df.groupby("node_id"):
        g     = g.sort_values("time_s")
        label = label_map.get(int(node_id), f"node{int(node_id)}")
        out[label] = _motion_from_xy(
            label,
            g["time_s"].to_numpy(dtype=np.float64),
            g["x"].to_numpy(dtype=np.float64),
            g["y"].to_numpy(dtype=np.float64),
        )
    return out, label_map


## @brief Compute initial-position pairwise Euclidean distances between all nodes.
#
# @param motion Dict of node label → @ref NodeMotion.
# @return Dict mapping sorted ``(a, b)`` tuples to distance in metres.
def _pairwise_distances(motion: dict[str, NodeMotion]) -> dict[tuple[str, str], float]:
    nodes = sorted(motion.keys())
    out: dict[tuple[str, str], float] = {}
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            ax, ay = motion[a].start_xy
            bx, by = motion[b].start_xy
            if any(not np.isfinite(v) for v in (ax, ay, bx, by)):
                out[(a, b)] = float("nan")
            else:
                out[(a, b)] = float(np.hypot(bx - ax, by - ay))
    return out


def _fmt(v: float, decimals: int = 1) -> str:
    return f"{v:.{decimals}f}" if np.isfinite(v) else "—"


def _render_table(header: tuple, body: list[tuple]) -> str:
    widths = [max(len(str(h)), max((len(str(r[i])) for r in body), default=0))
              for i, h in enumerate(header)]
    sep    = "  ".join("-" * w for w in widths)
    lines  = ["  ".join(h.ljust(w) for h, w in zip(header, widths)), sep]
    for row in body:
        lines.append("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))
    return "\n".join(lines)


## @brief Render the per-node mobility comparison table.
#
# @param field     Field motion dict.
# @param sim       Sim motion dict.
# @param sim_modes Sim mobility configuration strings.
# @return Tuple ``(table_string, mismatch_messages)``.
def _render_per_node(field: dict[str, NodeMotion],
                     sim:   dict[str, NodeMotion],
                     sim_modes: dict[str, str]) -> tuple[str, list[str]]:
    nodes  = sorted(set(field) | set(sim))
    header = ("node", "field motion", "sim motion", "field bbox WxH (m)",
              "sim bbox WxH (m)", "field path (m)", "sim path (m)",
              "sim cfg", "match?")
    body: list[tuple] = []
    mismatches: list[str] = []
    for n in nodes:
        f       = field.get(n)
        s       = sim.get(n)
        f_label = f.label if f else "—"
        s_label = s.label if s else "—"
        cfg     = sim_modes.get(n, "—")
        match   = (f.is_mobile == s.is_mobile) if (f and s) else False
        marker  = "✓" if match else "✗ MISMATCH"
        if not match:
            mismatches.append(f"{n}: field={f_label}, sim={s_label} (sim cfg: {cfg})")
        body.append((
            n, f_label, s_label,
            f"{_fmt(f.bbox_w)} x {_fmt(f.bbox_h)}" if f else "—",
            f"{_fmt(s.bbox_w)} x {_fmt(s.bbox_h)}" if s else "—",
            _fmt(f.path_m, 0) if f else "—",
            _fmt(s.path_m, 0) if s else "—",
            cfg, marker,
        ))
    return _render_table(header, body), mismatches


## @brief Render the pairwise initial-distance comparison table.
#
# @param field  Field motion dict.
# @param sim    Sim motion dict.
# @param tol_m  Distance tolerance in metres.
# @return Tuple ``(table_string, geometry_warnings)``.
def _render_pairwise(field: dict[str, NodeMotion],
                     sim:   dict[str, NodeMotion],
                     tol_m: float) -> tuple[str, list[str]]:
    fd    = _pairwise_distances(field)
    sd    = _pairwise_distances(sim)
    pairs = sorted(set(fd) | set(sd))
    header = ("pair", "field t=0 (m)", "sim t=0 (m)", "|Δ| (m)", "match?")
    body: list[tuple] = []
    geom_warnings: list[str] = []
    for a, b in pairs:
        f     = fd.get((a, b), float("nan"))
        s     = sd.get((a, b), float("nan"))
        if np.isfinite(f) and np.isfinite(s):
            delta = abs(f - s)
            ok    = delta <= tol_m
        else:
            delta = float("nan")
            ok    = False
        marker = "✓" if ok else "✗"
        if not ok and np.isfinite(delta):
            geom_warnings.append(
                f"{a}-{b}: field={f:.1f} m, sim={s:.1f} m, Δ={delta:.1f} m")
        body.append((f"{a}-{b}", _fmt(f), _fmt(s), _fmt(delta), marker))
    return _render_table(header, body), geom_warnings


## @brief Run the fidelity check for one scenario (scenario mode).
#
# @param scenario_out_dir Scenario output directory.
# @param tol_m            Pairwise distance tolerance in metres.
# @return Tuple ``(had_mismatch, report_text)``.
def _process_scenario(scenario_out_dir: Path, tol_m: float,
                      window_s: float | None = None) -> tuple[bool, str]:
    sim_name   = scenario_out_dir.name
    field_name = sim_to_field_scenario(sim_name)
    if field_name is None:
        return False, f"[{sim_name}] could not map to field scenario; skipping"

    field_motion   = _load_field_motion(FIELD_PER_DAY_ROOT / field_name, window_s)
    sim_motion, _  = _load_sim_motion(scenario_out_dir, window_s=window_s)
    sim_modes      = _sim_mobility_modes(scenario_out_dir)
    return _build_report(sim_name, field_name, field_motion, sim_motion,
                         sim_modes, tol_m)


## @brief Run the fidelity check for node mode (calfex).
#
# Bypasses scenario name derivation — field GPS is supplied directly via
# @p field_gps_path. The scenario dir is @p batch_root itself.
#
# @param batch_root    Scenario output directory (calfex flat layout).
# @param field_gps_path Direct path to the GPS trace CSV.
# @param tol_m         Pairwise distance tolerance in metres.
# @return Tuple ``(had_mismatch, report_text)``.
def _process_scenario_node(batch_root: Path, field_gps_path: Path,
                           tol_m: float,
                           window_s: float | None = None) -> tuple[bool, str]:
    #Loading Variables
    field_motion  = _load_field_motion_from_file(field_gps_path, window_s)
    sim_motion, _ = _load_sim_motion(batch_root, window_s=window_s)
    sim_modes     = _sim_mobility_modes(batch_root)
    
    #Main Logic
    return _build_report(batch_root.name, field_gps_path.name,
                         field_motion, sim_motion, sim_modes, tol_m)


## @brief Assemble the fidelity report string from pre-loaded motion dicts.
#
# Shared by both @ref _process_scenario and @ref _process_scenario_node.
#
# @param sim_name     Scenario label for the report header.
# @param field_name   Field label for the report header.
# @param field_motion Field motion dict.
# @param sim_motion   Sim motion dict.
# @param sim_modes    Sim mobility config strings.
# @param tol_m        Pairwise distance tolerance in metres.
# @return Tuple ``(had_mismatch, report_text)``.
def _build_report(sim_name: str, field_name: str,
                  field_motion: dict[str, NodeMotion],
                  sim_motion:   dict[str, NodeMotion],
                  sim_modes:    dict[str, str],
                  tol_m: float) -> tuple[bool, str]:
    
    lines = [f"[{sim_name}]  (field: {field_name})"]
    #Param Checker
    if not field_motion:
        lines.append("  no field GPS trace found")
    if not sim_motion:
        lines.append("  no sim positions.csv found under seed-1/")
    if not field_motion or not sim_motion:
        return True, "\n".join(lines)

    per_node_tbl, motion_mismatches = _render_per_node(field_motion, sim_motion,
                                                        sim_modes)
    pairwise_tbl, geom_warnings     = _render_pairwise(field_motion, sim_motion,
                                                        tol_m)
    lines += ["", "  per-node mobility:"]
    lines.extend("    " + ln for ln in per_node_tbl.splitlines())
    lines += ["", f"  pairwise t=0 distances  (tolerance ±{tol_m:g} m):"]
    lines.extend("    " + ln for ln in pairwise_tbl.splitlines())

    had_mismatch = bool(motion_mismatches or geom_warnings)
    if had_mismatch:
        lines.append("")
        lines.append("  flags:")
        for w in motion_mismatches:
            lines.append(f"    * motion mismatch — {w}")
        for w in geom_warnings:
            lines.append(f"    * geometry off  — {w}")
    return had_mismatch, "\n".join(lines)


## @brief Discover all scenario directories with a ``seed-1/positions.csv`` (scenario mode).
#
# @param batch_dir Batch output directory.
# @return Sorted list of qualifying subdirectory paths.
def _discover_scenarios(batch_dir: Path) -> list[Path]:
    return sorted(p for p in batch_dir.iterdir()
                  if p.is_dir() and (p / "seed-1" / "positions.csv").is_file())


## @brief CLI entry point for the scenario-fidelity checker.
#
# @param argv Argument list; defaults to ``sys.argv[1:]`` when ``None``.
# @return 0 if all scenarios passed, 1 if any had mismatches.
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Compare sim scenario layout/motion against field GPS.")
    p.add_argument("batch_root",
                   help="batch output dir; for node mode this is the scenario dir itself")
    p.add_argument("--mode", "-m", choices=["scenario", "node"], default="scenario",
                   help="scenario=spring_lake named subdirs, "
                        "node=calfex flat layout (default: scenario)")
    p.add_argument("--field-gps", default=None,
                   help="direct path to field GPS trace CSV (required for node mode); "
                        "e.g. data/arpo_extracted/_plots/per_day/gps_all_nodes_trace.csv")
    p.add_argument("--only", default=None,
                   help="restrict to one scenario name (scenario mode only)")
    p.add_argument("--tol-m", type=float, default=5.0,
                   help="pairwise initial-distance tolerance in metres (default: 5)")
    p.add_argument("--window", type=float, default=None,
                   help="use only the first N seconds of sim AND field GPS "
                        "(default: use the whole trace)")
    args = p.parse_args(argv)

    window_s = args.window

    batch_dir = Path(args.batch_root).resolve()
    if not batch_dir.is_dir():
        print(f"Error: batch dir not found: {batch_dir}", file=sys.stderr)
        return 1

    # --- node mode ---
    if args.mode == "node":
        if not args.field_gps: #< Check for field_gps arg
            print("Error: --field-gps is required for node mode", file=sys.stderr)
            return 1
        field_gps = Path(args.field_gps).resolve()
        if not field_gps.is_file(): #< Check for field_gps existence
            print(f"Error: field GPS file not found: {field_gps}", file=sys.stderr)
            return 1
        had_mismatch, report = _process_scenario_node(batch_dir, field_gps, args.tol_m,
                                                      window_s) #< Main Logic
        print(report)
        return 1 if had_mismatch else 0

    # --- scenario mode ---
    elif args.mode == "scenario":
        scenarios = _discover_scenarios(batch_dir)
        if args.only:
            scenarios = [s for s in scenarios if s.name == args.only]
        if not scenarios:
            print(f"Error: no scenarios with seed-1/positions.csv under {batch_dir}",
                file=sys.stderr)
            return 1

        print(f"== scenario fidelity  (batch = {batch_dir.name}) ==\n")
        flagged: list[str] = []
        for scen in scenarios:
            had_mismatch, report = _process_scenario(scen, args.tol_m, window_s)
            print(report)
            print()
            if had_mismatch:
                flagged.append(scen.name)

        if flagged:
            print(f"-- scenarios with flags ({len(flagged)}/{len(scenarios)}) --")
            for name in flagged:
                print(f"  {name}")
        else:
            print("all scenarios within tolerance")
        return 0

if __name__ == "__main__":
    sys.exit(main())