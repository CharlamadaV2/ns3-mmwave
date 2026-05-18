## @package docstring
# Compare each scenario's sim layout/motion against the field collect.
#
# More details.

#TODO: Finish Documentation for this page

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .compare import sim_to_field_scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
FIELD_PER_DAY_ROOT = REPO_ROOT / "data" / "arpo_extracted" / "_plots" / "per_day"

_MOBILE_BBOX_M = 20.0   # bbox max-dim threshold for "mobile"; path length alone is GPS-noise-prone

## Documentation for a class.
#
#  More details.
@dataclass(frozen=True)
class NodeMotion:
    node: str
    n_samples: int
    bbox_w: float          # east extent (m)
    bbox_h: float          # north extent (m)
    path_m: float          # cumulative path length (m)
    start_xy: tuple[float, float]
    duration_s: float

    @property
    def is_mobile(self) -> bool:
        return max(self.bbox_w, self.bbox_h) > _MOBILE_BBOX_M

    @property
    def label(self) -> str:
        return "mobile" if self.is_mobile else "static"

## @brief
def _motion_from_xy(node: str, t: np.ndarray, x: np.ndarray,
                    y: np.ndarray) -> NodeMotion:
    n = x.size
    if n == 0:
        return NodeMotion(node, 0, 0.0, 0.0, 0.0, (float("nan"), float("nan")), 0.0)
    bbox_w = float(x.max() - x.min())
    bbox_h = float(y.max() - y.min())
    if n >= 2:
        dx = np.diff(x)
        dy = np.diff(y)
        path = float(np.sum(np.hypot(dx, dy)))
        duration = float(t[-1] - t[0]) if t.size == n else 0.0
    else:
        path = 0.0
        duration = 0.0
    return NodeMotion(
        node=node, n_samples=n,
        bbox_w=bbox_w, bbox_h=bbox_h, path_m=path,
        start_xy=(float(x[0]), float(y[0])),
        duration_s=duration,
    )

## @brief
def _load_field_motion(field_scenario_dir: Path) -> dict[str, NodeMotion]:
    trace = field_scenario_dir / "csvs" / "gps_track_trace.csv"
    if not trace.is_file():
        return {}
    df = pd.read_csv(trace, usecols=["node", "sec_since_origin", "east_m", "north_m"])
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

## @brief
def _load_sim_motion(scenario_out_dir: Path,
                     seed_name: str = "seed-1") -> tuple[dict[str, NodeMotion], dict[int, str]]:
    ##Returns (per-node motion, node_id -> label map).##
    pos = scenario_out_dir / seed_name / "positions.csv"
    if not pos.is_file():
        return {}, {}
    df = pd.read_csv(pos, comment="#")

    label_map = _node_id_to_label(scenario_out_dir)
    out: dict[str, NodeMotion] = {}
    for node_id, g in df.groupby("node_id"):
        g = g.sort_values("time_s")
        label = label_map.get(int(node_id), f"node{int(node_id)}")
        out[label] = _motion_from_xy(
            label,
            g["time_s"].to_numpy(dtype=np.float64),
            g["x"].to_numpy(dtype=np.float64),
            g["y"].to_numpy(dtype=np.float64),
        )
    return out, label_map

## @brief
def _node_id_to_label(scenario_out_dir: Path) -> dict[int, str]:
    ##Pull rab labels from the snapshotted nodes.json (order = node_id).##
    nodes_json = scenario_out_dir / "inputs" / "nodes.json"
    if not nodes_json.is_file():
        return {}
    try:
        specs = json.loads(nodes_json.read_text())
    except json.JSONDecodeError:
        return {}
    return {i: str(s.get("id", f"node{i}")) for i, s in enumerate(specs)}

## @brief
def _sim_mobility_modes(scenario_out_dir: Path) -> dict[str, str]:
    ##rab -> mobility string from snapshotted nodes.json.##
    nodes_json = scenario_out_dir / "inputs" / "nodes.json"
    if not nodes_json.is_file():
        return {}
    try:
        specs = json.loads(nodes_json.read_text())
    except json.JSONDecodeError:
        return {}
    return {str(s.get("id", "?")): str(s.get("mobility", "?")) for s in specs}

## @brief
def _pairwise_distances(motion: dict[str, NodeMotion]) -> dict[tuple[str, str], float]:
    ##Initial-position pairwise Euclidean distance, sorted pair keys.##
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

## @brief
def _fmt(v: float, digits: int = 1) -> str:
    return "—" if not np.isfinite(v) else f"{v:.{digits}f}"

## @brief
def _render_table(header: tuple[str, ...], body: list[tuple[str, ...]]) -> str:
    if not body:
        return "  ".join(header) + "\n(no rows)"
    widths = [max(len(h), *(len(row[i]) for row in body)) for i, h in enumerate(header)]
    sep = "  ".join("-" * w for w in widths)
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths)), sep]
    for row in body:
        lines.append("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
    return "\n".join(lines)

## @brief
def _render_per_node(field: dict[str, NodeMotion],
                     sim: dict[str, NodeMotion],
                     sim_modes: dict[str, str]) -> tuple[str, list[str]]:
    nodes = sorted(set(field) | set(sim))
    header = ("rab", "field motion", "sim motion", "field bbox WxH (m)",
              "sim bbox WxH (m)", "field path (m)", "sim path (m)",
              "sim cfg", "match?")
    body = []
    mismatches: list[str] = []
    for n in nodes:
        f = field.get(n)
        s = sim.get(n)
        f_label = f.label if f else "—"
        s_label = s.label if s else "—"
        cfg = sim_modes.get(n, "—")
        if f and s:
            match = (f.is_mobile == s.is_mobile)
        else:
            match = False
        marker = "✓" if match else "✗ MISMATCH"
        if not match:
            mismatches.append(f"{n}: field={f_label}, sim={s_label} (sim cfg: {cfg})")
        body.append((
            n,
            f_label,
            s_label,
            f"{_fmt(f.bbox_w)} x {_fmt(f.bbox_h)}" if f else "—",
            f"{_fmt(s.bbox_w)} x {_fmt(s.bbox_h)}" if s else "—",
            _fmt(f.path_m, 0) if f else "—",
            _fmt(s.path_m, 0) if s else "—",
            cfg,
            marker,
        ))
    return _render_table(header, body), mismatches

## @brief
def _render_pairwise(field: dict[str, NodeMotion],
                     sim: dict[str, NodeMotion],
                     tol_m: float) -> tuple[str, list[str]]:
    fd = _pairwise_distances(field)
    sd = _pairwise_distances(sim)
    pairs = sorted(set(fd) | set(sd))
    header = ("pair", "field t=0 (m)", "sim t=0 (m)", "|Δ| (m)", "match?")
    body = []
    geom_warnings: list[str] = []
    for a, b in pairs:
        f = fd.get((a, b), float("nan"))
        s = sd.get((a, b), float("nan"))
        if np.isfinite(f) and np.isfinite(s):
            delta = abs(f - s)
            ok = delta <= tol_m
        else:
            delta = float("nan")
            ok = False
        marker = "✓" if ok else "✗"
        if not ok and np.isfinite(delta):
            geom_warnings.append(f"{a}-{b}: field={f:.1f} m, sim={s:.1f} m, Δ={delta:.1f} m")
        body.append((
            f"{a}-{b}",
            _fmt(f),
            _fmt(s),
            _fmt(delta),
            marker,
        ))
    return _render_table(header, body), geom_warnings

## @brief
def _process_scenario(scenario_out_dir: Path, tol_m: float) -> tuple[bool, str]:
    ##Returns (had_mismatch, report_text).##
    sim_name = scenario_out_dir.name
    field_name = sim_to_field_scenario(sim_name)
    if field_name is None:
        return False, f"[{sim_name}] could not map to field scenario; skipping"

    field_scen_dir = FIELD_PER_DAY_ROOT / field_name
    field_motion = _load_field_motion(field_scen_dir)
    sim_motion, _ = _load_sim_motion(scenario_out_dir)
    sim_modes = _sim_mobility_modes(scenario_out_dir)

    lines = [f"[{sim_name}]  (field: {field_name})"]
    if not field_motion:
        lines.append("  no field GPS trace found at "
                     f"{field_scen_dir.relative_to(REPO_ROOT)}/csvs/gps_track_trace.csv")
    if not sim_motion:
        lines.append("  no sim positions.csv found under seed-1/")
    if not field_motion or not sim_motion:
        return True, "\n".join(lines)

    per_node_tbl, motion_mismatches = _render_per_node(field_motion, sim_motion, sim_modes)
    pairwise_tbl, geom_warnings = _render_pairwise(field_motion, sim_motion, tol_m)

    lines.append("")
    lines.append("  per-node mobility:")
    lines.extend("    " + ln for ln in per_node_tbl.splitlines())
    lines.append("")
    lines.append(f"  pairwise t=0 distances  (tolerance ±{tol_m:g} m):")
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

## @brief
def _discover_scenarios(batch_dir: Path) -> list[Path]:
    return sorted(p for p in batch_dir.iterdir()
                  if p.is_dir() and (p / "seed-1" / "positions.csv").is_file())

## Documentation for a function.
#
#  More details.
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Compare each sim scenario's layout/motion against the field collect.")
    p.add_argument("batch_root", help="batch output dir (parent of arpo-* scenario dirs)")
    p.add_argument("--only", default=None,
                   help="restrict to one scenario name (e.g. arpo-2-2-los-obstruction-04172026)")
    p.add_argument("--tol-m", type=float, default=5.0,
                   help="pairwise initial-distance tolerance in metres (default: 5)")
    args = p.parse_args(argv)

    batch_dir = Path(args.batch_root).resolve()
    if not batch_dir.is_dir():
        print(f"batch dir not found: {batch_dir}", file=sys.stderr)
        return 1

    scenarios = _discover_scenarios(batch_dir)
    if args.only:
        scenarios = [s for s in scenarios if s.name == args.only]
    if not scenarios:
        print(f"no scenarios with seed-1/positions.csv under {batch_dir}", file=sys.stderr)
        return 1

    print(f"== scenario fidelity  (batch = {batch_dir.relative_to(REPO_ROOT)}) ==\n")
    flagged: list[str] = []
    for scen in scenarios:
        had_mismatch, report = _process_scenario(scen, args.tol_m)
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
