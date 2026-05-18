"""
Generate waypoint mobility for a sim node from its field GPS trace.

Reads the per-day `gps_track_trace.csv` (centroid-ENU metres) emitted by
`arpo_data.cli plot`, aligns the field frame to the sim frame using a
stationary anchor node (default: rab1), downsamples the moving node's
track to N waypoints, and patches them into the scenario's `nodes.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .compare import sim_to_field_scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_ROOT = REPO_ROOT / "inputs" / "custom" / "sherpa" / "spring_lake"
FIELD_PER_DAY_ROOT = REPO_ROOT / "data" / "arpo_extracted" / "_plots" / "per_day"

_MOBILE_BBOX_M = 20.0  # field bbox max-dim threshold to count as "mobile"


def _load_field_trace(scenario_field_dir: Path) -> pd.DataFrame:
    csv = scenario_field_dir / "csvs" / "gps_track_trace.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"no field GPS trace at {csv}")
    df = pd.read_csv(csv, usecols=["node", "sec_since_origin", "east_m", "north_m"])
    return df


def _anchor_offset(df: pd.DataFrame, anchor: str,
                   sim_anchor_xy: tuple[float, float]) -> tuple[float, float]:
    """Return (dx, dy) so that field_anchor mean -> sim_anchor_xy."""
    g = df[df["node"] == anchor]
    if g.empty:
        raise ValueError(f"anchor '{anchor}' not in field trace")
    f_mean_x = float(g["east_m"].mean())
    f_mean_y = float(g["north_m"].mean())
    return sim_anchor_xy[0] - f_mean_x, sim_anchor_xy[1] - f_mean_y


def _downsample_uniform_time(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                             n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Take n waypoints evenly spaced in time, always including first and last."""
    if t.size <= n:
        return t, x, y
    grid = np.linspace(t[0], t[-1], n)
    idx = np.searchsorted(t, grid)
    idx = np.clip(idx, 0, t.size - 1)
    idx = np.unique(idx)
    return t[idx], x[idx], y[idx]


def _scale_time(t: np.ndarray, mode: str, target_s: float | None) -> np.ndarray:
    """Map field timeline onto sim timeline."""
    t0 = t[0]
    rel = t - t0
    if mode == "raw":
        return rel
    if mode == "scale":
        if target_s is None or rel[-1] <= 0:
            return rel
        return rel * (target_s / rel[-1])
    if mode == "clip":
        if target_s is None:
            return rel
        return np.minimum(rel, target_s)
    raise ValueError(f"unknown time mode: {mode}")


def _load_nodes_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _save_nodes_json(path: Path, nodes: list[dict]) -> None:
    path.write_text(json.dumps(nodes, indent=2) + "\n")


def _patch_node(nodes: list[dict], target_id: str,
                waypoints: list[dict]) -> dict:
    for n in nodes:
        if n.get("id") == target_id:
            n["mobility"] = "waypoint"
            n["waypoints"] = waypoints
            # Keep position in sync with the first waypoint so logs/snapshots match.
            wp0 = waypoints[0]
            n.setdefault("position", {})
            n["position"]["x"] = wp0["x"]
            n["position"]["y"] = wp0["y"]
            n["position"].setdefault("z", 0.0)
            n["position"]["z"] = wp0["z"]
            return n
    raise KeyError(f"node id '{target_id}' not in nodes.json")


def _resolve_scenario_dir(name: str) -> Path:
    """Accept either the sim or field scenario name."""
    direct = SCENARIOS_ROOT / name
    if direct.is_dir():
        return direct
    for sim_dir in SCENARIOS_ROOT.iterdir():
        if sim_to_field_scenario(sim_dir.name) == name:
            return sim_dir
    raise FileNotFoundError(f"scenario dir not found for '{name}' under {SCENARIOS_ROOT}")


def _field_bbox_max_m(df: pd.DataFrame, node: str) -> float:
    g = df[df["node"] == node]
    if g.empty:
        return 0.0
    return float(max(g["east_m"].max() - g["east_m"].min(),
                     g["north_m"].max() - g["north_m"].min()))


def patch_scenario_waypoints(sim_dir: Path, *, node: str = "rab2", anchor: str = "rab1",
                             n_waypoints: int = 20, time_mode: str = "raw",
                             duration: float | None = None, field_z: float | None = None,
                             field_scenario: str | None = None,
                             dry_run: bool = False,
                             mobile_bbox_m: float = _MOBILE_BBOX_M) -> str:
    """Patch one scenario's nodes.json with field-derived waypoints.

    Returns a one-line status string: "patched", "skipped: ...", or "error: ..."
    """
    nodes_json = sim_dir / "nodes.json"
    if not nodes_json.is_file():
        return f"error: nodes.json not found at {nodes_json}"
    nodes = _load_nodes_json(nodes_json)

    field_name = field_scenario or sim_to_field_scenario(sim_dir.name)
    if field_name is None:
        return f"error: cannot derive field scenario from '{sim_dir.name}'"
    field_dir = FIELD_PER_DAY_ROOT / field_name
    try:
        df = _load_field_trace(field_dir)
    except FileNotFoundError as e:
        return f"error: {e}"

    bbox = _field_bbox_max_m(df, node)
    if bbox <= mobile_bbox_m:
        return f"skipped: field {node} bbox {bbox:.1f} m <= {mobile_bbox_m:g} m (static)"

    sim_anchor = next((n for n in nodes if n.get("id") == anchor), None)
    if sim_anchor is None:
        return f"error: anchor '{anchor}' not in nodes.json"
    sim_anchor_xy = (float(sim_anchor["position"]["x"]),
                     float(sim_anchor["position"]["y"]))
    dx, dy = _anchor_offset(df, anchor, sim_anchor_xy)

    g = df[df["node"] == node].sort_values("sec_since_origin")
    if g.empty:
        return f"skipped: no field rows for node '{node}'"
    t = g["sec_since_origin"].to_numpy(dtype=np.float64)
    x = g["east_m"].to_numpy(dtype=np.float64) + dx
    y = g["north_m"].to_numpy(dtype=np.float64) + dy

    t_sim = _scale_time(t, time_mode, duration)
    t_ds, x_ds, y_ds = _downsample_uniform_time(t_sim, x, y, n_waypoints)

    target_node = next((n for n in nodes if n.get("id") == node), None)
    z_default = float(target_node["position"].get("z", 0.0)) if target_node else 0.0
    z_val = field_z if field_z is not None else z_default

    waypoints = [{"t": float(ti), "x": float(xi), "y": float(yi), "z": z_val}
                 for ti, xi, yi in zip(t_ds, x_ds, y_ds)]

    path_m = float(np.sum(np.hypot(np.diff(x_ds), np.diff(y_ds))))
    summary = (f"patched: {len(waypoints)} waypoints, "
               f"bbox {x_ds.max() - x_ds.min():.0f}x{y_ds.max() - y_ds.min():.0f} m, "
               f"path {path_m:.0f} m, t {t_ds[0]:.0f}..{t_ds[-1]:.0f} s")
    if dry_run:
        return summary + " (dry-run)"

    _patch_node(nodes, node, waypoints)
    _save_nodes_json(nodes_json, nodes)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate waypoint mobility from field GPS for a sim node.")
    p.add_argument("scenario", nargs="?", default=None,
                   help="scenario name (sim or field form). Omit with --all to "
                        "patch every scenario under inputs/custom/sherpa/spring_lake/.")
    p.add_argument("--all", dest="all_scenarios", action="store_true",
                   help="iterate every scenario, skipping those where the field "
                        "node is static")
    p.add_argument("--node", default="rab2",
                   help="moving node id to author waypoints for (default: rab2)")
    p.add_argument("--anchor", default="rab1",
                   help="stationary node used for field-to-sim frame alignment "
                        "(default: rab1)")
    p.add_argument("--n-waypoints", type=int, default=20,
                   help="downsample target (default: 20)")
    p.add_argument("--time-mode", choices=("raw", "clip", "scale"), default="raw",
                   help="raw=field clock, clip=clip to --duration, "
                        "scale=stretch/compress to --duration (default: raw)")
    p.add_argument("--duration", type=float, default=None,
                   help="target duration in seconds for clip/scale modes")
    p.add_argument("--field-z", type=float, default=None,
                   help="z (m) for each waypoint; default: keep existing node z")
    p.add_argument("--field-scenario", default=None,
                   help="override the field scenario name (else derived from sim name)")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would be patched without writing nodes.json")
    args = p.parse_args(argv)

    if not args.all_scenarios and not args.scenario:
        p.error("provide a scenario name or pass --all")
    if args.all_scenarios and args.scenario:
        p.error("use either a scenario name OR --all, not both")

    if args.all_scenarios:
        sim_dirs = sorted(d for d in SCENARIOS_ROOT.iterdir()
                          if d.is_dir() and (d / "nodes.json").is_file())
    else:
        sim_dirs = [_resolve_scenario_dir(args.scenario)]

    n_patched = 0
    n_skipped = 0
    n_error = 0
    for sim_dir in sim_dirs:
        status = patch_scenario_waypoints(
            sim_dir,
            node=args.node, anchor=args.anchor, n_waypoints=args.n_waypoints,
            time_mode=args.time_mode, duration=args.duration,
            field_z=args.field_z, field_scenario=args.field_scenario,
            dry_run=args.dry_run,
        )
        print(f"  {sim_dir.name}: {status}")
        if status.startswith("patched"):
            n_patched += 1
        elif status.startswith("skipped"):
            n_skipped += 1
        else:
            n_error += 1

    print(f"\nsummary: {n_patched} patched, {n_skipped} skipped, {n_error} errors")
    return 0 if n_error == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
