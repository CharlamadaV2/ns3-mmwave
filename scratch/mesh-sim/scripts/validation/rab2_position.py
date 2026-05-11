"""
Reduce a field scenario's per-rab GPS tracks to ENU positions.

For static collects rab2 is sometimes a vehicle (no surveyed lat/lon in
the docx) and sometimes a mast (stable tracker fix). This helper takes
each rab's ``geotak_gps.csv``, computes the median lat/lon, the GPS
spread in metres, and projects to ENU with rab1's surveyed lat/lon as
origin. Output is a paste-ready ``nodes.json`` snippet plus a short
sanity table.

Run as a module:

    python3 -m scripts.validation.rab2_position 1-1_static_04172026

Override the rab1 origin lat/lon with --origin if a future scenario
uses a different reference mast.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# Surveyed lat/lon of the Sherpa-building fixed mast (rab1) from
# data/ARPO Spring Lake Event Description.docx -- 35°12'16.3"N
# 78°57'39.1"W. Used as the ENU origin so node coords stay comparable
# to inputs/scenarios/arpo-static-baseline-3/nodes.json.
DEFAULT_ORIGIN_LAT = 35.0 + 12.0 / 60.0 + 16.3 / 3600.0
DEFAULT_ORIGIN_LON = -(78.0 + 57.0 / 60.0 + 39.1 / 3600.0)
DEFAULT_Z_M = 10.0

M_PER_DEG_LAT = 111_320.0


def resolve_field_root() -> Path:
    env = os.environ.get("MESH_SIM_FIELD_ROOT")
    if env:
        return Path(env)
    in_repo = REPO_ROOT / "data" / "arpo_extracted" / "csv"
    if in_repo.is_dir():
        return in_repo
    return Path.home() / "Desktop" / "arpo-data" / "arpo_extracted" / "csv"


def load_geotak(node_dir: Path) -> Optional[pd.DataFrame]:
    fp = node_dir / "geotak_gps.csv"
    if not fp.is_file():
        return None
    df = pd.read_csv(fp, low_memory=False)
    if not {"lat", "lon"}.issubset(df.columns):
        return None
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"] != 0) | (df["lon"] != 0)]
    return df if not df.empty else None


def project_enu(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    m_per_deg_lon = M_PER_DEG_LAT * math.cos(math.radians(lat0))
    return ((lon - lon0) * m_per_deg_lon, (lat - lat0) * M_PER_DEG_LAT)


def reduce_node(node_dir: Path, lat0: float, lon0: float) -> Optional[dict]:
    df = load_geotak(node_dir)
    if df is None:
        return None
    med_lat = float(df["lat"].median())
    med_lon = float(df["lon"].median())
    m_per_deg_lon = M_PER_DEG_LAT * math.cos(math.radians(med_lat))
    spread_m = math.hypot(
        float(df["lat"].std(ddof=0)) * M_PER_DEG_LAT,
        float(df["lon"].std(ddof=0)) * m_per_deg_lon,
    )
    x, y = project_enu(med_lat, med_lon, lat0, lon0)
    uid = df["uid"].iloc[0] if "uid" in df.columns else "(no uid)"
    return {
        "node": node_dir.name,
        "uid": str(uid),
        "n": int(len(df)),
        "lat": med_lat,
        "lon": med_lon,
        "spread_m": spread_m,
        "x_m": x,
        "y_m": y,
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenario", help="dirname under MESH_SIM_FIELD_ROOT")
    p.add_argument("--origin-lat", type=float, default=DEFAULT_ORIGIN_LAT)
    p.add_argument("--origin-lon", type=float, default=DEFAULT_ORIGIN_LON)
    p.add_argument("--z", type=float, default=DEFAULT_Z_M)
    p.add_argument(
        "--reorigin-to",
        default="",
        help="after projecting, subtract this node's ENU from all "
             "(e.g. 'rab1') so the chosen node sits at (0,0). Useful "
             "when the docx-surveyed origin and tracker centroid disagree.",
    )
    args = p.parse_args(argv)

    field_root = resolve_field_root()
    scen_dir = field_root / args.scenario
    if not scen_dir.is_dir():
        raise SystemExit(f"scenario not found: {scen_dir}\n"
                         f"  (field root: {field_root}; override with $MESH_SIM_FIELD_ROOT)")

    rows = []
    for node in sorted(scen_dir.iterdir()):
        if not node.is_dir() or node.name == "sdwan":
            continue
        out = reduce_node(node, args.origin_lat, args.origin_lon)
        if out is not None:
            rows.append(out)
    if not rows:
        raise SystemExit(f"no geotak_gps.csv found under {scen_dir}")

    if args.reorigin_to:
        anchor = next((r for r in rows if r["node"] == args.reorigin_to), None)
        if anchor is None:
            raise SystemExit(f"--reorigin-to '{args.reorigin_to}' not in scenario")
        dx, dy = anchor["x_m"], anchor["y_m"]
        for r in rows:
            r["x_m"] -= dx
            r["y_m"] -= dy

    print(f"# scenario: {args.scenario}")
    print(f"# origin (lat,lon): {args.origin_lat:.6f}, {args.origin_lon:.6f}")
    if args.reorigin_to:
        print(f"# reorigined to: {args.reorigin_to} (subtracted its ENU from all)")
    print(f"# z (m, AGL convention): {args.z}")
    print()
    print(f"{'node':<6}{'uid':<22}{'n':>6}  {'lat':>11}  {'lon':>12}  "
          f"{'spread_m':>9}  {'x_m':>9}  {'y_m':>9}")
    for r in rows:
        print(f"{r['node']:<6}{r['uid']:<22}{r['n']:>6}  "
              f"{r['lat']:>11.6f}  {r['lon']:>12.6f}  "
              f"{r['spread_m']:>9.2f}  {r['x_m']:>9.2f}  {r['y_m']:>9.2f}")

    print()
    print("# paste into nodes.json")
    print("[")
    for i, r in enumerate(rows):
        comma = "," if i < len(rows) - 1 else ""
        print(f'  {{"id": "{r["node"]}", "role": "peer", "mobility": "fixed",')
        print(f'    "position": {{"x": {r["x_m"]:.1f}, "y": {r["y_m"]:.1f}, '
              f'"z": {args.z}}}}}{comma}')
    print("]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
