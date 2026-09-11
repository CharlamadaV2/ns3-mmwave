#!/usr/bin/env python3
# make_jammers.py
"""Generate a jammers.json for the mesh sim from the field EW trials CSV.

Reads the EW trial log (trial, start/end epoch, ew_type, lat/lon, heading,
strength W, ...) and a scenario's gps_all_nodes_trace.csv, and emits a
jammers.json whose JammerSpec entries are aligned to that scenario:

  * epoch start/end  -> sim seconds (relative to the scenario's first t_utc)
  * ew_strength (W)  -> tx_power_dbm = 10*log10(W) + 30
  * lat/lon          -> local ENU (x, y), fit empirically from the trace's own
                        paired lat_deg/lon_deg <-> east_m/north_m columns
  * ew_type          -> beamwidth_deg (directional => --beamwidth, else 360)
  * heading          -> azimuth_deg (already degrees from North)

Each overlapping trial becomes ONE JammerSpec with a single [start,end)
interval. Because the trials are time-disjoint, a window that spans a power
sweep yields several specs at the same position, each active only in its slice
— which correctly models a jammer whose power/heading change over time.
"""

from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd


def _fit_enu(trace: pd.DataFrame):
    """Return f(lat,lon)->(x,y) fit from the trace's lat/lon <-> east/north."""
    need = {"lat_deg", "lon_deg", "east_m", "north_m"}
    if not need.issubset(trace.columns):
        raise ValueError(f"trace needs {sorted(need)} to fit ENU; has {list(trace.columns)}")
    d = trace[["lat_deg", "lon_deg", "east_m", "north_m"]].apply(pd.to_numeric, errors="coerce").dropna()
    d = d[(d.lat_deg != 0) | (d.lon_deg != 0)]
    A = np.column_stack([d.lat_deg, d.lon_deg, np.ones(len(d))])
    cx, *_ = np.linalg.lstsq(A, d.east_m.to_numpy(), rcond=None)
    cy, *_ = np.linalg.lstsq(A, d.north_m.to_numpy(), rcond=None)
    return lambda lat, lon: (float(cx[0]*lat + cx[1]*lon + cx[2]),
                             float(cy[0]*lat + cy[1]*lon + cy[2]))


def _watt_to_dbm(w: float) -> float:
    return 10.0 * math.log10(w) + 30.0 if w > 0 else 0.0


def build(trials_csv: Path, trace_csv: Path, trial: str | None,
          beamwidth: float, gain: float, z: float) -> list[dict]:
    trace = pd.read_csv(trace_csv)
    t = pd.to_datetime(trace["t_utc"], utc=True)
    scen_start = t.min().timestamp()
    scen_end = t.max().timestamp()
    duration = scen_end - scen_start
    to_enu = _fit_enu(trace)

    tr = pd.read_csv(trials_csv)
    tr.columns = [c.strip().lstrip("\ufeff") for c in tr.columns]

    rows = tr[tr["trial"] == trial] if trial else \
        tr[(tr["start"] < scen_end) & (tr["end"] > scen_start)]

    jammers = []
    for _, r in rows.iterrows():
        if str(r["ew_type"]).strip().lower() == "none":
            continue
        i_start = max(0.0, float(r["start"]) - scen_start)
        i_end = min(duration, float(r["end"]) - scen_start)
        if i_end <= 0.0 or i_start >= duration or i_end <= i_start:
            continue
        x, y = to_enu(float(r["ew_approx_lat"]), float(r["ew_approx_lon"]))
        directional = str(r["ew_type"]).strip().lower() == "directional"
        g = r.get("ew_gain (dBi)")
        jammers.append({
            "id": str(r["trial"]),
            "enabled": True,
            "type": "constant",
            "target_freq_mhz": [],
            "tx_power_dbm": round(_watt_to_dbm(float(r["ew_strength (W)"])), 2),
            "tx_array_gain_dbi": float(g) if pd.notna(g) else gain,
            "duty_cycle": 1.0,
            "max_range_m": 0.0,
            "beamwidth_deg": beamwidth if directional else 360.0,
            "azimuth_deg": float(r["ew_approx_heading (deg from N)"]) if directional else 0.0,
            "zenith_deg": 0.0,
            "position": {"x": round(x, 2), "y": round(y, 2), "z": z},
            "intervals": [{"start": round(i_start, 2), "end": round(i_end, 2)}],
        })
    return jammers


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build jammers.json from the EW trials CSV.")
    p.add_argument("--trials", type=Path, required=True, help="trials CSV")
    p.add_argument("--trace", type=Path, required=True,
                   help="scenario gps_all_nodes_trace.csv (epoch ref + ENU fit)")
    p.add_argument("--trial", default=None,
                   help="single trial name; default: all trials overlapping the trace window")
    p.add_argument("-o", "--output", type=Path, required=True, help="jammers.json to write")
    p.add_argument("--beamwidth", type=float, default=60.0,
                   help="beamwidth_deg for directional jammers (default 60)")
    p.add_argument("--gain", type=float, default=12.0,
                   help="tx_array_gain_dbi when the CSV cell is blank (default 12)")
    p.add_argument("--z", type=float, default=2.0, help="jammer height in m (default 2)")
    p.add_argument("--run-ini", type=Path, default=None,
                   help="optional run.ini to patch with 'jammers_file = <name>' under [scenario]")
    args = p.parse_args(argv)

    jammers = build(args.trials, args.trace, args.trial, args.beamwidth, args.gain, args.z)
    if not jammers:
        print("No jammers overlap this scenario window (or all were ew_type=none).",
              file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jammers, indent=2))
    print(f"wrote {args.output}  ({len(jammers)} jammer spec(s))")
    for j in jammers:
        iv = j["intervals"][0]
        print(f"  {j['id']}: {j['tx_power_dbm']} dBm  az={j['azimuth_deg']}  "
              f"bw={j['beamwidth_deg']}  t=[{iv['start']},{iv['end']})s  "
              f"pos=({j['position']['x']},{j['position']['y']})")

    if args.run_ini and args.run_ini.is_file():
        txt = args.run_ini.read_text()
        if "jammers_file" not in txt:
            txt = txt.replace("[scenario]\n", f"[scenario]\njammers_file = {args.output.name}\n", 1)
            args.run_ini.write_text(txt)
            print(f"  patched {args.run_ini}: jammers_file = {args.output.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
