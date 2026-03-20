#!/usr/bin/env python3
"""Single mmwave-sim episode wrapper."""

import argparse
import configparser
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                    stream=sys.stderr)
log = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MMWAVE_SIM_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))


def parse_args():
    p = argparse.ArgumentParser(description="Run one mmwave-sim episode")
    p.add_argument("--config", required=True, help="Path to run.ini")
    p.add_argument("--seed", type=int, default=None,
                   help="Override seed (default: from run.ini)")
    p.add_argument("--run-id", type=int, default=None,
                   help="Override run_id (default: from run.ini)")
    p.add_argument("--output-dir", default=None,
                   help="Explicit output directory (skips timestamped layout).")
    p.add_argument("--out-base", default=os.path.join(_MMWAVE_SIM_DIR, "outputs"),
                   help="Base output directory for timestamped runs "
                        "(default: scratch/mmwave-sim/outputs/). "
                        "Ignored when --output-dir is given.")
    p.add_argument("--positions-override", default="",
                   help="JSON file overriding node positions (RL extension point)")
    p.add_argument("--ns3-root", default=None,
                   help="ns3-mmwave repo root (default: 3 levels up from this script)")
    return p.parse_args()


def find_ns3_root(hint):
    if hint:
        return os.path.abspath(hint)
    return os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))


def make_timestamped_dirs(base, seed):
    """Create outputs/YYYY-MM/DD/HH-MM-SS/seed-<N>/ under base."""
    now = datetime.datetime.now()
    run_dir  = os.path.join(base, now.strftime("%Y-%m"), now.strftime("%d"),
                            now.strftime("%H-%M-%S"))
    seed_dir = os.path.join(run_dir, f"seed-{seed}")
    inp_dir  = os.path.join(run_dir, "inputs")
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(inp_dir,  exist_ok=True)
    return run_dir, seed_dir, inp_dir


def make_explicit_dirs(output_dir):
    """Use the caller-specified directory; create inputs/ alongside it."""
    seed_dir = os.path.abspath(output_dir)
    inp_dir  = os.path.join(os.path.dirname(seed_dir), "inputs")
    os.makedirs(seed_dir, exist_ok=True)
    os.makedirs(inp_dir,  exist_ok=True)
    return seed_dir, inp_dir


def write_patched_ini(src_ini_path, dest_ini_path, seed, run_id, output_dir):
    cfg = configparser.RawConfigParser()
    cfg.optionxform = str
    cfg.read(src_ini_path)
    if not cfg.has_section("scenario"):
        cfg.add_section("scenario")
    if not cfg.has_section("output"):
        cfg.add_section("output")
    if seed is not None:
        cfg.set("scenario", "seed", str(seed))
    if run_id is not None:
        cfg.set("scenario", "run_id", str(run_id))
    cfg.set("output", "dir", output_dir)
    with open(dest_ini_path, "w") as f:
        cfg.write(f)


def run(args):
    ns3_root = find_ns3_root(args.ns3_root)
    ns3_bin  = os.path.join(ns3_root, "ns3")

    if not os.path.isfile(ns3_bin):
        log.error("ns3 binary not found at %s", ns3_bin)
        sys.exit(1)

    src_ini = os.path.abspath(args.config)
    if not os.path.isfile(src_ini):
        log.error("config file not found: %s", src_ini)
        sys.exit(1)

    ini = configparser.RawConfigParser()
    ini.optionxform = str
    ini.read(src_ini)
    seed   = args.seed   if args.seed   is not None else int(ini.get("scenario", "seed",   fallback="42"))
    run_id = args.run_id if args.run_id is not None else int(ini.get("scenario", "run_id", fallback="1"))

    # Determine output directories
    if args.output_dir:
        seed_dir, inp_dir = make_explicit_dirs(args.output_dir)
    else:
        out_base = os.path.abspath(args.out_base)
        _, seed_dir, inp_dir = make_timestamped_dirs(out_base, seed)

    log.info("Output directory: %s", seed_dir)

    # Archive scenario input files
    scenario_dir = os.path.dirname(src_ini)
    for fname in os.listdir(scenario_dir):
        shutil.copy2(os.path.join(scenario_dir, fname), inp_dir)

    patched_ini = os.path.join(inp_dir, "run_patched.ini")
    write_patched_ini(src_ini, patched_ini, seed, run_id, seed_dir)

    cmd = [ns3_bin, "run", "scratch/mmwave-sim/sim", "--",
           f"--run-config={patched_ini}"]
    if args.positions_override:
        cmd.append(f"--positions-override={os.path.abspath(args.positions_override)}")

    log.info("Running: %s", " ".join(cmd))

    stderr_path = os.path.join(seed_dir, "stderr.txt")
    with open(stderr_path, "w") as stderr_f:
        proc = subprocess.Popen(cmd, cwd=ns3_root, stderr=subprocess.PIPE, text=True)
        for line in proc.stderr:
            sys.stderr.write(line)
            stderr_f.write(line)
        proc.wait()

    if proc.returncode != 0:
        log.error("simulation exited with code %d — see %s", proc.returncode, stderr_path)
        sys.exit(proc.returncode)

    summary_path = os.path.join(seed_dir, "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        net = summary.get("network", {})
        log.info("Done. Network mean SINR: %.2f dB, sum DL tput: %.1f Mbps",
                 net.get("mean_sinr_db", float("nan")),
                 net.get("sum_dl_throughput_mbps", float("nan")))
    else:
        log.warning("summary.json not found in %s", seed_dir)

    log.info("Full output at: %s", seed_dir)
    return seed_dir


if __name__ == "__main__":
    run(parse_args())
