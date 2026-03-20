#!/usr/bin/env python3

import argparse
import configparser
import datetime
import json
import logging
import math
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s",
                    stream=sys.stderr)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep multiple seeds for one scenario",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True,
                   help="Path to run.ini scenario configuration file")
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated list of seed values to run, e.g. '1,2,3,4,5'. "
                        "When given, --num-seeds and --base-seed are ignored.")
    p.add_argument("--num-seeds", type=int, default=5,
                   help="Number of seeds to run when --seeds is not given (default: 5)")
    p.add_argument("--base-seed", type=int, default=1,
                   help="First seed value when using --num-seeds (default: 1). "
                        "Seeds will be base_seed, base_seed+1, ..., base_seed+num_seeds-1.")
    p.add_argument("--out-base", default=None,
                   help="Base output directory (default: scratch/mmwave-sim/outputs/)")
    p.add_argument("--ns3-root", default=None,
                   help="ns3-mmwave repo root (default: 3 levels up from this script)")
    return p.parse_args()


def find_ns3_root(hint, script_dir):
    if hint:
        return os.path.abspath(hint)
    return os.path.abspath(os.path.join(script_dir, "..", "..", ".."))


def aggregate(summary_paths, out_path):
    summaries = []
    for p in summary_paths:
        if os.path.isfile(p):
            with open(p) as f:
                summaries.append(json.load(f))

    if not summaries:
        log.warning("no summary files found to aggregate")
        return

    def stats(values):
        n = len(values)
        if n == 0:
            return {"mean": None, "std": None, "n": 0}
        mean = sum(values) / n
        # TODO: Should add a specified confidence interval in which to stop seeds early
        variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
        return {"mean": mean, "std": math.sqrt(variance), "n": n}

    network_keys = ["mean_sinr_db", "min_sinr_db", "corruption_rate",
                    "sum_dl_throughput_mbps"]
    net_agg = {}
    for key in network_keys:
        vals = [s["network"][key] for s in summaries
                if "network" in s and s["network"].get(key) is not None]
        net_agg[key] = stats(vals)

    all_ue_ids = set()
    for s in summaries:
        all_ue_ids.update(s.get("per_ue", {}).keys())

    ue_keys = ["mean_sinr_db", "min_sinr_db", "corruption_rate",
               "dl_throughput_mbps", "dl_delay_mean_ms"]
    per_ue_agg = {}
    for uid in sorted(all_ue_ids):
        ue_agg = {}
        for key in ue_keys:
            vals = [s["per_ue"][uid][key] for s in summaries
                    if uid in s.get("per_ue", {})
                    and s["per_ue"][uid].get(key) is not None]
            ue_agg[key] = stats(vals)
        per_ue_agg[uid] = ue_agg

    result = {
        "num_seeds": len(summaries),
        "seeds": [s.get("seed") for s in summaries],
        "network": net_agg,
        "per_ue": per_ue_agg,
        "seed_dirs": summary_paths,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Aggregated summary written to %s", out_path)


def run(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ns3_root   = find_ns3_root(args.ns3_root, script_dir)
    run_sim    = os.path.join(script_dir, "run_sim.py")

    # Determine seed list
    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = list(range(args.base_seed, args.base_seed + args.num_seeds))

    if not seeds:
        log.error("No seeds to run.")
        sys.exit(1)

    ini = configparser.RawConfigParser()
    ini.optionxform = str
    ini.read(args.config)
    scenario_name = ini.get("scenario", "name", fallback="unnamed")

    mmwave_sim_dir = os.path.abspath(os.path.join(script_dir, ".."))
    out_base = os.path.abspath(args.out_base) if args.out_base else os.path.join(mmwave_sim_dir, "outputs")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = os.path.join(out_base, f"{scenario_name}_sweep_{ts}")
    os.makedirs(parent_dir, exist_ok=True)

    log.info("Sweep: scenario=%s  seeds=%s", scenario_name, seeds)
    log.info("Output dir: %s", parent_dir)

    summary_paths = []
    for i, seed in enumerate(seeds):
        seed_out_dir = os.path.join(parent_dir, f"seed-{seed}")
        log.info("--- seed %d  (%d/%d) ---", seed, i + 1, len(seeds))
        cmd = [sys.executable, run_sim,
               "--config", args.config,
               "--seed", str(seed),
               "--run-id", "1",
               "--output-dir", seed_out_dir,
               "--ns3-root", ns3_root]
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            log.warning("seed=%d failed with exit code %d", seed, result.returncode)

        summary_path = os.path.join(seed_out_dir, "summary.json")
        if os.path.isfile(summary_path):
            summary_paths.append(summary_path)
        else:
            log.warning("summary.json not found for seed=%d at %s", seed, summary_path)

    agg_path = os.path.join(parent_dir, "aggregated_summary.json")
    aggregate(summary_paths, agg_path)
    log.info("Sweep complete. Results in: %s", parent_dir)


if __name__ == "__main__":
    run(parse_args())
