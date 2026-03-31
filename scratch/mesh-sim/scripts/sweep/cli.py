#!/usr/bin/env python3
"""CLI entry point for mesh-sim parameter sweeps.

Usage:
    python -m scripts.sweep.cli --config inputs/sweeps/example.ini
    python -m scripts.sweep.cli --config inputs/sweeps/example.ini --dry-run
    python -m scripts.sweep.cli --config inputs/sweeps/example.ini --resume
"""

import argparse

from .config import parse_sweep_config
from .runner import run_sweep


def main():
    parser = argparse.ArgumentParser(description="mesh-sim parameter sweep runner")
    parser.add_argument("--config", required=True,
                        help="Path to sweep INI config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sweep matrix without running simulations")
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed points (checks for summary.json)")
    parser.add_argument("--sim-binary",
                        help="Path to sim binary (default: auto-detect)")
    args = parser.parse_args()

    cfg = parse_sweep_config(args.config)
    run_sweep(cfg, sweep_config_path=args.config, sim_binary=args.sim_binary,
              dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
