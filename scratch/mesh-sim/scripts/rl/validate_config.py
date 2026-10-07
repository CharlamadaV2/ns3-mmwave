#!/usr/bin/env python3
"""Check an RL scenario, selection, and binary before training; --launch adds one reset."""

import argparse
import json
import os
import sys
from pathlib import Path

from scripts.rl.bootstrap_venv import DIRECT_DEPS, MESH_SIM_ROOT, read_pins
from scripts.rl.cli_common import (add_scenario_arguments, add_selection_arguments,
                                   has_previous_run, package_versions, resolve_seed,
                                   selection_from_args)
from scripts.rl.env.config import (read_control_mode, read_rl_bounds,
                                   read_scenario_identity)

VALIDATE_DIR = "validate"


class _Report:
    """Collects check results; a check is static unless it needed a launch."""

    def __init__(self):
        self.checks: list[dict] = []

    def add(self, name: str, status: str, detail: str, kind: str = "static") -> None:
        self.checks.append({"check": name, "status": status, "detail": detail,
                            "kind": kind})

    def ok(self, name, detail, kind="static"):
        self.add(name, "ok", detail, kind)

    def warn(self, name, detail, kind="static"):
        self.add(name, "warning", detail, kind)

    def error(self, name, detail, kind="static"):
        self.add(name, "error", detail, kind)

    @property
    def failed(self) -> bool:
        return any(check["status"] == "error" for check in self.checks)


def _check_run_config(report: _Report, run_config: str) -> bool:
    path = Path(run_config)
    if not path.is_file():
        report.error("run_config", f"run config not found: {run_config}")
        return False
    try:
        mode = read_control_mode(run_config)
    except Exception as exc:
        report.error("run_config", f"{type(exc).__name__}: {exc}")
        return False
    report.ok("run_config", f"readable: {path.resolve()}")
    report.ok("control_mode", mode)
    return True


def _check_selection(report: _Report, args):
    try:
        selection = selection_from_args(args)
    except ValueError as exc:
        report.error("selection", f"Invalid RL selection: {exc}")
        return None
    described = selection.describe()
    report.ok("selection", json.dumps(described, sort_keys=True))
    return selection


def _check_seed(report: _Report, args):
    try:
        seed, source = resolve_seed(args.seed, args.run_config)
    except ValueError as exc:
        report.error("seed", str(exc))
        return None
    report.ok("seed", f"{seed} (source: {source})")
    return seed


def _check_identity(report: _Report, run_config: str) -> dict | None:
    try:
        identity = read_scenario_identity(run_config)
    except (FileNotFoundError, OSError) as exc:
        report.error("scenario_identity", str(exc))
        return None
    digests = ", ".join(f"{key}={value}" for key, value in identity.items()
                        if key != "run_config")
    report.ok("scenario_identity", digests)
    return identity


def _check_bounds(report: _Report, run_config: str) -> None:
    try:
        bounds = read_rl_bounds(run_config)
    except Exception as exc:
        report.error("rl_bounds", f"{type(exc).__name__}: {exc}")
        return
    unordered = [axis for axis, (low, high) in zip("xyz", bounds) if not low < high]
    detail = ", ".join(f"{axis}=[{low}, {high}]"
                       for axis, (low, high) in zip("xyz", bounds))
    if unordered:
        report.error("rl_bounds", f"min must be < max for {unordered}; {detail}")
    else:
        report.ok("rl_bounds", detail)


def _check_binary(report: _Report, sim_binary: str) -> None:
    path = Path(sim_binary)
    if not path.is_file():
        report.error("sim_binary", f"simulator binary not found: {sim_binary}")
    elif not os.access(path, os.X_OK):
        report.error("sim_binary", f"simulator binary is not executable: {sim_binary}")
    else:
        report.ok("sim_binary", str(path.resolve()))


def _check_output_dir(report: _Report, output_dir: str | None) -> None:
    if not output_dir:
        report.ok("output_dir", "not given; no output-dir guard applied")
        return
    existing = has_previous_run(output_dir) if os.path.isdir(output_dir) else None
    if existing:
        report.error("output_dir",
                     f"{output_dir} already contains {existing}; choose a new directory")
    else:
        report.ok("output_dir", f"{output_dir} is free for a new run")


def _check_packages(report: _Report) -> None:
    try:
        pins = read_pins(MESH_SIM_ROOT / "requirements.txt")
    except (OSError, ValueError) as exc:
        report.warn("package_versions", f"cannot read requirements.txt: {exc}")
        return

    installed = package_versions()
    drift = []
    for module, distribution in DIRECT_DEPS.items():
        pinned, have = pins.get(distribution), installed.get(module)
        if have is None:
            drift.append(f"{distribution} not installed (pinned {pinned})")
        elif pinned != have:
            drift.append(f"{distribution} {have} != pinned {pinned}")
    if drift:
        report.warn("package_versions", "; ".join(drift))
    else:
        report.ok("package_versions", "installed versions match requirements.txt")


def _launch(report: _Report, args, selection, seed: int) -> dict | None:
    """One reset+close of a real MeshRlEnv; the only runtime-validated checks."""
    from scripts.rl.env.mesh_env import MeshRlEnv

    out_dir = os.path.join(args.output_dir, VALIDATE_DIR)
    env = None
    try:
        env = MeshRlEnv(args.sim_binary, args.run_config, seed=seed,
                        output_dir=out_dir, band=args.band, selection=selection)
        env.reset()
        contract = env.contract or {}
        observation = env.observation_schema or {}
        reward = env.reward_schema or {}
        launch = {
            "output_dir": os.path.abspath(out_dir),
            "control_mode": env.control_mode,
            "contract": contract.get("contract"),
            "num_mesh_nodes": contract.get("num_mesh_nodes"),
            "max_controlled_nodes": contract.get("max_controlled_nodes"),
            "num_decisions": contract.get("num_decisions"),
            "cpp_obs_dim": contract.get("obs_dim"),
            "cpp_mask_dim": contract.get("mask_dim"),
            "preset_obs_dim": observation.get("obs_dim"),
            "observation_schema_sha256": observation.get("sha256"),
            "reward_schema_sha256": reward.get("sha256"),
        }
    except Exception as exc:
        report.error("launch", f"{type(exc).__name__}: {exc}", kind="launch")
        return None
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    report.ok("launch", ", ".join(f"{key}={value}" for key, value in launch.items()),
              kind="launch")
    return launch


def print_report(report: _Report, launch: dict | None) -> None:
    for check in report.checks:
        print(f"[{check['status'].upper():<7}] {check['kind']:<6} "
              f"{check['check']:<19} {check['detail']}")
    if launch is None:
        print("launch checks: not run (static checks only)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Validate a mesh-sim RL run configuration before training")
    add_scenario_arguments(p)
    add_selection_arguments(p)
    p.add_argument("--seed", type=int, default=None,
                   help="Seed to validate; defaults to [scenario] seed in run.ini")
    p.add_argument("--output-dir", default="",
                   help="Directory checked by the output-dir guard and used by --launch")
    p.add_argument("--launch", action="store_true",
                   help="Also start the simulator once (requires --output-dir)")
    p.add_argument("--json", action="store_true", help="Print the report as JSON")
    args = p.parse_args(argv)

    if args.launch and not args.output_dir:
        print("--launch requires --output-dir", file=sys.stderr)
        return 1

    report = _Report()
    readable = _check_run_config(report, args.run_config)
    selection = _check_selection(report, args) if readable else None
    seed = _check_seed(report, args) if readable else None
    if readable:
        _check_identity(report, args.run_config)
        _check_bounds(report, args.run_config)
    _check_binary(report, args.sim_binary)
    _check_output_dir(report, args.output_dir)
    _check_packages(report)

    launch = None
    if args.launch:
        if report.failed:
            report.error("launch", "skipped: a static check failed", kind="launch")
        else:
            launch = _launch(report, args, selection, seed)

    payload = {"run_config": args.run_config, "sim_binary": args.sim_binary,
               "checks": report.checks, "launch": launch,
               "status": "failed" if report.failed else "ok"}
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_report(report, launch)
    if report.failed:
        for check in report.checks:
            if check["status"] == "error":
                print(f"{check['check']}: {check['detail']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
