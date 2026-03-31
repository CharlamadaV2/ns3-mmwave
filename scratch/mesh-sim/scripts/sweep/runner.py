"""Orchestrate a parameter sweep: generate configs, run sims, plot."""

import configparser
import glob
import json
import itertools
import os
import shutil
import subprocess
import sys
from datetime import datetime

from .config import SweepConfig
from .ini_writer import copy_scenario_files, write_point_ini


def _find_sim_binary(mesh_sim_root: str) -> str | None:
    """Auto-detect the sim binary from the ns-3 build directory."""
    # Walk up to ns-3 root (mesh-sim is at scratch/mesh-sim/)
    ns3_root = os.path.dirname(os.path.dirname(mesh_sim_root))
    pattern = os.path.join(ns3_root, "build", "scratch", "mesh-sim", "ns3*-sim-*")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[0]
    return None



def _point_label(index: int) -> str:
    """Build a directory name for a sweep point."""
    return f"point-{index:03d}"


def _is_point_complete(point_dir: str, seeds: list[int]) -> bool:
    """Check if all seed dirs have a summary.json (for --resume)."""
    for seed in seeds:
        summary = os.path.join(point_dir, f"seed-{seed}", "summary.json")
        if not os.path.isfile(summary):
            return False
    return True


def _read_nodes_file(base_run_ini: str) -> str:
    """Read the nodes_file value from a base run.ini."""
    cfg = configparser.ConfigParser()
    cfg.read(base_run_ini)
    return cfg.get("scenario", "nodes_file", fallback="nodes.json")


def _read_buildings_file(base_run_ini: str) -> str:
    """Read the buildings_file value from a base run.ini."""
    cfg = configparser.ConfigParser()
    cfg.read(base_run_ini)
    return cfg.get("scenario", "buildings_file", fallback="")


def _run_plotting(point_dir: str, plot_config: str, mesh_sim_root: str) -> None:
    """Generate a plot INI for a point and invoke the plotting CLI."""
    # Read the user's plot config as a base (or use defaults)
    plot_cfg = configparser.ConfigParser()
    if plot_config and os.path.isfile(os.path.join(mesh_sim_root, plot_config)):
        plot_cfg.read(os.path.join(mesh_sim_root, plot_config))

    if not plot_cfg.has_section("output"):
        plot_cfg.add_section("output")
    plot_cfg.set("output", "data_dir", os.path.abspath(point_dir))

    # Ensure at least some plots are enabled
    if not plot_cfg.has_section("plots"):
        plot_cfg.add_section("plots")
        for key in ["sinr_timeseries", "capacity_timeseries",
                     "throughput_timeseries", "latency_timeseries",
                     "per_node_sinr", "per_flow_throughput", "network_summary"]:
            plot_cfg.set("plots", key, "true")

    plot_ini_path = os.path.join(point_dir, "plot.ini")
    with open(plot_ini_path, "w") as f:
        plot_cfg.write(f)

    subprocess.run(
        [sys.executable, "-m", "scripts.plotting.cli", "--config", plot_ini_path],
        cwd=mesh_sim_root,
    )


def run_sweep(
    cfg: SweepConfig,
    sweep_config_path: str,
    sim_binary: str | None = None,
    dry_run: bool = False,
    resume: bool = False,
) -> None:
    """Execute the full sweep."""
    # Resolve mesh-sim root (base_scenario is absolute, walk up)
    mesh_sim_root = cfg.base_scenario
    while mesh_sim_root != "/" and not os.path.isfile(
        os.path.join(mesh_sim_root, "sim.cc")
    ):
        mesh_sim_root = os.path.dirname(mesh_sim_root)

    # Find sim binary
    if not sim_binary:
        sim_binary = _find_sim_binary(mesh_sim_root)
    if not sim_binary and not dry_run:
        print("Error: could not find sim binary. Use --sim-binary to specify.",
              file=sys.stderr)
        sys.exit(1)

    # Create sweep output directory (mirrors single-run structure)
    now = datetime.now()
    ts_month = now.strftime("%Y-%m")
    ts_day = now.strftime("%d")
    ts_time = now.strftime("%H-%M-%S")
    sweep_dir = os.path.join(
        mesh_sim_root, "outputs", ts_month, ts_day, ts_time
    )

    if not dry_run:
        os.makedirs(sweep_dir, exist_ok=True)
        shutil.copy2(os.path.abspath(sweep_config_path),
                      os.path.join(sweep_dir, "sweep.ini"))

    # Read base scenario files info
    base_run_ini = os.path.join(cfg.base_scenario, "run.ini")
    nodes_file = _read_nodes_file(base_run_ini)
    buildings_file = _read_buildings_file(base_run_ini)

    # Check if any override or sweep dimension overrides nodes_file
    for (section, key), value in cfg.overrides.items():
        if section == "scenario" and key == "nodes_file":
            nodes_file = value
    # (per-point nodes_file overrides handled in the loop)

    # Compute cartesian product
    dim_values = [d.values for d in cfg.dimensions]
    dim_keys = [(d.section, d.key) for d in cfg.dimensions]
    product = list(itertools.product(*dim_values))

    total_points = len(product)
    total_runs = total_points * len(cfg.seeds)
    seeds_str = ",".join(str(s) for s in cfg.seeds)

    print(f"Sweep: {cfg.label}")
    print(f"  Dimensions: {len(cfg.dimensions)}")
    for d in cfg.dimensions:
        print(f"    {d.section}.{d.key} = {d.values}")
    if cfg.overrides:
        print(f"  Overrides:")
        for (s, k), v in cfg.overrides.items():
            print(f"    {s}.{k} = {v}")
    print(f"  Points: {total_points} (cartesian product)")
    print(f"  Seeds per point: {cfg.seeds}")
    print(f"  Total simulation runs: {total_runs}")
    print(f"  Auto-plot: {cfg.auto_plot}")
    print()

    if dry_run:
        print("--- DRY RUN: sweep matrix ---")
        for i, combo in enumerate(product, 1):
            params = dict(zip(dim_keys, combo))
            label = _point_label(i)
            param_str = ", ".join(
                f"{s}.{k}={v}" for (s, k), v in params.items()
            )
            print(f"  {label}: {param_str}")
        print(f"\nWould run {total_runs} simulations. Exiting (dry run).")
        return

    os.makedirs(sweep_dir, exist_ok=True)

    # Build manifest
    manifest = {
        "label": cfg.label,
        "timestamp": now.isoformat(),
        "base_scenario": cfg.base_scenario,
        "seeds": cfg.seeds,
        "dimensions": [
            {"section": d.section, "key": d.key, "values": d.values}
            for d in cfg.dimensions
        ],
        "overrides": {f"{s}.{k}": v for (s, k), v in cfg.overrides.items()},
        "points": [],
    }

    succeeded = 0
    failed = 0
    skipped = 0

    for i, combo in enumerate(product, 1):
        point_params = dict(zip(dim_keys, combo))
        point_name = _point_label(i)
        point_dir = os.path.join(sweep_dir, point_name)

        param_str = ", ".join(f"{s}.{k}={v}" for (s, k), v in point_params.items())
        print(f"[{i}/{total_points}] {point_name}: {param_str}")

        # Resume: skip completed points
        if resume and os.path.isdir(point_dir) and _is_point_complete(point_dir, cfg.seeds):
            print(f"  Skipping (already complete)")
            manifest["points"].append({
                "index": i,
                "dir": point_name,
                "params": {f"{s}.{k}": v for (s, k), v in point_params.items()},
                "status": "completed",
            })
            skipped += 1
            continue

        os.makedirs(point_dir, exist_ok=True)

        # Determine nodes_file for this point
        pt_nodes_file = nodes_file
        for (s, k), v in point_params.items():
            if s == "scenario" and k == "nodes_file":
                pt_nodes_file = v

        # Generate run.ini and copy scenario files
        scenario_name = f"{cfg.label}_point-{i:03d}"
        write_point_ini(base_run_ini, point_dir, cfg.overrides, point_params,
                        scenario_name)
        copy_scenario_files(cfg.base_scenario, point_dir, pt_nodes_file,
                            buildings_file)

        # Run simulation
        cmd = [sim_binary, f"--run-config={os.path.join(point_dir, 'run.ini')}",
               f"--seeds={seeds_str}"]

        log_path = os.path.join(point_dir, "run.log")
        status = "completed"
        env = os.environ.copy()
        ns3_root = os.path.dirname(os.path.dirname(mesh_sim_root))
        lib_dir = os.path.join(ns3_root, "build", "lib")
        env["DYLD_LIBRARY_PATH"] = lib_dir + ":" + env.get("DYLD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
        with open(log_path, "w") as log_f:
            log_f.write(f"Command: {' '.join(cmd)}\n\n")
            log_f.flush()
            result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                    env=env)

        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode}). See {log_path}")
            status = "failed"
            failed += 1
        else:
            print(f"  OK")
            succeeded += 1

            # Auto-plot per point
            if cfg.auto_plot == "each":
                print(f"  Plotting...")
                _run_plotting(point_dir, cfg.plot_config, mesh_sim_root)

        manifest["points"].append({
            "index": i,
            "dir": point_name,
            "params": {f"{s}.{k}": v for (s, k), v in point_params.items()},
            "status": status,
        })

        # Update manifest after each point (for resume)
        with open(os.path.join(sweep_dir, "sweep_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    # Auto-plot all points after sweep
    if cfg.auto_plot == "all":
        print("\nPlotting all completed points...")
        for point in manifest["points"]:
            if point["status"] == "completed":
                point_dir = os.path.join(sweep_dir, point["dir"])
                print(f"  Plotting {point['dir']}...")
                _run_plotting(point_dir, cfg.plot_config, mesh_sim_root)

    # Final summary
    print(f"\n{'='*60}")
    print(f"Sweep complete: {cfg.label}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Output:    {sweep_dir}")
    print(f"{'='*60}")
