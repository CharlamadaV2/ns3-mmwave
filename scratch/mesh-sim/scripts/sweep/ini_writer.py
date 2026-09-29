"""Generate per-sweep-point run.ini files from a base scenario."""

import configparser
import os
import shutil


## @fn write_point_ini
# @brief Write one sweep point's `run.ini`, derived from the base scenario.
#
# @param base_run_ini  Path to the base scenario `run.ini`.
# @param point_dir     Existing directory for this point (must already exist).
# @param overrides     Constant `(section, key) -> value` overrides for all points.
# @param point_params  Swept `(section, key) -> value` for this point; wins over `overrides`.
# @return Path of the written `<point_dir>/run.ini`.
# @throws OSError if `point_dir` does not exist or is not writable.
#
# Copies the base INI, applies `overrides`, then `point_params`, creating missing
# sections. Then sets `[output] dir` to the absolute `point_dir` and `[scenario] name`
# to `scenario_name`. Comments in the base INI are not preserved.
def write_point_ini(
    base_run_ini: str,
    point_dir: str,
    overrides: dict[tuple[str, str], str],
    point_params: dict[tuple[str, str], str],
    scenario_name: str,
) -> str:
    """Read base run.ini, apply overrides + point params, write to point_dir.

    Returns the path to the written run.ini.
    """
    cfg = configparser.ConfigParser()
    cfg.read(base_run_ini)

    # Apply constant overrides first, then per-point swept values on top
    for (section, key), value in overrides.items():
        if not cfg.has_section(section):
            cfg.add_section(section)
        cfg.set(section, key, value)

    for (section, key), value in point_params.items():
        if not cfg.has_section(section):
            cfg.add_section(section)
        cfg.set(section, key, value)

    # Set output dir to the point directory (absolute)
    if not cfg.has_section("output"):
        cfg.add_section("output")
    cfg.set("output", "dir", os.path.abspath(point_dir))

    # Tag the scenario name
    if not cfg.has_section("scenario"):
        cfg.add_section("scenario")
    cfg.set("scenario", "name", scenario_name)

    out_path = os.path.join(point_dir, "run.ini")
    with open(out_path, "w") as f:
        cfg.write(f)

    return out_path


## @fn copy_scenario_files
# @brief Copy the node (and optional building) JSON files into a point directory.
#
# @param base_scenario_dir  Base scenario directory.
# @param point_dir          Destination directory.
# @param nodes_file         Nodes file name (default `nodes.json`).
# @param buildings_file     Buildings file name; empty (default) skips buildings.
# @return None.
#
# Files that do not exist in the base scenario are silently skipped.
# `jammers.json` is not copied.
def copy_scenario_files(base_scenario_dir: str, point_dir: str,
                        nodes_file: str = "nodes.json",
                        buildings_file: str = "") -> None:
    """Copy nodes.json and buildings.json from the base scenario to point_dir."""
    nodes_src = os.path.join(base_scenario_dir, nodes_file)
    if os.path.isfile(nodes_src):
        shutil.copy2(nodes_src, os.path.join(point_dir, nodes_file))

    if buildings_file:
        buildings_src = os.path.join(base_scenario_dir, buildings_file)
        if os.path.isfile(buildings_src):
            shutil.copy2(buildings_src, os.path.join(point_dir, buildings_file))
