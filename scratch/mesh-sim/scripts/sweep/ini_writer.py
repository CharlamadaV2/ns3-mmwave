"""Generate per-sweep-point run.ini files from a base scenario."""

import configparser
import os
import shutil


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
