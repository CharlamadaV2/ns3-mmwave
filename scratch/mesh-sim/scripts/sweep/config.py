"""Parse sweep INI configuration files."""

import configparser
import os
import sys
from dataclasses import dataclass, field


@dataclass
class SweepDimension:
    """A single swept parameter with its list of values."""

    section: str  # e.g. "channel"
    key: str  # e.g. "frequency_ghz"
    values: list[str] = field(default_factory=list)


@dataclass
class SweepConfig:
    """Parsed sweep configuration."""

    base_scenario: str
    seeds: list[int]
    auto_plot: str  # "each", "all", "none"
    plot_config: str  # path to plot INI (empty = use defaults)
    label: str
    overrides: dict[tuple[str, str], str] = field(default_factory=dict)
    dimensions: list[SweepDimension] = field(default_factory=list)


def _parse_section_key(dotted: str) -> tuple[str, str]:
    """Split 'section.key' into (section, key). Exits on bad format."""
    parts = dotted.split(".", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        print(f"Error: invalid parameter name '{dotted}' — expected section.key",
              file=sys.stderr)
        sys.exit(1)
    return parts[0], parts[1]


def parse_sweep_config(path: str) -> SweepConfig:
    """Parse a sweep INI file and return a validated SweepConfig."""
    if not os.path.isfile(path):
        print(f"Error: sweep config not found: {path}", file=sys.stderr)
        sys.exit(1)

    cfg = configparser.ConfigParser()
    cfg.read(path)

    # --- [sweep.meta] ---
    if not cfg.has_section("sweep.meta"):
        print("Error: sweep config missing [sweep.meta] section", file=sys.stderr)
        sys.exit(1)

    base_scenario = cfg.get("sweep.meta", "base_scenario")
    seeds_str = cfg.get("sweep.meta", "seeds", fallback="1")
    seeds = [int(s.strip()) for s in seeds_str.split(",")]
    auto_plot = cfg.get("sweep.meta", "auto_plot", fallback="none")
    plot_config = cfg.get("sweep.meta", "plot_config", fallback="")
    label = cfg.get("sweep.meta", "label", fallback="sweep")

    if auto_plot not in ("each", "all", "none"):
        print(f"Error: auto_plot must be 'each', 'all', or 'none', got '{auto_plot}'",
              file=sys.stderr)
        sys.exit(1)

    # Resolve base_scenario relative to sweep config directory
    sweep_dir = os.path.dirname(os.path.abspath(path))
    # Walk up to mesh-sim root (sweep configs are in inputs/sweeps/)
    mesh_sim_root = sweep_dir
    while mesh_sim_root != "/" and not os.path.isfile(
        os.path.join(mesh_sim_root, "sim.cc")
    ):
        mesh_sim_root = os.path.dirname(mesh_sim_root)

    if not os.path.isfile(os.path.join(mesh_sim_root, "sim.cc")):
        mesh_sim_root = os.path.dirname(os.path.abspath(path))

    base_scenario_abs = os.path.join(mesh_sim_root, base_scenario)
    run_ini = os.path.join(base_scenario_abs, "run.ini")
    if not os.path.isfile(run_ini):
        print(f"Error: base scenario run.ini not found: {run_ini}", file=sys.stderr)
        sys.exit(1)

    # --- [sweep.override] (optional) ---
    overrides: dict[tuple[str, str], str] = {}
    if cfg.has_section("sweep.override"):
        for dotted, value in cfg.items("sweep.override"):
            section, key = _parse_section_key(dotted)
            overrides[(section, key)] = value.strip()

    # --- [sweep] ---
    if not cfg.has_section("sweep"):
        print("Error: sweep config missing [sweep] section", file=sys.stderr)
        sys.exit(1)

    dimensions: list[SweepDimension] = []
    for dotted, values_str in cfg.items("sweep"):
        section, key = _parse_section_key(dotted)
        values = [v.strip() for v in values_str.split(",") if v.strip()]
        if not values:
            print(f"Error: sweep dimension '{dotted}' has no values", file=sys.stderr)
            sys.exit(1)
        dimensions.append(SweepDimension(section=section, key=key, values=values))

    if not dimensions:
        print("Error: [sweep] section has no dimensions", file=sys.stderr)
        sys.exit(1)

    return SweepConfig(
        base_scenario=base_scenario_abs,
        seeds=seeds,
        auto_plot=auto_plot,
        plot_config=plot_config,
        label=label,
        overrides=overrides,
        dimensions=dimensions,
    )
