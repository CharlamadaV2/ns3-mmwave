"""Strict [baseline] and mapping-file parsing that mirrors the simulator's INI rules."""

import configparser
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from scripts.sim_support import strip_inline_comment

PLACEMENT_METHODS = ("geometric", "optimization")
ALGORITHMS = ("none", *PLACEMENT_METHODS)
OBJECTIVES = ("coverage", "balanced", "resilience")
APPLICATIONS = ("initial_positions",)
WAYPOINT_POLICIES = ("reject", "translate")
PLATFORMS = ("ground", "aerial")
ALL_NODES = "all"
# key -> (parse kind, lower bound, lower bound inclusive, upper bound, default)
NUMERIC_KEYS = {
    "aerial_fixed_cost_m2": ("float", 0.0, True, None, 150000.0),
    "ground_fixed_cost_m2": ("float", 0.0, True, None, 150000.0),
    "aerial_cost_m2_per_m": ("float", 0.0, True, None, 500.0),
    "ground_cost_m2_per_m": ("float", 0.0, True, None, 100.0),
    "aerial_max_displacement_m": ("float", 0.0, False, None, None),
    "ground_max_displacement_m": ("float", 0.0, False, None, None),
    "candidate_grid_cells": ("int", 1, True, None, 400),
    "coverage_grid_cells": ("int", 1, True, None, 400),
    "grid_min_resolution_m": ("float", 0.0, False, None, 5.0),
    "coverage_probe_height_m": ("float", 0.0, True, None, 1.5),
    "coverage_probe_rx_gain_dbi": ("float", 0.0, True, None, None),
    "coverage_sinr_db": ("float", None, True, None, -6.7),
    "balanced_core_fraction": ("float", 0.0, True, 1.0, 0.5),
}
BASELINE_KEYS = ("algorithm", "objective", "application", "movable_nodes", "seed",
                 "max_iterations", "waypoint_policy", "mapping_file", *NUMERIC_KEYS)
REMOVED_KEYS = {
    "gateway_node_id": "removed: baselines are gateway-free; delete the key",
    "rf_config": "removed: candidates are scored by the simulator channel; delete the key",
}
# Mirrors the [scenario] seed and run_id defaults in src/config/config-loader.cc.
DEFAULT_SCENARIO_SEED = 42
DEFAULT_RUN_ID = 1

RECT_KEYS = ("x_min", "x_max", "y_min", "y_max")
# Mirrors the C++ RlConfig fallbacks for bounds that are not written in the INI.
RL_BOUND_DEFAULTS = {"x_min": -1000.0, "x_max": 2000.0, "y_min": -1000.0, "y_max": 1000.0}

MAPPING_VERSION = 2
GEOFENCE_SOURCES = ("rl_bounds", "rectangle_xy_m")
_MAPPING_KEYS = {"baseline_mapping_version", "geofence", "platforms"}
_MAPPING_REQUIRED = {"baseline_mapping_version", "geofence"}
_MAPPING_REMOVED = {"origin", "ground_datum", "radios"}
MAPPING_V1_MIGRATION = (
    "baseline_mapping_version 1 is no longer read; migrate to version 2: delete "
    "origin, ground_datum and radios (baselines are gateway-free and scored by the "
    "simulator channel), keep geofence and platforms, and set baseline_mapping_version "
    "to 2")
_POLYGON_KEYS = {"vertices", "polygon", "polygons", "points", "coordinates", "rings",
                 "holes", "geojson", "exterior", "interiors"}
POLYGON_TODO = ("only axis-aligned rectangle geofences are supported "
                "(geofence.source 'rl_bounds' or 'rectangle_xy_m'); polygon geofences "
                "are a documented TODO, and a bounding rectangle is never substituted")

# Keeps [DEFAULT] an ordinary section, as in the C++ parser; no header can contain '\n'.
_NO_DEFAULT_SECTION = "\n"
_UNSIGNED = re.compile(r"[0-9]+")
_DECIMAL = re.compile(r"[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?")


class ConfigError(ValueError):
    """Invalid [baseline] section, mapping file, or INI structure."""


@dataclass(frozen=True)
class BaselineConfig:
    """Validated [baseline] values; paths are absolute and not yet checked for existence."""

    run_config: Path
    algorithm: str = "none"
    objective: str | None = None
    application: str = "initial_positions"
    movable_nodes: tuple[str, ...] = ()
    seed: int | None = None
    max_iterations: int | None = None
    waypoint_policy: str = "reject"
    mapping_file: Path | None = None
    aerial_fixed_cost_m2: float = NUMERIC_KEYS["aerial_fixed_cost_m2"][4]
    ground_fixed_cost_m2: float = NUMERIC_KEYS["ground_fixed_cost_m2"][4]
    aerial_cost_m2_per_m: float = NUMERIC_KEYS["aerial_cost_m2_per_m"][4]
    ground_cost_m2_per_m: float = NUMERIC_KEYS["ground_cost_m2_per_m"][4]
    aerial_max_displacement_m: float | None = None
    ground_max_displacement_m: float | None = None
    candidate_grid_cells: int = NUMERIC_KEYS["candidate_grid_cells"][4]
    coverage_grid_cells: int = NUMERIC_KEYS["coverage_grid_cells"][4]
    grid_min_resolution_m: float = NUMERIC_KEYS["grid_min_resolution_m"][4]
    coverage_probe_height_m: float = NUMERIC_KEYS["coverage_probe_height_m"][4]
    coverage_probe_rx_gain_dbi: float | None = None
    coverage_sinr_db: float = NUMERIC_KEYS["coverage_sinr_db"][4]
    balanced_core_fraction: float = NUMERIC_KEYS["balanced_core_fraction"][4]
    present_keys: frozenset = field(default_factory=frozenset)

    @property
    def movable_all(self) -> bool:
        return self.movable_nodes == (ALL_NODES,)


@dataclass(frozen=True)
class Mapping:
    """Validated mapping file (or the defaults) with its geofence as a scenario-metre rectangle."""

    path: Path | None
    geofence_source: str
    rectangle: dict
    platforms: dict

    @property
    def geofence(self) -> dict:
        return {"source": self.geofence_source, **self.rectangle}


def read_ini(run_config: str | Path) -> configparser.ConfigParser:
    """Parse strictly: duplicates, ':' assignments, and continuation lines fail."""
    path = Path(run_config)
    if not path.is_file():
        raise ConfigError(f"run config not found: {path}")
    ini = configparser.ConfigParser(interpolation=None, strict=True, delimiters=("=",),
                                    default_section=_NO_DEFAULT_SECTION)
    ini.optionxform = str
    try:
        ini.read_string(path.read_text(encoding="utf-8"), source=str(path))
    except configparser.Error as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    for section in ini.sections():
        if section.strip() == "baseline" and section != "baseline":
            raise ConfigError(f"{path}: section header '[{section}]' has surrounding "
                              "whitespace; use '[baseline]'")
        for key, raw in ini.items(section, raw=True):
            if "\n" in raw:
                raise ConfigError(f"{path}: [{section}] {key} continues onto an indented "
                                  "line; the simulator reads each line separately")
    return ini


def ini_value(ini: configparser.ConfigParser, section: str, key: str) -> str | None:
    """Return the comment-stripped value, or None when absent or blank."""
    if not ini.has_option(section, key):
        return None
    value = strip_inline_comment(ini.get(section, key))
    return value or None


def _resolve(run_config: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else run_config.parent / path


def _unsigned(where: str, key: str, raw: str, minimum: int) -> int:
    if not _UNSIGNED.fullmatch(raw) or int(raw) < minimum:
        raise ConfigError(f"{where}: baseline.{key} must be an integer >= {minimum}, "
                          f"got {raw!r}")
    return int(raw)


def _choice(where: str, key: str, raw: str, allowed: tuple) -> str:
    if raw not in allowed:
        raise ConfigError(f"{where}: baseline.{key} must be one of {', '.join(allowed)}; "
                          f"got {raw!r}")
    return raw


def _node_id(where: str, key: str, raw: str) -> str:
    if not raw or raw == ALL_NODES or "," in raw or raw != raw.strip():
        raise ConfigError(f"{where}: baseline.{key} has an invalid node id {raw!r}")
    return raw


def _numeric(where: str, key: str, raw: str):
    kind, low, inclusive, high, _ = NUMERIC_KEYS[key]
    pattern = _UNSIGNED if kind == "int" else _DECIMAL
    value = (int if kind == "int" else float)(raw) if pattern.fullmatch(raw) else None
    if value is None or (kind == "float" and not math.isfinite(value)):
        raise ConfigError(f"{where}: baseline.{key} must be a finite "
                          f"{'integer' if kind == 'int' else 'number'}, got {raw!r}")
    if low is not None and (value < low if inclusive else value <= low):
        hint = "; omit the key for no cap" if key.endswith("_max_displacement_m") else ""
        raise ConfigError(f"{where}: baseline.{key} must be {'>=' if inclusive else '>'} "
                          f"{low}, got {raw!r}{hint}")
    if high is not None and value > high:
        raise ConfigError(f"{where}: baseline.{key} must be <= {high}, got {raw!r}")
    return value


def _movable(where: str, raw: str) -> tuple[str, ...]:
    tokens = [token.strip() for token in raw.split(",")]
    if tokens == [ALL_NODES]:
        return (ALL_NODES,)
    ids = tuple(_node_id(where, "movable_nodes", token) for token in tokens)
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ConfigError(f"{where}: baseline.movable_nodes lists "
                          f"{', '.join(duplicates)} more than once")
    return ids


def load_baseline(run_config: str | Path) -> BaselineConfig:
    """Read and validate [baseline]; an absent section or blank algorithm means none."""
    path = Path(run_config).resolve()
    ini = read_ini(path)
    where = str(path)
    if not ini.has_section("baseline"):
        return BaselineConfig(run_config=path)
    unknown = sorted(set(ini.options("baseline")) - set(BASELINE_KEYS))
    if unknown:
        named = [f"{key} ({REMOVED_KEYS[key]})" if key in REMOVED_KEYS else key
                 for key in unknown]
        raise ConfigError(f"{where}: unknown [baseline] key(s): {', '.join(named)}; "
                          f"allowed: {', '.join(BASELINE_KEYS)}")
    raw = {key: ini_value(ini, "baseline", key) for key in BASELINE_KEYS}
    values: dict = {"present_keys": frozenset(k for k, v in raw.items() if v is not None)}
    values["algorithm"] = _choice(where, "algorithm", raw["algorithm"] or "none", ALGORITHMS)
    if raw["objective"] is not None:
        values["objective"] = _choice(where, "objective", raw["objective"], OBJECTIVES)
    if raw["application"] is not None:
        values["application"] = _choice(where, "application", raw["application"],
                                        APPLICATIONS)
    if raw["waypoint_policy"] is not None:
        values["waypoint_policy"] = _choice(where, "waypoint_policy",
                                            raw["waypoint_policy"], WAYPOINT_POLICIES)
    if raw["movable_nodes"] is not None:
        values["movable_nodes"] = _movable(where, raw["movable_nodes"])
    if raw["seed"] is not None:
        values["seed"] = _unsigned(where, "seed", raw["seed"], 0)
    if raw["max_iterations"] is not None:
        values["max_iterations"] = _unsigned(where, "max_iterations",
                                             raw["max_iterations"], 1)
    if raw["mapping_file"] is not None:
        values["mapping_file"] = _resolve(path, raw["mapping_file"])
    for key in NUMERIC_KEYS:
        if raw[key] is not None:
            values[key] = _numeric(where, key, raw[key])
    return BaselineConfig(run_config=path, **values)


def require_method(cfg: BaselineConfig, method: str) -> None:
    """Check that every key the effective method needs is present."""
    if method not in ALGORITHMS:
        raise ConfigError(f"unknown placement method {method!r}; expected one of "
                          f"{', '.join(ALGORITHMS)}")
    if method == "none":
        return
    required = ["objective", "movable_nodes"]
    if method == "optimization":
        required += ["seed", "max_iterations"]
    missing = [key for key in required if key not in cfg.present_keys]
    if missing:
        raise ConfigError(f"{cfg.run_config}: method '{method}' requires [baseline] "
                          f"{', '.join(missing)}")


def scenario_int(ini: configparser.ConfigParser, key: str, default: int) -> int:
    """A non-negative integer [scenario] key such as seed or run_id, with its C++ default."""
    value = ini_value(ini, "scenario", key)
    if value is None:
        return default
    if not _UNSIGNED.fullmatch(value):
        raise ConfigError(f"[scenario] {key} is not a non-negative integer: {value!r}")
    return int(value)


def scenario_file(ini: configparser.ConfigParser, run_config: str | Path, key: str,
                  default: str | None = None) -> Path | None:
    """Resolve a [scenario] file key against the INI directory; blank or absent means none."""
    raw = ini_value(ini, "scenario", key) or default
    return _resolve(Path(run_config).resolve(), raw) if raw else None


def explicit_rl_bounds(ini: configparser.ConfigParser) -> dict:
    """Return only the [rl] x/y bounds written in the INI, as floats."""
    bounds = {}
    for key in RECT_KEYS:
        raw = ini_value(ini, "rl", key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigError(f"rl.{key} is not a number: {raw!r}") from exc
        if not math.isfinite(value):
            raise ConfigError(f"rl.{key} must be finite, got {raw!r}")
        bounds[key] = value
    return bounds


def effective_rl_bounds(ini: configparser.ConfigParser) -> dict:
    """Return the x/y bounds the simulator enforces, with its fallbacks for absent keys."""
    return {**RL_BOUND_DEFAULTS, **explicit_rl_bounds(ini)}


def rl_controlled_nodes(ini: configparser.ConfigParser) -> tuple[str, ...] | None:
    """Tokens of [rl] controlled_nodes, or None for legacy single-node control."""
    if not ini.has_option("rl", "controlled_nodes"):
        return None
    raw = strip_inline_comment(ini.get("rl", "controlled_nodes"))
    return tuple(token.strip() for token in raw.split(",") if token.strip())


def _object(where: str, value, allowed: set, required: set) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a JSON object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{where} has unknown key(s): {', '.join(unknown)}")
    missing = sorted(required - set(value))
    if missing:
        raise ConfigError(f"{where} is missing key(s): {', '.join(missing)}")
    return value


def _number(where: str, value, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ConfigError(f"{where} must be finite")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ConfigError(f"{where} must be within [{low}, {high}], got {value}")
    return value


def _rectangle(where: str, source: dict) -> dict:
    rect = {key: _number(f"{where}.{key}", source[key]) for key in RECT_KEYS}
    if rect["x_min"] >= rect["x_max"] or rect["y_min"] >= rect["y_max"]:
        raise ConfigError(f"{where} must have x_min < x_max and y_min < y_max")
    return rect


def _geofence(where: str, value, ini: configparser.ConfigParser) -> tuple[str, dict]:
    if isinstance(value, list):
        raise ConfigError(f"{where} is a point list: {POLYGON_TODO}")
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a JSON object")
    source = value.get("source")
    polygonal = sorted(set(value) & _POLYGON_KEYS)
    if polygonal or (isinstance(source, str) and source not in GEOFENCE_SOURCES
                     and any(word in source.lower()
                             for word in ("poly", "geojson", "vert", "point"))):
        raise ConfigError(f"{where} requests a polygon "
                          f"({', '.join(polygonal) or repr(source)}): {POLYGON_TODO}")
    if source not in GEOFENCE_SOURCES:
        raise ConfigError(f"{where}.source must be one of {', '.join(GEOFENCE_SOURCES)}, "
                          f"got {source!r}")
    if source == "rl_bounds":
        _object(where, value, {"source"}, {"source"})
        bounds = explicit_rl_bounds(ini)
        missing = [key for key in RECT_KEYS if key not in bounds]
        if missing:
            raise ConfigError(f"{where}.source 'rl_bounds' needs [rl] "
                              f"{', '.join(missing)} written in the INI; loader defaults "
                              "are not used")
        return source, _rectangle("[rl] bounds", bounds)
    _object(where, value, {"source", *RECT_KEYS}, {"source", *RECT_KEYS})
    return source, _rectangle(where, value)


def default_mapping(ini: configparser.ConfigParser) -> Mapping:
    """No mapping file: the [rl] bounds geofence and platforms by node_type."""
    source, rectangle = _geofence("default geofence", {"source": "rl_bounds"}, ini)
    return Mapping(path=None, geofence_source=source, rectangle=rectangle, platforms={})


def load_mapping(path: str | Path | None, ini: configparser.ConfigParser) -> Mapping:
    """Parse a baseline_mapping_version 2 file; None gives default_mapping()."""
    if path is None:
        return default_mapping(ini)
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"mapping file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"mapping file {path} is not valid JSON: {exc}") from exc
    where = f"mapping file {path}"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a JSON object")
    version = raw.get("baseline_mapping_version")
    if not isinstance(version, bool) and version == 1:
        raise ConfigError(f"{where}: {MAPPING_V1_MIGRATION}")
    if isinstance(version, bool) or version != MAPPING_VERSION:
        raise ConfigError(f"{where}: baseline_mapping_version must be {MAPPING_VERSION}, "
                          f"got {version!r}")
    removed = sorted(set(raw) & _MAPPING_REMOVED)
    if removed:
        raise ConfigError(f"{where}: {', '.join(removed)} removed in mapping version 2; "
                          "baselines are gateway-free and scored by the simulator channel, "
                          "so delete them")
    _object(where, raw, _MAPPING_KEYS, _MAPPING_REQUIRED)
    source, rectangle = _geofence(f"{where}: geofence", raw["geofence"], ini)
    platforms_raw = _object(f"{where}: platforms", raw.get("platforms", {}), {"nodes"}, set())
    platform_nodes = platforms_raw.get("nodes", {})
    if not isinstance(platform_nodes, dict):
        raise ConfigError(f"{where}: platforms.nodes must be an object of node id -> platform")
    for node, platform in platform_nodes.items():
        if platform not in PLATFORMS:
            raise ConfigError(f"{where}: platforms.nodes.{node} must be one of "
                              f"{', '.join(PLATFORMS)}, got {platform!r}")
    return Mapping(path=path, geofence_source=source, rectangle=rectangle,
                   platforms=dict(platform_nodes))
