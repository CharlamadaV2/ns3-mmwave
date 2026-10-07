"""Node records, penalty/grid/probe resolution, plan validation, and prepare()."""

import importlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from scripts.baselines import artifacts, config, effective_inputs
from scripts.baselines.config import PLACEMENT_METHODS, PLATFORMS, ConfigError

MODES = ("standalone", "evaluation")
EXECUTORS = {"standalone": "none", "evaluation": "hold"}
MODE_FLAGS = {"standalone": None, "evaluation": "--rl-mode"}
ROUND_TRIP_TOL_M = 0.01
Z_TOL_M = 1e-6
PLATFORM_BY_NODE_TYPE = {"drone": "aerial", "vehicle": "ground", "pedestrian": "ground"}
MOBILITIES = ("fixed", "constant_velocity", "random_walk", "waypoint")
# Mirrors RandomWalkParams defaults in src/domain/node-spec.h.
RANDOM_WALK_DEFAULTS = {"x_min": -100.0, "x_max": 100.0, "y_min": -100.0, "y_max": 100.0}
DEFAULT_BAND = "mmwave"
EVAL_MANIFEST_NAME = "eval_manifest.json"


class BaselinePreparationError(RuntimeError):
    """Placement preparation failed; the baseline manifest records status failed."""


@dataclass(frozen=True)
class NodeRecord:
    """One nodes.json entry with its planner role (movable or fixed) and movement platform."""

    id: str
    roster_index: int
    node_type: str
    mobility: str
    role: str
    platform: str
    platform_source: str
    x: float
    y: float
    z: float
    random_walk_bounds: dict | None = None
    has_waypoints: bool = False
    slot: int | None = None

    @property
    def selected(self) -> bool:
        return self.role == "movable"


@dataclass(frozen=True)
class PreparedBaseline:
    """Paths and the eval_manifest `baseline` block for one prepared method."""

    prep_dir: Path
    mode: str
    method: str
    manifest_path: Path
    plan_path: Path
    effective_run_config: Path
    metadata: dict


def _solver():
    return importlib.import_module("scripts.baselines.solver")


def _objective():
    return importlib.import_module("scripts.baselines.planners.objective")


def load_nodes(path: str | Path) -> list:
    """Read nodes.json as a list of objects with unique string ids."""
    try:
        nodes = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"nodes file {path} is not valid JSON: {exc}") from exc
    if not isinstance(nodes, list) or not all(isinstance(n, dict) for n in nodes):
        raise ConfigError(f"nodes file {path} must be a JSON array of objects")
    ids = [n.get("id") for n in nodes]
    if not all(isinstance(i, str) and i for i in ids):
        raise ConfigError(f"nodes file {path}: every node needs a non-empty string id")
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ConfigError(f"nodes file {path} repeats node id(s) {', '.join(duplicates)}")
    return nodes


def _require_ids(label: str, ids, known: set) -> None:
    unknown = sorted(set(ids) - known)
    if unknown:
        raise ConfigError(f"{label} names unknown node id(s): {', '.join(unknown)}")


def _slot(node_id: str, index: int, controlled: tuple | None) -> int | None:
    if controlled is None:
        return None
    if controlled == (config.ALL_NODES,):
        return index
    return controlled.index(node_id) if node_id in controlled else None


def resolve_ids(ids: tuple, roster: list) -> list:
    """`all` -> the whole roster; otherwise the given ids in roster order."""
    if ids == (config.ALL_NODES,):
        return list(roster)
    return [node_id for node_id in roster if node_id in ids]


def _random_walk_bounds(entry: dict) -> dict:
    bounds = (entry.get("random_walk") or {}).get("bounds") or {}
    return {key: float(bounds.get(key, default))
            for key, default in RANDOM_WALK_DEFAULTS.items()}


def build_records(nodes: list, cfg: config.BaselineConfig, mapping: config.Mapping,
                  controlled: tuple | None = None) -> tuple:
    """Map every node to a role, platform, start position, bounds, and RL slot."""
    known = {n["id"] for n in nodes}
    if not cfg.movable_all:
        _require_ids("baseline.movable_nodes", cfg.movable_nodes, known)
    _require_ids("mapping platforms.nodes", mapping.platforms, known)
    movable = set(resolve_ids(cfg.movable_nodes, [n["id"] for n in nodes]))
    records = []
    for index, entry in enumerate(nodes):
        node_id = entry["id"]
        mobility = entry.get("mobility", "fixed")
        if mobility not in MOBILITIES:
            raise ConfigError(f"node '{node_id}' has unsupported mobility {mobility!r}")
        node_type = entry.get("node_type", "drone")
        if node_id in mapping.platforms:
            platform, platform_source = mapping.platforms[node_id], "mapping"
        elif node_type in PLATFORM_BY_NODE_TYPE:
            platform, platform_source = PLATFORM_BY_NODE_TYPE[node_type], "node_type"
        else:
            raise ConfigError(f"node '{node_id}' has node_type {node_type!r}; set its "
                              "platform in a mapping file (platforms.nodes)")
        x, y, z = effective_inputs.start_position(entry)
        if not all(math.isfinite(v) for v in (x, y, z)):
            raise ConfigError(f"node '{node_id}' has a non-finite start position")
        records.append(NodeRecord(
            id=node_id, roster_index=index, node_type=node_type, mobility=mobility,
            role="movable" if node_id in movable else "fixed", platform=platform,
            platform_source=platform_source, x=x, y=y, z=z,
            random_walk_bounds=(_random_walk_bounds(entry) if mobility == "random_walk"
                                else None),
            has_waypoints=mobility == "waypoint" and bool(entry.get("waypoints")),
            slot=_slot(node_id, index, controlled)))
    return tuple(records)


def check_datum(records: tuple) -> None:
    """Accept only z as height above flat ground at z = 0."""
    for record in records:
        if record.z < 0:
            raise ConfigError(f"node '{record.id}' has z={record.z} below the ground "
                              "datum; z is height above flat ground at z = 0 and must be "
                              ">= 0")


def check_ownership(cfg: config.BaselineConfig, controlled: tuple | None,
                    roster: list) -> list:
    """Evaluation: resolved movable_nodes must equal resolved [rl] controlled_nodes."""
    if controlled is None:
        raise ConfigError("placement evaluation needs [rl] controlled_nodes (centralized "
                          "control); legacy single-node control is not supported")
    known = set(roster)
    if not cfg.movable_all:
        _require_ids("baseline.movable_nodes", cfg.movable_nodes, known)
    if controlled != (config.ALL_NODES,):
        _require_ids("[rl] controlled_nodes", controlled, known)
        duplicates = sorted({i for i in controlled if controlled.count(i) > 1})
        if duplicates:
            raise ConfigError(f"[rl] controlled_nodes lists {', '.join(duplicates)} more "
                              "than once")
    movable = set(resolve_ids(cfg.movable_nodes, roster))
    controlled_ids = _controlled_ids(controlled, roster)
    not_controlled = [node for node in roster if node in movable
                      and node not in controlled_ids]
    not_movable = [node for node in roster if node in controlled_ids
                   and node not in movable]
    if not_controlled or not_movable:
        raise ConfigError(
            "placement evaluation holds every RL slot, so [baseline] movable_nodes must "
            "equal [rl] controlled_nodes; movable but not controlled: "
            f"{', '.join(not_controlled) or '-'}; controlled but not movable: "
            f"{', '.join(not_movable) or '-'}. Set [baseline] movable_nodes and [rl] "
            "controlled_nodes to the same ids, or `all`")
    return controlled_ids


def evaluation_region(rectangle: dict, rl_bounds: dict) -> dict:
    """Evaluation placement area: the geofence rectangle intersected with the [rl] x/y bounds."""
    region = {"x_min": max(rectangle["x_min"], rl_bounds["x_min"]),
              "x_max": min(rectangle["x_max"], rl_bounds["x_max"]),
              "y_min": max(rectangle["y_min"], rl_bounds["y_min"]),
              "y_max": min(rectangle["y_max"], rl_bounds["y_max"])}
    if region["x_min"] >= region["x_max"] or region["y_min"] >= region["y_max"]:
        raise ConfigError(f"the geofence rectangle {rectangle} and the [rl] x/y bounds "
                          f"{rl_bounds} do not overlap; evaluation places nodes only "
                          "inside both")
    return region


def resolve_penalties(cfg: config.BaselineConfig) -> dict:
    """Platform -> objective.MovementCost from the [baseline] cost keys."""
    movement_cost = _objective().MovementCost
    return {platform: movement_cost(
                fixed_cost_m2=getattr(cfg, f"{platform}_fixed_cost_m2"),
                cost_m2_per_m=getattr(cfg, f"{platform}_cost_m2_per_m"),
                max_displacement_m=getattr(cfg, f"{platform}_max_displacement_m"))
            for platform in PLATFORMS}


def penalties_block(penalties: dict, aoi_m2: float) -> dict:
    """Manifest `penalties`: per-platform costs plus the AOI scale check."""
    block = {platform: {"fixed_cost_m2": cost.fixed_cost_m2,
                        "cost_m2_per_m": cost.cost_m2_per_m,
                        "max_displacement_m": cost.max_displacement_m}
             for platform, cost in penalties.items()}
    block["aoi_m2"] = aoi_m2
    block["fixed_cost_over_aoi"] = {platform: penalties[platform].fixed_cost_m2 / aoi_m2
                                    for platform in PLATFORMS}
    return block


def _area(rect: dict) -> float:
    return (rect["x_max"] - rect["x_min"]) * (rect["y_max"] - rect["y_min"])


def _inside(x: float, y: float, rect: dict) -> bool:
    return rect["x_min"] <= x <= rect["x_max"] and rect["y_min"] <= y <= rect["y_max"]


def validate_result(records: tuple, result, rectangle: dict, penalties: dict,
                    rl_bounds: dict | None = None) -> dict:
    """Check the plan and return id -> final (x, y, z); fails on any violation, never clamps."""
    positions = result.positions
    expected = {record.id for record in records}
    missing = sorted(expected - set(positions))
    extra = sorted(set(positions) - expected)
    if missing or extra:
        raise ValueError(f"planner result ids do not match the scenario (missing: "
                         f"{', '.join(missing) or '-'}; unexpected: {', '.join(extra) or '-'})")
    final = {}
    for record in records:
        values = positions[record.id]
        if len(values) != 3 or not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f"planner returned a non-finite position for '{record.id}'")
        x, y, z = (float(v) for v in values)
        if abs(z - record.z) > Z_TOL_M:
            raise ValueError(f"planner changed z of '{record.id}' from {record.z} to {z}")
        moved = math.hypot(x - record.x, y - record.y)
        if not record.selected:
            if moved > ROUND_TRIP_TOL_M:
                raise ValueError(f"planner moved {record.role} node '{record.id}' by "
                                 f"{moved:.6f} m; only movable_nodes may move")
            final[record.id] = (record.x, record.y, record.z)
            continue
        if moved <= ROUND_TRIP_TOL_M:
            x, y, moved = record.x, record.y, 0.0
        if not _inside(x, y, rectangle):
            raise ValueError(f"planned position of '{record.id}' ({x}, {y}) is outside the "
                             f"declared geofence rectangle {rectangle}")
        cap = penalties[record.platform].max_displacement_m
        if cap is not None and moved > cap + ROUND_TRIP_TOL_M:
            raise ValueError(f"'{record.id}' moved {moved:.3f} m, beyond [baseline] "
                             f"{record.platform}_max_displacement_m {cap}")
        if rl_bounds is not None and not _inside(x, y, rl_bounds):
            raise ValueError(f"planned position of '{record.id}' ({x}, {y}) is outside the "
                             f"[rl] x/y bounds {rl_bounds}")
        if record.random_walk_bounds is not None and not _inside(x, y,
                                                                 record.random_walk_bounds):
            raise ValueError(f"planned position of '{record.id}' ({x}, {y}) is outside "
                             f"its random_walk bounds {record.random_walk_bounds}")
        final[record.id] = (x, y, record.z)
    return final


def _xyz(values) -> dict:
    return {"x": values[0], "y": values[1], "z": values[2]}


def plan_nodes(records: tuple, final: dict) -> list:
    """baseline-plan.json node entries."""
    entries = []
    for record in records:
        planned = final[record.id]
        entries.append({
            "id": record.id, "roster_index": record.roster_index, "slot": record.slot,
            "role": record.role, "platform": record.platform, "selected": record.selected,
            "original": _xyz((record.x, record.y, record.z)), "planned": _xyz(planned),
            "displacement_m": math.hypot(planned[0] - record.x, planned[1] - record.y),
        })
    return entries


def _unplanned_nodes(nodes: list) -> list:
    entries = []
    for index, entry in enumerate(nodes):
        start = effective_inputs.start_position(entry)
        entries.append({"id": entry["id"], "roster_index": index, "slot": None,
                        "role": "fixed", "platform": None, "selected": False,
                        "original": _xyz(start), "planned": _xyz(start),
                        "displacement_m": 0.0})
    return entries


def resolve_band(ini, band: str | None) -> tuple[str, str]:
    """Effective band and its source, in the simulator's order: CLI, run.ini, default."""
    ini_band = config.ini_value(ini, "channel", "band") if ini.has_section("channel") else None
    if band:
        return band, "cli"
    if ini_band:
        return ini_band, "run.ini"
    return DEFAULT_BAND, "default"


def planning_identity(ini, simulation_seeds: list | None) -> tuple[int, int]:
    """(planning seed, run_id): the first simulation seed, else [scenario] seed."""
    if simulation_seeds:
        seed = simulation_seeds[0]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ConfigError(f"simulation seed {seed!r} is not a non-negative integer")
    else:
        seed = config.scenario_int(ini, "seed", config.DEFAULT_SCENARIO_SEED)
    return seed, config.scenario_int(ini, "run_id", config.DEFAULT_RUN_ID)


def _controlled_ids(controlled: tuple | None, roster: list) -> list | None:
    if controlled is None:
        return None
    return list(roster) if controlled == (config.ALL_NODES,) else list(controlled)


def _seed_fields(method: str, cfg: config.BaselineConfig) -> dict:
    if method == "optimization":
        return {"planner_seed": cfg.seed, "planner_seed_status": "used",
                "max_iterations": cfg.max_iterations, "max_iterations_status": "used"}
    return {"planner_seed": None,
            "planner_seed_status": "unused" if method == "geometric" else "not_applicable",
            "max_iterations": None, "max_iterations_status": "not_applicable"}


def _resolve_method(mode: str, method: str | None, cfg: config.BaselineConfig) -> str:
    if mode == "evaluation":
        if method not in PLACEMENT_METHODS:
            raise ConfigError(f"evaluation placement method must be one of "
                              f"{', '.join(PLACEMENT_METHODS)}, got {method!r}")
        return method
    resolved = method or cfg.algorithm
    if resolved not in config.ALGORITHMS:
        raise ConfigError(f"unknown placement method {resolved!r}; expected one of "
                          f"{', '.join(config.ALGORITHMS)}")
    return resolved


def _require_binary(method: str, sim_binary) -> Path:
    if sim_binary is None:
        raise ConfigError(f"method '{method}' needs the simulator binary (sim_binary / "
                          "--sim-binary): candidates are scored by its --channel-query mode")
    path = Path(sim_binary)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ConfigError(f"simulator binary {path} is not an executable file")
    return path.resolve()


def prepare(run_config: str | Path, method: str | None, prep_dir: str | Path,
            mode: str = "evaluation", *, band: str | None = None,
            simulation_seeds: list | None = None, sim_binary: str | Path | None = None,
            eval_root: str | Path | None = None) -> PreparedBaseline:
    """Validate, plan once, and write the manifest, snapshots, and effective inputs."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    prep_dir = Path(prep_dir)
    if prep_dir.exists() and (not prep_dir.is_dir() or any(prep_dir.iterdir())):
        raise BaselinePreparationError(f"preparation directory {prep_dir} is not empty")
    prep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = prep_dir / artifacts.MANIFEST_NAME
    artifacts.write_json(manifest_path, artifacts.new_manifest(
        mode, EXECUTORS[mode], method, None))
    log_path = prep_dir / artifacts.PLANNER_LOG_NAME
    try:
        return _prepare(Path(run_config), method, prep_dir, mode, manifest_path, log_path,
                        band, simulation_seeds, sim_binary, eval_root)
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        if log_path.exists():
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"FAILED: {message}\n")
        _record_end(manifest_path, "failed", message)
        raise BaselinePreparationError(message) from exc
    except BaseException:
        _record_end(manifest_path, "interrupted", "interrupted")
        raise


def _record_end(manifest_path: Path, status: str, error: str) -> None:
    if artifacts.read_json(manifest_path).get("status") == "preparing":
        artifacts.set_status(manifest_path, status, error=error)


@dataclass(frozen=True)
class _Setup:
    """Validated planning inputs for an active method, before staging."""

    records: tuple
    rectangle: dict
    rl_bounds: dict | None
    region: dict | None
    penalties: dict
    grid: object
    probe: object
    planning_seed: int
    run_id: int
    aoi_m2: float


def _setup(cfg: config.BaselineConfig, mode: str, ini, files, nodes: list,
           band: str | None, simulation_seeds, manifest_path: Path) -> _Setup:
    mapping = config.load_mapping(files.mapping, ini)
    controlled = config.rl_controlled_nodes(ini)
    roster = [entry["id"] for entry in nodes]
    rectangle = dict(mapping.rectangle)
    rl_bounds = region = None
    if mode == "evaluation":
        check_ownership(cfg, controlled, roster)
        rl_bounds = config.effective_rl_bounds(ini)
        region = evaluation_region(rectangle, rl_bounds)
    records = build_records(nodes, cfg, mapping, controlled if mode == "evaluation" else None)
    check_datum(records)
    if cfg.waypoint_policy == "reject":
        for record in records:
            if record.selected and record.mobility == "waypoint":
                raise ConfigError(f"node '{record.id}' uses waypoint mobility and is "
                                  "selected; set [baseline] waypoint_policy = "
                                  "translate or leave it out of movable_nodes")
    objective = _objective()
    penalties = resolve_penalties(cfg)
    grid = objective.GridSettings(candidate_cells=cfg.candidate_grid_cells,
                                  coverage_cells=cfg.coverage_grid_cells,
                                  min_resolution_m=cfg.grid_min_resolution_m)
    probe = objective.ProbeSettings(height_m=cfg.coverage_probe_height_m,
                                    rx_gain_dbi=cfg.coverage_probe_rx_gain_dbi,
                                    sinr_db=cfg.coverage_sinr_db)
    aoi_m2 = _area(rectangle)
    planning_seed, run_id = planning_identity(ini, simulation_seeds)
    band_value, band_source = resolve_band(ini, band)
    probe_points, probe_cell, _ = objective.rectangle_grid(
        rectangle, cfg.coverage_grid_cells, cfg.grid_min_resolution_m)
    candidate_points, candidate_cell, _ = objective.rectangle_grid(
        rectangle, cfg.candidate_grid_cells, cfg.grid_min_resolution_m)
    artifacts.update_manifest(
        manifest_path, geofence=mapping.geofence,
        penalties=penalties_block(penalties, aoi_m2),
        ownership={"mode": mode,
                   "movable_resolved": resolve_ids(cfg.movable_nodes, roster),
                   "controlled_resolved": (_controlled_ids(controlled, roster)
                                           if mode == "evaluation" else None)},
        channel_scoring={
            "contract": None, "isolation": None, "planning_seed": planning_seed,
            "planning_run_id": run_id, "jammer_seed": planning_seed,
            "mode_flag": MODE_FLAGS[mode], "band": band_value, "band_source": band_source,
            "sinr_threshold_db": None, "coverage_sinr_db": cfg.coverage_sinr_db,
            "probe": {"height_m": cfg.coverage_probe_height_m,
                      "rx_gain_dbi": cfg.coverage_probe_rx_gain_dbi,
                      "grid_cells": cfg.coverage_grid_cells,
                      "min_resolution_m": cfg.grid_min_resolution_m,
                      "cell_m": float(probe_cell), "count": len(probe_points)},
            "candidate_grid": {"cells": cfg.candidate_grid_cells,
                               "min_resolution_m": cfg.grid_min_resolution_m,
                               "cell_m": float(candidate_cell),
                               "count": len(candidate_points)},
            "queries": None},
        planner_code_sha256=artifacts.planner_code_identity()["aggregate_sha256"])
    return _Setup(records=records, rectangle=rectangle, rl_bounds=rl_bounds, region=region,
                  penalties=penalties, grid=grid, probe=probe, planning_seed=planning_seed,
                  run_id=run_id, aoi_m2=aoi_m2)


def _log_scale(log, setup: _Setup) -> None:
    log.write(f"aoi_m2={setup.aoi_m2:.1f}\n")
    for platform in PLATFORMS:
        fixed = setup.penalties[platform].fixed_cost_m2
        ratio = fixed / setup.aoi_m2
        log.write(f"{platform} fixed_cost_m2/aoi_m2={ratio:.4f}\n")
        if ratio >= 1.0:
            log.write(f"WARNING: {platform}_fixed_cost_m2 {fixed} >= aoi_m2 "
                      f"{setup.aoi_m2:.1f}; no coverage gain can pay for moving a "
                      f"{platform} node, so it will stay put\n")


def _scoring_result(manifest_path: Path, result) -> None:
    manifest = artifacts.read_json(manifest_path)
    scoring = manifest["channel_scoring"]
    facts = result.channel or {}
    if facts.get("band") is not None and facts["band"] != scoring["band"]:
        raise ValueError(f"the channel query resolved band {facts['band']!r}, but this run "
                         f"resolves {scoring['band']!r}")
    scoring.update(contract=facts.get("contract"), isolation=facts.get("isolation"),
                   sinr_threshold_db=facts.get("sinr_threshold_db"),
                   queries=dict(result.query_stats))
    if facts.get("probe_rx_gain_dbi") is not None:
        scoring["probe"]["rx_gain_dbi"] = facts["probe_rx_gain_dbi"]
    artifacts.update_manifest(manifest_path, channel_scoring=scoring,
                              planner_settings=result.planner_settings)


def _prepare(run_config: Path, method: str | None, prep_dir: Path, mode: str,
             manifest_path: Path, log_path: Path, band, simulation_seeds, sim_binary,
             eval_root) -> PreparedBaseline:
    cfg = config.load_baseline(run_config)
    method = _resolve_method(mode, method, cfg)
    config.require_method(cfg, method)
    ini = config.read_ini(cfg.run_config)
    files = effective_inputs.scenario_files(ini, cfg)
    nodes = load_nodes(files.nodes)
    active = method in PLACEMENT_METHODS
    if active:
        sim_binary = _require_binary(method, sim_binary)
    artifacts.update_manifest(
        manifest_path, requested_algorithm=cfg.algorithm, method=method,
        objective=cfg.objective if active else None, application=cfg.application,
        waypoint_policy=cfg.waypoint_policy, source_run_config_abs=str(cfg.run_config),
        simulation_seeds=list(simulation_seeds) if simulation_seeds is not None else None,
        sim_binary_sha256=artifacts.sha256_file(sim_binary) if sim_binary else None,
        mapping_sha256=artifacts.sha256_file(files.mapping) if files.mapping else None,
        sim_log=artifacts.SIM_LOG_NAME if mode == "standalone" else None,
        **_seed_fields(method, cfg))

    setup = None
    if active:
        setup = _setup(cfg, mode, ini, files, nodes, band, simulation_seeds, manifest_path)
        runtime = artifacts.channel_runtime_identity(sim_binary)
        artifacts.update_manifest(manifest_path, channel_runtime_sha256=runtime["sha256"],
                                  channel_runtime_files=runtime["files"])

    source_identity = effective_inputs.snapshot_sources(files, prep_dir)
    artifacts.update_manifest(manifest_path, source_scenario_identity=source_identity,
                              source_inputs=effective_inputs.SOURCE_DIR)

    staged_ini = effective_inputs.stage_effective_inputs(files, prep_dir, mode)
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            artifacts.update_manifest(manifest_path, planner_log=artifacts.PLANNER_LOG_NAME)
            if not active:
                log.write("method none: no planner was run\n")
                plan_entries, predictions, moved, wall_s = (_unplanned_nodes(nodes), None,
                                                            None, None)
            else:
                plan_entries, predictions, moved, wall_s = _plan(
                    cfg, method, mode, band, setup, staged_ini, sim_binary, log,
                    manifest_path)
        plan = artifacts.build_plan(method, cfg.objective if active else None, plan_entries,
                                    predictions)
        rewritten = (effective_inputs.rewrite_nodes(nodes, moved, cfg.waypoint_policy)
                     if moved else None)
        effective_identity = effective_inputs.finish_effective_inputs(files, prep_dir,
                                                                      rewritten, plan)
    except BaseException:
        effective_inputs.discard_staged_inputs(prep_dir)
        raise
    plan_rel = f"{effective_inputs.EFFECTIVE_DIR}/{artifacts.PLAN_NAME}"
    fields = {
        "effective_scenario_identity": effective_identity,
        "effective_inputs": effective_inputs.EFFECTIVE_DIR,
        "plan": plan_rel,
        "initial_displacement_m_total": plan["initial_displacement_m_total"],
        "planner_wall_s": wall_s,
    }
    root = prep_dir
    if mode == "evaluation":
        root = Path(eval_root) if eval_root is not None else prep_dir.parent.parent
        fields["eval_manifest"] = Path(os.path.relpath(
            root.resolve() / EVAL_MANIFEST_NAME, prep_dir.resolve())).as_posix()
    manifest = artifacts.read_json(manifest_path)
    manifest.update(fields)
    fields["fingerprint"] = artifacts.fingerprint(manifest)
    manifest = artifacts.set_status(manifest_path, "prepared", final=mode == "evaluation",
                                    **fields)
    metadata = artifacts.eval_metadata(
        manifest, artifacts.run_relative(manifest_path, root),
        artifacts.run_relative(prep_dir / plan_rel, root))
    return PreparedBaseline(prep_dir=prep_dir, mode=mode, method=method,
                            manifest_path=manifest_path, plan_path=prep_dir / plan_rel,
                            effective_run_config=(prep_dir / effective_inputs.EFFECTIVE_DIR
                                                  / effective_inputs.RUN_INI),
                            metadata=metadata)


def _plan(cfg: config.BaselineConfig, method: str, mode: str, band, setup: _Setup,
          staged_ini: Path, sim_binary: Path, log, manifest_path: Path) -> tuple:
    solver = _solver()
    request = solver.PlanRequest(
        method=method, objective=cfg.objective, nodes=setup.records,
        rectangle=setup.rectangle, rl_bounds=setup.region, mode=mode,
        penalties=setup.penalties, grid=setup.grid, probe=setup.probe,
        planner_seed=cfg.seed if method == "optimization" else None,
        max_iterations=cfg.max_iterations if method == "optimization" else None,
        balanced_core_fraction=cfg.balanced_core_fraction,
        waypoint_policy=cfg.waypoint_policy, planning_seed=setup.planning_seed,
        run_id=setup.run_id, band=band, query_run_config=staged_ini, sim_binary=sim_binary)
    log.write(f"method={method} objective={cfg.objective} "
              f"planner_seed={request.planner_seed} max_iterations={request.max_iterations} "
              f"planning_seed={request.planning_seed} run_id={request.run_id} mode={mode}\n")
    if setup.region is not None:
        log.write(f"evaluation_region={json.dumps(setup.region, sort_keys=True)}\n")
    _log_scale(log, setup)
    log.flush()
    result = solver.solve(request, log)
    final = validate_result(setup.records, result, setup.rectangle, setup.penalties,
                            setup.rl_bounds)
    _scoring_result(manifest_path, result)
    plan_entries = plan_nodes(setup.records, final)
    moved = {entry["id"]: (entry["planned"]["x"], entry["planned"]["y"])
             for entry in plan_entries if entry["displacement_m"] > 0.0}
    for note in result.notes:
        log.write(f"note: {note}\n")
    log.write(f"queries={json.dumps(result.query_stats, sort_keys=True)}\n")
    log.write(f"planner_wall_s={result.planner_wall_s:.3f}\n")
    return plan_entries, result.predictions, moved, result.planner_wall_s
