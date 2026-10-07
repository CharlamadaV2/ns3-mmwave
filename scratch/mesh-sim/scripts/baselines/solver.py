"""Run one gateway-free placement planner with a channel-query scorer and package its result."""

import importlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from scripts.baselines.config import PLACEMENT_METHODS
from scripts.baselines.planners.channel import ChannelScorer, PlannerError

__all__ = ["PlanRequest", "PlanResult", "PlannerError", "solve"]

POSITION_AXES = 3


@dataclass(frozen=True)
class PlanRequest:
    """Everything a planner needs in scenario metres; nodes are NodeRecords in roster order."""

    method: str
    objective: str
    nodes: tuple
    rectangle: dict
    rl_bounds: dict | None
    mode: str
    penalties: dict
    grid: object
    probe: object
    planner_seed: int | None
    max_iterations: int | None
    balanced_core_fraction: float
    waypoint_policy: str
    planning_seed: int
    run_id: int
    band: str | None
    query_run_config: Path
    sim_binary: Path


@dataclass(frozen=True)
class PlanResult:
    """Planner output: node id -> (x, y, z), diagnostics, query statistics and channel facts."""

    positions: dict
    predictions: dict | None = None
    planner_wall_s: float = 0.0
    query_stats: dict = field(default_factory=dict)
    planner_settings: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    channel: dict = field(default_factory=dict)


def _strategy(method: str):
    if method not in PLACEMENT_METHODS:
        raise PlannerError(f"unknown placement method {method!r}; expected one of "
                           f"{', '.join(PLACEMENT_METHODS)}")
    return importlib.import_module(f"scripts.baselines.planners.{method}")


def _objective():
    return importlib.import_module("scripts.baselines.planners.objective")


def _channel_facts(scorer: ChannelScorer) -> dict:
    init = scorer.init
    return {
        "contract": init.get("contract"),
        "isolation": init.get("isolation"),
        "band": init.get("band"),
        "band_source": init.get("band_source"),
        "seed": init.get("seed"),
        "run_id": init.get("run_id"),
        "jammer_seed": init.get("jammer_seed"),
        "sinr_threshold_db": init.get("sinr_threshold_db"),
        "probe_rx_gain_dbi": scorer.probe_rx_gain_dbi,
    }


def _positions(request: PlanRequest, raw) -> dict:
    rows = [list(map(float, row)) for row in raw]
    if len(rows) != len(request.nodes) or any(len(row) != POSITION_AXES for row in rows):
        raise PlannerError(f"{request.method} planner returned {len(rows)} positions for "
                           f"{len(request.nodes)} nodes")
    return {record.id: tuple(row) for record, row in zip(request.nodes, rows)}


def solve(request: PlanRequest, log: TextIO) -> PlanResult:
    """Score candidates through one channel-query worker and return the planned layout."""
    strategy = _strategy(request.method)
    objective = _objective()
    started = time.perf_counter()
    with ChannelScorer(request.sim_binary, request.query_run_config, request.planning_seed,
                       request.run_id, request.band, request.mode,
                       [record.id for record in request.nodes],
                       [(record.x, record.y, record.z) for record in request.nodes],
                       jammer_seed=request.planning_seed, log=log) as scorer:
        objective.prepare_scorer(request, scorer)
        raw, diagnostics, notes = strategy.solve(request, scorer, log)
        stats = scorer.stats
        channel = _channel_facts(scorer)
    wall_s = time.perf_counter() - started
    return PlanResult(positions=_positions(request, raw), predictions=diagnostics,
                      planner_wall_s=wall_s, query_stats=stats,
                      planner_settings=strategy.settings(request), notes=list(notes),
                      channel=channel)
