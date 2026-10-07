"""Gateway-free placement objective: grids, components/core, coverage, vulnerability, costs."""

import math
from dataclasses import dataclass

import numpy as np

from scripts.baselines.planners.channel import LayoutResult, PlannerError

OBJECTIVES = ("coverage", "balanced", "resilience")
SEPARATION_FRAC = 0.4
BALANCED_VULNERABILITY_FRAC = 0.02
BIG_AOI_FACTOR = 2.0
STAY_PUT_TOL_M = 1e-6
PROBE_MATCH_TOL_M = 1e-9
# Absolute m^2 margin so float rounding between equal-valued layouts is never a "gain".
SCORE_TIE_TOL_M2 = 1e-6


@dataclass(frozen=True)
class MovementCost:
    """Coverage-equivalent m^2 cost of moving one node, plus an optional hard cap."""

    fixed_cost_m2: float
    cost_m2_per_m: float
    max_displacement_m: float | None = None

    def cost(self, displacement_m: float) -> float:
        if displacement_m <= STAY_PUT_TOL_M:
            return 0.0
        return self.fixed_cost_m2 + self.cost_m2_per_m * displacement_m

    def within_cap(self, displacement_m: float) -> bool:
        return self.max_displacement_m is None or displacement_m <= self.max_displacement_m


@dataclass(frozen=True)
class GridSettings:
    candidate_cells: int
    coverage_cells: int
    min_resolution_m: float


@dataclass(frozen=True)
class ProbeSettings:
    height_m: float
    rx_gain_dbi: float | None
    sinr_db: float


@dataclass(frozen=True)
class LayoutScore:
    coverage_m2: float
    sep_m2: float
    move_cost_m2: float
    disconnected: int
    vuln: int
    total: float

    def as_dict(self) -> dict:
        return {"coverage_m2": self.coverage_m2, "sep_m2": self.sep_m2,
                "move_cost_m2": self.move_cost_m2, "disconnected": self.disconnected,
                "vuln": self.vuln, "total": self.total}


def vulnerability_weight(objective: str, aoi_m2: float) -> float:
    """m^2 charged per vulnerability pair."""
    if objective == "resilience":
        return float(aoi_m2)
    if objective == "balanced":
        return BALANCED_VULNERABILITY_FRAC * float(aoi_m2)
    if objective == "coverage":
        return 0.0
    raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")


def BIG(aoi_m2: float) -> float:  # noqa: N802
    """Penalty per selected node outside the core."""
    return BIG_AOI_FACTOR * float(aoi_m2)


def rectangle_area(rect: dict) -> float:
    return (rect["x_max"] - rect["x_min"]) * (rect["y_max"] - rect["y_min"])


def inside(x: float, y: float, rect: dict) -> bool:
    return rect["x_min"] <= x <= rect["x_max"] and rect["y_min"] <= y <= rect["y_max"]


def rectangle_grid(rect: dict, cells: int, min_resolution_m: float):
    """Clipped-cell centres, cell size and per-cell clipped area; areas sum to the AOI."""
    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 1:
        raise ValueError(f"grid cells must be an integer >= 1, got {cells!r}")
    if not (math.isfinite(min_resolution_m) and min_resolution_m > 0):
        raise ValueError("grid min_resolution_m must be positive")
    x_min, x_max, y_min, y_max = (float(rect[k]) for k in ("x_min", "x_max", "y_min", "y_max"))
    width, height = x_max - x_min, y_max - y_min
    if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
        raise ValueError(f"degenerate rectangle {rect}")
    cell = max(float(min_resolution_m), math.sqrt(width * height / cells))

    def edges(lo: float, span: float) -> np.ndarray:
        count = max(1, math.ceil(span / cell - 1e-9))
        result = np.minimum(lo + cell * np.arange(count + 1), lo + span)
        result[-1] = lo + span
        return result

    xe, ye = edges(x_min, width), edges(y_min, height)
    xc, yc = (xe[:-1] + xe[1:]) / 2.0, (ye[:-1] + ye[1:]) / 2.0
    xx, yy = np.meshgrid(xc, yc)
    ww, hh = np.meshgrid(np.diff(xe), np.diff(ye))
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return points, cell, (ww * hh).ravel()


def components(connected: np.ndarray) -> list[frozenset[int]]:
    """Connected components ordered by their smallest index; the diagonal is ignored."""
    adjacency = np.asarray(connected, dtype=bool)
    n = len(adjacency)
    label = np.full(n, -1)
    result = []
    for root in range(n):
        if label[root] >= 0:
            continue
        label[root] = len(result)
        stack, members = [root], [root]
        while stack:
            node = stack.pop()
            for other in np.nonzero(adjacency[node])[0]:
                if label[other] < 0:
                    label[other] = len(result)
                    stack.append(int(other))
                    members.append(int(other))
        result.append(frozenset(members))
    return result


def core(parts: list[frozenset[int]]) -> frozenset[int]:
    """Largest component; ties go to the one holding the smallest roster index."""
    if not parts:
        raise ValueError("no components: the roster is empty")
    return min(parts, key=lambda part: (-len(part), min(part)))


def vulnerability_pairs(connected: np.ndarray, core_members, victims) -> int:
    """Sum over victims of core pairs (victim excluded) split once the victim is removed."""
    adjacency = np.asarray(connected, dtype=bool)
    members = frozenset(core_members)
    total = 0
    for victim in victims:
        survivors = sorted(members - {victim})
        if len(survivors) < 2:
            continue
        reduced = adjacency.copy()
        reduced[victim, :] = False
        reduced[:, victim] = False
        label = {}
        for index, part in enumerate(components(reduced)):
            for node in part:
                label[node] = index
        sizes: dict[int, int] = {}
        for node in survivors:
            sizes[label[node]] = sizes.get(label[node], 0) + 1
        m = len(survivors)
        total += m * (m - 1) // 2 - sum(k * (k - 1) // 2 for k in sizes.values())
    return total


@dataclass(frozen=True)
class ScoringContext:
    """Everything the shared score needs that does not depend on a layout."""

    objective: str
    ids: tuple
    platforms: tuple
    rectangle: dict
    rl_bounds: dict | None
    aoi_m2: float
    diag_m: float
    coverage_points: np.ndarray
    coverage_weights: np.ndarray
    cell_m: float
    candidate_points: np.ndarray
    candidate_cell_m: float
    big: float
    w_vuln: float
    selected: tuple
    starts: np.ndarray
    costs: tuple
    walk_bounds: tuple
    mobility: tuple
    waypoint_policy: str
    balanced_core_fraction: float
    penalties: dict

    @classmethod
    def build(cls, request, scorer) -> "ScoringContext":
        if request.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got "
                             f"{request.objective!r}")
        nodes = tuple(request.nodes)
        if [node.roster_index for node in nodes] != list(range(len(nodes))):
            raise ValueError("request nodes must be in roster order")
        selected = tuple(k for k, node in enumerate(nodes) if node.role == "movable")
        if not selected:
            raise PlannerError("no movable node is selected; [baseline] movable_nodes must "
                               "name at least one node")
        if request.waypoint_policy == "reject":
            for k in selected:
                if nodes[k].mobility == "waypoint":
                    raise PlannerError(
                        f"node '{nodes[k].id}' uses waypoint mobility and is selected; set "
                        "[baseline] waypoint_policy = translate or leave it out of "
                        "movable_nodes")
        missing = sorted({node.platform for node in nodes} - set(request.penalties))
        if missing:
            raise ValueError(f"no movement penalties for platform(s) {missing}")
        probes = scorer.probes
        if probes is None:
            raise ValueError("the scorer has no probes; call prepare_scorer first")
        rect = dict(request.rectangle)
        points, cell_m, weights = rectangle_grid(rect, request.grid.coverage_cells,
                                                 request.grid.min_resolution_m)
        sent = np.asarray(probes["points"], dtype=float)
        if sent.shape != points.shape or not np.allclose(sent, points, rtol=0,
                                                         atol=PROBE_MATCH_TOL_M):
            raise ValueError("the scorer's probes are not this request's coverage grid")
        candidates, candidate_cell_m, _ = rectangle_grid(rect, request.grid.candidate_cells,
                                                         request.grid.min_resolution_m)
        aoi = rectangle_area(rect)
        return cls(
            objective=request.objective,
            ids=tuple(str(node.id) for node in nodes),
            platforms=tuple(node.platform for node in nodes),
            rectangle=rect,
            rl_bounds=dict(request.rl_bounds) if request.mode == "evaluation"
            and request.rl_bounds is not None else None,
            aoi_m2=aoi,
            diag_m=math.hypot(rect["x_max"] - rect["x_min"], rect["y_max"] - rect["y_min"]),
            coverage_points=points, coverage_weights=weights, cell_m=cell_m,
            candidate_points=candidates, candidate_cell_m=candidate_cell_m,
            big=BIG(aoi), w_vuln=vulnerability_weight(request.objective, aoi),
            selected=selected,
            starts=np.array([[node.x, node.y, node.z] for node in nodes], dtype=float),
            costs=tuple(request.penalties[node.platform] for node in nodes),
            walk_bounds=tuple(dict(node.random_walk_bounds) if node.random_walk_bounds
                              else None for node in nodes),
            mobility=tuple(node.mobility for node in nodes),
            waypoint_policy=request.waypoint_policy,
            balanced_core_fraction=float(request.balanced_core_fraction),
            penalties=dict(request.penalties))

    def allowed(self, node: int, x: float, y: float) -> bool:
        """Rectangle, evaluation [rl] bounds and the node's random-walk bounds."""
        if not inside(x, y, self.rectangle):
            return False
        if self.rl_bounds is not None and not inside(x, y, self.rl_bounds):
            return False
        walk = self.walk_bounds[node]
        return walk is None or inside(x, y, walk)

    def displacement(self, node: int, xy) -> float:
        return float(np.hypot(xy[0] - self.starts[node, 0], xy[1] - self.starts[node, 1]))


def prepare_scorer(request, scorer) -> None:
    """Send the coverage grid to the scorer once, before any evaluation."""
    points, _, _ = rectangle_grid(request.rectangle, request.grid.coverage_cells,
                                  request.grid.min_resolution_m)
    scorer.set_probes(points, height_m=request.probe.height_m,
                      rx_gain_dbi=request.probe.rx_gain_dbi, sinr_db=request.probe.sinr_db)


def covered_mask(ctx: ScoringContext, result: LayoutResult, core_members) -> np.ndarray:
    """Probes covered by at least one core node."""
    if result.coverage is None:
        raise ValueError("layout result has no coverage; probes were not set")
    mask = np.zeros(len(ctx.coverage_weights), dtype=bool)
    for node in core_members:
        mask[result.coverage[node]] = True
    return mask


def check_layout(ctx: ScoringContext, layout: np.ndarray) -> None:
    """Unselected nodes keep their start exactly and no z ever changes."""
    layout = np.asarray(layout, dtype=float)
    if layout.shape != ctx.starts.shape:
        raise PlannerError(f"layout shape {layout.shape} != {ctx.starts.shape}")
    if not np.array_equal(layout[:, 2], ctx.starts[:, 2]):
        raise PlannerError("planner changed a node's z")
    fixed = [k for k in range(len(ctx.ids)) if k not in ctx.selected]
    if not np.array_equal(layout[fixed], ctx.starts[fixed]):
        raise PlannerError("planner moved an unselected node")


def _separation(ctx: ScoringContext, layout: np.ndarray) -> float:
    xy = layout[:, :2]
    ratios = []
    for node in ctx.selected:
        distance = np.hypot(*(xy - xy[node]).T)
        distance[node] = np.inf
        ratios.append(min(float(distance.min()) / ctx.diag_m, 1.0))
    return SEPARATION_FRAC * ctx.cell_m ** 2 * float(np.mean(ratios))


def score(ctx: ScoringContext, result: LayoutResult, layout: np.ndarray) -> LayoutScore:
    """coverage + separation - movement - BIG x disconnected - w x vulnerability pairs."""
    layout = np.asarray(layout, dtype=float)
    members = core(components(result.connected))
    coverage_m2 = float(ctx.coverage_weights[covered_mask(ctx, result, members)].sum())
    sep_m2 = _separation(ctx, layout)
    move_m2 = sum(ctx.costs[k].cost(ctx.displacement(k, layout[k])) for k in ctx.selected)
    disconnected = sum(1 for k in ctx.selected if k not in members)
    vuln = vulnerability_pairs(result.connected, members, sorted(members & set(ctx.selected)))
    total = coverage_m2 + sep_m2 - move_m2 - ctx.big * disconnected - ctx.w_vuln * vuln
    return LayoutScore(coverage_m2=coverage_m2, sep_m2=sep_m2, move_cost_m2=float(move_m2),
                       disconnected=disconnected, vuln=vuln, total=float(total))


def scale_ratios(ctx: ScoringContext) -> dict:
    return {platform: cost.fixed_cost_m2 / ctx.aoi_m2
            for platform, cost in sorted(ctx.penalties.items())}


def diagnose(ctx: ScoringContext, result: LayoutResult, layout: np.ndarray,
             layout_score: LayoutScore, log=None) -> dict:
    """JSON-serializable graph, coverage, resilience, movement and scale facts."""
    layout = np.asarray(layout, dtype=float)
    parts = components(result.connected)
    members = core(parts)
    controlled = sorted(members & set(ctx.selected))
    ratios = scale_ratios(ctx)
    warnings = [f"{platform} fixed_cost_m2 is {ratio:.3g} x the AOI ({ctx.aoi_m2:.6g} m2); "
                "any move must buy at least that much coverage"
                for platform, ratio in ratios.items() if ratio >= 1.0]
    if log is not None:
        for warning in warnings:
            log.write(f"planner scale warning: {warning}\n")
        log.flush()
    nodes = {}
    for k, node_id in enumerate(ctx.ids):
        displacement = ctx.displacement(k, layout[k])
        selected = k in ctx.selected
        nodes[node_id] = {"selected": selected, "platform": ctx.platforms[k],
                          "in_core": k in members, "displacement_m": displacement,
                          "move_cost_m2": ctx.costs[k].cost(displacement) if selected
                          else 0.0}
    return {
        "aoi_m2": ctx.aoi_m2,
        "coverage_m2": layout_score.coverage_m2,
        "coverage_fraction": layout_score.coverage_m2 / ctx.aoi_m2,
        "component_sizes": [len(part) for part in parts],
        "core": [ctx.ids[k] for k in sorted(members)],
        "disconnected_selected": [ctx.ids[k] for k in ctx.selected if k not in members],
        "vulnerability_pairs": vulnerability_pairs(result.connected, members, controlled),
        "survives_single_node_loss":
            vulnerability_pairs(result.connected, members, sorted(members)) == 0,
        "controlled_mesh_survives_single_loss":
            vulnerability_pairs(result.connected, members, controlled) == 0,
        "nodes": nodes,
        "score": {**layout_score.as_dict(),
                  "connectivity_penalty_m2": ctx.big * layout_score.disconnected,
                  "vulnerability_penalty_m2": ctx.w_vuln * layout_score.vuln},
        "fixed_cost_over_aoi": ratios,
        "scale_warnings": warnings,
    }


def candidate_positions(ctx: ScoringContext, node_index: int, current=None) -> np.ndarray:
    """Allowed candidate-grid points within the cap, after the node's current position."""
    start = ctx.starts[node_index]
    here = start if current is None else np.asarray(current, dtype=float)
    cost = ctx.costs[node_index]
    keep = []
    for x, y in ctx.candidate_points:
        if not ctx.allowed(node_index, x, y):
            continue
        if not cost.within_cap(ctx.displacement(node_index, (x, y))):
            continue
        if math.hypot(x - here[0], y - here[1]) <= STAY_PUT_TOL_M:
            continue
        keep.append((x, y))
    rows = [(float(here[0]), float(here[1]))] + keep
    return np.array([[x, y, start[2]] for x, y in rows], dtype=float)


class LayoutCache:
    """Full-layout query results keyed by layout bytes; misses go out in one batch."""

    def __init__(self, scorer):
        self._scorer = scorer
        self._results: dict[bytes, LayoutResult] = {}

    def evaluate(self, layouts) -> list[LayoutResult]:
        arrays = [np.ascontiguousarray(layout, dtype=float) for layout in layouts]
        keys = [array.tobytes() for array in arrays]
        missing = {key: array for key, array in zip(keys, arrays) if key not in self._results}
        if missing:
            fresh = self._scorer.evaluate(list(missing.values()))
            self._results.update(zip(missing, fresh))
        return [self._results[key] for key in keys]
