"""Keep-best simulated annealing over selected x/y, warm-started from the greedy plan."""

import math

import numpy as np

from scripts.baselines.planners import geometric, objective
from scripts.baselines.planners.channel import PlannerError
from scripts.baselines.planners.objective import (ScoringContext, check_layout, components,
                                                  core, covered_mask, diagnose, score)

# Desktop OptimizerConfig defaults (core/rf.py) and annealing schedule (optimization.py).
T0_AREA_FRAC = 0.02
TF_AREA_FRAC = 0.0002
NUDGE_SIGMA_FRAC = 0.08
SIGMA_FINAL_RATIO = 0.05
SIGMA_FLOOR_M = 2.0
TELEPORT_JITTER_FRAC = 0.15
CAP_PULLBACK = 0.999
DESKTOP_MOVE_WEIGHTS = {"nudge": 0.55, "teleport": 0.20, "swap": 0.10, "altitude": 0.15}
MOVES = ("nudge", "teleport", "swap")
MOVE_WEIGHTS = {move: DESKTOP_MOVE_WEIGHTS[move] /
                sum(DESKTOP_MOVE_WEIGHTS[m] for m in MOVES) for move in MOVES}


def settings(request) -> dict:
    return {
        "strategy": "simulated_annealing_keep_best",
        "warm_start": geometric.settings(request),
        "second_seed": "current_layout",
        "t0_area_frac": T0_AREA_FRAC,
        "tf_area_frac": TF_AREA_FRAC,
        "nudge_sigma_frac": NUDGE_SIGMA_FRAC,
        "sigma_final_ratio": SIGMA_FINAL_RATIO,
        "sigma_floor_m": SIGMA_FLOOR_M,
        "teleport_jitter_frac": TELEPORT_JITTER_FRAC,
        "cap_pullback": CAP_PULLBACK,
        "move_weights": dict(MOVE_WEIGHTS),
        "termination": {"max_iterations": request.max_iterations},
        "balanced_core_fraction": float(request.balanced_core_fraction),
        "separation_frac": objective.SEPARATION_FRAC,
        "balanced_vulnerability_frac": objective.BALANCED_VULNERABILITY_FRAC,
        "big_aoi_factor": objective.BIG_AOI_FACTOR,
    }


def _clamp(ctx: ScoringContext, node: int, xy: np.ndarray) -> np.ndarray | None:
    """Pull back inside the displacement cap; None outside rectangle, [rl] or walk bounds."""
    cap = ctx.costs[node].max_displacement_m
    start = ctx.starts[node, :2]
    if cap is not None:
        offset = xy - start
        distance = float(np.hypot(*offset))
        if distance > cap:
            xy = start + offset * (cap / distance) * CAP_PULLBACK
    if not ctx.allowed(node, float(xy[0]), float(xy[1])):
        return None
    return xy


def propose(ctx: ScoringContext, rng, layout: np.ndarray, covered: np.ndarray, move: str,
            sigma: float) -> np.ndarray | None:
    """One x/y proposal over selected nodes; z is never touched."""
    chosen = np.array(ctx.selected)
    new = layout.copy()
    if move == "swap" and len(chosen) >= 2:
        i, j = (int(k) for k in rng.choice(chosen, size=2, replace=False))
        moved = {i: layout[j, :2].copy(), j: layout[i, :2].copy()}
        for node, xy in moved.items():
            clamped = _clamp(ctx, node, xy)
            if clamped is None:
                return None
            new[node, :2] = clamped
        return new
    node = int(rng.choice(chosen))
    if move == "teleport":
        uncovered = np.nonzero(~covered)[0]
        if not len(uncovered):
            return None
        target = ctx.coverage_points[int(rng.choice(uncovered))]
        xy = target + rng.normal(0.0, TELEPORT_JITTER_FRAC * sigma, size=2)
    else:
        xy = layout[node, :2] + rng.normal(0.0, sigma, size=2)
    clamped = _clamp(ctx, node, xy)
    if clamped is None:
        return None
    new[node, :2] = clamped
    return new


def _covered(ctx, result) -> np.ndarray:
    return covered_mask(ctx, result, core(components(result.connected)))


def solve(request, scorer, log):
    """Anneal from the better of the greedy plan and the start; returns the best layout seen."""
    if request.max_iterations is None or request.max_iterations < 1:
        raise PlannerError("optimization needs [baseline] max_iterations >= 1")
    warm, warm_diagnostics, warm_notes = geometric.solve(request, scorer, log)
    ctx = ScoringContext.build(request, scorer)
    starts = ctx.starts.copy()
    warm_result, start_result = scorer.evaluate([warm, starts])
    warm_score = score(ctx, warm_result, warm)
    start_score = score(ctx, start_result, starts)
    if start_score.total > warm_score.total:
        layout, result, current, seed_name = starts, start_result, start_score, "current"
    else:
        layout, result, current, seed_name = warm, warm_result, warm_score, "geometric"
    best_layout, best_result, best = layout.copy(), result, current
    covered = _covered(ctx, result)
    rng = np.random.default_rng(request.planner_seed)
    t0 = T0_AREA_FRAC * ctx.aoi_m2
    tf = max(TF_AREA_FRAC * ctx.aoi_m2, 1e-9)
    sigma0 = NUDGE_SIGMA_FRAC * ctx.diag_m
    names = list(MOVES)
    weights = np.array([MOVE_WEIGHTS[name] for name in names])
    iterations = accepts = rejected = queried = 0
    while iterations < request.max_iterations:
        frac = iterations / request.max_iterations
        temperature = t0 * (tf / t0) ** frac
        sigma = sigma0 * SIGMA_FINAL_RATIO ** frac + SIGMA_FLOOR_M
        move = names[int(rng.choice(len(names), p=weights))]
        proposal = propose(ctx, rng, layout, covered, move, sigma)
        iterations += 1
        if proposal is None:
            rejected += 1
            continue
        proposal_result = scorer.evaluate([proposal])[0]
        queried += 1
        proposal_score = score(ctx, proposal_result, proposal)
        delta = proposal_score.total - current.total
        if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-9)):
            layout, result, current = proposal, proposal_result, proposal_score
            covered = _covered(ctx, result)
            accepts += 1
            if current.total > best.total:
                best_layout, best_result, best = layout.copy(), result, current
    check_layout(ctx, best_layout)
    diagnostics = diagnose(ctx, best_result, best_layout, best)
    diagnostics["geometric"] = warm_diagnostics["geometric"]
    diagnostics["optimization"] = {
        "seed_layout": seed_name, "geometric_total": warm_score.total,
        "current_total": start_score.total, "best_total": best.total,
        "iterations": iterations, "accepted": accepts, "rejected_proposals": rejected,
        "queried_proposals": queried}
    note = (f"sa iters={iterations} accepts={accepts} rejected={rejected} "
            f"score={best.total:,.0f} m2-equivalent (seed: {seed_name})")
    log.write(f"optimization: {note}\n")
    log.flush()
    return best_layout, diagnostics, [*warm_notes, note]
