"""Sequential greedy placement over a rootless peer-anchor schedule, scored by the channel."""

import math

import numpy as np

from scripts.baselines.planners import objective
from scripts.baselines.planners.objective import (LayoutCache, ScoringContext,
                                                  candidate_positions, check_layout,
                                                  components, core, diagnose, score)


def needs(objective_name: str, count: int, balanced_core_fraction: float) -> list[int]:
    """Anchor links each selected node should have, in roster order."""
    if objective_name == "coverage":
        return [1] * count
    if objective_name == "resilience":
        return [2] * count
    if objective_name == "balanced":
        twos = math.ceil(count * balanced_core_fraction)
        return [2] * twos + [1] * (count - twos)
    raise ValueError(f"objective must be one of {objective.OBJECTIVES}, got "
                     f"{objective_name!r}")


def settings(request) -> dict:
    count = sum(1 for node in request.nodes if node.role == "movable")
    return {
        "strategy": "sequential_greedy",
        "order": "roster",
        "needs": needs(request.objective, count, request.balanced_core_fraction),
        "balanced_core_fraction": float(request.balanced_core_fraction),
        "acceptance": "strictly_positive_total_gain",
        "score_tie_tol_m2": objective.SCORE_TIE_TOL_M2,
        "separation_frac": objective.SEPARATION_FRAC,
        "balanced_vulnerability_frac": objective.BALANCED_VULNERABILITY_FRAC,
        "big_aoi_factor": objective.BIG_AOI_FACTOR,
    }


def _pool(ctx: ScoringContext, connected: np.ndarray, processed: list[int],
          node: int) -> list[int]:
    members = core(components(connected))
    chosen = set(processed)
    return sorted(k for k in members if k != node and
                  (k not in ctx.selected or k in chosen))


def _place(ctx, cache, layout, current, current_score, node, need, processed, log):
    pool = _pool(ctx, current.connected, processed, node)
    eff_need = min(need, len(pool))
    step = {"node": ctx.ids[node], "need": need, "eff_need": eff_need,
            "pool": [ctx.ids[k] for k in pool], "bootstrap": eff_need == 0,
            "degraded": False, "candidates": 0, "feasible": 0, "accepted": False,
            "marginal_m2": None, "anchor_links": None}
    moves = candidate_positions(ctx, node)[1:]
    step["candidates"] = len(moves)
    if not len(moves):
        return layout, current, current_score, step
    layouts = []
    for position in moves:
        trial = layout.copy()
        trial[node] = position
        layouts.append(trial)
    results = cache.evaluate(layouts)
    links = np.array([int(result.connected[node, pool].sum()) for result in results])
    feasible = np.ones(len(moves), dtype=bool) if eff_need == 0 else links >= eff_need
    if eff_need >= 2 and not feasible.any():
        feasible = links >= 1
        step["degraded"] = True
    step["feasible"] = int(feasible.sum())
    best = None
    for k in np.nonzero(feasible)[0]:
        trial_score = score(ctx, results[k], layouts[k])
        if best is None or trial_score.total > best[1].total:
            best = (int(k), trial_score)
    if best is None:
        log.write(f"geometric: {ctx.ids[node]} stays put (no anchor-feasible candidate)\n")
        return layout, current, current_score, step
    k, best_score = best
    step["marginal_m2"] = best_score.total - current_score.total
    if step["marginal_m2"] <= objective.SCORE_TIE_TOL_M2:
        log.write(f"geometric: {ctx.ids[node]} stays put (best marginal "
                  f"{step['marginal_m2']:.6g} m2)\n")
        return layout, current, current_score, step
    step["accepted"] = True
    step["anchor_links"] = int(links[k])
    log.write(f"geometric: {ctx.ids[node]} -> ({moves[k][0]:.3f}, {moves[k][1]:.3f}) "
              f"marginal {step['marginal_m2']:.6g} m2\n")
    return layouts[k], results[k], best_score, step


def solve(request, scorer, log):
    """Place selected nodes one at a time in roster order; returns (N x 3, diagnostics, notes)."""
    ctx = ScoringContext.build(request, scorer)
    cache = LayoutCache(scorer)
    layout = ctx.starts.copy()
    current = cache.evaluate([layout])[0]
    current_score = score(ctx, current, layout)
    start_total = current_score.total
    schedule = needs(ctx.objective, len(ctx.selected), ctx.balanced_core_fraction)
    processed, steps, notes = [], [], []
    for node, need in zip(ctx.selected, schedule):
        layout, current, current_score, step = _place(ctx, cache, layout, current,
                                                      current_score, node, need, processed,
                                                      log)
        processed.append(node)
        steps.append(step)
        if not step["accepted"]:
            notes.append(f"{step['node']}: stay put")
        elif step["degraded"]:
            notes.append(f"{step['node']}: anchor need degraded 2 -> 1")
    log.flush()
    check_layout(ctx, layout)
    diagnostics = diagnose(ctx, current, layout, current_score, log=log)
    diagnostics["geometric"] = {"steps": steps, "start_total": start_total}
    notes.append(f"geometric: {sum(s['accepted'] for s in steps)}/{len(steps)} moved, "
                 f"score {current_score.total:,.0f} m2-equivalent")
    return layout, diagnostics, notes
