"""
OptimizationModel — model 2: simulated annealing over continuous
positions, warm-started from the geometric solution.

Objective (maximized, all terms in m^2 so weights are physical):

    score = coverage_m2
          - movement_cost_m2                 (soft cost, from config)
          - BIG * n_disconnected_movables    (hard: everyone reaches c2)
          - w_vuln * vulnerability_pairs     (per-COA redundancy term)

``vulnerability_pairs`` counts (victim, orphan) pairs over the
controlled mesh: for each movable node whose loss would disconnect
other movable nodes from the c2, one pair per orphaned node. It is a
*graded* version of controlled_mesh_survives_single_loss (0 pairs ==
survivable), which gives the annealer a slope to descend instead of a
binary cliff. Per-COA weights: coverage 0; resilience 1.0 x AOI area
per pair (a vulnerability outweighs any possible coverage gain, i.e. a
hard constraint expressed softly); balanced a config fraction of AOI.

Structured moves (the reason this beats generic optimizers here):
  nudge     — gaussian step on one node, sigma annealed large -> small
  teleport  — move one node next to a currently-uncovered eval point
              (repairs greedy's myopia: reachable coverage gaps get
              claimed directly instead of waiting for a random walk)
  swap      — exchange two nodes' positions + altitudes (pays off with
              heterogeneous radios or movement-cost classes)
  altitude  — resample one node's AGL within its band

Every proposal is constraint-native: clamped to the geofence, the
per-mobility displacement cap, and the altitude band — so no penalty
tuning for feasibility, and no evaluations wasted outside it.

Warm start + keep-best means the result is never worse than model 1
under the same objective. Time-budgeted; deterministic under a seed.
"""

from __future__ import annotations

import math
import time

import numpy as np
from shapely import contains_xy

from core.context import PlanningContext
from planners.base import COAResult, PlacementModel
from planners.geometric import GeometricModel


class _Evaluator:
    """Vectorized objective over full node-state arrays."""

    def __init__(self, ctx: PlanningContext, coa: str):
        self.ctx = ctx
        self.nodes = ctx.nodes
        self.n = len(self.nodes)
        self.ids = [pn.node_id for pn in self.nodes]
        self.idx = {nid: i for i, nid in enumerate(self.ids)}
        self.movable_idx = np.array(
            [i for i, pn in enumerate(self.nodes) if pn.movable], dtype=int
        )
        self.gateway_idx = [i for i, pn in enumerate(self.nodes) if pn.is_gateway]

        # pairwise max link range matrix (squared)
        R = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                R[i, j] = R[j, i] = ctx.link_range(self.nodes[i], self.nodes[j])
        self.R2 = R**2

        rr = ctx.rf.config.reference_receiver
        self.has_cov = np.array(
            [ctx.rf.has_coverage_radio(pn.radio_types) for pn in self.nodes]
        )
        bt = ctx.backhaul_types()
        self.has_blos = np.array(
            [bool(pn.radio_types & bt) and not pn.is_gateway
             for pn in self.nodes]
        )
        self.slant2 = ctx.rf.coverage_slant_range**2
        self.rx_h = rr.height_agl_m

        self.eval_pts = ctx.eval_points
        self.cell_area = ctx.eval_cell_m**2
        minx, miny, maxx, maxy = ctx.geofence_m.bounds
        self._diag = math.hypot(maxx - minx, maxy - miny)
        # Tie-break weight: strictly less than ONE eval
        # cell of coverage, so it can only ever decide exact ties —
        # never trade real coverage for spread. Fixes issue
        # when footprints dwarf the fence.
        self.w_sep = (
            getattr(ctx.config.optimizer, "separation_tiebreak_frac", 0.4)
            * self.cell_area
        )
        self.aoi_m2 = ctx.geofence_m.area

        # movement cost per movable node
        self.start_pos = np.array([pn.xy for pn in self.nodes])
        self.cost_models = [ctx.movement_cost(pn) for pn in self.nodes]

        # per-COA vulnerability weight (m^2 per orphan pair)
        if coa == "resilience":
            self.w_vuln = self.aoi_m2
        elif coa == "balanced":
            self.w_vuln = (
                ctx.config.optimizer.balanced_vulnerability_frac * self.aoi_m2
            )
        else:
            self.w_vuln = 0.0
        self.big = 2.0 * self.aoi_m2  # per disconnected movable node

    # -- graph pieces -------------------------------------------

    def _adjacency(self, pos: np.ndarray, agl: np.ndarray) -> np.ndarray:
        dxy2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
        dz2 = (agl[:, None] - agl[None, :]) ** 2
        adj = (dxy2 + dz2) <= self.R2
        np.fill_diagonal(adj, False)
        return adj

    def _reachable(self, adj: np.ndarray, skip: int = -1) -> np.ndarray:
        seen = np.zeros(self.n, dtype=bool)
        stack = [g for g in self.gateway_idx if g != skip]
        while stack:
            i = stack.pop()
            if seen[i]:
                continue
            seen[i] = True
            nbrs = np.nonzero(adj[i])[0]
            for j in nbrs:
                if not seen[j] and j != skip:
                    stack.append(j)
        return seen

    # -- objective ----------------------------------------------

    def score(
        self, pos: np.ndarray, agl: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Returns (score, covered_mask) — the mask feeds teleports."""
        adj = self._adjacency(pos, agl)
        reach = self._reachable(adj)

        # coverage from connected coverage-capable nodes
        covered = np.zeros(len(self.eval_pts), dtype=bool)
        active = np.nonzero(reach & self.has_cov)[0]
        for i in active:
            r2 = self.slant2 - (agl[i] - self.rx_h) ** 2
            if r2 <= 0:
                continue
            d2 = ((self.eval_pts - pos[i]) ** 2).sum(1)
            covered |= d2 <= r2
        coverage_m2 = covered.sum() * self.cell_area
        sep_m2 = 0.0
        if self.w_sep > 0 and len(self.movable_idx) > 0:
            ratios = []
            for i in self.movable_idx:
                d = np.linalg.norm(pos - pos[i], axis=1)
                d[i] = np.inf
                ratios.append(min(float(d.min()) / self._diag, 1.0))
            sep_m2 = self.w_sep * float(np.mean(ratios))

        # movement cost
        move_cost = 0.0
        for i in self.movable_idx:
            disp = float(np.linalg.norm(pos[i] - self.start_pos[i]))
            move_cost += self.cost_models[i].cost(disp)

        # connectivity (hard, expressed as dominant penalty)
        disconnected = int((~reach[self.movable_idx]).sum())

        # vulnerability pairs over the controlled mesh (victims whose
        # loss no placement could mitigate — the sole backhaul
        # terminal — are excluded; see PlanningContext._unmitigable_victims)
        vuln = 0
        if self.w_vuln > 0:
            core = [i for i in self.movable_idx if reach[i]]
            core_set = set(core)
            # sole connected backhaul terminal = unmitigable victim;
            # computed from the reach array already in hand (cheap)
            terms = np.nonzero(self.has_blos & reach)[0]
            skip = {int(terms[0])} if len(terms) == 1 else set()
            for victim in [v for v in core if v not in skip]:
                seen = self._reachable(adj, skip=victim)
                vuln += sum(
                    1 for i in core_set if i != victim and not seen[i]
                )

        score = (
            coverage_m2
            + sep_m2
            - move_cost
            - self.big * disconnected
            - self.w_vuln * vuln
        )
        return score, covered


class OptimizationModel(PlacementModel):
    name = "optimization"

    COA_NAMES = ("coverage", "balanced", "resilience")

    def __init__(self, warm_start: PlacementModel | None = None):
        self.warm_start = warm_start or GeometricModel()

    # -- entry point ---------------------------------------------

    def solve(self, ctx: PlanningContext, coa: str) -> COAResult:
        opt = ctx.config.optimizer
        rng = np.random.default_rng(opt.seed)
        ev = _Evaluator(ctx, coa)

        # Two seeds, keep-best throughout:
        #   * geometric warm start (fresh plan), and
        #   * the CURRENT layout (zero displacement by definition).
        # With movement costs configured, "hold position" is therefore
        # always a scored candidate — the service can never command a
        # net-negative shuffle just because the fresh plan drifted.
        warm = self.warm_start.solve(ctx, coa)
        g_pos = np.array([warm.pos[nid] for nid in ev.ids], dtype=float)
        g_agl = np.array([warm.agl[nid] for nid in ev.ids], dtype=float)
        g_score, g_cov = ev.score(g_pos, g_agl)

        cur_pos_d, cur_agl_d = ctx.current_positions()
        c_pos = np.array([cur_pos_d[nid] for nid in ev.ids], dtype=float)
        c_agl = np.array([cur_agl_d[nid] for nid in ev.ids], dtype=float)
        c_score, c_cov = ev.score(c_pos, c_agl)

        if c_score > g_score:
            pos, agl, cur_score, covered = c_pos, c_agl, c_score, c_cov
        else:
            pos, agl, cur_score, covered = g_pos, g_agl, g_score, g_cov
        best_pos, best_agl = pos.copy(), agl.copy()
        best_score = cur_score

        if len(ev.movable_idx) == 0:
            return self._result(ctx, ev, best_pos, best_agl)

        minx, miny, maxx, maxy = ctx.geofence_m.bounds
        diag = math.hypot(maxx - minx, maxy - miny)
        t0 = opt.t0_area_frac * ev.aoi_m2
        tf = max(opt.tf_area_frac * ev.aoi_m2, 1e-9)
        sigma0 = opt.nudge_sigma_frac * diag

        move_names = [k for k, _ in opt.move_weights]
        move_p = np.array([v for _, v in opt.move_weights])
        move_p = move_p / move_p.sum()

        budget = opt.budget_s_per_coa
        t_start = time.perf_counter()
        iters = accepts = 0

        while True:
            if opt.max_iters is not None:
                frac = iters / opt.max_iters
            else:
                frac = (time.perf_counter() - t_start) / budget
            if frac >= 1.0:
                break
            temp = t0 * (tf / t0) ** frac
            sigma = sigma0 * (0.05 / 1.0) ** frac + 2.0  # anneal to ~2 m

            move = move_names[int(rng.choice(len(move_names), p=move_p))]
            prop = self._propose(
                ctx, ev, rng, pos, agl, covered, move, sigma
            )
            iters += 1
            if prop is None:
                continue
            new_pos, new_agl = prop
            new_score, new_covered = ev.score(new_pos, new_agl)
            d = new_score - cur_score
            if d >= 0 or rng.random() < math.exp(d / max(temp, 1e-9)):
                pos, agl, cur_score, covered = (
                    new_pos, new_agl, new_score, new_covered
                )
                accepts += 1
                if cur_score > best_score:
                    best_score, best_pos, best_agl = (
                        cur_score, pos.copy(), agl.copy()
                    )

        result = self._result(ctx, ev, best_pos, best_agl)
        result.diagnostics.notes = (
            f"sa iters={iters} accepts={accepts} "
            f"score={best_score:,.0f} m2-equivalent"
        )
        return result

    # -- proposals -----------------------------------------------

    def _propose(self, ctx, ev, rng, pos, agl, covered, move, sigma):
        m = ev.movable_idx
        if move == "swap" and len(m) >= 2:
            i, j = rng.choice(m, size=2, replace=False)
            new_pos, new_agl = pos.copy(), agl.copy()
            new_pos[[i, j]] = new_pos[[j, i]]
            new_agl[[i, j]] = new_agl[[j, i]]
            # both must respect their own displacement caps
            for k in (i, j):
                if not ev.cost_models[k].within_cap(
                    float(np.linalg.norm(new_pos[k] - ev.start_pos[k]))
                ):
                    return None
            return new_pos, new_agl

        i = int(rng.choice(m))
        pn = ev.nodes[i]

        if move == "altitude":
            lo, hi = pn.band
            if hi - lo < 1e-6:
                return None
            new_agl = agl.copy()
            new_agl[i] = rng.uniform(lo, hi)
            return pos.copy(), new_agl

        if move == "teleport":
            uncovered = np.nonzero(~covered)[0]
            if not len(uncovered):
                return None
            target = ev.eval_pts[int(rng.choice(uncovered))]
            jitter = rng.normal(0.0, 0.15 * sigma, size=2)
            cand = target + jitter
        else:  # nudge
            cand = pos[i] + rng.normal(0.0, sigma, size=2)

        cand = self._clamp(ctx, ev, i, cand)
        if cand is None:
            return None
        new_pos = pos.copy()
        new_pos[i] = cand
        return new_pos, agl.copy()

    @staticmethod
    def _clamp(ctx, ev, i, cand):
        # displacement cap: pull back toward the start position
        cm = ev.cost_models[i]
        if cm.max_displacement_m is not None:
            v = cand - ev.start_pos[i]
            d = float(np.linalg.norm(v))
            if d > cm.max_displacement_m:
                cand = ev.start_pos[i] + v * (cm.max_displacement_m / d) * 0.999
        # geofence: reject if outside (holes included)
        if not contains_xy(ctx.geofence_m, cand[0], cand[1]):
            return None
        return cand

    # -- packaging -----------------------------------------------

    def _result(self, ctx, ev, pos, agl) -> COAResult:
        pos_d = {nid: pos[i].copy() for i, nid in enumerate(ev.ids)}
        agl_d = {nid: float(agl[i]) for i, nid in enumerate(ev.ids)}
        return COAResult(pos_d, agl_d, self._diagnose(ctx, pos_d, agl_d))
