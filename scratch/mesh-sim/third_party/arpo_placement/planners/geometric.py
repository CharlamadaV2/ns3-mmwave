"""
GeometricModel — model 1: pure Friis + geometry heuristics.

All three COAs are the same greedy marginal-coverage placement
(submodular objective, so greedy carries the classic (1 - 1/e)
approximation guarantee) run under different *anchor schedules*. A
node's anchor requirement is how many already-guaranteed nodes it must
be within safe link range of when placed:

coverage    every node needs 1 anchor — spread wide, tree topology.
resilience  every node needs 2 anchors. Attaching each new vertex to
            two vertices of a 2-connected graph keeps it 2-connected,
            so single-node-loss survivability holds by construction
            (once >= 2 movable nodes are placed). Denser footprint —
            that's the tradeoff.
balanced    two-phase: the first ``balanced_core_fraction`` of movable
            nodes are placed under the 2-anchor rule (resilient core),
            the remainder under the 1-anchor rule (coverage fringe).
            The core survives single loss; the fringe buys area.

Anchor bookkeeping is deliberately split in two:
  * connected set — anything currently reachable from a c2 node.
    Counts for coverage and can serve as a 1-anchor.
  * anchor set — only nodes whose redundancy is guaranteed by
    construction (c2 nodes + movables placed under the schedule).
    Unmovable relays that get incidentally bridged in may hang off a
    single link, so they must never serve as 2-anchor redundancy
    anchors.

A movable c2 (type=c2, mobility=ground/aerial) is placed first with an
anchor requirement of zero — it anchors itself, everything else
anchors to it.

Movement cost (config ``movement_cost``, keyed by mobility class):
candidates beyond ``max_displacement_m`` from a node's current
position are infeasible (hard cap); feasible candidates are scored as
``coverage_gain_m2 - (fixed_cost_m2 + cost_m2_per_m * displacement)``
(soft penalty, both sides in m^2 so the tradeoff is physical: "moving
this node 1 km must buy at least X m^2 of new area"). A node stays put
when no placement earns its movement cost. Defaults are zero, which
reproduces cost-free behavior exactly.

Altitude: chosen within each node's AGL band to maximize its ground
coverage footprint. Under pure Friis that resolves to the band minimum
(height only adds slant distance); it's computed rather than hardcoded
so the preference flips correctly once LOS/terrain models arrive.
"""

from __future__ import annotations

import math

import numpy as np

from core.context import PlanNode, PlanningContext
from core.geometry import grid_points_in
from planners.base import COAResult, PlacementModel


class GeometricModel(PlacementModel):
    name = "geometric"

    COA_NAMES = ("coverage", "balanced", "resilience")

    # -- entry point ---------------------------------------------

    def solve(self, ctx: PlanningContext, coa: str) -> COAResult:
        ordered = self._order_movable(ctx)
        if coa == "coverage":
            needs = [1] * len(ordered)
        elif coa == "resilience":
            needs = [2] * len(ordered)
        elif coa == "balanced":
            core = math.ceil(len(ordered) * ctx.config.balanced_core_fraction)
            needs = [2] * core + [1] * (len(ordered) - core)
        else:
            raise ValueError(
                f"unknown COA {coa!r}; expected one of {self.COA_NAMES}"
            )
        # Movable c2 nodes anchor themselves.
        needs = [0 if pn.is_gateway else n for pn, n in zip(ordered, needs)]
        pos, agl = self._greedy_place(ctx, ordered, needs)
        return COAResult(pos, agl, self._diagnose(ctx, pos, agl))

    # -- ordering & candidates -----------------------------------

    @staticmethod
    def _order_movable(ctx: PlanningContext) -> list[PlanNode]:
        """Movable c2 first (it's the anchor), then longest mesh range
        first — long-haul nodes make the best early relays."""

        def key(pn: PlanNode):
            best = max((ctx.rf.mesh_range(t) for t in pn.radio_types), default=0.0)
            return (0 if pn.is_gateway else 1, -best)

        return sorted(ctx.movable, key=key)

    def _candidates(self, ctx: PlanningContext) -> np.ndarray:
        cg = ctx.config.candidate_grid
        pts, _ = grid_points_in(
            ctx.geofence_m,
            target_cells=int(cg["target_cells"]),
            min_resolution_m=float(cg["min_resolution_m"]),
        )
        return pts

    # -- greedy placement under an anchor schedule ---------------

    def _greedy_place(
        self,
        ctx: PlanningContext,
        ordered: list[PlanNode],
        needs: list[int],
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        pos, agl = ctx.current_positions()
        candidates = self._candidates(ctx)
        eval_pts = ctx.eval_points
        covered = np.zeros(len(eval_pts), dtype=bool)

        movable_ids = {pn.node_id for pn in ordered}
        connected = self._connected_set(ctx, pos, agl)
        # Guaranteed anchor set (grows with each placed movable):
        #  * unmovable c2 nodes, and
        #  * satellite-rooted unmovables — connected nodes carrying a
        #    backhaul (blos) radio, e.g. a fixed terminal mast. Their
        #    link to the c2 rides the constellation, not mesh
        #    geometry, so they are root-equivalent. Without this, a
        #    satellite-only c2 leaves the anchor set empty and every
        #    node parks at spawn.
        # For 1-anchor placements (coverage / balanced fringe) ANY
        # connected node is a sound anchor; the guaranteed set is only
        # required for the >=2-anchor redundancy rule.
        bt = ctx.backhaul_types()
        anchor_ids = {g.node_id for g in ctx.gateways if not g.movable}
        anchor_ids |= {
            pn.node_id
            for pn in ctx.nodes
            if not pn.movable
            and pn.node_id in connected
            and bool(pn.radio_types & bt)
        }

        # Existing coverage from connected, non-movable nodes counts
        # before we place anything (movables will be OR'd on placement
        # at their new positions).
        for nid in connected - movable_ids:
            self._or_footprint(
                ctx, covered, eval_pts, pos[nid], agl[nid], ctx.by_id[nid]
            )

        for pn, need in zip(ordered, needs):
            target_agl = self._coverage_altitude(ctx, pn)
            foot_r = (
                ctx.rf.ground_footprint_radius(target_agl)
                if ctx.rf.has_coverage_radio(pn.radio_types)
                else 0.0
            )

            # Movement cost: the hard cap is a physical constraint —
            # apply it FIRST, then find anchor-feasible spots within
            # the reachable set (degrading 2->1 anchors if needed).
            cost_model = ctx.movement_cost(pn)
            reachable = candidates
            if cost_model.max_displacement_m is not None:
                disp_all = np.linalg.norm(candidates - pos[pn.node_id], axis=1)
                reachable = candidates[disp_all <= cost_model.max_displacement_m]

            # Stable anchors only: connected UNMOVABLE nodes (their
            # position won't change) — never yet-unplaced movables,
            # whose spawn-cluster links evaporate as they move.
            stable = {
                nid for nid in connected if not ctx.by_id[nid].movable
            }
            eff_need = min(need, len(anchor_ids))
            pool = anchor_ids if eff_need >= 2 else (anchor_ids | stable)
            anchors_xy, anchors_r = self._anchor_arrays(
                ctx, pn, pool - {pn.node_id}, pos, agl, target_agl
            )
            feasible = self._feasible(reachable, anchors_xy, anchors_r, eff_need)
            if not len(feasible) and eff_need > 1:
                # Degrade gracefully: a singly-linked node beats a
                # parked one; diagnostics will report the weakness.
                loose_xy, loose_r = self._anchor_arrays(
                    ctx, pn, (anchor_ids | stable) - {pn.node_id},
                    pos, agl, target_agl,
                )
                feasible = self._feasible(reachable, loose_xy, loose_r, 1)
            if not len(feasible):
                continue  # nowhere viable — leave the node where it is

            displacement = np.linalg.norm(feasible - pos[pn.node_id], axis=1)
            moved = displacement > 1e-6
            penalty_m2 = np.where(
                moved,
                cost_model.fixed_cost_m2
                + cost_model.cost_m2_per_m * displacement,
                0.0,
            )

            cell_area = ctx.eval_cell_m**2  # eval points -> m^2
            if foot_r > 0:
                d2 = (
                    np.sum(
                        (eval_pts[None, :, :] - feasible[:, None, :]) ** 2,
                        axis=2,
                    )
                    <= foot_r * foot_r
                )
                gains_m2 = (d2 & ~covered[None, :]).sum(axis=1) * cell_area
                # Separation tie-break (< one eval cell): when coverage
                # cannot distinguish candidates — tiny fences, saturated
                # scenarios — prefer spread over stacking.
                w_sep = getattr(
                    ctx.config.optimizer, "separation_tiebreak_frac", 0.4
                ) * cell_area
                if w_sep > 0:
                    others = np.array(
                        [pos[nid] for nid in pos if nid != pn.node_id]
                    )
                    dnn = np.linalg.norm(
                        feasible[:, None, :] - others[None, :, :], axis=2
                    ).min(axis=1)
                    minx, miny, maxx, maxy = ctx.geofence_m.bounds
                    diag = math.hypot(maxx - minx, maxy - miny)
                    sep_bonus = w_sep * np.minimum(dnn / diag, 1.0)
                else:
                    sep_bonus = 0.0
                scores = gains_m2 - penalty_m2 + sep_bonus
                best_idx = int(np.argmax(scores))
                if scores[best_idx] <= 0 and cost_model.cost(1.0) > 0:
                    # No placement earns its movement cost; staying put
                    # is the correct move under this cost model.
                    continue
                best_xy = feasible[best_idx]
            else:
                # No coverage radio: pure relay — push the connected
                # frontier outward from the (first) c2. Frontier gain
                # is in meters, not m^2, so only the hard cap applies
                # (soft cost is documented as coverage-domain only).
                gw = pos[ctx.gateway.node_id]
                best_xy = feasible[
                    int(np.argmax(np.linalg.norm(feasible - gw, axis=1)))
                ]

            pos[pn.node_id] = best_xy.astype(float)
            agl[pn.node_id] = target_agl
            connected.add(pn.node_id)
            anchor_ids.add(pn.node_id)
            self._or_footprint(ctx, covered, eval_pts, best_xy, target_agl, pn)

            # Placing a node may bridge previously-unreachable relays
            # into the component: they count for coverage/connectivity
            # but never as redundancy anchors.
            newly = self._connected_set(ctx, pos, agl) - connected
            for nid in newly - movable_ids:
                self._or_footprint(
                    ctx, covered, eval_pts, pos[nid], agl[nid], ctx.by_id[nid]
                )
            connected |= newly

        return pos, agl

    # -- helpers -------------------------------------------------

    @staticmethod
    def _coverage_altitude(ctx: PlanningContext, pn: PlanNode) -> float:
        """AGL within the node's band maximizing ground footprint."""
        lo, hi = pn.band
        candidates = np.linspace(lo, hi, 7)
        radii = [ctx.rf.ground_footprint_radius(a) for a in candidates]
        return float(candidates[int(np.argmax(radii))])

    def _anchor_arrays(self, ctx, pn, anchor_ids, pos, agl, target_agl):
        anchors_xy, anchors_r = [], []
        for nid in anchor_ids:
            other = ctx.by_id[nid]
            r = ctx.safe_link_range(pn, other)
            if r <= 0:
                continue
            # budget horizontal distance for the altitude delta
            dz = target_agl - agl[nid]
            rh2 = r * r - dz * dz
            if rh2 <= 0:
                continue
            anchors_xy.append(pos[nid])
            anchors_r.append(np.sqrt(rh2))
        return np.array(anchors_xy), np.array(anchors_r)

    @staticmethod
    def _feasible(candidates, anchors_xy, anchors_r, need) -> np.ndarray:
        if need <= 0:
            return candidates
        if not len(anchors_xy):
            return np.empty((0, 2))
        d = np.linalg.norm(candidates[:, None, :] - anchors_xy[None, :, :], axis=2)
        count = (d <= anchors_r[None, :]).sum(axis=1)
        return candidates[count >= need]

    @staticmethod
    def _or_footprint(ctx, covered, eval_pts, xy, agl_m, pn):
        if not ctx.rf.has_coverage_radio(pn.radio_types):
            return
        r = ctx.rf.ground_footprint_radius(agl_m)
        if r <= 0:
            return
        d2 = np.sum((eval_pts - np.asarray(xy)) ** 2, axis=1)
        covered |= d2 <= r * r

    @staticmethod
    def _connected_set(ctx, pos, agl) -> set[str]:
        """C2 nodes plus everything currently reachable from one."""
        return set(ctx.connected_to_gateway(pos, agl)) | {
            g.node_id for g in ctx.gateways
        }

