"""
PlacementModel — the interface every backend implements.

Backends planned:
  * GeometricModel     (model 1: Friis + geometry heuristics)  — done
  * OptimizationModel  (model 2: meshopt-style optimizer)      — done
  * RLModel            (model 3: learned policy per COA)       — in progress
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.context import PlanningContext
from core.geometry import bearing_deg
from models import COADiagnostics, NodePlacement


class COAResult:
    """A single COA solution in the metric frame."""

    def __init__(
        self,
        pos: dict[str, np.ndarray],
        agl: dict[str, float],
        diagnostics: COADiagnostics,
    ):
        self.pos = pos
        self.agl = agl
        self.diagnostics = diagnostics


class PlacementModel(ABC):
    """Solves one or more COAs against a PlanningContext."""

    COA_NAMES = ("coverage", "balanced", "resilience")

    @abstractmethod
    def solve(self, ctx: PlanningContext, coa: str) -> COAResult:
        ...

    # -- shared: diagnostics -------------------------------------

    def _diagnose(self, ctx: PlanningContext, pos, agl) -> COADiagnostics:
        connected = ctx.connected_to_gateway(pos, agl)
        all_ids = {pn.node_id for pn in ctx.nodes}
        return COADiagnostics(
            coverage_fraction=round(ctx.coverage_fraction(pos, agl), 4),
            max_extent_from_gateway_m=round(
                ctx.max_extent_from_gateway(pos, agl), 1
            ),
            survives_single_node_loss=ctx.survives_single_loss(pos, agl),
            controlled_mesh_survives_single_loss=(
                ctx.controlled_mesh_survives_single_loss(pos, agl)
            ),
            single_backhaul_terminal=ctx.single_backhaul_terminal(pos, agl),
            redundant_link_fraction=round(
                ctx.redundant_link_fraction(pos, agl), 4
            ),
            connected_node_ids=sorted(connected),
            unreachable_node_ids=sorted(all_ids - connected),
        )

    # -- shared: metric frame -> wire schema ---------------------

    def to_placements(
        self, ctx: PlanningContext, result: COAResult
    ) -> dict[str, NodePlacement]:
        out: dict[str, NodePlacement] = {}
        gw_xy = result.pos[ctx.gateway.node_id]
        for pn in ctx.nodes:
            xy = result.pos[pn.node_id]
            agl = result.agl[pn.node_id]
            lat, lon = ctx.frame.to_latlon(float(xy[0]), float(xy[1]))
            if pn.is_gateway or np.allclose(xy, gw_xy):
                heading = pn.node.position.heading_deg % 360.0
            else:
                heading = bearing_deg(xy, gw_xy)  # face the hub
            out[pn.node_id] = NodePlacement(
                node_id=pn.node_id,
                lat=float(lat),
                lon=float(lon),
                elevation_m=max(0.0, ctx.terrain_elev_m + agl),
                heading_deg=float(heading % 360.0),
            )
        return out
