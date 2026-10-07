"""
PlanningContext — the shared, metric-frame view of a request.

Every model backend (geometric, optimization, RL) consumes this same
object, so schema parsing, projection, altitude handling, connectivity
math, and coverage evaluation are written once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Optional

import numpy as np

from core.geometry import (
    GeofenceGeom,
    LocalFrame,
    geofence_to_lonlat_geom,
    grid_points_in,
)
from core.rf import RFConfig, RFEngine, ZERO_COST, MovementCost
from models import ARPOInput, Mobility, Node


@dataclass
class PlanNode:
    """A node in the metric frame."""

    node: Node
    xy: np.ndarray            # [x, y] meters in local frame
    agl_m: float              # current height above (flat) terrain
    band: tuple[float, float]  # (min_agl, max_agl) allowed
    radio_types: set[str] = field(default_factory=set)

    @property
    def node_id(self) -> str:
        return self.node.node_id

    @property
    def movable(self) -> bool:
        return self.node.is_movable

    @property
    def is_gateway(self) -> bool:
        return self.node.is_gateway


class PlanningContext:
    def __init__(self, request: ARPOInput, config: Optional[RFConfig] = None):
        self.request = request
        self.config = config or RFConfig.load()
        self.rf = RFEngine(self.config)

        # Geofence: lon/lat -> local metric frame
        lonlat_geom = geofence_to_lonlat_geom(request.geofence)
        self.frame = LocalFrame.centered_on(lonlat_geom)
        self.geofence_m: GeofenceGeom = self.frame.project_geom(lonlat_geom)

        # Flat-earth terrain elevation (m ASL)
        self.terrain_elev_m = self._infer_terrain_elevation()

        # Nodes into the metric frame
        self.nodes: list[PlanNode] = [self._to_plan_node(n) for n in request.nodes]
        self.by_id = {pn.node_id: pn for pn in self.nodes}
        self.gateway = next(pn for pn in self.nodes if pn.is_gateway)
        self.gateways = [pn for pn in self.nodes if pn.is_gateway]
        self.movable = [pn for pn in self.nodes if pn.movable]
        self.unmovable = [pn for pn in self.nodes if not pn.movable]

        # Coverage evaluation grid
        cg = self.config.coverage_grid
        self.eval_points, self.eval_cell_m = grid_points_in(
            self.geofence_m,
            target_cells=int(cg["target_cells"]),
            min_resolution_m=float(cg["min_resolution_m"]),
        )

    # -- construction helpers ------------------------------------

    def _infer_terrain_elevation(self) -> float:
        if self.config.terrain_elevation_m is not None:
            return self.config.terrain_elevation_m
        ground = [
            n.position.elevation_m
            for n in self.request.nodes
            if n.mobility in (Mobility.GROUND, Mobility.FIXED)
        ]
        if ground:
            return float(median(ground))
        return float(min(n.position.elevation_m for n in self.request.nodes))

    def _to_plan_node(self, node: Node) -> PlanNode:
        x, y = self.frame.to_xy(node.position.lat, node.position.lon)
        agl = max(0.0, node.position.elevation_m - self.terrain_elev_m)
        if node.altitude_band_m is not None:
            band = (node.altitude_band_m.min_agl_m, node.altitude_band_m.max_agl_m)
        elif node.mobility == Mobility.FIXED:
            band = (agl, agl)  # installed in place, current height honored
        else:
            d = self.config.altitude_defaults.get(node.mobility.value)
            band = (d.min_agl_m, d.max_agl_m) if d else (agl, agl)
        return PlanNode(
            node=node,
            xy=np.array([float(x), float(y)]),
            agl_m=agl,
            band=band,
            radio_types=node.radio_types(),
        )

    def movement_cost(self, pn: "PlanNode") -> MovementCost:
        """Cost model for repositioning this node (ZERO_COST when the
        node's mobility class has none configured)."""
        return self.config.movement_cost.get(
            pn.node.mobility.value, ZERO_COST
        )

    # -- RF / geometry helpers -----------------------------------

    def link_range(self, a: PlanNode, b: PlanNode) -> float:
        """Max 3D link distance between two nodes (m); 0 = no shared radio."""
        return self.rf.pair_mesh_range(a.radio_types, b.radio_types)

    def safe_link_range(self, a: PlanNode, b: PlanNode) -> float:
        return self.link_range(a, b) * self.config.range_safety_factor

    def linked(
        self,
        a: PlanNode,
        b: PlanNode,
        pos: dict[str, np.ndarray],
        agl: dict[str, float],
    ) -> bool:
        r = self.link_range(a, b)
        if r <= 0:
            return False
        dxy = pos[a.node_id] - pos[b.node_id]
        dz = agl[a.node_id] - agl[b.node_id]
        return math.hypot(math.hypot(dxy[0], dxy[1]), dz) <= r

    def adjacency(
        self, pos: dict[str, np.ndarray], agl: dict[str, float]
    ) -> dict[str, set[str]]:
        adj: dict[str, set[str]] = {pn.node_id: set() for pn in self.nodes}
        for i, a in enumerate(self.nodes):
            for b in self.nodes[i + 1 :]:
                if self.linked(a, b, pos, agl):
                    adj[a.node_id].add(b.node_id)
                    adj[b.node_id].add(a.node_id)
        return adj

    def connected_to_gateway(
        self, pos: dict[str, np.ndarray], agl: dict[str, float]
    ) -> set[str]:
        """IDs of nodes reachable from any gateway."""
        adj = self.adjacency(pos, agl)
        seen: set[str] = set()
        stack = [g.node_id for g in self.gateways]
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            stack.extend(adj[nid] - seen)
        return seen

    # -- coverage evaluation -------------------------------------

    def coverage_mask(
        self, pos: dict[str, np.ndarray], agl: dict[str, float]
    ) -> np.ndarray:
        """Boolean mask over eval_points: covered by >=1 connected node
        carrying the reference receiver's radio type."""
        connected = self.connected_to_gateway(pos, agl)
        mask = np.zeros(len(self.eval_points), dtype=bool)
        for pn in self.nodes:
            if pn.node_id not in connected:
                continue
            if not self.rf.has_coverage_radio(pn.radio_types):
                continue
            r = self.rf.ground_footprint_radius(agl[pn.node_id])
            if r <= 0:
                continue
            d2 = np.sum((self.eval_points - pos[pn.node_id]) ** 2, axis=1)
            mask |= d2 <= r * r
        return mask

    def coverage_fraction(
        self, pos: dict[str, np.ndarray], agl: dict[str, float]
    ) -> float:
        m = self.coverage_mask(pos, agl)
        return float(m.mean()) if len(m) else 0.0

    def max_extent_from_gateway(
        self, pos: dict[str, np.ndarray], agl: dict[str, float]
    ) -> float:
        """Max distance from any connected node to its nearest mesh
        root. Roots are the connected backhaul terminals when a BLOS
        c2 setup is in play (measuring from a command post 50 km
        outside the fence would be meaningless), else the gateways."""
        connected = self.connected_to_gateway(pos, agl)
        roots = self._connected_terminals(pos, agl) or [
            g.node_id for g in self.gateways
        ]
        root_xy = [pos[r] for r in roots]
        best = 0.0
        for nid in connected:
            if nid in roots or self.by_id[nid].is_gateway:
                continue
            d = min(float(np.linalg.norm(pos[nid] - r)) for r in root_xy)
            best = max(best, d)
        return best

    def survives_single_loss(
        self, pos: dict[str, np.ndarray], agl: dict[str, float]
    ) -> bool:
        """True if removing any single non-gateway node leaves every
        remaining node connected to a gateway."""
        base = self.connected_to_gateway(pos, agl)
        others = [pn for pn in self.nodes if not pn.is_gateway]
        for victim in others:
            adj = self.adjacency(pos, agl)
            # BFS from gateways skipping the victim
            seen: set[str] = set()
            stack = [g.node_id for g in self.gateways]
            while stack:
                nid = stack.pop()
                if nid in seen or nid == victim.node_id:
                    continue
                seen.add(nid)
                stack.extend(adj[nid] - seen)
            expected = (base - {victim.node_id}) | {
                g.node_id for g in self.gateways
            }
            if not expected <= seen:
                return False
        return True

    def backhaul_types(self) -> set[str]:
        return {n for n, s in self.config.radios.items() if s.blos}

    def _connected_terminals(
        self, pos: dict[str, np.ndarray], agl: dict[str, float]
    ) -> list[str]:
        """Non-c2 connected nodes carrying a blos backhaul radio."""
        bt = self.backhaul_types()
        if not bt:
            return []
        connected = self.connected_to_gateway(pos, agl)
        return [
            pn.node_id
            for pn in self.nodes
            if pn.node_id in connected
            and not pn.is_gateway
            and pn.radio_types & bt
        ]

    def single_backhaul_terminal(
        self, pos: dict[str, np.ndarray], agl: dict[str, float]
    ):
        """True if exactly one connected field terminal bridges the
        backhaul (an unavoidable single point of failure); False if
        several; None when no blos radios are in play."""
        if not self.backhaul_types():
            return None
        return len(self._connected_terminals(pos, agl)) == 1

    def _unmitigable_victims(
        self, pos: dict[str, np.ndarray], agl: dict[str, float]
    ) -> set[str]:
        """Victims whose loss no placement could mitigate: the sole
        connected backhaul terminal. Excluded from planner-guarantee
        metrics and from the vulnerability objective (a constant,
        unfixable penalty would pollute both); the strict
        survives_single_node_loss metric still counts them honestly."""
        terms = self._connected_terminals(pos, agl)
        return {terms[0]} if len(terms) == 1 else set()

    def controlled_mesh_survives_single_loss(
        self, pos: dict[str, np.ndarray], agl: dict[str, float]
    ) -> bool:
        """Single-loss survivability restricted to the mesh we plan:
        c2 nodes + connected movable nodes. Relays are excluded — we
        don't control them, so their redundancy can't be guaranteed."""
        connected = self.connected_to_gateway(pos, agl)
        core_ids = {pn.node_id for pn in self.movable} & connected
        skip = self._unmitigable_victims(pos, agl)
        for victim in core_ids - skip:
            adj = self.adjacency(pos, agl)
            seen: set[str] = set()
            stack = [g.node_id for g in self.gateways]
            while stack:
                nid = stack.pop()
                if nid in seen or nid == victim:
                    continue
                seen.add(nid)
                stack.extend(adj[nid] - seen)
            if not (core_ids - {victim}) <= seen:
                return False
        return True

    def redundant_link_fraction(
        self, pos: dict[str, np.ndarray], agl: dict[str, float]
    ) -> float:
        """Fraction of connected non-c2 nodes with >= 2 mesh links."""
        connected = self.connected_to_gateway(pos, agl)
        adj = self.adjacency(pos, agl)
        ids = [
            pn.node_id
            for pn in self.nodes
            if pn.node_id in connected and not pn.is_gateway
        ]
        if not ids:
            return 1.0
        return sum(1 for nid in ids if len(adj[nid]) >= 2) / len(ids)

    # -- output helpers ------------------------------------------

    def current_positions(self) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        pos = {pn.node_id: pn.xy.copy() for pn in self.nodes}
        agl = {pn.node_id: pn.agl_m for pn in self.nodes}
        return pos, agl
