"""
Pydantic models for the ARPO Model Input/Output schemas (v1.1.0).

Extends the v1.0.0 schema with:
  - ``ARPOInput.geofence``  — GeoJSON geometry *or* a bare list of points
  - ``Node.mobility``       — controlled | fixed | independent
  - ``Node.role``           — standard | gateway  (the "hub")
  - ``Node.altitude_band_m``— optional per-node min/max AGL band
  - ``ARPOOutput.diagnostics`` — optional per-COA metrics (non-breaking)

All additions are optional or defaulted, so v1.0.0 payloads that also
carry a geofence remain valid.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Union

from pydantic import (AliasChoices, BaseModel, Field,
                      field_validator, model_validator)


# -- Input Schema ------------------------------------------------


class NodeType(str, Enum):
    """Command relationship for a node.

    c2         — the hub; connectivity anchor for the whole mesh.
    controlled — ours; the planner may reposition it (if mobility allows).
    relay      — carries our radios but we don't control its movement
                 (vehicle/person-mounted, partner asset). Position is
                 honored as reported; radios still count toward the
                 mesh. Future planners may plan around its
                 predicted motion.
    """

    C2 = "c2"
    CONTROLLED = "controlled"
    RELAY = "relay"


class Mobility(str, Enum):
    """How a node moves. ``ground``/``aerial`` are movable platforms
    (and select the AGL band defaults); ``fixed`` is installed in
    place and never repositioned regardless of type."""

    GROUND = "ground"
    AERIAL = "aerial"
    FIXED = "fixed"


class AltitudeBand(BaseModel):
    """Allowed altitude band, meters above ground level."""

    min_agl_m: float = Field(..., ge=0)
    max_agl_m: float = Field(..., ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "AltitudeBand":
        if self.max_agl_m < self.min_agl_m:
            raise ValueError("max_agl_m must be >= min_agl_m")
        return self


class Position(BaseModel):
    """Current location of a node."""

    lat: float = Field(..., ge=-90, le=90, description="Latitude (WGS84)")
    lon: float = Field(..., ge=-180, le=180, description="Longitude (WGS84)")
    # AR-69 aggregates guarantee only lat/lon (position.yaml); default
    # the rest so field telemetry validates. Flat-earth planning takes
    # AGL from the altitude bands defined in the config.
    elevation_m: float = Field(
        0.0, ge=0, le=10_000_000, description="Elevation above sea level (m)"
    )
    heading_deg: float = Field(
        0.0, ge=0, le=360, description="Compass heading (deg)"
    )


class RadioEndpoint(BaseModel):
    node_id: str
    radio_id: str


class Throughput(BaseModel):
    tx_mbps: float = Field(..., ge=0)
    rx_mbps: float = Field(..., ge=0)


class Latency(BaseModel):
    rtt_ms: float = Field(..., ge=0)
    rtt_max_ms: Optional[float] = Field(None, ge=0)


class ConnectionMetrics(BaseModel):
    # AR-69 connection.yaml names this "snr"
    snr_db: float = Field(
        validation_alias=AliasChoices("snr_db", "snr")
    )
    throughput: Optional[Throughput] = None
    latency: Optional[Latency] = None


class Connection(BaseModel):
    source: RadioEndpoint
    target: RadioEndpoint
    metrics: ConnectionMetrics


class RadioMetrics(BaseModel):
    link_quality: Optional[float] = None
    noise_floor_dbm: Optional[float] = Field(None, ge=-120, le=0)


class Radio(BaseModel):
    radio_id: str
    radio_type: str
    metrics: Optional[RadioMetrics] = None


class Node(BaseModel):
    node_id: str
    position: Position
    radios: list[Radio] = Field(..., min_length=1)
    # AR-69 node metadata (state net.nodes) uses node_type /
    # mobility_type; enum VALUES are identical. Accept both names.
    type: NodeType = Field(
        validation_alias=AliasChoices("type", "node_type")
    )
    mobility: Mobility = Field(
        validation_alias=AliasChoices("mobility", "mobility_type")
    )
    altitude_band_m: Optional[AltitudeBand] = Field(
        None,
        description="Optional per-node AGL band; falls back to config "
        "defaults for the node's mobility.",
    )
    connections: Optional[list[Connection]] = None
    timestamp: Optional[int] = Field(None, description="Unix ms (aggregated)")

    @property
    def is_gateway(self) -> bool:
        return self.type == NodeType.C2

    @property
    def is_movable(self) -> bool:
        # relays are never ours to move; fixed never moves. A mobile
        # c2 (c2 + ground/aerial) IS movable — it anchors itself.
        return self.type != NodeType.RELAY and self.mobility != Mobility.FIXED

    def radio_types(self) -> set[str]:
        return {r.radio_type for r in self.radios}


# -- Geofence ----------------------------------------------------


class GeoPoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class Geofence(BaseModel):
    """Area of operation and coverage target.

    Exactly one of the two representations must be provided:

    - ``geojson``: a GeoJSON Geometry (Polygon or MultiPolygon, holes
      allowed as keep-out zones), Feature, or FeatureCollection.
      Coordinates follow the GeoJSON spec: **[lon, lat]** order.
    - ``points``: a bare ring as a list of points, either
      ``{"lat": .., "lon": ..}`` objects or ``[lat, lon]`` pairs
      (note: lat first — the opposite of GeoJSON order). The ring is
      closed automatically if the last point differs from the first.
    """

    geojson: Optional[dict[str, Any]] = None
    points: Optional[list[GeoPoint]] = None

    @field_validator("points", mode="before")
    @classmethod
    def _coerce_points(cls, v: Any) -> Any:
        if v is None:
            return v
        out = []
        for p in v:
            if isinstance(p, (list, tuple)):
                if len(p) != 2:
                    raise ValueError(f"point pair must have 2 items, got {p!r}")
                out.append({"lat": p[0], "lon": p[1]})
            else:
                out.append(p)
        return out

    @model_validator(mode="after")
    def _exactly_one(self) -> "Geofence":
        if (self.geojson is None) == (self.points is None):
            raise ValueError("provide exactly one of 'geojson' or 'points'")
        if self.points is not None and len(self.points) < 3:
            raise ValueError("a point-list geofence needs at least 3 points")
        return self


class ARPOInput(BaseModel):
    """Top-level ARPO model input."""

    nodes: list[Node] = Field(..., min_length=1)
    geofence: Geofence

    @field_validator("geofence", mode="before")
    @classmethod
    def _coerce_bare_geofence(cls, v):
        """AR-69 sends geofence as a bare array of {lat, lon} positions
        (transforms/schemas/arpo/geofence.yaml); wrap it as points."""
        if isinstance(v, list):
            return {"points": v}
        return v
    coas: Optional[list[str]] = Field(
        None,
        description="COAs to generate; defaults to all "
        "(coverage, distance, resilience).",
    )

    @model_validator(mode="after")
    def _has_gateway(self) -> "ARPOInput":
        if not any(n.is_gateway for n in self.nodes):
            raise ValueError("input must contain at least one c2 node")
        return self

    def gateways(self) -> list[Node]:
        return [n for n in self.nodes if n.is_gateway]


# -- Output Schema -----------------------------------------------


class NodePlacement(BaseModel):
    """Predicted node placement for a specific course of action."""

    node_id: str
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    elevation_m: float = Field(..., ge=0, le=10_000_000)
    heading_deg: float = Field(..., ge=0, le=360)


class COADiagnostics(BaseModel):
    """Optional per-COA quality metrics (additive; safe to ignore)."""

    coverage_fraction: Optional[float] = None
    max_extent_from_gateway_m: Optional[float] = None
    survives_single_node_loss: Optional[bool] = None
    controlled_mesh_survives_single_loss: Optional[bool] = Field(
        None,
        description="Single-node-loss survivability of the mesh the "
        "planner actually controls (c2 + movable nodes). This is the "
        "planner's guarantee; the stricter survives_single_node_loss "
        "also counts relays, whose redundancy we cannot control.",
    )
    single_backhaul_terminal: Optional[bool] = Field(
        None,
        description="True when exactly one field node bridges the BLOS "
        "backhaul — an unavoidable single point of failure the planner "
        "cannot mitigate (bring a second terminal). None when no BLOS "
        "radios are in play.",
    )
    redundant_link_fraction: Optional[float] = Field(
        None,
        description="Fraction of connected non-c2 nodes with >= 2 "
        "mesh links (1.0 = every node has a redundant path).",
    )
    connected_node_ids: Optional[list[str]] = None
    unreachable_node_ids: Optional[list[str]] = None
    notes: Optional[str] = None


class ARPOOutput(BaseModel):
    """Top-level ARPO model output: COA name → node placements."""

    coas: dict[str, dict[str, NodePlacement]] = Field(
        ...,
        description="Map of COA name to a map of node_id → NodePlacement",
    )
    diagnostics: Optional[dict[str, COADiagnostics]] = Field(
        None, description="Map of COA name to quality metrics"
    )
