"""
Coordinate handling: everything internal is meters in a local frame.

Project WGS84 lat/lon into a local azimuthal equidistant (AEQD) 
frame centered on the geofence centroid, do all geometry/RF 
math in meters, and convert back to WGS84 only at the
output boundary. Distortion over AOIs of tens of km is negligible.
"""

from __future__ import annotations

import logging
import math

from dataclasses import dataclass
from typing import Any, Union

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import transform as shp_transform
from shapely.validation import make_valid

from models import Geofence

GeofenceGeom = Union[Polygon, MultiPolygon]


# -- Geofence normalization (lat/lon domain) ---------------------


def _repair_ring(ring: list[tuple[float, float]]) -> Polygon:
    """Salvage a self-intersecting point-list fence.

    Some emitters (e.g. the ATAK plugin) send vertices out of
    perimeter order — corners first, then edge midpoints — which
    reads as a self-crossing 'bowtie' ring and, if merely
    make_valid'ed, fragments into slivers covering a fraction of the
    intended fence. Repair strategy, most to least faithful:

      1. Valid as-given: never touched (preserves concave fences
         drawn in proper order).
      2. Angle-sort around the centroid: exactly reconstructs the
         perimeter for any convex fence regardless of emission order
         (corners+midpoints included).
      3. Convex hull of the points: last resort.

    Repairs are logged loudly — a reordered fence should be fixed at
    the source even though the model can cope.
    """
    poly = Polygon(ring)
    if poly.is_valid:
        return poly

    pts = list(dict.fromkeys(ring[:-1] if ring[0] == ring[-1] else ring))
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    ordered = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    candidate = Polygon(ordered + [ordered[0]])
    if candidate.is_valid and candidate.area > 0:
        logging.getLogger("arpo-model").warning(
            "geofence ring was self-intersecting (%d vertices out of "
            "perimeter order); repaired by angle-sort around centroid "
            "(salvageable area %.3g -> %.3g units^2, +%.0f%%). "
            "Fix the emitter.",
            len(pts), make_valid(poly).area, candidate.area,
            100.0 * (candidate.area / max(make_valid(poly).area, 1e-12) - 1),
        )
        return candidate

    hull = Polygon(pts).convex_hull
    logging.getLogger("arpo-model").warning(
        "geofence ring was self-intersecting and not angle-sortable; "
        "using convex hull of its points. Fix the emitter.",
    )
    return hull


def geofence_to_lonlat_geom(geofence: Geofence) -> GeofenceGeom:
    """Normalize either representation to a shapely (Multi)Polygon in
    lon/lat (x=lon, y=lat) coordinates."""
    if geofence.points is not None:
        ring = [(p.lon, p.lat) for p in geofence.points]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        geom: GeofenceGeom = _repair_ring(ring)
    else:
        geom = _geojson_to_geom(geofence.geojson)

    geom = make_valid(geom)
    geom = _keep_polygonal(geom)
    if geom.is_empty:
        raise ValueError("geofence resolves to an empty geometry")
    return geom


def _geojson_to_geom(gj: dict[str, Any]) -> GeofenceGeom:
    t = gj.get("type")
    if t == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in gj.get("features", [])]
        polys: list[Polygon] = []
        for g in geoms:
            polys.extend(_polygons_of(g))
        if not polys:
            raise ValueError("FeatureCollection contains no polygonal geometry")
        return MultiPolygon(polys) if len(polys) > 1 else polys[0]
    if t == "Feature":
        return _keep_polygonal(shape(gj["geometry"]))
    return _keep_polygonal(shape(gj))


def _polygons_of(geom) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        out: list[Polygon] = []
        for g in geom.geoms:
            out.extend(_polygons_of(g))
        return out
    return []


def _keep_polygonal(geom) -> GeofenceGeom:
    polys = _polygons_of(geom)
    if not polys:
        raise ValueError(
            f"geofence must be polygonal, got {geom.geom_type!r}"
        )
    return MultiPolygon(polys) if len(polys) > 1 else polys[0]


# -- Local metric frame ------------------------------------------


@dataclass
class LocalFrame:
    """Bidirectional WGS84 <-> local AEQD (meters) transform."""

    center_lat: float
    center_lon: float

    def __post_init__(self) -> None:
        aeqd = CRS.from_proj4(
            f"+proj=aeqd +lat_0={self.center_lat} +lon_0={self.center_lon} "
            "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        wgs84 = CRS.from_epsg(4326)
        self._fwd = Transformer.from_crs(wgs84, aeqd, always_xy=True)
        self._inv = Transformer.from_crs(aeqd, wgs84, always_xy=True)

    @classmethod
    def centered_on(cls, geom_lonlat: GeofenceGeom) -> "LocalFrame":
        c = geom_lonlat.centroid
        return cls(center_lat=c.y, center_lon=c.x)

    # scalar / array friendly (pyproj handles both)
    def to_xy(self, lat, lon):
        """lat/lon (deg) -> x/y (m). Returns (x, y)."""
        return self._fwd.transform(lon, lat)

    def to_latlon(self, x, y):
        """x/y (m) -> (lat, lon) in degrees."""
        lon, lat = self._inv.transform(x, y)
        return lat, lon

    def project_geom(self, geom_lonlat: GeofenceGeom) -> GeofenceGeom:
        return shp_transform(self._fwd.transform, geom_lonlat)


# -- Grid sampling -----------------------------------------------


def grid_points_in(
    geom_m: GeofenceGeom,
    target_cells: int = 10_000,
    min_resolution_m: float = 5.0,
) -> tuple[np.ndarray, float]:
    """Regular grid of points inside a metric-frame geofence.

    Returns (points[N,2], cell_size_m). Cell size is chosen so the
    bounding box holds roughly ``target_cells`` cells, floored at
    ``min_resolution_m``.
    """
    minx, miny, maxx, maxy = geom_m.bounds
    w, h = maxx - minx, maxy - miny
    if w <= 0 or h <= 0:
        raise ValueError("degenerate geofence bounds")
    cell = max(min_resolution_m, float(np.sqrt(w * h / target_cells)))
    xs = np.arange(minx + cell / 2, maxx, cell)
    ys = np.arange(miny + cell / 2, maxy, cell)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])

    from shapely import contains_xy  # shapely >= 2.0 vectorized predicate

    mask = contains_xy(geom_m, pts[:, 0], pts[:, 1])
    inside = pts[mask]
    if len(inside) == 0:
        # Geofence smaller than one cell — fall back to a point
        # guaranteed interior. (NOT the centroid: for concave shapes
        # like an L or C the centroid can lie outside the polygon.)
        c = geom_m.representative_point()
        inside = np.array([[c.x, c.y]])
    return inside, cell


def bearing_deg(from_xy: np.ndarray, to_xy: np.ndarray) -> float:
    """Compass bearing (0=N, 90=E) from one metric point to another."""
    dx = float(to_xy[0] - from_xy[0])
    dy = float(to_xy[1] - from_xy[1])
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return float(np.degrees(np.arctan2(dx, dy)) % 360.0)
