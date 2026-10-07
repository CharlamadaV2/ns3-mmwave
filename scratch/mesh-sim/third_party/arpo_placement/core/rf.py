"""
Flat-earth free-space (Friis) RF engine.

Everything reduces to closed-form max ranges:

    FSPL(d, f) = 20 log10(d) + 20 log10(f) + 20 log10(4*pi/c)
    budget     = Ptx + Gtx + Grx - sensitivity - fade_margin
    d_max      = 10 ** ((budget - 20 log10(f) - K) / 20)

Two range flavors:
  * mesh range  — node radio <-> node radio of the same radio_type
  * coverage range — node radio -> reference receiver (ground user)

Coverage range is a slant range; the usable ground footprint radius of
a node at height h above the receiver plane is sqrt(d_max^2 - dh^2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml

_C = 299_792_458.0
_K = 20.0 * math.log10(4.0 * math.pi / _C)  # ≈ -147.55 dB


# -- Config models -----------------------------------------------


@dataclass(frozen=True)
class RadioSpec:
    radio_type: str
    frequency_hz: float = 0.0
    tx_power_dbm: float = 0.0
    tx_gain_dbi: float = 0.0
    rx_gain_dbi: float = 0.0
    rx_sensitivity_dbm: float = 0.0
    blos: bool = False
    """Beyond-line-of-sight backhaul (e.g. a satellite terminal).
    Any two nodes sharing a blos radio type are linked regardless of
    distance (full clique through the constellation); Friis params are
    ignored. v1 treats the backhaul as always-up and ideal."""


@dataclass(frozen=True)
class ReferenceReceiver:
    radio_type: str
    height_agl_m: float
    antenna_gain_dbi: float
    rx_sensitivity_dbm: float


@dataclass(frozen=True)
class AltitudeDefault:
    min_agl_m: float
    max_agl_m: float


@dataclass(frozen=True)
class OptimizerConfig:
    budget_s_per_coa: float = 5.0
    max_iters: Optional[int] = None  # set for fully reproducible runs
    seed: Optional[int] = None
    t0_area_frac: float = 0.02
    tf_area_frac: float = 0.0002
    nudge_sigma_frac: float = 0.08
    move_weights: tuple = (("nudge", 0.55), ("teleport", 0.20),
                           ("swap", 0.10), ("altitude", 0.15))
    balanced_vulnerability_frac: float = 0.02


@dataclass(frozen=True)
class BackhaulOverrides:
    radio_type: str = "starlink"
    node_ids: tuple = ()


@dataclass(frozen=True)
class SyntheticC2:
    node_id: str
    lat: float
    lon: float
    elevation_m: float = 0.0


@dataclass(frozen=True)
class MovementCost:
    """Cost of repositioning a node, in coverage-equivalent m^2
    (comparable to coverage gain), plus an optional hard cap."""

    fixed_cost_m2: float = 0.0
    cost_m2_per_m: float = 0.0
    max_displacement_m: Optional[float] = None

    def cost(self, displacement_m: float) -> float:
        if displacement_m <= 1e-6:
            return 0.0
        return self.fixed_cost_m2 + self.cost_m2_per_m * displacement_m

    def within_cap(self, displacement_m: float) -> bool:
        return (
            self.max_displacement_m is None
            or displacement_m <= self.max_displacement_m
        )


ZERO_COST = MovementCost()


@dataclass
class RFConfig:
    radios: dict[str, RadioSpec]
    fade_margin_db: float
    range_safety_factor: float
    reference_receiver: ReferenceReceiver
    altitude_defaults: dict[str, AltitudeDefault]
    terrain_elevation_m: Optional[float]
    balanced_core_fraction: float
    movement_cost: dict[str, MovementCost]
    optimizer: OptimizerConfig
    coverage_grid: dict[str, float]
    candidate_grid: dict[str, float]
    backhaul_overrides: BackhaulOverrides = BackhaulOverrides()
    synthetic_c2: Optional[SyntheticC2] = None

    @classmethod
    def load(cls, path: Union[str, Path, None] = None) -> "RFConfig":
        if path is None:
            path = Path(__file__).parent.parent / "rf_config.yaml"
        raw = yaml.safe_load(Path(path).read_text())

        radios = {}
        for name, spec in raw["radios"].items():
            if spec.get("blos", False):
                radios[name] = RadioSpec(radio_type=name, blos=True)
            else:
                radios[name] = RadioSpec(
                    radio_type=name,
                    frequency_hz=float(spec["frequency_ghz"]) * 1e9,
                    tx_power_dbm=float(spec["tx_power_dbm"]),
                    tx_gain_dbi=float(spec["tx_gain_dbi"]),
                    rx_gain_dbi=float(spec["rx_gain_dbi"]),
                    rx_sensitivity_dbm=float(spec["rx_sensitivity_dbm"]),
                )
        rr = raw["reference_receiver"]
        alts = {
            k: AltitudeDefault(float(v["min_agl_m"]), float(v["max_agl_m"]))
            for k, v in raw["altitude_defaults"].items()
        }
        return cls(
            radios=radios,
            fade_margin_db=float(raw["fade_margin_db"]),
            range_safety_factor=float(raw["range_safety_factor"]),
            reference_receiver=ReferenceReceiver(
                radio_type=rr["radio_type"],
                height_agl_m=float(rr["height_agl_m"]),
                antenna_gain_dbi=float(rr["antenna_gain_dbi"]),
                rx_sensitivity_dbm=float(rr["rx_sensitivity_dbm"]),
            ),
            altitude_defaults=alts,
            terrain_elevation_m=(
                float(raw["terrain_elevation_m"])
                if raw.get("terrain_elevation_m") is not None
                else None
            ),
            balanced_core_fraction=float(raw.get("balanced_core_fraction", 0.5)),
            optimizer=_parse_optimizer(raw.get("optimizer", {})),
            backhaul_overrides=BackhaulOverrides(
                radio_type=raw.get("backhaul_overrides", {}).get(
                    "radio_type", "starlink"),
                node_ids=tuple(raw.get("backhaul_overrides", {}).get(
                    "node_ids", []) or []),
            ),
            synthetic_c2=(
                SyntheticC2(
                    node_id=str(raw["synthetic_c2"]["node_id"]),
                    lat=float(raw["synthetic_c2"]["lat"]),
                    lon=float(raw["synthetic_c2"]["lon"]),
                    elevation_m=float(
                        raw["synthetic_c2"].get("elevation_m", 0.0)),
                )
                if raw.get("synthetic_c2") else None
            ),
            movement_cost={
                k: MovementCost(
                    fixed_cost_m2=float(v.get("fixed_cost_m2", 0.0)),
                    cost_m2_per_m=float(v.get("cost_m2_per_m", 0.0)),
                    max_displacement_m=(
                        float(v["max_displacement_m"])
                        if v.get("max_displacement_m") is not None
                        else None
                    ),
                )
                for k, v in raw.get("movement_cost", {}).items()
            },
            coverage_grid=dict(raw["coverage_grid"]),
            candidate_grid=dict(raw["candidate_grid"]),
        )


# -- Friis math --------------------------------------------------


def fspl_db(distance_m: float, frequency_hz: float) -> float:
    if distance_m <= 0:
        return -math.inf
    return 20.0 * math.log10(distance_m) + 20.0 * math.log10(frequency_hz) + _K


def max_range_m(
    tx_power_dbm: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    rx_sensitivity_dbm: float,
    frequency_hz: float,
    fade_margin_db: float,
) -> float:
    budget = (
        tx_power_dbm + tx_gain_dbi + rx_gain_dbi - rx_sensitivity_dbm - fade_margin_db
    )
    if budget <= 0:
        return 0.0
    return 10.0 ** ((budget - 20.0 * math.log10(frequency_hz) - _K) / 20.0)


class RFEngine:
    """Precomputes per-radio-type ranges from an RFConfig."""

    def __init__(self, config: RFConfig):
        self.config = config
        self._mesh_range: dict[str, float] = {}
        for name, spec in config.radios.items():
            if spec.blos:
                self._mesh_range[name] = math.inf
                continue
            self._mesh_range[name] = max_range_m(
                spec.tx_power_dbm,
                spec.tx_gain_dbi,
                spec.rx_gain_dbi,
                spec.rx_sensitivity_dbm,
                spec.frequency_hz,
                config.fade_margin_db,
            )
        rr = config.reference_receiver
        node_spec = config.radios.get(rr.radio_type)
        if node_spec is not None and node_spec.blos:
            raise ValueError(
                "reference_receiver.radio_type cannot be a blos backhaul"
            )
        if node_spec is None:
            raise ValueError(
                f"reference_receiver.radio_type {rr.radio_type!r} is not in "
                f"the radio registry ({sorted(config.radios)})"
            )
        # Node transmits, reference receiver listens.
        self._coverage_slant_range = max_range_m(
            node_spec.tx_power_dbm,
            node_spec.tx_gain_dbi,
            rr.antenna_gain_dbi,
            rr.rx_sensitivity_dbm,
            node_spec.frequency_hz,
            config.fade_margin_db,
        )

    # -- mesh (node <-> node) ------------------------------------

    def mesh_range(self, radio_type: str) -> float:
        return self._mesh_range.get(radio_type, 0.0)

    def pair_mesh_range(
        self, types_a: set[str], types_b: set[str]
    ) -> float:
        """Best link range between two nodes = max over shared types.
        Cross-type links are assumed impossible."""
        shared = types_a & types_b
        if not shared:
            return 0.0
        return max(self._mesh_range.get(t, 0.0) for t in shared)

    # -- coverage (node -> reference receiver) -------------------

    @property
    def coverage_slant_range(self) -> float:
        return self._coverage_slant_range

    def ground_footprint_radius(self, node_height_agl_m: float) -> float:
        """Radius on the receiver plane covered by a node at the given
        AGL height. Zero if the node carries no reference-compatible
        radio type — callers should gate on has_coverage_radio()."""
        dh = node_height_agl_m - self.config.reference_receiver.height_agl_m
        r2 = self._coverage_slant_range**2 - dh**2
        return math.sqrt(r2) if r2 > 0 else 0.0

    def has_coverage_radio(self, radio_types: set[str]) -> bool:
        return self.config.reference_receiver.radio_type in radio_types


def _parse_optimizer(raw: dict) -> OptimizerConfig:
    mw = raw.get("move_weights", {})
    defaults = dict(OptimizerConfig.move_weights)
    weights = tuple((k, float(mw.get(k, v))) for k, v in defaults.items())
    return OptimizerConfig(
        budget_s_per_coa=float(raw.get("budget_s_per_coa", 5.0)),
        max_iters=(int(raw["max_iters"]) if raw.get("max_iters") is not None
                   else None),
        seed=(int(raw["seed"]) if raw.get("seed") is not None else None),
        t0_area_frac=float(raw.get("t0_area_frac", 0.02)),
        tf_area_frac=float(raw.get("tf_area_frac", 0.0002)),
        nudge_sigma_frac=float(raw.get("nudge_sigma_frac", 0.08)),
        move_weights=weights,
        balanced_vulnerability_frac=float(
            raw.get("balanced_vulnerability_frac", 0.02)
        ),
    )
