"""Composable reward components computed from the facts window sums."""

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from .observations import SINR_CLIP_DB, SINR_INVALID_DB, schema_sha256

DEMAND_EPS = 1e-9
SUCCESS_DELIVERY = 0.95
SUCCESS_ROUTABLE = 0.95


def _delivery_ratio(window: dict, msg_reward: float, contract: dict) -> tuple[float, bool]:
    demand = float(window["demand_mbps_sum"])
    if demand <= DEMAND_EPS:
        return 0.0, False
    return float(window["delivered_mbps_sum"]) / demand, True


def _delivery_binary(window: dict, msg_reward: float, contract: dict) -> tuple[float, bool]:
    """Reward any measured delivery and penalize a complete service failure."""
    demand = float(window["demand_mbps_sum"])
    if demand <= DEMAND_EPS:
        return 0.0, False
    delivered = float(window["delivered_mbps_sum"])
    return (1.0 if delivered > DEMAND_EPS else -1.0), True


def _signed_delivery_ratio(window: dict, msg_reward: float,
                           contract: dict) -> tuple[float, bool]:
    """Map delivered/offered demand from [0, 1] to a symmetric [-1, 1]."""
    ratio, valid = _delivery_ratio(window, msg_reward, contract)
    if not valid:
        return 0.0, False
    return 2.0 * max(0.0, min(1.0, ratio)) - 1.0, True


def _connectivity(window: dict, msg_reward: float, contract: dict) -> tuple[float, bool]:
    pairs = int(window["ticks"]) * int(contract["num_links"])
    if pairs <= 0:
        return 0.0, True
    return float(window["connected_pairs_sum"]) / pairs, True


def _throughput_mbps(window: dict, msg_reward: float, contract: dict) -> tuple[float, bool]:
    return float(window["delivered_mbps_sum"]) / int(window["ticks"]), True


def _legacy(window: dict, msg_reward: float, contract: dict) -> tuple[float, bool]:
    return float(window["legacy_reward_sum"]) / int(window["ticks"]), True


def _service_success(window: dict, msg_reward: float, contract: dict) -> tuple[float, bool]:
    demand = float(window["demand_mbps_sum"])
    flows = int(window["flow_ticks_with_demand"])
    if demand <= DEMAND_EPS or flows == 0:
        return -1.0, True
    delivered = float(window["delivered_mbps_sum"]) / demand
    routable = 1.0 - int(window["unroutable_flow_ticks"]) / flows
    return (1.0 if delivered >= SUCCESS_DELIVERY and routable >= SUCCESS_ROUTABLE
            else -1.0), True


def _service_failure(window: dict, msg_reward: float, contract: dict) -> tuple[float, bool]:
    success, _ = _service_success(window, msg_reward, contract)
    return (1.0 if success < 0 else 0.0), True


def position_context(facts: dict, previous_nodes: list, initial_nodes: list,
                     contract: dict) -> dict:
    """Measured position costs for the action just completed, from existing facts."""
    slots = [node_id for node_id in contract["slot_node_ids"] if node_id is not None]
    node_ids = contract["node_ids"]
    interval = int(facts["window"]["ticks"]) * float(contract["tick_s"])
    bounds = contract["bounds"]
    diagonal = math.hypot(float(bounds["x_max"]) - float(bounds["x_min"]),
                          float(bounds["y_max"]) - float(bounds["y_min"]))
    if interval <= 0 or diagonal <= 0:
        raise ValueError("movement reward needs a positive completed window and xy bounds")
    travel, displacement = [], []
    for slot, node_id in enumerate(slots):
        index = node_ids.index(node_id)
        current, previous, initial = (facts["nodes"][index], previous_nodes[index],
                                      initial_nodes[index])
        speed = float(contract["slot_speed_mps"][slot])
        travel.append(min(1.0, math.hypot(current[0] - previous[0],
                                           current[1] - previous[1]) / (speed * interval)))
        displacement.append(min(1.0, math.hypot(current[0] - initial[0],
                                                 current[1] - initial[1]) / diagonal))
    qualities = [max(0.0, min(1.0, (float(link[0]) - SINR_CLIP_DB[0]) /
                                  (SINR_CLIP_DB[1] - SINR_CLIP_DB[0])))
                 if float(link[0]) > SINR_INVALID_DB else 0.0
                 for link in facts["links"]]
    return {"travel_fraction": sum(travel) / len(travel),
            "origin_fraction": sum(displacement) / len(displacement),
            "sinr_quality": sum(qualities) / len(qualities) if qualities else 0.0}


def _travel_fraction(window: dict, msg_reward: float, contract: dict,
                     context: dict) -> tuple[float, bool]:
    return float(context["travel_fraction"]), True


def _origin_fraction(window: dict, msg_reward: float, contract: dict,
                     context: dict) -> tuple[float, bool]:
    return float(context["origin_fraction"]), True


def _sinr_quality(window: dict, msg_reward: float, contract: dict,
                  context: dict) -> tuple[float, bool]:
    return float(context["sinr_quality"]), True


def _unmet_sinr_quality(window: dict, msg_reward: float, contract: dict,
                        context: dict) -> tuple[float, bool]:
    """Shaping disappears once measured service meets the success threshold."""
    demand = float(window["demand_mbps_sum"])
    ratio = (float(window["delivered_mbps_sum"]) / demand
             if demand > DEMAND_EPS else 0.0)
    unmet = max(0.0, min(1.0, (SUCCESS_DELIVERY - ratio) / SUCCESS_DELIVERY))
    return float(context["sinr_quality"]) * unmet, True


@dataclass(frozen=True)
class RewardComponent:
    """One named scalar term with its validity rule and declared range."""

    name: str
    _value: Callable[..., tuple[float, bool]]
    range: tuple[float, float | None]

    def value(self, window: dict, msg_reward: float, contract: dict,
              context: dict | None = None) -> tuple[float, bool]:
        if self.name in ("travel_fraction", "origin_fraction", "sinr_quality",
                         "unmet_sinr_quality"):
            if context is None:
                raise ValueError(f"{self.name} requires measured position context")
            return self._value(window, msg_reward, contract, context)
        return self._value(window, msg_reward, contract)


COMPONENTS = {
    "delivery_ratio": RewardComponent("delivery_ratio", _delivery_ratio, (0.0, 1.0)),
    "delivery_binary": RewardComponent("delivery_binary", _delivery_binary,
                                        (-1.0, 1.0)),
    "signed_delivery_ratio": RewardComponent("signed_delivery_ratio",
                                               _signed_delivery_ratio,
                                               (-1.0, 1.0)),
    "connectivity": RewardComponent("connectivity", _connectivity, (0.0, 1.0)),
    "throughput_mbps": RewardComponent("throughput_mbps", _throughput_mbps, (0.0, None)),
    "legacy": RewardComponent("legacy", _legacy, (None, None)),
    "service_success": RewardComponent("service_success", _service_success, (-1.0, 1.0)),
    "service_failure": RewardComponent("service_failure", _service_failure, (0.0, 1.0)),
    "travel_fraction": RewardComponent("travel_fraction", _travel_fraction, (0.0, 1.0)),
    "origin_fraction": RewardComponent("origin_fraction", _origin_fraction, (0.0, 1.0)),
    "sinr_quality": RewardComponent("sinr_quality", _sinr_quality, (0.0, 1.0)),
    "unmet_sinr_quality": RewardComponent("unmet_sinr_quality",
                                            _unmet_sinr_quality, (0.0, 1.0)),
}


def get_component(name: str) -> RewardComponent:
    """Look up a registered reward component by name."""
    component = COMPONENTS.get(name)
    if component is None:
        raise ValueError(
            f"Unknown reward_components entry {name!r}; "
            f"valid choices: {sorted(COMPONENTS)}"
        )
    return component


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-component values, validity, and the weighted total returned to the policy."""

    total: float
    components: dict
    valid: dict
    weights: dict
    legacy: float


class RewardComposer:
    """Weighted sum of registered components over one decision window."""

    def __init__(self, components: Sequence[str], weights: Sequence[float]):
        names = list(components)
        values = [float(w) for w in weights]
        if len(names) != len(values):
            raise ValueError(
                f"reward_weights has {len(values)} entries but reward_components has "
                f"{len(names)}"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"reward_components has duplicate entries: {names}")
        bad = next((w for w in values if not math.isfinite(w)), None)
        if bad is not None:
            raise ValueError(f"reward_weights must all be finite, got {bad!r}")
        self.components = [get_component(name) for name in names]
        self.weights = values

    def compose(self, window: dict, msg_reward: float, contract: dict,
                context: dict | None = None) -> RewardBreakdown:
        values, valid, weights = {}, {}, {}
        total = 0.0
        for component, weight in zip(self.components, self.weights):
            value, ok = component.value(window, msg_reward, contract, context)
            values[component.name] = float(value)
            valid[component.name] = int(bool(ok))
            weights[component.name] = float(weight)
            if ok:
                total += weight * float(value)
        if not math.isfinite(total):
            raise ValueError(f"composed reward is not finite: {total!r} from {values}")
        return RewardBreakdown(float(total), values, valid, weights, float(msg_reward))


def reward_schema(components: Sequence[str], weights: Sequence[float], *,
                  reward_type: str, reward_window: str) -> dict:
    """Reward identity; C++ authority (reward_type/reward_window from init) when empty."""
    if not list(components):
        schema = {
            "authority": "cpp",
            "reward_type": reward_type,
            "reward_window": reward_window,
        }
        schema["sha256"] = schema_sha256(schema)
        return schema
    names = list(components)
    schema = {
        "authority": "python",
        "components": names,
        "weights": [float(w) for w in weights],
        "zero_demand_rule": "masked",
        "window": "sums_over_window_ticks",
        "ranges": {name: list(get_component(name).range) for name in names},
    }
    if "service_success" in names or "service_failure" in names:
        schema["success_rule"] = {"delivery_ratio_at_least": SUCCESS_DELIVERY,
                                  "routable_flow_fraction_at_least": SUCCESS_ROUTABLE,
                                  "zero_demand": "failure"}
    if "delivery_binary" in names:
        schema["delivery_binary_rule"] = (
            "+1 when offered demand is positive and any demand is delivered; "
            "-1 when offered demand is positive and none is delivered; "
            "zero demand is masked")
    if "signed_delivery_ratio" in names:
        schema["signed_delivery_ratio_rule"] = (
            "2 * clip(delivered_mbps_sum / demand_mbps_sum, 0, 1) - 1; "
            "zero demand is masked")
    if "travel_fraction" in names or "origin_fraction" in names:
        schema["movement_rule"] = {
            "travel_fraction": "mean per controlled slot of actual xy travel / (slot_speed_mps * window_ticks * tick_s), clipped [0,1]",
            "origin_fraction": "mean per controlled slot of xy distance from reset / bounds xy diagonal, clipped [0,1]",
        }
    if "sinr_quality" in names:
        schema["sinr_quality_rule"] = "mean per-link SINR clipped [-20,40] dB then mapped to [0,1]; invalid links score zero"
    if "unmet_sinr_quality" in names:
        schema["unmet_sinr_quality_rule"] = (
            "sinr_quality * clip((0.95 - delivered/demand)/0.95, 0, 1); "
            "zero demand uses delivery ratio zero")
    schema["sha256"] = schema_sha256(schema)
    return schema
