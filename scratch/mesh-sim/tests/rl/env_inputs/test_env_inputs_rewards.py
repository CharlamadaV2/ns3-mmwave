"""Edge cases for reward components, RewardComposer, and reward_schema (rewards.py).

Spec: src/rl/policy-inputs.md "One action, several ticks, one reward" and the
facts.window invariants in src/rl/README.md.
"""


import pytest

from scripts.rl.env.rewards import (
    COMPONENTS, DEMAND_EPS, RewardComposer, get_component, reward_schema,
)
from scripts.rl.env.observations import schema_sha256

CONTRACT = {"num_links": 3, "tick_s": 0.1}
WINDOW = {"ticks": 5, "demand_mbps_sum": 150.0, "delivered_mbps_sum": 120.0,
          "flow_ticks_with_demand": 15, "unroutable_flow_ticks": 0,
          "connected_pairs_sum": 12, "los_pairs_sum": 15, "legacy_reward_sum": 5.0}
WINDOW_COMPONENTS = ["delivery_ratio", "delivery_binary", "signed_delivery_ratio",
                     "connectivity", "throughput_mbps", "legacy", "service_success",
                     "service_failure"]


def window(**overrides) -> dict:
    return dict(WINDOW, **overrides)


# --- Zero-demand threshold ----------------------------------------------------------

@pytest.mark.parametrize("demand,valid", [
    (0.0, False),
    (DEMAND_EPS, False),          # "invalid ... if demand sum <= 1e-9"
    (DEMAND_EPS * 0.5, False),
    (DEMAND_EPS * 1.5, True),
    (1.0, True),
])
def test_delivery_ratio_demand_threshold(demand, valid):
    value, ok = get_component("delivery_ratio").value(
        window(demand_mbps_sum=demand, delivered_mbps_sum=demand), 0.0, CONTRACT)
    assert ok is valid
    assert value == (1.0 if valid else 0.0)


def test_delivery_ratio_uses_window_sums_not_tick_means():
    # Same sums over a different tick count give the same ratio.
    one = get_component("delivery_ratio").value(window(ticks=1), 0.0, CONTRACT)
    five = get_component("delivery_ratio").value(window(ticks=5), 0.0, CONTRACT)
    assert one == five == (pytest.approx(0.8), True)


def test_connectivity_full_and_empty_windows():
    full = get_component("connectivity").value(
        window(connected_pairs_sum=15), 0.0, CONTRACT)
    empty = get_component("connectivity").value(
        window(connected_pairs_sum=0), 0.0, CONTRACT)
    assert full == (1.0, True) and empty == (0.0, True)


def test_legacy_component_preserves_negative_reward():
    assert get_component("legacy").value(window(legacy_reward_sum=-2.5), 0.0,
                                         CONTRACT) == (-0.5, True)


def test_throughput_is_per_tick_mean_of_delivered_sum():
    assert get_component("throughput_mbps").value(window(), 0.0, CONTRACT) == (24.0, True)


# --- Composer -----------------------------------------------------------------------

def test_composer_all_invalid_total_is_zero_and_keeps_cpp_reward():
    composer = RewardComposer(["delivery_ratio", "signed_delivery_ratio",
                               "delivery_binary"], [1.0, 2.0, 3.0])
    zero = window(demand_mbps_sum=0.0, delivered_mbps_sum=0.0,
                  flow_ticks_with_demand=0)
    breakdown = composer.compose(zero, -0.75, CONTRACT)
    assert breakdown.total == 0.0
    assert breakdown.valid == {"delivery_ratio": 0, "signed_delivery_ratio": 0,
                               "delivery_binary": 0}
    assert breakdown.components == {"delivery_ratio": 0.0, "signed_delivery_ratio": 0.0,
                                    "delivery_binary": 0.0}
    assert breakdown.legacy == -0.75


def test_composer_preserves_component_order_and_float_weights():
    names = ["throughput_mbps", "legacy", "connectivity", "delivery_ratio"]
    composer = RewardComposer(names, [1, 0, -2, "0.5"])
    breakdown = composer.compose(window(), 1.0, CONTRACT)
    assert list(breakdown.components) == names
    assert list(breakdown.weights) == names
    assert breakdown.weights == {"throughput_mbps": 1.0, "legacy": 0.0,
                                 "connectivity": -2.0, "delivery_ratio": 0.5}
    assert all(type(w) is float for w in breakdown.weights.values())
    assert breakdown.total == pytest.approx(24.0 + 0.0 - 2.0 * 0.8 + 0.5 * 0.8)


def test_composer_negative_total_is_returned_as_is():
    breakdown = RewardComposer(["connectivity"], [-3.0]).compose(window(), 0.0, CONTRACT)
    assert breakdown.total == pytest.approx(-3.0 * 0.8)


@pytest.mark.parametrize("name", WINDOW_COMPONENTS)
def test_window_component_values_respect_declared_range(name):
    component = get_component(name)
    low, high = component.range
    for w in (window(), window(delivered_mbps_sum=0.0), window(demand_mbps_sum=0.0,
                                                               delivered_mbps_sum=0.0,
                                                               flow_ticks_with_demand=0),
              window(unroutable_flow_ticks=15), window(connected_pairs_sum=15)):
        value, valid = component.value(w, 0.0, CONTRACT)
        if not valid:
            assert value == 0.0
            continue
        if low is not None:
            assert value >= low
        if high is not None:
            assert value <= high


# --- reward_schema -------------------------------------------------------------------

def test_cpp_schema_ignores_weights_and_tracks_reward_type_and_window():
    base = reward_schema([], [], reward_type="all_links_los", reward_window="mean")
    assert base == reward_schema([], [5.0], reward_type="all_links_los",
                                 reward_window="mean")
    assert base["sha256"] != reward_schema([], [], reward_type="throughput",
                                           reward_window="mean")["sha256"]
    assert base["sha256"] != reward_schema([], [], reward_type="all_links_los",
                                           reward_window="sum")["sha256"]
    assert schema_sha256(base) == base["sha256"]


def test_python_schema_hash_sensitivity():
    def sha(components, weights):
        return reward_schema(components, weights, reward_type="x",
                             reward_window="mean")["sha256"]
    base = sha(["delivery_ratio", "connectivity"], [1.0, 0.5])
    assert base == sha(["delivery_ratio", "connectivity"], [1, 0.5])   # int == float
    assert base != sha(["connectivity", "delivery_ratio"], [0.5, 1.0])  # order matters
    assert base != sha(["delivery_ratio", "connectivity"], [1.0, 0.25])
    assert base != sha(["delivery_ratio"], [1.0])


def test_every_component_has_a_schema_range():
    schema = reward_schema(sorted(COMPONENTS), [1.0] * len(COMPONENTS),
                           reward_type="x", reward_window="mean")
    assert set(schema["ranges"]) == set(COMPONENTS)
    assert schema_sha256(schema) == schema["sha256"]
