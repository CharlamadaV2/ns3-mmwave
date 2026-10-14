"""Direct tests of step/facts/window validation, joint actions, and LegacyProtocol.

Expectations come from src/rl/README.md (step fields, mask order, cadence,
partial windows, facts tables, window invariants, "Python replies with exactly
M integers in [0,4]") and the "raise ProtocolError, never coerce" rule in
scripts/rl/env/CLAUDE.md.
"""


import numpy as np
import pytest

from scripts.rl.env.protocol import CentralizedProtocol, LegacyProtocol, ProtocolError

from ._helpers import episode_steps, make_init, make_legacy_step, make_step


@pytest.fixture
def init() -> dict:
    return make_init()


@pytest.fixture
def protocol(init) -> CentralizedProtocol:
    return CentralizedProtocol(init)


def _reset(protocol, init):
    return protocol.validate_step(make_step(init, 0, 0, 1), first=True)


# Happy path ----------------------------------------------------------------------

def test_full_episode_validates_and_exposes_mask_and_facts(protocol, init):
    steps = episode_steps(init)
    obs = protocol.validate_step(steps[0], first=True)
    assert obs.dtype == np.float64 and obs.shape == (24,)
    assert protocol.mask.dtype == np.int8 and protocol.mask.shape == (15,)
    assert protocol.facts is steps[0]["facts"]
    for step in steps[1:]:
        protocol.validate_step(step)
    assert steps[-1]["done"] and steps[-1]["tick"] == 10


def test_partial_final_window_is_required():
    # num_ticks 7, k 5 -> windows of 5 then 2 ticks (README "Partial windows").
    init = make_init(num_ticks=7, num_decisions=2)
    protocol = CentralizedProtocol(init)
    steps = episode_steps(init)
    assert [s["ticks_in_step"] for s in steps] == [1, 5, 2]
    for index, step in enumerate(steps):
        protocol.validate_step(step, first=index == 0)


def test_single_tick_cadence_episode():
    init = make_init(num_ticks=3, decision_interval_ticks=1, decision_interval_s=0.1,
                     num_decisions=3)
    protocol = CentralizedProtocol(init)
    for index, step in enumerate(episode_steps(init)):
        protocol.validate_step(step, first=index == 0)


def test_extra_message_keys_are_tolerated(protocol, init):
    # The fake simulator already sends `last_action`; unknown keys are not a violation.
    protocol.validate_step(make_step(init, 0, 0, 1, last_action=None, extra=1),
                           first=True)


# Type, obs, and mask checks ------------------------------------------------------


def test_sinr_sentinel_passes_through_obs(protocol, init):
    msg = make_step(init, 0, 0, 1)
    msg["obs"][4] = -999.0
    obs = protocol.validate_step(msg, first=True)
    assert obs[4] == -999.0


def test_active_slot_with_only_hold_accepted(protocol, init):
    msg = make_step(init, 0, 0, 1)
    msg["mask"][0:5] = [0, 0, 0, 0, 1]
    protocol.validate_step(msg, first=True)
    assert list(protocol.mask[0:5]) == [0, 0, 0, 0, 1]


# Reward / done / counters --------------------------------------------------------


def test_revalidated_padded_slot_index_accepted(protocol, init):
    # README example: padding action 0 is revalidated and reported as slot 2.
    _reset(protocol, init)
    protocol.validate_step(make_step(init, 5, 1, 5, revalidated_slots=[2]))


# Monotonicity and cadence ----------------------------------------------------------


def test_time_s_within_tolerance_accepted(protocol, init):
    _reset(protocol, init)
    protocol.validate_step(make_step(init, 5, 1, 5, time_s=0.5 + 1e-7))


# Facts: nodes ----------------------------------------------------------------------

def _with_facts(init, mutate):
    msg = make_step(init, 0, 0, 1)
    mutate(msg["facts"])
    return msg


# Facts: links ----------------------------------------------------------------------


def test_fact_link_sinr_sentinel_accepted(protocol, init):
    def mutate(facts):
        facts["links"][2][0] = -999.0
    protocol.validate_step(_with_facts(init, mutate), first=True)


def test_fact_link_zero_capacity_accepted(protocol, init):
    def mutate(facts):
        facts["links"][1][1] = 0
    protocol.validate_step(_with_facts(init, mutate), first=True)


# Facts: window ---------------------------------------------------------------------

def _window(init, **changes):
    msg = make_step(init, 0, 0, 1)
    msg["facts"]["window"].update(changes)
    return msg


def test_window_integral_mbps_sums_accepted(protocol, init):
    protocol.validate_step(_window(init, demand_mbps_sum=30, delivered_mbps_sum=15),
                           first=True)


def test_zero_demand_window_accepted(protocol, init):
    protocol.validate_step(_window(init, demand_mbps_sum=0.0, delivered_mbps_sum=0.0,
                                   flow_ticks_with_demand=0, unroutable_flow_ticks=0),
                           first=True)


def test_window_mean_reward_over_five_ticks(protocol, init):
    # README: three tick rewards +1,-1,+1 average; here five ticks summing to 1 -> 0.2.
    _reset(protocol, init)
    msg = make_step(init, 5, 1, 5, reward=0.2)
    msg["facts"]["window"]["legacy_reward_sum"] = 1.0
    protocol.validate_step(msg)
    bad = CentralizedProtocol(init)
    _reset(bad, init)
    msg = make_step(init, 5, 1, 5, reward=1.0)
    msg["facts"]["window"]["legacy_reward_sum"] = 1.0     # mean would be 0.2, not 1.0
    with pytest.raises(ProtocolError, match="legacy_reward_sum/ticks"):
        bad.validate_step(msg)


def test_negative_reward_window_accepted(protocol, init):
    protocol.validate_step(make_step(init, 0, 0, 1, reward=-1.0), first=True)


# Joint action ----------------------------------------------------------------------

@pytest.mark.parametrize("action", [[0, 1, 4], (4, 4, 4), np.array([3, 2, 4]),
                                    np.array([4, 4, 4], dtype=np.int64),
                                    np.array([1, 0, 4], dtype=np.uint8)])
def test_joint_action_accepts_m_integers(protocol, action):
    joint = protocol.joint_action(action)
    assert joint == [int(a) for a in np.asarray(action)]
    assert all(type(a) is int for a in joint)


# Legacy protocol -------------------------------------------------------------------

def test_legacy_valid_stream_and_parse_obs():
    protocol = LegacyProtocol(None)
    first = make_legacy_step(0, pos=(1.0, 2.0, 3.0), links=2)
    protocol.validate_step(first, first=True)
    obs = LegacyProtocol.parse_obs(first)
    # [x, y, z, sinr0, cap0, sinr1, cap1]
    assert obs.tolist() == [1.0, 2.0, 3.0, 15.0, 80.0, 15.0, 80.0]
    assert obs.dtype == np.float64
    protocol.validate_step(make_legacy_step(1, links=2))
    protocol.validate_step(make_legacy_step(2, links=2, done=True))
