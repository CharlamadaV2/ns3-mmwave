"""Direct tests of CentralizedProtocol.validate_init against hand-built init messages.

Expectations come from src/rl/README.md ("Centralized contract", "Message
fields", "Per-decision facts") and the "validate, never coerce" rule in
scripts/rl/env/CLAUDE.md.
"""


from scripts.rl.env.protocol import CentralizedProtocol

from ._helpers import make_init


def _accepts(init: dict) -> CentralizedProtocol:
    return CentralizedProtocol(init)


def test_valid_init_is_accepted_and_exposes_properties():
    protocol = _accepts(make_init())
    assert protocol.mask is None and protocol.facts is None
    assert protocol.node_ids == ["node-a", "node-b", "node-c"]
    assert protocol.mask_dim == 15
    assert protocol.bounds["x_max"] == 100.0


# Contract identity -------------------------------------------------------------


# Integer / number field types ----------------------------------------------------


# Slot counts ---------------------------------------------------------------------


def test_all_slots_active_no_padding_accepted():
    init = make_init(max_controlled_nodes=2, obs_dim=16, mask_dim=10,
                     slot_node_ids=["node-b", "node-c"], slot_speed_mps=[1.0, 2.0])
    _accepts(init)


# Slot ids and speeds -------------------------------------------------------------


# Facts metadata ------------------------------------------------------------------
