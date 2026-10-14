"""Edge cases for observation presets and schemas (scripts/rl/env/observations.py).

Spec: src/rl/policy-inputs.md ("What the observation contains", "Decode the trace
header", "Decode the raw facts") and src/rl/README.md (facts layout).
"""

import copy
import json
import math

import numpy as np
import pytest

from scripts.rl.env.observations import (
    PRESETS, SINR_INVALID_DB, SchemaMismatchError, canonical_json, check_schema,
    get_preset, observation_schema, schema_sha256,
)

BOUNDS = {"x_min": 0.0, "x_max": 100.0, "y_min": -50.0, "y_max": 100.0,
          "z_min": 0.0, "z_max": 50.0}
FLOAT32_PRESETS = ["local_links_v1", "geometry_v1", "service_v1", "full_facts_v1"]


def make_contract(num_nodes=3, slot_nodes=("node-b", "node-c", None), bounds=None):
    """A protocol-consistent init contract for N nodes and the given slot mapping."""
    node_ids = [f"node-{chr(ord('a') + i)}" for i in range(num_nodes)]
    slots = len(slot_nodes)
    return {
        "type": "init",
        "contract": "mesh_move_2d_v1",
        "dimensions": 2,
        "action_meanings": ["west", "east", "south", "north", "hold"],
        "max_controlled_nodes": slots,
        "num_controlled": sum(1 for s in slot_nodes if s is not None),
        "slot_node_ids": list(slot_nodes),
        "slot_speed_mps": [10.0 if s is not None else None for s in slot_nodes],
        "num_mesh_nodes": num_nodes,
        "obs_dim": slots * (4 + 2 * (num_nodes - 1)),
        "mask_dim": 5 * slots,
        "tick_s": 0.1,
        "decision_interval_s": 0.5,
        "decision_interval_ticks": 5,
        "num_ticks": 10,
        "num_decisions": 2,
        "reward_type": "all_links_los",
        "reward_window": "mean",
        "wall_policy": "clip",
        "facts_schema": "mesh_facts_v1",
        "facts_columns": {"nodes": ["x", "y", "z", "vx", "vy", "vz", "slot"],
                          "links": ["sinr_db", "capacity_mbps", "is_los"]},
        "node_ids": node_ids,
        "num_links": num_nodes * (num_nodes - 1) // 2,
        "bounds": dict(bounds or BOUNDS),
        "band": "mmwave",
        "jammer_path_enabled": False,
        "warmup_s": 0.0,
    }


def make_facts(contract, positions=None, links=None, window=None):
    """Facts whose link row k is [10+k, 100*(k+1), k%2] unless overridden."""
    n = contract["num_mesh_nodes"]
    ids = contract["node_ids"]
    slot_of = {node: s for s, node in enumerate(contract["slot_node_ids"]) if node}
    positions = positions or [(10.0 * (i + 1), 5.0 * i, 10.0) for i in range(n)]
    nodes = [[float(p[0]), float(p[1]), float(p[2]), 0.0, 0.0, 0.0,
              slot_of.get(ids[i], -1)] for i, p in enumerate(positions)]
    if links is None:
        links = [[10.0 + k, 100.0 * (k + 1), k % 2] for k in range(contract["num_links"])]
    window = window or {"ticks": 5, "demand_mbps_sum": 150.0, "delivered_mbps_sum": 120.0,
                        "flow_ticks_with_demand": 15, "unroutable_flow_ticks": 0,
                        "connected_pairs_sum": 12, "los_pairs_sum": 15,
                        "legacy_reward_sum": 5.0}
    return {"nodes": nodes, "links": links, "window": window}


def pair_row(n, i, j):
    """Row of pair (i, j) in the i<j, i-outer link table, enumerated explicitly."""
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    return pairs.index((min(i, j), max(i, j)))


# --- Documented decoding example ---------------------------------------------------

def test_documented_decoding_example_local_links():
    # policy-inputs.md "Decode the raw facts": slot 0 -> node B, row
    # [100,-5,10,0,-10,0,0] -> (x_n,y_n,z_n)=(1,-0.4,-0.6); (A,B) link
    # [40.62,5397.59,1] -> present=1, sinr_valid=1, sinr_n=1, cap_n~0.933.
    contract = make_contract(3, ("node-b", None))
    facts = make_facts(contract,
                       positions=[(50.0, 20.0, 10.0), (100.0, -5.0, 10.0), (0.0, 0.0, 0.0)],
                       links=[[40.62, 5397.59, 1], [0.0, 0.0, 0], [0.0, 0.0, 0]])
    facts["nodes"][1] = [100.0, -5.0, 10.0, 0.0, -10.0, 0.0, 0]
    obs = get_preset("local_links_v1").build(facts, contract)
    assert obs[:4].tolist() == pytest.approx([1.0, 1.0, -0.4, -0.6], abs=1e-6)
    assert obs[4:8].tolist() == pytest.approx([1.0, 1.0, 1.0, 0.933], abs=5e-4)


# --- Position normalization --------------------------------------------------------

@pytest.mark.parametrize("x,expected", [
    (0.0, -1.0),      # at x_min
    (100.0, 1.0),     # at x_max
    (50.0, 0.0),      # midpoint
    (-1e6, -1.0),     # far below: clamped
    (1e6, 1.0),       # far above: clamped ("x_max has x_n = 1 even if raw x is 100 m")
    (25.0, -0.5),
])
def test_local_position_maps_bounds_to_unit_interval_and_clamps(x, expected):
    contract = make_contract(2, ("node-a",))
    facts = make_facts(contract, positions=[(x, -50.0, 50.0), (0.0, 0.0, 0.0)])
    obs = get_preset("local_links_v1").build(facts, contract)
    assert obs[1] == pytest.approx(expected, abs=1e-6)
    assert obs[2] == pytest.approx(-1.0)   # y at y_min
    assert obs[3] == pytest.approx(1.0)    # z at z_max


# --- Link normalization and validity -------------------------------------------------

@pytest.mark.parametrize("sinr,expected", [(-20.0, 0.0), (10.0, 0.5), (40.0, 1.0),
                                           (-21.0, 0.0), (41.0, 1.0)])
def test_sinr_n_documented_points(sinr, expected):
    # policy-inputs.md: "-20 dB becomes 0, 10 dB becomes 0.5, and 40 dB or higher becomes 1".
    contract = make_contract(2, ("node-a",))
    obs = get_preset("local_links_v1").build(make_facts(contract, links=[[sinr, 1.0, 1]]),
                                             contract)
    assert obs[5] == 1.0 and obs[6] == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("capacity,expected", [(0.0, 0.0), (9.0, 0.25), (9999.0, 1.0),
                                               (1e9, 1.0)])
def test_cap_n_log_scale_points(capacity, expected):
    contract = make_contract(2, ("node-a",))
    obs = get_preset("local_links_v1").build(
        make_facts(contract, links=[[0.0, capacity, 1]]), contract)
    assert obs[7] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("sinr,capacity,valid", [
    (SINR_INVALID_DB, 10.0, False),        # "at or below -900 dB" is invalid
    (-900.5, 10.0, False),
    (-899.5, 10.0, True),                  # just above: valid, but sinr_n clamps to 0
    (math.nan, 10.0, False),
    (math.inf, 10.0, False),
    (-math.inf, 10.0, False),
    (10.0, math.inf, False),               # non-finite capacity
    (10.0, math.nan, False),
])
def test_link_validity_rules(sinr, capacity, valid):
    contract = make_contract(2, ("node-a",))
    obs = get_preset("local_links_v1").build(
        make_facts(contract, links=[[sinr, capacity, 1]]), contract)
    present, sinr_valid, sinr_n, cap_n = obs[4:8].tolist()
    assert present == 1.0          # present means the peer exists, not that the link works
    assert sinr_valid == (1.0 if valid else 0.0)
    if valid:
        assert sinr_n == 0.0 and cap_n == pytest.approx(math.log10(11.0) / 4.0, abs=1e-6)
    else:
        assert (sinr_n, cap_n) == (0.0, 0.0)
    assert np.isfinite(obs).all()


# --- Layout: peer order, slot order, dtypes, dimensions ------------------------------

@pytest.mark.parametrize("node_index", [0, 2, 3])
def test_raw_links_peer_order_four_nodes(node_index):
    # README: links are i<j with i outer; obs peers are in node-index order skipping self.
    contract = make_contract(4, (f"node-{'abcd'[node_index]}",))
    facts = make_facts(contract)
    obs = get_preset("raw_links_v1").build(facts, contract)
    peers = [p for p in range(4) if p != node_index]
    expected = [1.0, *facts["nodes"][node_index][:3]]
    for peer in peers:
        row = facts["links"][pair_row(4, node_index, peer)]
        expected += [row[0], row[1]]
    np.testing.assert_array_equal(obs, np.asarray(expected, dtype=np.float64))


def test_slot_blocks_follow_slot_node_ids_not_node_order():
    contract = make_contract(3, ("node-c", "node-a"))
    facts = make_facts(contract, positions=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0),
                                            (7.0, 8.0, 9.0)])
    obs = get_preset("raw_links_v1").build(facts, contract)
    width = 4 + 2 * 2
    assert obs[0:4].tolist() == [1.0, 7.0, 8.0, 9.0]
    assert obs[width:width + 4].tolist() == [1.0, 1.0, 2.0, 3.0]


SHAPES = [
    (2, ("node-a",)),                          # minimal: two nodes, one slot
    (3, ("node-b", "node-c", None)),           # padded slot
    (4, ("node-d", "node-b")),                 # slot order differs from node order
]


@pytest.mark.parametrize("preset_name", sorted(PRESETS))
@pytest.mark.parametrize("num_nodes,slots", SHAPES)
def test_dtype_dimension_and_space_agree(preset_name, num_nodes, slots):
    contract = make_contract(num_nodes, slots)
    preset = get_preset(preset_name)
    obs = preset.build(make_facts(contract), contract)
    space = preset.space(contract)
    schema = observation_schema(preset_name, contract)
    names = preset.feature_names(contract)
    assert str(obs.dtype) == preset.dtype == schema["dtype"] == str(space.dtype)
    assert obs.shape == space.shape == (schema["obs_dim"],) == (len(names),)
    assert len(set(names)) == len(names)
    assert schema["feature_names"] == names
    assert len(schema["low"]) == len(schema["high"]) == len(names)
    assert space.contains(obs)
    if preset_name == "raw_links_v1":
        assert obs.shape[0] == contract["obs_dim"]   # reproduces the C++ obs size


def test_documented_sizes_for_three_nodes_three_slots():
    # policy-inputs.md: N=3, M=3 -> raw 24 values, local 36.
    contract = make_contract(3, ("node-a", "node-b", "node-c"))
    assert observation_schema("raw_links_v1", contract)["obs_dim"] == 24
    assert observation_schema("local_links_v1", contract)["obs_dim"] == 36


@pytest.mark.parametrize("preset_name", ["service_v1", "full_facts_v1"])
def test_zero_demand_and_zero_flow_window_is_finite(preset_name):
    contract = make_contract(3, ("node-b", "node-c", None))
    window = {"ticks": 1, "demand_mbps_sum": 0.0, "delivered_mbps_sum": 0.0,
              "flow_ticks_with_demand": 0, "unroutable_flow_ticks": 0,
              "connected_pairs_sum": 0, "los_pairs_sum": 0, "legacy_reward_sum": 0.0}
    preset = get_preset(preset_name)
    obs = preset.build(make_facts(contract, window=window), contract)
    assert np.isfinite(obs).all() and preset.space(contract).contains(obs)


@pytest.mark.parametrize("preset_name", sorted(PRESETS))
def test_build_is_pure_and_deterministic(preset_name):
    contract = make_contract(3, ("node-b", "node-c", None))
    facts = make_facts(contract)
    before = (copy.deepcopy(facts), copy.deepcopy(contract))
    preset = get_preset(preset_name)
    first = preset.build(facts, contract)
    second = preset.build(facts, contract)
    assert first.tobytes() == second.tobytes()
    assert (facts, contract) == before


# --- Schema hashing ----------------------------------------------------------------


@pytest.mark.parametrize("change", [
    {"bounds": dict(BOUNDS, z_max=60.0)},
    {"node_ids": ["node-a", "node-x", "node-c"],
     "slot_node_ids": ["node-x", "node-c", None]},
    {"slot_node_ids": ["node-c", "node-b", None]},
])
@pytest.mark.parametrize("preset_name", ["raw_links_v1", "local_links_v1"])
def test_schema_hash_changes_with_recorded_fields(preset_name, change):
    base = observation_schema(preset_name, make_contract())
    changed = observation_schema(preset_name, dict(make_contract(), **change))
    assert base["sha256"] != changed["sha256"]


def test_schema_hash_differs_between_presets():
    contract = make_contract()
    hashes = {observation_schema(name, contract)["sha256"] for name in PRESETS}
    assert len(hashes) == len(PRESETS)


@pytest.mark.parametrize("change", [{"num_ticks": 50, "num_decisions": 10},
                                    {"tick_s": 0.2}, {"band": "sub-6"},
                                    {"warmup_s": 1.0}])
def test_schema_hash_ignores_fields_outside_the_schema(change):
    # Fingerprints cover layout/normalization/identity, not episode length or band.
    base = observation_schema("local_links_v1", make_contract())
    changed = observation_schema("local_links_v1", dict(make_contract(), **change))
    assert base["sha256"] == changed["sha256"]


def test_schema_sha256_recomputes_and_excludes_its_own_field():
    schema = observation_schema("local_links_v1", make_contract())
    assert schema_sha256(schema) == schema["sha256"]
    assert schema_sha256(dict(schema, sha256="tampered")) == schema["sha256"]
    assert len(schema["sha256"]) == 64 and schema["sha256"] == schema["sha256"].lower()


def test_local_schema_low_high_match_box():
    contract = make_contract()
    schema = observation_schema("local_links_v1", contract)
    box = get_preset("local_links_v1").space(contract)
    assert schema["low"] == box.low.tolist() and schema["high"] == box.high.tolist()
    json.loads(canonical_json(schema))


# --- check_schema ----------------------------------------------------------------

def test_check_schema_ignores_bounds_for_raw_links():
    saved = observation_schema("raw_links_v1", make_contract())
    live = observation_schema("raw_links_v1",
                              make_contract(bounds=dict(BOUNDS, x_max=500.0)))
    assert check_schema(saved, live) == []


@pytest.mark.parametrize("preset_name", FLOAT32_PRESETS)
def test_check_schema_rejects_bounds_for_normalized_presets(preset_name):
    saved = observation_schema(preset_name, make_contract())
    live = observation_schema(preset_name, make_contract(bounds=dict(BOUNDS, y_min=-60.0)))
    with pytest.raises(SchemaMismatchError) as excinfo:
        check_schema(saved, live)
    assert excinfo.value.fields == ["bounds"]


def test_check_schema_slot_reassignment_only_warns():
    saved = observation_schema("local_links_v1", make_contract())
    live = observation_schema("local_links_v1",
                              make_contract(slot_nodes=("node-a", "node-c", None)))
    warnings = check_schema(saved, live)
    assert len(warnings) == 1 and warnings[0].startswith("slot_node_ids changed")
