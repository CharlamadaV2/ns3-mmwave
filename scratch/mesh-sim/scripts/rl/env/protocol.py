"""Validation of the simulator's RL wire protocol and action shape."""

import math

import numpy as np

_CONTRACTS = {
    "mesh_move_2d_v1": {
        "dimensions": 2,
        "action_meanings": ["west", "east", "south", "north", "hold"],
    },
}
SLOT_ACTIONS = 5
_HOLD_ACTION = 4
_SLOT_STATE_VALUES = 4
_MAX_SLOTS = 64
_PADDED_SLOT_MASK = [0, 0, 0, 0, 1]
_TIME_TOL = 1e-6

FACTS_SCHEMA = "mesh_facts_v1"
_FACTS_COLUMNS = {
    "nodes": ["x", "y", "z", "vx", "vy", "vz", "slot"],
    "links": ["sinr_db", "capacity_mbps", "is_los"],
}
_NODE_COLUMNS = len(_FACTS_COLUMNS["nodes"])
_LINK_COLUMNS = len(_FACTS_COLUMNS["links"])
_SLOT_COLUMN = 6
_BOUND_AXES = ("x", "y", "z")
# Counts must be integers; the Mbps and reward sums may be integral floats.
_WINDOW_COUNTS = ("ticks", "flow_ticks_with_demand", "unroutable_flow_ticks",
                  "connected_pairs_sum", "los_pairs_sum")
_WINDOW_SUMS = ("demand_mbps_sum", "delivered_mbps_sum", "legacy_reward_sum")
_DELIVERED_REL_TOL = 1e-6
_REWARD_TOL = 1e-9


class ProtocolError(ValueError):
    """A simulator message violates the selected wire contract."""


def _fail(detail: str):
    raise ProtocolError(detail)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


class CentralizedProtocol:
    """Validates the `init` and per-step messages of centralized control and the joint action."""

    def __init__(self, init: dict):
        self.validate_init(init)
        self._num_slots = int(init["max_controlled_nodes"])
        self._num_controlled = int(init["num_controlled"])
        self._num_nodes = int(init["num_mesh_nodes"])
        self._num_links = int(init["num_links"])
        self._node_ids = list(init["node_ids"])
        self._slot_node_ids = list(init["slot_node_ids"])
        self._bounds = dict(init["bounds"])
        self._obs_dim = int(init["obs_dim"])
        self._mask_dim = int(init["mask_dim"])
        self._tick_s = float(init["tick_s"])
        self._interval_ticks = int(init["decision_interval_ticks"])
        self._num_ticks = int(init["num_ticks"])
        self._num_decisions = int(init["num_decisions"])
        self._last_tick: int | None = None
        self._last_time: float | None = None
        self._last_decision: int | None = None
        self._mask: np.ndarray | None = None
        self._facts: dict | None = None

    @property
    def mask(self) -> np.ndarray | None:
        return self._mask

    @property
    def facts(self) -> dict | None:
        return self._facts

    @property
    def node_ids(self) -> list[str]:
        return list(self._node_ids)

    @property
    def bounds(self) -> dict:
        return dict(self._bounds)

    @property
    def mask_dim(self) -> int:
        return self._mask_dim

    def validate_init(self, init: dict) -> None:
        contract = init.get("contract")
        spec = _CONTRACTS.get(contract) if isinstance(contract, str) else None
        if spec is None:
            _fail(
                f"Unknown init contract {contract!r}; supported: {sorted(_CONTRACTS)}"
            )
        if init.get("dimensions") != spec["dimensions"]:
            _fail(
                f"init dimensions {init.get('dimensions')!r} != "
                f"{spec['dimensions']} for contract {contract}"
            )
        if init.get("action_meanings") != spec["action_meanings"]:
            _fail(
                f"init action_meanings {init.get('action_meanings')!r} != "
                f"{spec['action_meanings']} for contract {contract}"
            )

        for field in ("max_controlled_nodes", "num_controlled", "num_mesh_nodes",
                      "obs_dim", "mask_dim", "decision_interval_ticks",
                      "num_ticks", "num_decisions"):
            if not _is_int(init.get(field)):
                _fail(
                    f"init {field} must be an integer, got {init.get(field)!r}"
                )
        for field in ("tick_s", "decision_interval_s"):
            value = init.get(field)
            if not _is_finite_number(value) or value <= 0:
                _fail(
                    f"init {field} must be finite and > 0, got {value!r}"
                )
        for field in ("reward_type", "reward_window", "wall_policy"):
            if not isinstance(init.get(field), str) or not init[field]:
                _fail(
                    f"init {field} must be a non-empty string, got {init.get(field)!r}"
                )

        slots = init["max_controlled_nodes"]
        count = init["num_controlled"]
        nodes = init["num_mesh_nodes"]
        if not 1 <= count <= slots <= _MAX_SLOTS:
            _fail(
                f"init requires 1 <= num_controlled ({count}) <= max_controlled_nodes "
                f"({slots}) <= {_MAX_SLOTS}"
            )
        if nodes < 2:
            _fail(f"init num_mesh_nodes must be >= 2, got {nodes}")
        for field in ("decision_interval_ticks", "num_ticks", "num_decisions"):
            if init[field] < 1:
                _fail(f"init {field} must be >= 1, got {init[field]}")

        expected_obs = slots * (_SLOT_STATE_VALUES + 2 * (nodes - 1))
        if init["obs_dim"] != expected_obs:
            _fail(
                f"init obs_dim {init['obs_dim']} != max_controlled_nodes*(4+2*(N-1)) "
                f"= {expected_obs}"
            )
        if init["mask_dim"] != SLOT_ACTIONS * slots:
            _fail(
                f"init mask_dim {init['mask_dim']} != {SLOT_ACTIONS}*"
                f"max_controlled_nodes = {SLOT_ACTIONS * slots}"
            )
        ticks, k = init["num_ticks"], init["decision_interval_ticks"]
        if k > ticks:
            _fail(
                f"init decision_interval_ticks {k} exceeds num_ticks {ticks}"
            )
        expected_decisions = -(-ticks // k)
        if init["num_decisions"] != expected_decisions:
            _fail(
                f"init num_decisions {init['num_decisions']} != "
                f"ceil(num_ticks/decision_interval_ticks) = {expected_decisions}"
            )

        ids = init.get("slot_node_ids")
        speeds = init.get("slot_speed_mps")
        for field, value in (("slot_node_ids", ids), ("slot_speed_mps", speeds)):
            if not isinstance(value, list) or len(value) != slots:
                _fail(
                    f"init {field} must be a list of {slots} entries, got {value!r}"
                )
        active_ids = ids[:count]
        if not all(isinstance(i, str) and i for i in active_ids):
            _fail(
                f"init slot_node_ids has a non-string active entry: {active_ids!r}"
            )
        if len(set(active_ids)) != count:
            _fail(
                f"init slot_node_ids has duplicate active entries: {active_ids!r}"
            )
        if any(i is not None for i in ids[count:]):
            _fail(
                f"init slot_node_ids padding must be null, got {ids[count:]!r}"
            )
        if not all(_is_finite_number(s) and s > 0 for s in speeds[:count]):
            _fail(
                f"init slot_speed_mps active entries must be finite and > 0, "
                f"got {speeds[:count]!r}"
            )
        if any(s is not None for s in speeds[count:]):
            _fail(
                f"init slot_speed_mps padding must be null, got {speeds[count:]!r}"
            )
        self._validate_facts_metadata(init, nodes)

    @staticmethod
    def _validate_facts_metadata(init: dict, nodes: int) -> None:
        schema = init.get("facts_schema")
        if schema != FACTS_SCHEMA:
            _fail(
                f"init facts_schema {'missing' if schema is None else repr(schema)}: "
                f"centralized RL requires a simulator binary emitting "
                f"{FACTS_SCHEMA!r}"
            )
        if init.get("facts_columns") != _FACTS_COLUMNS:
            _fail(
                f"init facts_columns {init.get('facts_columns')!r} != "
                f"{_FACTS_COLUMNS}"
            )
        expected_links = nodes * (nodes - 1) // 2
        if not _is_int(init.get("num_links")) or init["num_links"] != expected_links:
            _fail(
                f"init num_links {init.get('num_links')!r} != N(N-1)/2 = "
                f"{expected_links}"
            )
        ids = init.get("node_ids")
        if not isinstance(ids, list) or len(ids) != nodes:
            _fail(f"init node_ids must be a list of {nodes} entries, got {ids!r}")
        if not all(isinstance(i, str) and i for i in ids):
            _fail(f"init node_ids has an empty or non-string entry: {ids!r}")
        if len(set(ids)) != nodes:
            _fail(f"init node_ids has duplicate entries: {ids!r}")
        bounds = init.get("bounds")
        if not isinstance(bounds, dict):
            _fail(f"init bounds must be an object, got {bounds!r}")
        for axis in _BOUND_AXES:
            low, high = bounds.get(f"{axis}_min"), bounds.get(f"{axis}_max")
            if not _is_finite_number(low) or not _is_finite_number(high):
                _fail(
                    f"init bounds {axis} endpoints must be finite, got "
                    f"{low!r}, {high!r}"
                )
            if not low < high:
                _fail(
                    f"init bounds requires {axis}_min < {axis}_max, got {low}, {high}"
                )

    def validate_step(self, msg: dict, first: bool = False) -> np.ndarray:
        """Check one step message against the init contract; raise ProtocolError on any violation."""
        if msg.get("type") != "step":
            _fail(
                f"Expected a 'step' message, got type {msg.get('type')!r}"
            )

        obs = msg.get("obs")
        if not isinstance(obs, list) or len(obs) != self._obs_dim:
            length = len(obs) if isinstance(obs, list) else None
            _fail(f"obs length {length} != obs_dim {self._obs_dim}")
        bad = next((i for i, v in enumerate(obs) if not _is_finite_number(v)), None)
        if bad is not None:
            _fail(f"obs[{bad}] is not a finite number: {obs[bad]!r}")

        mask = msg.get("mask")
        if not isinstance(mask, list) or len(mask) != self._mask_dim:
            length = len(mask) if isinstance(mask, list) else None
            _fail(f"mask length {length} != mask_dim {self._mask_dim}")
        bad = next((i for i, v in enumerate(mask) if not _is_int(v) or v not in (0, 1)),
                   None)
        if bad is not None:
            _fail(f"mask[{bad}] is not 0 or 1: {mask[bad]!r}")

        if not _is_finite_number(msg.get("reward")):
            _fail(f"reward is not a finite number: {msg.get('reward')!r}")
        if not isinstance(msg.get("done"), bool):
            _fail(f"done must be a boolean, got {msg.get('done')!r}")

        slot_obs = self._obs_dim // self._num_slots if self._num_slots else 0
        for slot in range(self._num_controlled, self._num_slots):
            window = obs[slot * slot_obs:(slot + 1) * slot_obs]
            if any(v != 0 for v in window):
                _fail(
                    f"padded slot {slot} observation must be all zero, got {window!r}"
                )
        for slot in range(self._num_slots):
            window = mask[slot * SLOT_ACTIONS:(slot + 1) * SLOT_ACTIONS]
            if window[_HOLD_ACTION] != 1:
                _fail(f"hold is masked out in slot {slot}: {window!r}")
            if slot >= self._num_controlled and window != _PADDED_SLOT_MASK:
                _fail(
                    f"padded slot {slot} mask must be {_PADDED_SLOT_MASK}, got {window!r}"
                )

        for field in ("tick", "decision", "ticks_in_step"):
            if not _is_int(msg.get(field)) or msg[field] < 0:
                _fail(
                    f"{field} must be a non-negative integer, got {msg.get(field)!r}"
                )
        if not _is_finite_number(msg.get("time_s")):
            _fail(f"time_s is not a finite number: {msg.get('time_s')!r}")
        revalidated = msg.get("revalidated_slots")
        if not isinstance(revalidated, list) or not all(
                _is_int(s) and 0 <= s < self._num_slots for s in revalidated):
            _fail(
                f"revalidated_slots must be slot indices in [0,{self._num_slots}), "
                f"got {revalidated!r}"
            )

        tick, decision, window_ticks = msg["tick"], msg["decision"], msg["ticks_in_step"]
        if first:
            if tick != 0 or decision != 0 or window_ticks != 1:
                _fail(
                    f"reset message must be tick 0, decision 0, ticks_in_step 1; got "
                    f"tick {tick}, decision {decision}, ticks_in_step {window_ticks}"
                )
        else:
            if tick <= self._last_tick:
                _fail(
                    f"tick {tick} does not advance past {self._last_tick}"
                )
            if decision != self._last_decision + 1:
                _fail(
                    f"decision {decision} does not follow {self._last_decision}"
                )
            if msg["time_s"] <= self._last_time:
                _fail(
                    f"time_s {msg['time_s']} does not advance past {self._last_time}"
                )
            expected = min(self._interval_ticks, self._num_ticks - self._last_tick)
            if window_ticks != expected or tick - self._last_tick != window_ticks:
                _fail(
                    f"ticks_in_step {window_ticks} is not the expected window "
                    f"{expected} for ticks ({self._last_tick},{tick}]"
                )
        if decision > self._num_decisions:
            _fail(
                f"decision {decision} exceeds num_decisions {self._num_decisions}"
            )
        if tick > self._num_ticks:
            _fail(f"tick {tick} exceeds num_ticks {self._num_ticks}")
        if abs(msg["time_s"] - tick * self._tick_s) > _TIME_TOL:
            _fail(
                f"time_s {msg['time_s']} != tick*tick_s = {tick * self._tick_s}"
            )
        if msg["done"] != (tick == self._num_ticks):
            _fail(
                f"done {msg['done']} disagrees with tick {tick} of {self._num_ticks}"
            )

        self._facts = self._validated_facts(msg)
        self._last_tick, self._last_decision = tick, decision
        self._last_time = msg["time_s"]
        self._mask = np.asarray(mask, dtype=np.int8)
        return np.asarray(obs, dtype=np.float64)

    def _validated_facts(self, msg: dict) -> dict:
        facts = msg.get("facts")
        if not isinstance(facts, dict):
            _fail(f"step facts must be an object, got {facts!r}")
        self._check_fact_nodes(facts.get("nodes"))
        self._check_fact_links(facts.get("links"))
        self._check_fact_window(facts.get("window"), msg["ticks_in_step"],
                                float(msg["reward"]))
        return facts

    def _check_fact_nodes(self, nodes) -> None:
        if not isinstance(nodes, list) or len(nodes) != self._num_nodes:
            _fail(
                f"facts.nodes must have {self._num_nodes} rows, got "
                f"{len(nodes) if isinstance(nodes, list) else nodes!r}"
            )
        owner: dict[int, int] = {}
        for index, row in enumerate(nodes):
            if not isinstance(row, list) or len(row) != _NODE_COLUMNS:
                _fail(
                    f"facts.nodes[{index}] must have {_NODE_COLUMNS} columns, "
                    f"got {row!r}"
                )
            bad = next((c for c, v in enumerate(row) if not _is_finite_number(v)), None)
            if bad is not None:
                _fail(f"facts.nodes[{index}][{bad}] is not finite: {row[bad]!r}")
            slot = row[_SLOT_COLUMN]
            if not _is_int(slot) or not -1 <= slot < self._num_slots:
                _fail(
                    f"facts.nodes[{index}] slot must be an integer in "
                    f"[-1,{self._num_slots}), got {slot!r}"
                )
            if slot < 0:
                continue
            if slot in owner:
                _fail(
                    f"facts.nodes slot {slot} appears on nodes {owner[slot]} and "
                    f"{index}"
                )
            owner[slot] = index
            if self._slot_node_ids[slot] != self._node_ids[index]:
                _fail(
                    f"facts.nodes[{index}] claims slot {slot}, which init assigns to "
                    f"{self._slot_node_ids[slot]!r}, not {self._node_ids[index]!r}"
                )
        missing = [s for s in range(self._num_controlled) if s not in owner]
        if missing:
            _fail(f"facts.nodes is missing the active slots {missing}")

    def _check_fact_links(self, links) -> None:
        if not isinstance(links, list) or len(links) != self._num_links:
            _fail(
                f"facts.links must have {self._num_links} rows, got "
                f"{len(links) if isinstance(links, list) else links!r}"
            )
        for index, row in enumerate(links):
            if not isinstance(row, list) or len(row) != _LINK_COLUMNS:
                _fail(
                    f"facts.links[{index}] must have {_LINK_COLUMNS} columns, "
                    f"got {row!r}"
                )
            sinr, capacity, is_los = row
            if not _is_finite_number(sinr):
                _fail(f"facts.links[{index}] sinr_db is not finite: {sinr!r}")
            if not _is_finite_number(capacity) or capacity < 0:
                _fail(
                    f"facts.links[{index}] capacity_mbps must be finite and >= 0, "
                    f"got {capacity!r}"
                )
            if not _is_int(is_los) or is_los not in (0, 1):
                _fail(f"facts.links[{index}] is_los is not 0 or 1: {is_los!r}")

    def _check_fact_window(self, window, ticks_in_step: int, reward: float) -> None:
        if not isinstance(window, dict):
            _fail(f"facts.window must be an object, got {window!r}")
        expected = set(_WINDOW_COUNTS) | set(_WINDOW_SUMS)
        if set(window) != expected:
            _fail(
                f"facts.window keys {sorted(window)} != {sorted(expected)}"
            )
        for key in _WINDOW_COUNTS:
            if not _is_int(window[key]) or window[key] < 0:
                _fail(
                    f"facts.window {key} must be a non-negative integer, got "
                    f"{window[key]!r}"
                )
        for key in ("demand_mbps_sum", "delivered_mbps_sum"):
            if not _is_finite_number(window[key]) or window[key] < 0:
                _fail(
                    f"facts.window {key} must be finite and >= 0, got {window[key]!r}"
                )
        if not _is_finite_number(window["legacy_reward_sum"]):
            _fail(
                "facts.window legacy_reward_sum must be finite, got "
                f"{window['legacy_reward_sum']!r}"
            )

        ticks = window["ticks"]
        if ticks != ticks_in_step:
            _fail(
                f"facts.window ticks {ticks} != ticks_in_step {ticks_in_step}"
            )
        demand, delivered = window["demand_mbps_sum"], window["delivered_mbps_sum"]
        if delivered > demand + _DELIVERED_REL_TOL * max(1.0, demand):
            _fail(
                f"facts.window delivered_mbps_sum {delivered} exceeds "
                f"demand_mbps_sum {demand}"
            )
        pairs = ticks * self._num_links
        for key in ("connected_pairs_sum", "los_pairs_sum"):
            if window[key] > pairs:
                _fail(
                    f"facts.window {key} {window[key]} exceeds ticks*num_links = "
                    f"{pairs}"
                )
        if window["unroutable_flow_ticks"] > window["flow_ticks_with_demand"]:
            _fail(
                f"facts.window unroutable_flow_ticks "
                f"{window['unroutable_flow_ticks']} exceeds flow_ticks_with_demand "
                f"{window['flow_ticks_with_demand']}"
            )
        if ticks and abs(reward - window["legacy_reward_sum"] / ticks) > _REWARD_TOL:
            _fail(
                f"reward {reward} != legacy_reward_sum/ticks = "
                f"{window['legacy_reward_sum'] / ticks}"
            )

    def joint_action(self, action) -> list[int]:
        """Convert an action to one integer in [0, 4] per slot; raise ValueError otherwise."""
        array = np.asarray(action)
        if array.ndim == 0:
            raise ValueError(
                f"centralized mode needs {self._num_slots} slot actions, got a scalar"
            )
        values = array.reshape(-1).tolist()
        if len(values) != self._num_slots:
            raise ValueError(
                f"centralized action must have {self._num_slots} entries, "
                f"got {len(values)}"
            )
        joint = []
        for slot, value in enumerate(values):
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"slot {slot} action must be an integer, got {value!r}")
            index = int(value)
            if not 0 <= index < SLOT_ACTIONS:
                raise ValueError(
                    f"slot {slot} action {index} is outside [0,{SLOT_ACTIONS - 1}]"
                )
            joint.append(index)
        return joint


class LegacyProtocol:
    """Validates single-node legacy step messages, which have no `init` message."""

    def __init__(self, obs_width: int | None):
        self.obs_width = obs_width
        self._last_tick: int | None = None

    def validate_step(self, msg: dict, first: bool = False) -> None:
        if msg.get("type") != "step":
            _fail(
                f"Expected a 'step' message, got type {msg.get('type')!r}"
            )
        obs = msg.get("obs")
        if not isinstance(obs, dict):
            _fail(f"legacy obs must be an object, got {obs!r}")
        pos = obs.get("controlled_pos")
        if not isinstance(pos, list) or not all(_is_finite_number(v) for v in pos):
            _fail(f"legacy controlled_pos is not finite: {pos!r}")
        sinrs, caps = obs.get("link_sinrs"), obs.get("link_capacities")
        if not isinstance(sinrs, list) or not isinstance(caps, list):
            _fail("legacy obs needs link_sinrs and link_capacities lists")
        if len(sinrs) != len(caps):
            _fail(
                f"legacy link_sinrs ({len(sinrs)}) and link_capacities ({len(caps)}) "
                "differ in length"
            )
        if not all(_is_finite_number(v) for v in (*sinrs, *caps)):
            _fail("legacy link values are not all finite numbers")
        if not _is_finite_number(msg.get("reward")):
            _fail(f"reward is not a finite number: {msg.get('reward')!r}")
        if not isinstance(msg.get("done"), bool):
            _fail(f"done must be a boolean, got {msg.get('done')!r}")
        if not _is_int(msg.get("tick")) or not _is_finite_number(msg.get("time_s")):
            _fail(
                f"legacy tick/time_s invalid: {msg.get('tick')!r}, {msg.get('time_s')!r}"
            )
        if not first and self._last_tick is not None and msg["tick"] <= self._last_tick:
            _fail(
                f"tick {msg['tick']} does not advance past {self._last_tick}"
            )
        if not first and self.obs_width is not None:
            width = len(pos) + 2 * len(sinrs)
            if width != self.obs_width:
                _fail(
                    f"legacy observation width {width} != "
                    f"{self.obs_width}"
                )
        self._last_tick = msg["tick"]

    @staticmethod
    def parse_obs(msg: dict) -> np.ndarray:
        obs = msg["obs"]
        parts = list(obs["controlled_pos"])
        for sinr, capacity in zip(obs["link_sinrs"], obs["link_capacities"]):
            parts.extend((sinr, capacity))
        return np.asarray(parts, dtype=np.float64)
