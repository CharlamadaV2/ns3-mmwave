"""Gymnasium adapter for the mesh simulator's RL protocol."""

import math
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium import spaces

from .config import read_rl_bounds, read_scenario_seed
from .episode import EpisodeSession
from .observations import get_preset, observation_schema
from .protocol import CentralizedProtocol, LegacyProtocol, ProtocolError, SLOT_ACTIONS
from .rewards import RewardComposer, reward_schema
from .selection import (RlSelection, resolve_selection, uses_composed_reward,
                        uses_custom_observation)

_TOTAL_TOL = 1e-9


class MeshRlEnv(gymnasium.Env):
    """Run one simulator process per episode with a fixed training seed."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, sim_binary: str, run_config: str, seed: int | None = None,
                 output_dir: str = "", band: str | None = None,
                 render_mode: str | None = None,
                 selection: RlSelection | None = None):
        super().__init__()
        if not output_dir:
            raise ValueError("MeshRlEnv requires output_dir (the training output root)")

        self._run_config = run_config
        self._selection = (selection if selection is not None
                           else resolve_selection(run_config))
        self._preset = (get_preset(self._selection.observation_preset)
                        if uses_custom_observation(self._selection) else None)
        self._composer = (RewardComposer(self._selection.reward_components,
                                         self._selection.reward_weights)
                          if uses_composed_reward(self._selection) else None)
        self._observation_schema: dict | None = None
        self._reward_schema: dict | None = None
        self._output_dir = Path(output_dir)
        self._session = EpisodeSession(sim_binary, run_config, self._output_dir, band)
        if seed is not None:
            self.seed_value = int(seed)
            self.seed_source = "cli"
        else:
            ini_seed = read_scenario_seed(run_config)
            if ini_seed is None:
                raise ValueError(
                    "No seed available: pass seed=<int> to MeshRlEnv or set "
                    f"[scenario] seed in {run_config}"
                )
            self.seed_value = ini_seed
            self.seed_source = "run.ini"

        self.action_space: spaces.Space | None = None
        self.observation_space: spaces.Space | None = None
        self._action_type: str | None = None
        self._signature: dict | None = None
        self._control_mode: str | None = None
        self._contract: dict | None = None
        self._protocol: CentralizedProtocol | LegacyProtocol | None = None

        self._x_range: tuple[float, float] | None = None
        self._y_range: tuple[float, float] | None = None
        self._z_range: tuple[float, float] | None = None
        self._ctrl_pos: np.ndarray | None = None

        self.window_size = 512
        self.render_mode = render_mode

    @property
    def control_mode(self) -> str | None:
        return self._control_mode

    @property
    def contract(self) -> dict | None:
        return dict(self._contract) if self._contract is not None else None

    @property
    def selection(self) -> RlSelection:
        return self._selection

    @property
    def observation_schema(self) -> dict | None:
        return self._observation_schema

    @property
    def reward_schema(self) -> dict | None:
        return self._reward_schema

    @property
    def _proc(self):
        return self._session.proc

    @property
    def _cmd(self):
        return self._session.command

    def reset(self, *, seed=None, options=None):
        """Stop any running episode, launch a fresh simulator, and return the first observation."""
        if seed is not None and isinstance(options, dict) and "seed_source" in options:
            self.seed_value = int(seed)
            self.seed_source = str(options["seed_source"])
        elif seed is not None and int(seed) != self.seed_value:
            self.seed_value = int(seed)
            self.seed_source = "gym"

        self._session.stop("interrupted", "reset")
        self._session.start(self.seed_value, self.seed_source)
        first = self._session.read_message()
        kind = first.get("type")
        if kind == "init":
            return self._reset_centralized(first)
        if kind == "step":
            return self._reset_legacy(first)
        self._session.protocol_error(
            f"Unexpected first message type {kind!r}; expected 'init' (centralized) "
            "or 'step' (legacy)"
        )

    def step(self, action):
        """Send one action, validate the reply, and return (obs, reward, terminated, False, info)."""
        if self._control_mode == "centralized":
            assert isinstance(self._protocol, CentralizedProtocol)
            action_value = self._protocol.joint_action(action)
        elif self._action_type == "continuous":
            action_value = [float(action[0]), float(action[1])]
        else:
            action_value = int(action)
        self._session.send_action(action_value)
        msg = self._session.read_message()

        detail = None
        if self._control_mode == "centralized":
            assert isinstance(self._protocol, CentralizedProtocol)
            obs = self._validated_step(self._protocol, msg)
            info = self._centralized_info(msg)
            if self._preset is not None:
                obs = self._preset.build(self._protocol.facts, self._contract)
            reward = float(msg["reward"])
            if self._composer is not None:
                breakdown = self._composer.compose(
                    self._protocol.facts["window"], reward, self._contract)
                self._check_total(breakdown)
                reward = breakdown.total
                info["reward"] = {
                    "total": breakdown.total,
                    "components": dict(breakdown.components),
                    "valid": dict(breakdown.valid),
                    "weights": dict(breakdown.weights),
                    "legacy": breakdown.legacy,
                }
                detail = {"obs": obs, "breakdown": breakdown}
            else:
                detail = {"obs": obs}
        else:
            assert isinstance(self._protocol, LegacyProtocol)
            self._validated_step(self._protocol, msg)
            obs = self._protocol.parse_obs(msg)
            self._ctrl_pos = np.asarray(msg["obs"]["controlled_pos"], dtype=float)
            info = {"time_s": msg["time_s"], "tick": msg["tick"]}
            reward = float(msg["reward"])
        terminated = bool(msg["done"])
        self._session.record_step(msg, reward, detail)
        if terminated:
            self._session.stop("completed", "done")
        return obs, reward, terminated, False, info

    def render(self):
        pass

    def _render_frame(self):
        pass

    def action_masks(self) -> np.ndarray:
        """Simulator mask in centralized mode; bounds-derived Discrete(7) mask in legacy mode."""
        if self._control_mode == "centralized":
            assert isinstance(self._protocol, CentralizedProtocol)
            mask = self._protocol.mask
            if mask is None:
                return np.ones(self._protocol.mask_dim, dtype=bool)
            return mask.astype(bool)

        n = self.action_space.n if isinstance(self.action_space, spaces.Discrete) else 7
        mask = np.ones(n, dtype=bool)
        if self._ctrl_pos is None or self._x_range is None:
            return mask

        x, y = float(self._ctrl_pos[0]), float(self._ctrl_pos[1])
        xmin, xmax = self._x_range
        ymin, ymax = self._y_range
        if n > 0:
            mask[0] = x > xmin
        if n > 1:
            mask[1] = x < xmax
        if n > 2:
            mask[2] = y > ymin
        if n > 3:
            mask[3] = y < ymax
        if n > 5 and self._z_range is not None and self._ctrl_pos.shape[0] >= 3:
            z = float(self._ctrl_pos[2])
            zmin, zmax = self._z_range
            mask[4] = z > zmin
            mask[5] = z < zmax
        return mask

    def valid_action_mask(self) -> np.ndarray:
        """Alias of `action_masks`."""
        return self.action_masks()

    def close(self):
        """Stop the simulator and mark the episode interrupted."""
        self._session.stop("interrupted", "close")

    def _reset_legacy(self, msg: dict):
        width = self.observation_space.shape[0] if self.observation_space else None
        protocol = LegacyProtocol(width)
        self._validated_step(protocol, msg, first=True)
        action_type = msg.get("action_type", "discrete")
        if action_type not in ("discrete", "continuous"):
            self._session.protocol_error(f"Unknown legacy action_type {action_type!r}")

        obs = protocol.parse_obs(msg)
        signature = {
            "control_mode": "legacy",
            "action_type": action_type,
            "obs_dim": int(obs.shape[0]),
        }
        self._check_signature(signature)
        self._control_mode = "legacy"
        self._contract = None
        self._action_type = action_type
        self._protocol = protocol
        self._ctrl_pos = np.asarray(msg["obs"]["controlled_pos"], dtype=float)
        info = {"time_s": msg["time_s"], "tick": msg["tick"]}

        if self._x_range is None:
            self._x_range, self._y_range, self._z_range = read_rl_bounds(self._run_config)
        if self.observation_space is None:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs.shape[0],), dtype=np.float64
            )
        if self.action_space is None:
            if action_type == "continuous":
                self.action_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                )
            else:
                self.action_space = spaces.Discrete(7)
        return obs, info

    def _reset_centralized(self, init: dict):
        try:
            protocol = CentralizedProtocol(init)
        except ProtocolError as exc:
            self._session.protocol_error(str(exc))
        slots = int(init["max_controlled_nodes"])
        signature = {
            "control_mode": "centralized",
            "contract": init["contract"],
            "dimensions": init["dimensions"],
            "action_meanings": tuple(init["action_meanings"]),
            "num_mesh_nodes": init["num_mesh_nodes"],
            "facts_schema": init["facts_schema"],
            "node_ids": tuple(init["node_ids"]),
            "bounds": tuple(sorted(init["bounds"].items())),
            "max_controlled_nodes": slots,
            "obs_dim": init["obs_dim"],
            "mask_dim": init["mask_dim"],
            "nvec": (SLOT_ACTIONS,) * slots,
            "slot_node_ids": tuple(init["slot_node_ids"]),
            "slot_speed_mps": tuple(init["slot_speed_mps"]),
            "tick_s": init["tick_s"],
            "decision_interval_s": init["decision_interval_s"],
            "decision_interval_ticks": init["decision_interval_ticks"],
            "num_ticks": init["num_ticks"],
            "num_decisions": init["num_decisions"],
            "reward_type": init["reward_type"],
            "reward_window": init["reward_window"],
            "wall_policy": init["wall_policy"],
        }
        self._check_signature(signature)
        self._control_mode = "centralized"
        self._contract = dict(init)
        self._action_type = "discrete"
        self._protocol = protocol
        self.observation_space = (
            self._preset.space(init) if self._preset is not None
            else spaces.Box(low=-np.inf, high=np.inf, shape=(init["obs_dim"],),
                            dtype=np.float64)
        )
        self.action_space = spaces.MultiDiscrete([SLOT_ACTIONS] * slots)
        self._session.set_contract(init)
        self._observation_schema = observation_schema(
            self._selection.observation_preset, init)
        self._reward_schema = reward_schema(
            self._selection.reward_components, self._selection.reward_weights,
            reward_type=init["reward_type"], reward_window=init["reward_window"])
        self._session.set_selection(self._selection, self._observation_schema,
                                    self._reward_schema)

        msg = self._session.read_message()
        obs = self._validated_step(protocol, msg, first=True)
        if self._preset is not None:
            obs = self._preset.build(protocol.facts, init)
        self._session.record_reset(msg, obs)
        return obs, self._centralized_info(msg)

    @staticmethod
    def _check_total(breakdown) -> None:
        expected = sum(breakdown.weights[name] * value
                       for name, value in breakdown.components.items()
                       if breakdown.valid[name])
        if (not math.isfinite(breakdown.total) or not math.isfinite(expected)
                or abs(breakdown.total - expected) >= _TOTAL_TOL):
            raise ValueError("Composed reward does not match its weighted components")

    def _validated_step(self, protocol, msg: dict, first: bool = False):
        try:
            return protocol.validate_step(msg, first=first)
        except ProtocolError as exc:
            self._session.protocol_error(str(exc))

    @staticmethod
    def _centralized_info(msg: dict) -> dict:
        return {
            "tick": msg["tick"],
            "time_s": msg["time_s"],
            "decision": msg["decision"],
            "ticks_in_step": msg["ticks_in_step"],
            "revalidated_slots": list(msg["revalidated_slots"]),
        }

    def _check_signature(self, signature: dict) -> None:
        if self._signature is None:
            self._signature = signature
            return
        if signature == self._signature:
            return
        keys = sorted(set(self._signature) | set(signature))
        diffs = "; ".join(
            f"{key}: first={self._signature.get(key)!r} now={signature.get(key)!r}"
            for key in keys if self._signature.get(key) != signature.get(key)
        )
        self._session.protocol_error(
            f"Simulator contract changed between resets ({diffs})"
        )
