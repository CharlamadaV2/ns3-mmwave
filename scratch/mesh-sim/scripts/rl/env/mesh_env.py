"""Gymnasium environment wrapping the C++ mesh simulator via stdin/stdout JSON."""

import configparser
import json
import subprocess

import gymnasium
import numpy as np
from gymnasium import spaces

#Update to work with updated sim
class MeshRlEnv(gymnasium.Env):
    ## Mesh simulator RL environment.
    
    # Spawns the C++ mesh-sim binary as a subprocess.  Each tick the sim writes
    # an observation+reward JSON line to stdout; this env reads it, returns it to
    # the agent, then writes the agent's action back to the sim's stdin.

    # Parameters
    # ----------
    # sim_binary : str
    #     Path to the built ns3 mesh-sim executable.
    # run_config : str
    #     Path to the run.ini with an ``[rl]`` section.
    # seed : int
    #    RNG seed passed to the sim via ``--seed``.

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, sim_binary: str, run_config: str, seed: int | None = None,
                 output_dir: str = "", render_mode: str | None = None):
        super().__init__()
        self._sim_binary = sim_binary
        self._run_config = run_config
        self._seed = seed
        self._output_dir = output_dir
        self._proc: subprocess.Popen | None = None
        self._stderr_file = None

        # Spaces are set dynamically on first reset once we know N and action_type.
        self.action_space: spaces.Space | None = None
        self.observation_space: spaces.Space | None = None
        self._action_type: str | None = None

        # Boundary info for action masking (read from [rl] on first reset).
        self._x_range: tuple[float, float] | None = None
        self._y_range: tuple[float, float] | None = None
        self._z_range: tuple[float, float] | None = None   # None -> z unbounded/unknown
        self._ctrl_pos: np.ndarray | None = None           # latest controlled-node position

        # Rendering (deferred to a later day).
        self.window_size = 512
        self.render_mode = render_mode

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        #Note: Seed is passed to config, for ns-3 sim 
        if seed is not None:
            self._seed = seed

        self._kill_proc()

        cmd = [
            self._sim_binary,
            f"--run-config={self._run_config}",
            "--rl-mode",
        ]
        if self._seed is not None:
            cmd.append(f"--seed={self._seed}")
        if self._output_dir:
            cmd.append(f"--output-dir={self._output_dir}")

        self._stderr_file = open(
            f"{self._output_dir}/sim_stderr.log" if self._output_dir else "/dev/null",
            "w",
        )
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,  # line-buffered
        )

        msg = self._read_message()
        self._action_type = msg.get("action_type", "discrete")

        obs = self._parse_obs(msg)
        self._ctrl_pos = np.asarray(msg["obs"]["controlled_pos"], dtype=float)
        info = {"time_s": msg["time_s"], "tick": msg["tick"]}

        if self._x_range is None:
            self._read_rl_bounds()

        if self.observation_space is None:
            n = len(obs)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(n,), dtype=np.float64
            )

        if self.action_space is None:
            if self._action_type == "continuous":
                self.action_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                )
            else:
                self.action_space = spaces.Discrete(7) # 3D Positioning(-X,+X,-Y,+Y,-Z,+Z,Stay)

        return obs, info

    def step(self, action):
        self._send_action(action)
        msg = self._read_message()

        obs = self._parse_obs(msg)
        self._ctrl_pos = np.asarray(msg["obs"]["controlled_pos"], dtype=float)
        reward = float(msg["reward"])
        terminated = bool(msg["done"])
        truncated = False
        info = {"time_s": msg["time_s"], "tick": msg["tick"]}

        if terminated:
            self._wait_proc()

        return obs, reward, terminated, truncated, info
    
    #NOTE: Use Claude for Rendering code
    def render(self):
        #TODO: Create modes for rendering environment
        pass
        
    def _render_frame(self):
        #TODO: Produce Graph that shows nodes current position with link information
        #TODO: Produce Line Graph for nodes radio quality update, trying to match line similar to PID
        pass
    
    ## @brief Boolean mask of legal discrete actions from the current 3-D position.
    #
    # A move is masked out (False) only when the node is already at the arena
    # boundary in that direction, so the move would be a wasted no-op the sim
    # clamps. "Stay" is always legal. X/Y are checked against x_min/x_max and
    # y_min/y_max; Z against z_min/z_max only when the observation carries a z
    # coordinate AND z bounds are configured (else Z moves stay unmasked).
    #
    # Action index -> direction. MUST match rl-bridge.cc ApplyAction:
    #   0:-X  1:+X  2:-Y  3:+Y  4:-Z  5:+Z  6:Stay
    # (MaskablePPO/ActionMasker look for this exact method name.)
    def action_masks(self) -> np.ndarray:
        n = self.action_space.n if isinstance(self.action_space, spaces.Discrete) else 7
        mask = np.ones(n, dtype=bool)
        if self._ctrl_pos is None or self._x_range is None:
            return mask  # called before first reset -> allow everything

        pos = self._ctrl_pos
        x, y = float(pos[0]), float(pos[1])
        xmin, xmax = self._x_range
        ymin, ymax = self._y_range

        if n > 0: mask[0] = x > xmin       # -X
        if n > 1: mask[1] = x < xmax       # +X
        if n > 2: mask[2] = y > ymin       # -Y
        if n > 3: mask[3] = y < ymax       # +Y
        if n > 5 and self._z_range is not None and pos.shape[0] >= 3:
            z = float(pos[2]); zmin, zmax = self._z_range
            mask[4] = z > zmin             # -Z
            mask[5] = z < zmax             # +Z
        # index 6 (Stay) always True
        return mask

    ## @brief Alias some ActionMasker setups expect.
    def valid_action_mask(self) -> np.ndarray:
        return self.action_masks()

    def close(self):
        self._kill_proc()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    ## Observation Function
    # @brief Grabs output message from sim
    def _read_message(self) -> dict:
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            self._proc.wait()
            rc = self._proc.returncode
            raise RuntimeError(f"Sim process ended unexpectedly (exit code {rc})")
        return json.loads(line)

    ## Action Function
    # @brief Produces action message for sim to make changes
    def _send_action(self, action) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        if self._action_type == "continuous":
            action_val = [float(action[0]), float(action[1])]
        else:
            action_val = int(action)
        msg = json.dumps({"action": action_val})
        self._proc.stdin.write(msg + "\n")
        self._proc.stdin.flush()

    ## Message Parser
    # @brief Parses through output message from env and collects radio quality value
    @staticmethod
    def _parse_obs(msg: dict) -> np.ndarray:
        """Flatten the observation dict into a numpy array.

        Layout: [ctrl_x, ctrl_y, sinr_0, cap_0, sinr_1, cap_1, ...]
        """
        obs_dict = msg["obs"]
        parts = list(obs_dict["controlled_pos"])
        sinrs = obs_dict["link_sinrs"]
        caps = obs_dict["link_capacities"]
        for s, c in zip(sinrs, caps):
            parts.append(s)
            parts.append(c)
        return np.array(parts, dtype=np.float64)
    #TODO: Add render modes and render_fps

    def _wait_proc(self) -> None:
        """Wait for the sim to exit naturally after sending done=true."""
        if self._proc is not None:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
            self._proc = None

    def _kill_proc(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None

    ## @brief Read arena bounds (x/y required, z optional) from the [rl] section.
    def _read_rl_bounds(self) -> None:
        ini = configparser.ConfigParser()
        ini.read(self._run_config)

        def rng(lo, hi, dlo, dhi):
            if ini.has_option("rl", lo) and ini.has_option("rl", hi):
                return (ini.getfloat("rl", lo), ini.getfloat("rl", hi))
            return (dlo, dhi)

        self._x_range = rng("x_min", "x_max", 0.0, 500.0)
        self._y_range = rng("y_min", "y_max", -250.0, 250.0)
        if ini.has_option("rl", "z_min") and ini.has_option("rl", "z_max"):
            self._z_range = (ini.getfloat("rl", "z_min"), ini.getfloat("rl", "z_max"))
        else:
            self._z_range = None
