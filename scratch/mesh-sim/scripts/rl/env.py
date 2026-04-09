"""Gymnasium environment wrapping the C++ mesh simulator via stdin/stdout JSON."""

import json
import subprocess

import gymnasium
import numpy as np
from gymnasium import spaces


class MeshRlEnv(gymnasium.Env):
    """Mesh simulator RL environment.

    Spawns the C++ mesh-sim binary as a subprocess.  Each tick the sim writes
    an observation+reward JSON line to stdout; this env reads it, returns it to
    the agent, then writes the agent's action back to the sim's stdin.

    Parameters
    ----------
    sim_binary : str
        Path to the built ns3 mesh-sim executable.
    run_config : str
        Path to the run.ini with an ``[rl]`` section.
    seed : int
        RNG seed passed to the sim via ``--seed``.
    """

    metadata = {"render_modes": []}

    def __init__(self, sim_binary: str, run_config: str, seed: int | None = None,
                 output_dir: str = ""):
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

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
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
        info = {"time_s": msg["time_s"], "tick": msg["tick"]}

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
                self.action_space = spaces.Discrete(5)

        return obs, info

    def step(self, action):
        self._send_action(action)
        msg = self._read_message()

        obs = self._parse_obs(msg)
        reward = float(msg["reward"])
        terminated = bool(msg["done"])
        truncated = False
        info = {"time_s": msg["time_s"], "tick": msg["tick"]}

        if terminated:
            self._wait_proc()

        return obs, reward, terminated, truncated, info

    def close(self):
        self._kill_proc()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_message(self) -> dict:
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            self._proc.wait()
            rc = self._proc.returncode
            raise RuntimeError(f"Sim process ended unexpectedly (exit code {rc})")
        return json.loads(line)

    def _send_action(self, action) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        if self._action_type == "continuous":
            action_val = [float(action[0]), float(action[1])]
        else:
            action_val = int(action)
        msg = json.dumps({"action": action_val})
        self._proc.stdin.write(msg + "\n")
        self._proc.stdin.flush()

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
