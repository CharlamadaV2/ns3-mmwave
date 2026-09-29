"""Per-episode simulator process, diagnostics, and manifest ownership."""

import json
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.sim_support import find_mesh_root, simulator_env, tail_lines

from .telemetry import TELEMETRY_FILE, StepRecorder, make_header, make_record

_EPISODE_RE = re.compile(r"^episode-(\d+)$")
_STDERR_TAIL_LINES = 40
_BAD_LINE_CHARS = 200
_MAX_ERROR_CHARS = 1000
_CLEANUP_BUDGET_S = 10.0
_NATURAL_EXIT_S = 5.0
_ESCALATION_WAIT_S = 2.0
_DRAIN_CHUNK = 64 * 1024


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EpisodeSession:
    """Owns one simulator subprocess and its `rl_episode.json` manifest."""

    def __init__(self, sim_binary: str, run_config: str, output_dir: Path,
                 band: str | None):
        self._sim_binary = sim_binary
        self._run_config = run_config
        self._output_dir = output_dir
        self._band = band
        self._next_index: int | None = None
        self._proc: subprocess.Popen | None = None
        self._stderr_file = None
        self._stderr_path: Path | None = None
        self._episode_dir: Path | None = None
        self._episode_index: int | None = None
        self._manifest: dict | None = None
        self._cmd: list[str] = []
        self._msg_count = 0
        self._last_tail = "(no stderr captured)"
        self._recorder: StepRecorder | None = None
        self._last_action = None

    @property
    def proc(self) -> subprocess.Popen | None:
        return self._proc

    @property
    def command(self) -> list[str]:
        return self._cmd

    @property
    def manifest(self) -> dict | None:
        return self._manifest

    def start(self, seed: int, seed_source: str) -> None:
        """Allocate `episode-NNNN/`, write a `running` manifest, and launch the simulator."""
        self._episode_dir, self._episode_index = self._allocate_episode_dir()
        self._last_action = None
        self._cmd = [
            self._sim_binary,
            f"--run-config={self._run_config}",
            "--rl-mode",
            f"--seed={seed}",
            f"--output-dir={self._episode_dir}",
        ]
        if self._band is not None:
            self._cmd.append(f"--band={self._band}")
        self._manifest = {
            "manifest_version": 1,
            "episode": self._episode_index,
            "seed": seed,
            "seed_source": seed_source,
            "command": list(self._cmd),
            "started_at": _now_iso(),
            "ended_at": None,
            "status": "running",
            "exit_code": None,
            "steps": 0,
            "cumulative_reward": 0.0,
        }
        self._write_manifest()
        self._stderr_path = self._episode_dir / "sim_stderr.log"
        self._stderr_file = open(self._stderr_path, "w")
        self._msg_count = 0
        try:
            self._proc = subprocess.Popen(
                self._cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_file,
                text=True,
                bufsize=1,
                env=self._child_env(),
            )
        except OSError as exc:
            self._finalize_manifest("failed", None)
            self._close_stderr()
            raise RuntimeError(f"Failed to launch simulator {self._cmd}: {exc}") from exc

    def set_contract(self, init: dict) -> None:
        """Upgrade the manifest to centralized fields (version 2)."""
        assert self._manifest is not None
        self._manifest.update({
            "manifest_version": 2,
            "control_mode": "centralized",
            "contract": dict(init),
            "decisions": 0,
            "last_tick": None,
            "stop_reason": None,
        })
        self._write_manifest()

    def set_selection(self, selection, observation_schema: dict,
                      reward_schema: dict) -> None:
        """Record policy schemas and open optional step telemetry."""
        if self._manifest is None:
            return
        self._manifest.update({
            "manifest_version": 3,
            "selection": selection.describe(),
            "observation_schema_sha256": observation_schema["sha256"],
            "reward_schema_sha256": reward_schema["sha256"],
            "reward_components_sum": {name: 0.0
                                      for name in selection.reward_components},
            "telemetry": None,
        })
        if selection.telemetry == "steps" and self._episode_dir is not None:
            self._recorder = StepRecorder(self._episode_dir / TELEMETRY_FILE,
                                          selection.telemetry_every)
            self._recorder.write_header(make_header(
                self._manifest["contract"], selection.describe(), observation_schema,
                reward_schema))
            self._manifest["telemetry"] = {"file": TELEMETRY_FILE, "records": 0,
                                           "every": selection.telemetry_every}
        self._write_manifest()

    def record_reset(self, msg: dict, obs) -> None:
        """The reset observation is telemetry only: no policy reward, no totals."""
        if self._recorder is not None:
            self._append_record(msg, None, obs)
            self._write_manifest()

    def record_step(self, msg: dict, reward: float, detail: dict | None = None) -> None:
        """Update manifest totals and, when due, append a telemetry record."""
        if self._manifest is None:
            return
        self._manifest["steps"] += 1
        self._manifest["cumulative_reward"] += reward
        if self._manifest["manifest_version"] >= 2:
            self._manifest["decisions"] = msg["decision"]
            self._manifest["last_tick"] = msg["tick"]
        breakdown = (detail or {}).get("breakdown")
        sums = self._manifest.get("reward_components_sum")
        if breakdown is not None and sums is not None:
            for name, value in breakdown.components.items():
                sums[name] = sums.get(name, 0.0) + value
        if self._recorder is not None and self._recorder.should_save(
                msg["decision"], msg["done"]):
            self._append_record(msg, breakdown if breakdown is not None else reward,
                                (detail or {}).get("obs"))
        self._write_manifest()

    def _append_record(self, msg: dict, reward, obs) -> None:
        assert self._recorder is not None
        self._recorder.append(make_record(
            msg["decision"], msg["tick"], msg["time_s"], msg["ticks_in_step"],
            self._last_action, msg["mask"], msg["revalidated_slots"], msg["facts"],
            msg["reward"], reward, obs))
        if self._manifest is not None and self._manifest.get("telemetry"):
            self._manifest["telemetry"]["records"] = self._recorder.records

    def _close_recorder(self) -> None:
        recorder, self._recorder = self._recorder, None
        if recorder is None:
            return
        recorder.close()
        if self._manifest is not None and self._manifest.get("telemetry"):
            self._manifest["telemetry"]["records"] = recorder.records

    def protocol_error(self, detail: str):
        """Mark the episode failed, stop the simulator, and raise RuntimeError with stderr tail."""
        message = f"{detail} on message line {self._msg_count}"
        if self._manifest is not None:
            self._manifest["error"] = message[:_MAX_ERROR_CHARS]
        rc = self.stop("failed", "error")
        raise RuntimeError(
            f"{message}\n"
            f"command: {self._cmd}\n"
            f"exit code: {rc}\n"
            f"last {_STDERR_TAIL_LINES} stderr lines:\n{self._last_tail}"
        )

    def send_action(self, action) -> None:
        """Write `{"action": ...}` to the simulator; a broken pipe is ignored so the next read reports it."""
        assert self._proc is not None and self._proc.stdin is not None
        self._last_action = action
        message = json.dumps({"action": action})
        try:
            self._proc.stdin.write(message + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass

    def _allocate_episode_dir(self) -> tuple[Path, int]:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if self._next_index is None:
            used = (int(match.group(1)) for path in self._output_dir.iterdir()
                    if (match := _EPISODE_RE.match(path.name)))
            self._next_index = max(used, default=-1) + 1
        while True:
            index = self._next_index
            self._next_index += 1
            path = self._output_dir / f"episode-{index:04d}"
            try:
                path.mkdir(exist_ok=False)
            except FileExistsError:
                continue
            return path, index

    def _child_env(self) -> dict:
        return simulator_env(find_mesh_root(__file__))

    def _write_manifest(self) -> None:
        if self._manifest is None or self._episode_dir is None:
            return
        with open(self._episode_dir / "rl_episode.json", "w") as fh:
            json.dump(self._manifest, fh, indent=2)
            fh.write("\n")

    def _finalize_manifest(self, status: str, exit_code: int | None,
                           stop_reason: str | None = None,
                           escalation: str | None = None) -> None:
        if self._manifest is None:
            return
        self._manifest["status"] = status
        self._manifest["exit_code"] = exit_code
        self._manifest["ended_at"] = _now_iso()
        if self._manifest["manifest_version"] >= 2:
            self._manifest["stop_reason"] = stop_reason
            self._manifest["escalation"] = escalation
        self._write_manifest()
        self._manifest = None

    def _stderr_tail(self) -> str:
        self._close_stderr()
        if self._stderr_path is None or not self._stderr_path.is_file():
            return "(no stderr captured)"
        return tail_lines(self._stderr_path, _STDERR_TAIL_LINES) or "(stderr empty)"

    def read_message(self) -> dict:
        """Read one JSON line from the simulator; EOF or bad JSON fails the episode."""
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        self._msg_count += 1
        if not line:
            if self._manifest is not None:
                self._manifest["error"] = "sim process ended unexpectedly"
            rc = self.stop("failed", "error")
            raise RuntimeError(
                f"Sim process ended unexpectedly (exit code {rc})\n"
                f"command: {self._cmd}\n"
                f"last {_STDERR_TAIL_LINES} stderr lines:\n{self._last_tail}"
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            bad = line.rstrip("\n")[:_BAD_LINE_CHARS]
            message = f"Invalid JSON from sim on message line {self._msg_count}: {exc}"
            if self._manifest is not None:
                self._manifest["error"] = message[:_MAX_ERROR_CHARS]
            rc = self.stop("failed", "error")
            raise RuntimeError(
                f"{message}\n"
                f"offending line: {bad!r}\n"
                f"command: {self._cmd}\n"
                f"exit code: {rc}\n"
                f"last {_STDERR_TAIL_LINES} stderr lines:\n{self._last_tail}"
            ) from exc

    def _close_stderr(self) -> None:
        if self._stderr_file is not None:
            try:
                self._stderr_file.flush()
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None

    @staticmethod
    def _start_drain(proc: subprocess.Popen) -> threading.Thread:
        stream = proc.stdout

        def drain() -> None:
            if stream is None:
                return
            try:
                while stream.read(_DRAIN_CHUNK):
                    pass
            except (ValueError, OSError):
                pass

        thread = threading.Thread(target=drain, name="mesh-sim-drain", daemon=True)
        thread.start()
        return thread

    @staticmethod
    def _close_stream(stream) -> None:
        if stream is None or stream.closed:
            return
        try:
            stream.close()
        except Exception:
            pass

    def stop(self, status: str, stop_reason: str | None = None) -> int | None:
        """Close stdin, escalate exit -> terminate -> kill, join the drain thread, and finalize the manifest."""
        self._close_recorder()
        proc, self._proc = self._proc, None
        if proc is None:
            if status == "failed":
                self._last_tail = self._stderr_tail()
            self._close_stderr()
            if self._manifest is not None:
                self._finalize_manifest(status, None, stop_reason, None)
            return None

        deadline = time.monotonic() + _CLEANUP_BUDGET_S
        reader = self._start_drain(proc)
        self._close_stream(proc.stdin)

        escalation = "exited"
        rc = self._reap(proc, min(_NATURAL_EXIT_S, self._remaining(deadline)))
        if rc is None:
            proc.terminate()
            escalation = "terminated"
            rc = self._reap(proc, min(_ESCALATION_WAIT_S, self._remaining(deadline)))
        if rc is None:
            proc.kill()
            escalation = "killed"
            rc = self._reap(proc, min(_ESCALATION_WAIT_S, self._remaining(deadline)))

        reader.join(timeout=self._remaining(deadline))
        self._close_stream(proc.stdout)
        if status == "failed" or stop_reason == "done":
            self._last_tail = self._stderr_tail()
        self._close_stderr()

        if stop_reason == "done":
            status = "completed" if rc == 0 else "failed"
        self._finalize_manifest(status, rc, stop_reason, escalation)
        return rc

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    @staticmethod
    def _reap(proc: subprocess.Popen, timeout: float) -> int | None:
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
