"""Client of the simulator's `mesh_channel_query_v1` candidate-layout scoring worker."""

import atexit
import json
import math
import os
import selectors
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np

from scripts.sim_support import find_mesh_root, simulator_env, stop_process

CONTRACT = "mesh_channel_query_v1"
MODES = ("standalone", "evaluation")
BANDS = ("mmwave", "sub-6")
SINR_MIN_DB = -6.7
POSITION_TOL_M = 1e-6
REQUEST_BASE_TIMEOUT_S = 60.0
REQUEST_PER_LAYOUT_S = 0.5
INIT_TIMEOUT_S = 60.0
INIT_LINE_LIMIT_BYTES = 16 * 1024 * 1024
SHUTDOWN_GRACE_S = 5.0
TERMINATE_WAIT_S = 5.0
READ_CHUNK_BYTES = 1024 * 1024
WRITE_CHUNK_BYTES = 64 * 1024
REQUIRED_LIMITS = ("max_layouts", "max_probes", "max_request_line_bytes",
                   "max_child_response_bytes")
# Response-line bound: each layout may carry one full child body plus re-framing.
RESPONSE_LAYOUT_SLACK_BYTES = 4096
RESPONSE_ENVELOPE_BYTES = 64 * 1024
STAT_KEYS = ("requests", "layouts", "links", "probe_links", "wall_s")


class PlannerError(RuntimeError):
    """Placement planning failed; no partial plan is usable."""


@dataclass(frozen=True)
class LayoutResult:
    """t = 0 channel facts for one layout; diagonals are -inf dB, 0 Mbps and False."""

    connected: np.ndarray
    sinr_db: np.ndarray
    capacity_mbps: np.ndarray
    is_los: np.ndarray
    coverage: list[list[int]] | None
    wall_s: float


def _reject_constant(token: str):
    raise ValueError(f"non-finite JSON token {token}")


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and \
        math.isfinite(value)


def _wait_fd(fd: int, event: int, timeout_s: float) -> bool:
    with selectors.DefaultSelector() as selector:
        selector.register(fd, event)
        return bool(selector.select(max(timeout_s, 0.0)))


def query_command(sim_binary, query_run_config, planning_seed: int, band: str | None,
                  mode: str) -> list[str]:
    """Worker argv: --rl-mode only in evaluation, --band only when given."""
    command = [str(sim_binary), f"--run-config={query_run_config}", "--channel-query",
               f"--seed={int(planning_seed)}"]
    if band is not None:
        command.append(f"--band={band}")
    if mode == "evaluation":
        command.append("--rl-mode")
    return command


class ChannelScorer:
    """Owns one channel-query worker: init handshake, batched evaluation, cleanup, stats."""

    def __init__(self, sim_binary, query_run_config, planning_seed: int, run_id: int,
                 band: str | None, mode: str, roster, start_positions, jammer_seed: int,
                 log, *, request_timeout_s: float | None = None,
                 max_layouts: int | None = None, init_timeout_s: float = INIT_TIMEOUT_S,
                 shutdown_grace_s: float = SHUTDOWN_GRACE_S,
                 terminate_wait_s: float = TERMINATE_WAIT_S):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if band is not None and band not in BANDS:
            raise ValueError(f"band must be one of {BANDS} or None, got {band!r}")
        self._roster = [str(node_id) for node_id in roster]
        n = len(self._roster)
        if n < 2 or len(set(self._roster)) != n:
            raise ValueError("roster needs at least two distinct node ids")
        self._start = np.asarray(start_positions, dtype=float)
        if self._start.shape != (n, 3) or not np.isfinite(self._start).all():
            raise ValueError(f"start_positions must be {n} finite [x, y, z] rows")
        if request_timeout_s is not None and not request_timeout_s > 0:
            raise ValueError("request_timeout_s must be positive")
        if max_layouts is not None and (not _is_int(max_layouts) or max_layouts < 1):
            raise ValueError("max_layouts must be an integer >= 1")
        self._expected = {"seed": int(planning_seed), "run_id": int(run_id),
                          "jammer_seed": int(jammer_seed)}
        self._band_requested = band
        self._mode = mode
        self._log = log
        self._request_timeout_s = request_timeout_s
        self._max_layouts_override = max_layouts
        self._shutdown_grace_s = shutdown_grace_s
        self._terminate_wait_s = terminate_wait_s
        self._pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        self._stats = dict.fromkeys(STAT_KEYS, 0)
        self._stats["wall_s"] = 0.0
        self._request_id = 0
        self._buffer = bytearray()
        self._probes = None
        self._probe_json = ""
        self._init: dict = {}
        self._in_flight = False
        self._closed = False
        self._stderr_thread = None
        self._proc = None
        self.command = query_command(sim_binary, query_run_config, planning_seed, band, mode)
        self._launch()
        try:
            line = self._read_line(time.monotonic() + init_timeout_s, INIT_LINE_LIMIT_BYTES,
                                   "the init line")
            self._check_init(self._decode(line, "init line"))
        except BaseException:
            self._abort()
            raise

    def _launch(self) -> None:
        self._log.write("channel query: " + " ".join(self.command) + "\n")
        self._log.flush()
        try:
            stderr_fd = self._log.fileno()
        except (AttributeError, OSError, ValueError):
            stderr_fd = None
        try:
            # start_new_session gives the worker its own group so close() can reach every
            # descendant. If this process dies, the worker reads EOF on stdin and exits.
            self._proc = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE if stderr_fd is None else stderr_fd, bufsize=0,
                start_new_session=True, env=simulator_env(find_mesh_root()))
        except OSError as exc:
            self._closed = True
            raise PlannerError(f"cannot start the channel query worker: {exc}") from exc
        atexit.register(self.close)
        os.set_blocking(self._proc.stdin.fileno(), False)
        if stderr_fd is None:
            self._stderr_thread = threading.Thread(target=self._copy_stderr, daemon=True)
            self._stderr_thread.start()

    def _copy_stderr(self) -> None:
        for raw in iter(self._proc.stderr.readline, b""):
            self._log.write(raw.decode("utf-8", errors="replace"))

    def __enter__(self) -> "ChannelScorer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def pid(self) -> int | None:
        return None if self._proc is None else self._proc.pid

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def init(self) -> dict:
        return json.loads(json.dumps(self._init))

    @property
    def channel(self) -> dict:
        return dict(self._init.get("channel", {}))

    @property
    def limits(self) -> dict:
        return dict(self._init.get("limits", {}))

    @property
    def band(self) -> str | None:
        return self._init.get("band")

    @property
    def sinr_threshold_db(self) -> float | None:
        return self._init.get("sinr_threshold_db")

    @property
    def max_layouts(self) -> int:
        published = self._init["limits"]["max_layouts"]
        if self._max_layouts_override is None:
            return published
        return min(published, self._max_layouts_override)

    @property
    def probes(self) -> dict | None:
        """Resolved probe settings (rx gain filled from init when it was None)."""
        if self._probes is None:
            return None
        return {**self._probes, "points": [list(p) for p in self._probes["points"]]}

    @property
    def probe_rx_gain_dbi(self) -> float | None:
        return None if self._probes is None else self._probes["rx_gain_dbi"]

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def _check_init(self, init) -> None:
        if not isinstance(init, dict) or init.get("type") != "init":
            raise PlannerError("channel query worker did not start with an init line")
        problems = []
        if init.get("contract") != CONTRACT:
            problems.append(f"contract {init.get('contract')!r} != {CONTRACT!r}")
        if init.get("node_ids") != self._roster:
            problems.append(f"node_ids {init.get('node_ids')} != roster {self._roster}")
        try:
            starts = np.asarray(init.get("start_positions"), dtype=float)
        except (TypeError, ValueError):
            starts = None
        if starts is None or starts.shape != self._start.shape or not np.all(
                np.abs(starts - self._start) <= POSITION_TOL_M):
            problems.append("start_positions differ from the roster's start positions "
                            f"by more than {POSITION_TOL_M} m")
        for key, expected in self._expected.items():
            if not _is_int(init.get(key)) or init[key] != expected:
                problems.append(f"{key} {init.get(key)!r} != {expected}")
        rl_expected = self._mode == "evaluation"
        if init.get("rl_enabled") is not rl_expected:
            problems.append(f"rl_enabled {init.get('rl_enabled')!r} != {rl_expected} "
                            f"for mode {self._mode}")
        if init.get("band") not in BANDS:
            problems.append(f"band {init.get('band')!r} is not one of {BANDS}")
        elif self._band_requested is not None and (
                init["band"] != self._band_requested or init.get("band_source") != "cli"):
            problems.append(f"band {init['band']!r} (source {init.get('band_source')!r}) "
                            f"!= requested {self._band_requested!r} from the CLI")
        if not _is_finite(init.get("sinr_threshold_db")):
            problems.append("sinr_threshold_db is not a finite number")
        channel = init.get("channel")
        if not isinstance(channel, dict) or not _is_finite(channel.get("rx_array_gain_dbi")):
            problems.append("channel.rx_array_gain_dbi is not a finite number")
        limits = init.get("limits")
        if not isinstance(limits, dict) or not all(
                _is_int(limits.get(key)) and limits[key] > 0 for key in REQUIRED_LIMITS):
            problems.append(f"limits must give positive integers {list(REQUIRED_LIMITS)}")
        if problems:
            raise PlannerError("channel query init does not match this plan: " +
                               "; ".join(problems))
        self._init = init

    def set_probes(self, points, *, height_m: float = 1.5, rx_gain_dbi: float | None = None,
                   sinr_db: float = SINR_MIN_DB) -> None:
        """Fix the coverage probe grid once, before the first evaluate()."""
        self._require_open()
        if self._probes is not None:
            raise ValueError("probes are already set")
        if self._stats["requests"]:
            raise ValueError("probes must be set before the first evaluate()")
        grid = np.asarray(points, dtype=float)
        if grid.ndim != 2 or grid.shape[1] != 2 or not len(grid) or \
                not np.isfinite(grid).all():
            raise ValueError("probe points must be a non-empty G x 2 array of finite x, y")
        if rx_gain_dbi is None:
            rx_gain_dbi = float(self._init["channel"]["rx_array_gain_dbi"])
        for name, value in (("height_m", height_m), ("rx_gain_dbi", rx_gain_dbi),
                            ("sinr_db", sinr_db)):
            if not _is_finite(value):
                raise ValueError(f"probe {name} must be finite")
        if height_m < 0:
            raise ValueError("probe height_m must be >= 0")
        if len(grid) > self._init["limits"]["max_probes"]:
            raise PlannerError(f"{len(grid)} coverage probes exceed the worker's max_probes "
                               f"{self._init['limits']['max_probes']}; use fewer coverage "
                               "grid cells or a coarser grid_min_resolution_m")
        self._probes = {"height_m": float(height_m), "rx_gain_dbi": float(rx_gain_dbi),
                        "sinr_db": float(sinr_db), "points": [tuple(p) for p in grid.tolist()]}
        payload = {**self._probes, "points": grid.tolist()}
        self._probe_json = ',"probes":' + json.dumps(payload, separators=(",", ":"))

    def evaluate(self, layouts) -> list[LayoutResult]:
        """Score full N x 3 layouts in order; any worker failure aborts with PlannerError."""
        self._require_open()
        arrays = [self._check_layout(layout, index) for index, layout in enumerate(layouts)]
        fragments = [json.dumps(a.tolist(), separators=(",", ":")) for a in arrays]
        results: list[LayoutResult] = []
        try:
            for start, stop in self._batches(fragments):
                results.extend(self._request(fragments[start:stop], start))
        except BaseException:
            self._abort()
            raise
        return results

    def _check_layout(self, layout, index: int) -> np.ndarray:
        array = np.asarray(layout, dtype=float)
        if array.shape != self._start.shape or not np.isfinite(array).all():
            raise ValueError(f"layout {index} must be {len(self._roster)} finite "
                             "[x, y, z] rows")
        return array

    def _line(self, request_id: int, fragments: list[str]) -> bytes:
        return ('{"type":"evaluate","request_id":%d,"layouts":[%s]%s}\n' % (
            request_id, ",".join(fragments), self._probe_json)).encode()

    def _batches(self, fragments: list[str]):
        limit = self._init["limits"]["max_request_line_bytes"]
        fixed = len(self._line(10 ** 18, []))
        start, size = 0, fixed
        for index, fragment in enumerate(fragments):
            if fixed + len(fragment) > limit:
                raise PlannerError(f"one layout request needs {fixed + len(fragment)} bytes, "
                                   f"over the worker's max_request_line_bytes {limit}; "
                                   "reduce the coverage probe count")
            extra = len(fragment) + 1
            if index > start and (index - start >= self.max_layouts or size + extra > limit):
                yield start, index
                start, size = index, fixed
            size += extra if index > start else len(fragment)
        if start < len(fragments):
            yield start, len(fragments)

    def _request(self, fragments: list[str], offset: int) -> list[LayoutResult]:
        self._request_id += 1
        request_id = self._request_id
        count = len(fragments)
        timeout = self._request_timeout_s if self._request_timeout_s is not None else (
            REQUEST_BASE_TIMEOUT_S + REQUEST_PER_LAYOUT_S * count)
        started = time.monotonic()
        deadline = started + timeout
        what = f"the response to request {request_id} ({count} layouts)"
        self._in_flight = True
        self._write(self._line(request_id, fragments), deadline, what)
        child = self._init["limits"]["max_child_response_bytes"]
        limit = count * (child + RESPONSE_LAYOUT_SLACK_BYTES) + RESPONSE_ENVELOPE_BYTES
        raw = self._read_line(deadline, limit, what)
        self._in_flight = False
        results = self._parse_response(self._decode(raw, what), request_id, count, offset)
        n, probes = len(self._roster), self._probes
        self._stats["requests"] += 1
        self._stats["layouts"] += count
        self._stats["links"] += count * len(self._pairs)
        self._stats["probe_links"] += count * n * (len(probes["points"]) if probes else 0)
        self._stats["wall_s"] += time.monotonic() - started
        return results

    def _parse_response(self, response, request_id: int, count: int,
                        offset: int) -> list[LayoutResult]:
        if not isinstance(response, dict):
            raise PlannerError(f"request {request_id}: response is not a JSON object")
        if response.get("type") == "error":
            raise PlannerError(f"request {request_id} rejected by the channel query worker "
                               f"(request_id {response.get('request_id')!r}): "
                               f"{response.get('message')}")
        if response.get("type") != "result":
            raise PlannerError(f"request {request_id}: unexpected response type "
                               f"{response.get('type')!r}")
        if not _is_int(response.get("request_id")) or response["request_id"] != request_id:
            raise PlannerError(f"request {request_id}: response echoes request_id "
                               f"{response.get('request_id')!r}")
        if not _is_finite(response.get("wall_s")):
            raise PlannerError(f"request {request_id}: wall_s is not a finite number")
        layouts = response.get("layouts")
        if not isinstance(layouts, list) or len(layouts) != count:
            raise PlannerError(f"request {request_id}: expected {count} layout results")
        return [self._parse_layout(entry, offset + k) for k, entry in enumerate(layouts)]

    def _parse_layout(self, entry, index: int) -> LayoutResult:
        where = f"layout {index}"
        if not isinstance(entry, dict):
            raise PlannerError(f"{where}: result is not a JSON object")
        if "error" in entry:
            raise PlannerError(f"{where} failed in the channel query worker: "
                               f"{entry['error']}")
        links = entry.get("links")
        if not isinstance(links, list) or len(links) != len(self._pairs):
            raise PlannerError(f"{where}: expected {len(self._pairs)} links in i<j order")
        n = len(self._roster)
        sinr = np.full((n, n), -np.inf)
        capacity = np.zeros((n, n))
        is_los = np.zeros((n, n), dtype=bool)
        connected = np.zeros((n, n), dtype=bool)
        for (i, j), link in zip(self._pairs, links):
            if not isinstance(link, list) or len(link) != 6 or not (
                    _is_int(link[0]) and _is_int(link[1]) and (link[0], link[1]) == (i, j)):
                raise PlannerError(f"{where}: link {link!r} is not [{i}, {j}, sinr_db, "
                                   "capacity_mbps, is_los, connected]")
            if not (_is_finite(link[2]) and _is_finite(link[3])):
                raise PlannerError(f"{where}: link ({i}, {j}) has a non-finite value")
            if not (isinstance(link[4], bool) and isinstance(link[5], bool)):
                raise PlannerError(f"{where}: link ({i}, {j}) is_los/connected not boolean")
            sinr[i, j] = sinr[j, i] = link[2]
            capacity[i, j] = capacity[j, i] = link[3]
            is_los[i, j] = is_los[j, i] = link[4]
            connected[i, j] = connected[j, i] = link[5]
        if not _is_finite(entry.get("wall_s")):
            raise PlannerError(f"{where}: wall_s is not a finite number")
        return LayoutResult(connected=connected, sinr_db=sinr, capacity_mbps=capacity,
                            is_los=is_los, coverage=self._parse_coverage(entry, where),
                            wall_s=float(entry["wall_s"]))

    def _parse_coverage(self, entry: dict, where: str) -> list[list[int]] | None:
        coverage = entry.get("coverage")
        if self._probes is None:
            if coverage is not None:
                raise PlannerError(f"{where}: coverage returned without probes")
            return None
        count = len(self._probes["points"])
        if not isinstance(coverage, list) or len(coverage) != len(self._roster):
            raise PlannerError(f"{where}: coverage must hold one list per node")
        parsed = []
        for node, covered in enumerate(coverage):
            if not isinstance(covered, list) or not all(
                    _is_int(k) and 0 <= k < count for k in covered):
                raise PlannerError(f"{where}: coverage of node {node} has a probe index "
                                   f"outside 0..{count - 1}")
            parsed.append(sorted(set(covered)))
        return parsed

    def _decode(self, raw: bytes, what: str):
        try:
            return json.loads(raw, parse_constant=_reject_constant)
        except (UnicodeDecodeError, ValueError) as exc:
            raise PlannerError(f"malformed JSON in {what}: {exc}") from exc

    def _worker_status(self) -> str:
        try:
            code = self._proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            return "its stdout closed while it is still running"
        return f"it exited with code {code}"

    def _write(self, data: bytes, deadline: float, what: str) -> None:
        fd = self._proc.stdin.fileno()
        view = memoryview(data)
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PlannerError(f"timed out sending the request for {what}")
            if not _wait_fd(fd, selectors.EVENT_WRITE, remaining):
                continue
            try:
                written = os.write(fd, view[:WRITE_CHUNK_BYTES])
            except BlockingIOError:
                continue
            except OSError as exc:
                raise PlannerError(f"channel query worker stopped reading ({exc}) before "
                                   f"{what}; {self._worker_status()}") from exc
            view = view[written:]

    def _read_line(self, deadline: float, limit: int, what: str) -> bytes:
        fd = self._proc.stdout.fileno()
        scanned = 0
        while True:
            newline = self._buffer.find(b"\n", scanned)
            if newline >= 0:
                if newline + 1 > limit:
                    break
                line = bytes(self._buffer[:newline])
                del self._buffer[:newline + 1]
                return line
            scanned = len(self._buffer)
            if scanned >= limit:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PlannerError(f"timed out waiting for {what}")
            if not _wait_fd(fd, selectors.EVENT_READ, remaining):
                continue
            chunk = os.read(fd, READ_CHUNK_BYTES)
            if not chunk:
                raise PlannerError(f"channel query worker ended before {what}; "
                                   f"{self._worker_status()} (see the planner log)")
            self._buffer += chunk
        raise PlannerError(f"{what} exceeds the {limit}-byte line limit")

    def _require_open(self) -> None:
        if self._closed:
            raise PlannerError("the channel query worker is closed")

    def _abort(self) -> None:
        self._in_flight = True
        self.close()

    def close(self) -> None:
        """Shut the worker down and reap its whole process group; safe to call twice."""
        if self._closed or self._proc is None:
            return
        self._closed = True
        atexit.unregister(self.close)
        proc = self._proc
        graceful = not self._in_flight
        try:
            if graceful:
                try:
                    self._write(b'{"type":"shutdown"}\n',
                                time.monotonic() + self._shutdown_grace_s, "shutdown")
                except PlannerError:
                    graceful = False
            try:
                proc.stdin.close()
            except OSError:
                pass
            if graceful:
                try:
                    proc.wait(timeout=self._shutdown_grace_s)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            stop_process(proc, self._terminate_wait_s, process_group=True)
            proc.stdout.close()
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=self._terminate_wait_s)
                proc.stderr.close()
