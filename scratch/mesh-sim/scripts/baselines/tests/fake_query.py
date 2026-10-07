"""Stand-in `mesh_channel_query_v1` worker with a distance-threshold channel and fault modes."""

# Channel: a pair is connected iff its 3-D distance <= FAKE_QUERY_RANGE_M (default 100);
# sinr_db = SINR_MIN_DB + 20 log10(range / max(d, 1)), so connected == (sinr_db >= -6.7).
# A probe is covered iff the same formula, with the probe at `height_m`, is >= probes.sinr_db.
# Environment:
#   FAKE_QUERY_FAULT       comma list: hang, crash, nonfinite, malformed, layout_error,
#                          request_error, wrong_id, short_links, bad_coverage, oversized,
#                          ignore_term, grandchild, grandchild_ignore_term
#   FAKE_QUERY_FAULT_AT    1-based evaluate-request ordinal for per-request faults (default 1)
#   FAKE_QUERY_PAD_BYTES   whitespace bytes added to the `oversized` response (default 2 MiB)
#   FAKE_QUERY_INIT_PATCH  JSON object merged over the init line (nested dicts merged)
#   FAKE_QUERY_PID_FILE    JSON {"worker": pid, "grandchild": pid|null} written at start
#   FAKE_QUERY_RECORD      JSON lines: argv at start, then one entry per request line

import configparser
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

CONTRACT = "mesh_channel_query_v1"
SINR_MIN_DB = -6.7
MIB = 1024 * 1024
LIMITS = {"max_layouts": 1024, "max_probes": 10000, "max_request_line_bytes": 16 * MIB,
          "max_child_response_bytes": 16 * MIB, "child_deadline_s": 60.0}
CRASH_EXIT_CODE = 3
GRANDCHILD = ("import signal, sys, time\n"
              "if sys.argv[1] == 'ignore':\n"
              "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
              "while True:\n"
              "    time.sleep(0.1)\n")


def _die(message: str) -> None:
    print(f"fake_query: {message}", file=sys.stderr, flush=True)
    sys.exit(2)


def _parse_argv(argv: list[str]) -> dict:
    flags = {}
    for arg in argv:
        if arg in ("--channel-query", "--rl-mode"):
            key, value = arg, True
        else:
            key, sep, value = arg.partition("=")
            if key not in ("--run-config", "--seed", "--band") or not sep:
                _die(f"unexpected argument {arg!r}")
        if key in flags:
            _die(f"repeated argument {arg!r}")
        flags[key] = value
    for required in ("--run-config", "--channel-query", "--seed"):
        if required not in flags:
            _die(f"missing {required}")
    return flags


def _value(ini: configparser.ConfigParser, section: str, key: str, default=None):
    if not ini.has_option(section, key):
        return default
    value = ini.get(section, key).split("#", 1)[0].split(";", 1)[0].strip()
    return value if value else default


def _start(entry: dict) -> list[float]:
    point = (entry.get("waypoints") or [{}])[0] if entry.get("mobility") == "waypoint" \
        else entry.get("position") or {}
    return [float(point.get(axis, 0.0)) for axis in ("x", "y", "z")]


def _merge(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def _init(flags: dict) -> tuple[dict, list]:
    run_config = Path(flags["--run-config"])
    ini = configparser.ConfigParser(interpolation=None, default_section="\n")
    ini.optionxform = str
    ini.read_string(run_config.read_text(encoding="utf-8"))
    nodes = json.loads((run_config.parent / _value(ini, "scenario", "nodes_file",
                                                     "nodes.json")).read_text())
    ids = [entry["id"] for entry in nodes]
    if "--band" in flags:
        band, band_source = flags["--band"], "cli"
    elif _value(ini, "channel", "band") is not None:
        band, band_source = _value(ini, "channel", "band"), "run.ini"
    else:
        band, band_source = "mmwave", "default"
    rl_enabled = bool(flags.get("--rl-mode"))
    controlled = []
    if rl_enabled:
        raw = _value(ini, "rl", "controlled_nodes", "")
        names = ids if raw == "all" else [t.strip() for t in raw.split(",") if t.strip()]
        controlled = [ids.index(name) for name in names if name in ids]
    buildings = _value(ini, "scenario", "buildings_file")
    num_buildings = len(json.loads((run_config.parent / buildings).read_text())) \
        if buildings else 0
    channel = {
        "frequency_ghz": float(_value(ini, "channel", "frequency_ghz", 28.0)),
        "tx_power_dbm": float(_value(ini, "channel", "tx_power_dbm", 30.0)),
        "bandwidth_mhz": float(_value(ini, "channel", "bandwidth_mhz", 400.0)),
        "noise_figure_db": float(_value(ini, "channel", "noise_figure_db", 5.0)),
        "amc_model": _value(ini, "channel", "amc_model", "shannon"),
        "channel_model": _value(ini, "channel", "channel_model", "3gpp"),
        "scenario": _value(ini, "channel", "scenario", "UMi"),
        "condition_model": _value(ini, "channel", "condition_model", "auto"),
        "tx_array_gain_dbi": float(_value(ini, "channel", "tx_array_gain_dbi", 12.0)),
        "rx_array_gain_dbi": float(_value(ini, "channel", "rx_array_gain_dbi", 12.0)),
    }
    init = {
        "type": "init", "contract": CONTRACT, "isolation": "fork_per_layout",
        "node_ids": ids, "node_types": [e.get("node_type", "drone") for e in nodes],
        "mobility": [e.get("mobility", "fixed") for e in nodes],
        "start_positions": [_start(e) for e in nodes], "controlled_indices": controlled,
        "rl_enabled": rl_enabled, "band": band, "band_source": band_source,
        "seed": int(flags["--seed"]), "run_id": int(_value(ini, "scenario", "run_id", 1)),
        "jammer_seed": int(flags["--seed"]),
        "sinr_threshold_db": SINR_MIN_DB,
        "jammer_path_enabled": band == "sub-6" and bool(_value(ini, "scenario",
                                                                "jammers_file")),
        "num_buildings": num_buildings, "channel": channel, "limits": dict(LIMITS),
        "time_s": 0.0,
    }
    patch = os.environ.get("FAKE_QUERY_INIT_PATCH")
    if patch:
        _merge(init, json.loads(patch))
    return init, nodes


def _sinr(distance: float, range_m: float) -> float:
    return SINR_MIN_DB + 20.0 * math.log10(range_m / max(distance, 1.0))


def _evaluate_layout(layout, nodes, init, probes, range_m) -> dict:
    started = time.monotonic()
    for index, (entry, point) in enumerate(zip(nodes, layout)):
        if abs(point[2] - init["start_positions"][index][2]) > 1e-6:
            return {"error": f"node '{entry['id']}': z differs from its start"}
        if entry.get("mobility") == "random_walk":
            b = entry.get("random_walk", {}).get("bounds", {})
            if not (b.get("x_min", -100) <= point[0] <= b.get("x_max", 100)
                    and b.get("y_min", -100) <= point[1] <= b.get("y_max", 100)):
                return {"error": f"node '{entry['id']}': outside its random_walk bounds"}
    bandwidth = init["channel"]["bandwidth_mhz"]
    links = []
    for i in range(len(layout)):
        for j in range(i + 1, len(layout)):
            distance = math.dist(layout[i], layout[j])
            sinr = _sinr(distance, range_m)
            capacity = bandwidth * math.log2(1.0 + 10.0 ** (sinr / 10.0))
            links.append([i, j, sinr, capacity, True, distance <= range_m])
    coverage = None
    if probes is not None:
        coverage = []
        for point in layout:
            coverage.append([k for k, (px, py) in enumerate(probes["points"])
                             if _sinr(math.dist(point, (px, py, probes["height_m"])),
                                      range_m) >= probes["sinr_db"]])
    return {"links": links, "coverage": coverage, "wall_s": time.monotonic() - started}


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def _check_request(request, n: int, limits: dict) -> str | None:
    layouts = request.get("layouts")
    if not isinstance(request.get("request_id"), int) or isinstance(request["request_id"],
                                                                    bool):
        return "request_id must be an integer"
    if not isinstance(layouts, list) or not layouts:
        return "layouts must be a non-empty list"
    if len(layouts) > limits["max_layouts"]:
        return f"{len(layouts)} layouts exceed max_layouts {limits['max_layouts']}"
    for layout in layouts:
        if not isinstance(layout, list) or len(layout) != n or not all(
                isinstance(p, list) and len(p) == 3 and all(map(_finite_number, p))
                for p in layout):
            return f"each layout must be {n} finite [x,y,z] points"
    probes = request.get("probes")
    if probes is not None:
        points = probes.get("points") if isinstance(probes, dict) else None
        if not isinstance(points, list) or not all(
                isinstance(p, list) and len(p) == 2 and all(map(_finite_number, p))
                for p in points):
            return "probes.points must be finite [x,y] points"
        if len(points) > limits["max_probes"]:
            return f"{len(points)} probes exceed max_probes {limits['max_probes']}"
        if not all(_finite_number(probes.get(k)) for k in ("height_m", "rx_gain_dbi",
                                                           "sinr_db")):
            return "probes height_m, rx_gain_dbi and sinr_db must be finite"
    return None


def _reject_constant(token: str):
    raise ValueError(f"non-finite token {token}")


def _record(entry: dict) -> None:
    path = os.environ.get("FAKE_QUERY_RECORD")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")


def _emit(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _spawn_grandchild(faults: set) -> int | None:
    if not faults & {"grandchild", "grandchild_ignore_term"}:
        return None
    mode = "ignore" if "grandchild_ignore_term" in faults else "term"
    child = subprocess.Popen([sys.executable, "-c", GRANDCHILD, mode],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    return child.pid


def _respond(request: dict, ordinal: int, faults: set, fault_at: int, nodes, init,
             range_m: float) -> None:
    request_id = request["request_id"]
    hit = ordinal == fault_at
    if hit and "hang" in faults:
        while True:
            time.sleep(0.1)
    if hit and "crash" in faults:
        print("fake_query: injected crash", file=sys.stderr, flush=True)
        os._exit(CRASH_EXIT_CODE)
    if hit and "malformed" in faults:
        _emit('{"type": "result", "request_id": ')
        return
    if hit and "request_error" in faults:
        _emit(json.dumps({"type": "error", "request_id": request_id,
                          "message": "injected request error"}))
        return
    started = time.monotonic()
    results = [_evaluate_layout(layout, nodes, init, request.get("probes"), range_m)
               for layout in request["layouts"]]
    if hit and "layout_error" in faults:
        results[0] = {"error": "injected layout error"}
    if hit and "nonfinite" in faults and "links" in results[0]:
        results[0]["links"][0][2] = float("nan")
    if hit and "short_links" in faults and "links" in results[0]:
        results[0]["links"].pop()
    if hit and "bad_coverage" in faults and results[0].get("coverage") is not None:
        results[0]["coverage"][0].append(len(request["probes"]["points"]))
    response = {"type": "result",
                "request_id": request_id + 1 if hit and "wrong_id" in faults else request_id,
                "layouts": results, "wall_s": time.monotonic() - started}
    text = json.dumps(response)
    if hit and "oversized" in faults:
        pad = int(os.environ.get("FAKE_QUERY_PAD_BYTES", 2 * MIB))
        text = text[:-1] + " " * pad + "}"
    _emit(text)


def main() -> int:
    flags = _parse_argv(sys.argv[1:])
    _record({"argv": sys.argv[1:]})
    faults = {f.strip() for f in os.environ.get("FAKE_QUERY_FAULT", "").split(",")
              if f.strip()}
    fault_at = int(os.environ.get("FAKE_QUERY_FAULT_AT", "1"))
    range_m = float(os.environ.get("FAKE_QUERY_RANGE_M", "100"))
    if "ignore_term" in faults:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    grandchild = _spawn_grandchild(faults)
    pid_file = os.environ.get("FAKE_QUERY_PID_FILE")
    if pid_file:
        Path(pid_file).write_text(json.dumps({"worker": os.getpid(),
                                              "grandchild": grandchild}))
    init, nodes = _init(flags)
    limits = init["limits"]
    _emit(json.dumps(init))
    ordinal = 0
    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline(limits["max_request_line_bytes"] + 2)
        if not line:
            return 0
        if len(line.rstrip(b"\n")) > limits["max_request_line_bytes"]:
            while line and not line.endswith(b"\n"):
                line = stdin.readline(MIB)
            _record({"oversized": True})
            _emit(json.dumps({"type": "error", "request_id": None,
                              "message": "request line exceeds max_request_line_bytes"}))
            continue
        try:
            request = json.loads(line, parse_constant=_reject_constant)
        except ValueError as exc:
            _emit(json.dumps({"type": "error", "request_id": None, "message": str(exc)}))
            continue
        kind = request.get("type") if isinstance(request, dict) else None
        if kind == "shutdown":
            _record({"type": "shutdown"})
            return 0
        if kind != "evaluate":
            _emit(json.dumps({"type": "error", "request_id": None,
                              "message": f"unknown request type {kind!r}"}))
            continue
        problem = _check_request(request, len(nodes), limits)
        _record({"type": "evaluate", "request_id": request.get("request_id"),
                 "layouts": len(request.get("layouts") or []), "bytes": len(line),
                 "probes": request.get("probes"), "problem": problem})
        if problem:
            request_id = request.get("request_id")
            _emit(json.dumps({"type": "error", "message": problem,
                              "request_id": request_id if isinstance(request_id, int)
                              else None}))
            continue
        ordinal += 1
        _respond(request, ordinal, faults, fault_at, nodes, init, range_m)


if __name__ == "__main__":
    sys.exit(main())
