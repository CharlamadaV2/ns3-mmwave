#!/usr/bin/env python3
"""Stand-in for the mesh-sim binary: same flags and RL protocol, no radio model."""

import configparser
import json
import math
import os
import sys
from pathlib import Path

VALUE_FLAGS = ("--run-config", "--seed", "--output-dir", "--band")
BOOL_FLAGS = ("--rl-mode",)
STEP_M = 5.0
STDERR_MARKER = "fake-sim: simulated fatal error"

CONTRACT = "mesh_move_2d_v1"
ACTION_MEANINGS = ["west", "east", "south", "north", "hold"]
HOLD = 4
SPEED_CAP_MPS = 20.0
EPS = 1e-6
BOUND_DEFAULTS = {"x": (-1000.0, 2000.0), "y": (-1000.0, 1000.0), "z": (0.0, 100.0)}
LONG_MODE_TICKS = 3000       # enough valid post-EOF lines to fill a pipe buffer

FACTS_SCHEMA = "mesh_facts_v1"
FACTS_COLUMNS = {"nodes": ["x", "y", "z", "vx", "vy", "vz", "slot"],
                 "links": ["sinr_db", "capacity_mbps", "is_los"]}
SINR_MIN_DB = -6.7           # connectivity/LOS threshold for the synthetic links
DEMAND_MBPS_DEFAULT = 10.0
DELIVERED_FRACTION = 0.5
LEGACY_REWARD_TICK = 1.0


def parse_args(argv: list[str]) -> dict:
    args = {"rl-mode": False}
    i = 0
    while i < len(argv):
        token = argv[i]
        name, sep, value = token.partition("=")
        if name in BOOL_FLAGS and not sep:
            args["rl-mode"] = True
        elif name in VALUE_FLAGS:
            if not sep:
                i += 1
                if i >= len(argv):
                    sys.exit(f"fake-sim: missing value for {name}")
                value = argv[i]
            args[name.lstrip("-")] = value
        else:
            sys.exit(f"fake-sim: unknown flag {token}")
        i += 1
    return args


def write_outputs(out_dir: Path, seed: int, band: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.log").write_text(
        "fake-sim run log\n"
        f"seeds:               [{seed}]\n"
        f"--band:              {band if band is not None else 'default'}\n"
    )
    seed_dir = out_dir / f"seed-{seed}"
    seed_dir.mkdir(exist_ok=True)
    (seed_dir / "summary.json").write_text(
        json.dumps({"seed": seed, "fake": True}, indent=2) + "\n"
    )
    (seed_dir / "links.csv").write_text("tick,src,dst,sinr_db,capacity_mbps\n")


def message(tick: int, tick_s: float, pos: list[float], done: bool) -> str:
    return json.dumps({
        "type": "step",
        "tick": tick,
        "time_s": tick * tick_s,
        "obs": {
            "controlled_pos": pos,
            "link_sinrs": [20.0 - pos[0] * 0.01, 18.0],
            "link_capacities": [100.0, 90.0],
        },
        "reward": 1.0,
        "done": done,
        "action_type": "discrete",
    })


def emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# --------------------------------------------------------------------------
# Centralized mode
# --------------------------------------------------------------------------


def _bounds(ini: configparser.ConfigParser) -> dict:
    ranges = {}
    for axis, (low, high) in BOUND_DEFAULTS.items():
        ranges[axis] = (
            ini.getfloat("rl", f"{axis}_min", fallback=low),
            ini.getfloat("rl", f"{axis}_max", fallback=high),
        )
    return ranges


def _load_nodes(run_config: str, ini: configparser.ConfigParser) -> list[dict]:
    nodes_file = ini.get("scenario", "nodes_file", fallback="") or "nodes.json"
    path = Path(nodes_file)
    if not path.is_absolute():
        path = Path(run_config).resolve().parent / path
    nodes = []
    for entry in json.loads(path.read_text()):
        position = entry.get("position")
        if position is None and entry.get("waypoints"):
            position = entry["waypoints"][0]
        position = position or {}
        nodes.append({
            "id": entry["id"],
            "pos": [float(position.get("x", 0.0)), float(position.get("y", 0.0)),
                    float(position.get("z", 0.0))],
        })
    return nodes


def _resolve_slots(ini: configparser.ConfigParser, nodes: list[dict]) -> list[int]:
    raw = ini.get("rl", "controlled_nodes", fallback="")
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if tokens == ["all"]:
        return list(range(len(nodes)))
    ids = [node["id"] for node in nodes]
    return [ids.index(token) for token in tokens]


def _mask(nodes: list[dict], slots: list[int], num_slots: int, bounds: dict) -> list[int]:
    flat = []
    for slot in range(num_slots):
        if slot >= len(slots):
            flat.extend([0, 0, 0, 0, 1])
            continue
        x, y, _ = nodes[slots[slot]]["pos"]
        flat.extend([
            int(x - bounds["x"][0] > EPS),
            int(bounds["x"][1] - x > EPS),
            int(y - bounds["y"][0] > EPS),
            int(bounds["y"][1] - y > EPS),
            1,
        ])
    return flat


def _link_values(nodes: list[dict], i: int, j: int) -> tuple[float, float]:
    """Synthetic (sinr_db, capacity_mbps) shared by obs and facts, so both agree."""
    distance = math.dist(nodes[i]["pos"], nodes[j]["pos"])
    return 20.0 - 0.01 * distance, max(0.0, 100.0 - 0.1 * distance)


def _observation(nodes: list[dict], slots: list[int], num_slots: int) -> list[float]:
    flat = []
    for slot in range(num_slots):
        if slot >= len(slots):
            flat.extend([0.0] * (4 + 2 * (len(nodes) - 1)))
            continue
        index = slots[slot]
        x, y, z = nodes[index]["pos"]
        flat.extend([1.0, x, y, z])
        for peer in range(len(nodes)):
            if peer == index:
                continue
            flat.extend(_link_values(nodes, index, peer))
    return flat


def _facts(nodes: list[dict], slots: list[int], velocities: dict, ticks: int,
           demand_mbps: float, reward_tick: float) -> dict:
    """Build facts consistent with this tick's observation and flow demand."""
    slot_of = {index: slot for slot, index in enumerate(slots)}
    fact_nodes = []
    for index, node in enumerate(nodes):
        vx, vy = velocities.get(index, (0.0, 0.0))
        fact_nodes.append([*node["pos"], vx, vy, 0.0, slot_of.get(index, -1)])

    fact_links, connected = [], 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            sinr, capacity = _link_values(nodes, i, j)
            is_los = int(sinr >= SINR_MIN_DB)
            connected += is_los
            fact_links.append([sinr, capacity, is_los])

    flows = len(fact_links)
    flow_ticks = flows * ticks if demand_mbps > 0.0 else 0
    demand_sum = demand_mbps * flow_ticks
    return {
        "nodes": fact_nodes,
        "links": fact_links,
        "window": {
            "ticks": ticks,
            "demand_mbps_sum": demand_sum,
            "delivered_mbps_sum": demand_sum * DELIVERED_FRACTION,
            "flow_ticks_with_demand": flow_ticks,
            "unroutable_flow_ticks": 0,
            "connected_pairs_sum": connected * ticks,
            "los_pairs_sum": connected * ticks,
            "legacy_reward_sum": reward_tick * ticks,
        },
    }


def _apply(nodes: list[dict], slots: list[int], num_slots: int, action: list[int],
           mask: list[int]) -> tuple[list[tuple[float, float]], list[int]]:
    """Return per-slot velocity and the slots whose action was replaced by hold."""
    velocities = []
    revalidated = []
    for slot in range(num_slots):
        choice = action[slot] if slot < len(action) else HOLD
        if mask[slot * 5 + choice] != 1:
            revalidated.append(slot)
            choice = HOLD
        if slot >= len(slots):
            continue
        velocities.append({
            0: (-1.0, 0.0), 1: (1.0, 0.0), 2: (0.0, -1.0), 3: (0.0, 1.0),
            HOLD: (0.0, 0.0),
        }[choice])
    return velocities, revalidated


def _advance(nodes: list[dict], slots: list[int], velocities, speed: float,
             tick_s: float, bounds: dict) -> None:
    for slot, index in enumerate(slots):
        vx, vy = velocities[slot]
        pos = nodes[index]["pos"]
        pos[0] = min(max(pos[0] + vx * speed * tick_s, bounds["x"][0]), bounds["x"][1])
        pos[1] = min(max(pos[1] + vy * speed * tick_s, bounds["y"][0]), bounds["y"][1])


def _init_message(mode: str, num_slots: int, count: int, nodes: list[dict],
                  slots: list[int], speed: float, tick_s: float, interval_s: float,
                  k: int, num_ticks: int, ini: configparser.ConfigParser,
                  bounds: dict, band: str | None) -> dict:
    ids = [nodes[i]["id"] for i in slots] + [None] * (num_slots - count)
    speeds = [speed] * count + [None] * (num_slots - count)
    init = {
        "type": "init",
        "contract": CONTRACT,
        "dimensions": 2,
        "action_meanings": list(ACTION_MEANINGS),
        "max_controlled_nodes": num_slots,
        "num_controlled": count,
        "slot_node_ids": ids,
        "slot_speed_mps": speeds,
        "num_mesh_nodes": len(nodes),
        "obs_dim": num_slots * (4 + 2 * (len(nodes) - 1)),
        "mask_dim": 5 * num_slots,
        "tick_s": tick_s,
        "decision_interval_s": interval_s,
        "decision_interval_ticks": k,
        "num_ticks": num_ticks,
        "num_decisions": -(-num_ticks // k),
        "reward_type": ini.get("rl", "reward_type", fallback="all_links_los"),
        "reward_window": "mean",
        "wall_policy": "clip",
    }
    if mode != "no_facts":
        init.update({
            "facts_schema": FACTS_SCHEMA,
            "facts_columns": FACTS_COLUMNS,
            "node_ids": [node["id"] for node in nodes],
            "num_links": len(nodes) * (len(nodes) - 1) // 2,
            "bounds": {f"{axis}_{end}": bounds[axis][i]
                       for axis in ("x", "y", "z")
                       for i, end in enumerate(("min", "max"))},
            "band": band if band is not None else "mmwave",
            "jammer_path_enabled": False,
            "warmup_s": 0.0,
        })
    if mode == "bad_contract":
        init["contract"] = "mesh_move_9d_v9"
    elif mode == "bad_meanings":
        init["action_meanings"] = ["north", "south", "east", "west", "hold"]
    elif mode == "bad_counts":
        init["num_controlled"] = num_slots + 1
    return init


def _fail_seeds() -> set[int]:
    raw = os.environ.get("FAKE_SIM_FAIL_SEEDS", "")
    return {int(token) for token in raw.split(",") if token.strip()}


def run_centralized(args: dict, ini: configparser.ConfigParser, mode: str,
                    seed: int) -> int:
    nodes = _load_nodes(args["run-config"], ini)
    slots = _resolve_slots(ini, nodes)
    count = len(slots)
    num_slots = max(ini.getint("rl", "max_controlled_nodes", fallback=0), count)
    bounds = _bounds(ini)
    tick_s = ini.getfloat("scenario", "tick_s", fallback=0.1)
    duration_s = ini.getfloat("scenario", "duration_s", fallback=0.4)
    speed = min(ini.getfloat("rl", "step_size_m", fallback=5.0) / tick_s, SPEED_CAP_MPS)
    interval_s = ini.getfloat("rl", "decision_interval_s", fallback=0.0) or tick_s
    k = max(1, int(round(interval_s / tick_s)))
    num_ticks = max(1, int(round(duration_s / tick_s)))
    demand_mbps = ini.getfloat("traffic", "demand_mbps", fallback=DEMAND_MBPS_DEFAULT)
    reward_tick = -1.0 if mode == "negative_reward" else LEGACY_REWARD_TICK
    if mode == "long":
        num_ticks, k, interval_s = LONG_MODE_TICKS, 1, tick_s

    emit(json.dumps(_init_message(mode, num_slots, count, nodes, slots, speed,
                                  tick_s, interval_s, k, num_ticks, ini, bounds,
                                  args.get("band"))))

    tick, decision, revalidated, last_action = 0, 0, [], None
    window_len = 1
    node_velocities: dict = {}
    eof = False
    while True:
        if mode == "malformed" and decision == 1:
            emit("this is not json {")
            return 0
        mask = _mask(nodes, slots, num_slots, bounds)
        obs = _observation(nodes, slots, num_slots)
        if mode == "bad_obs_len" and decision == 0:
            obs = obs[:-1]
        if mode == "bad_mask" and decision == 0:
            mask = list(mask)
            mask[0] = 2
        if mode == "non_finite" and decision == 0:
            obs = list(obs)
            obs[1] = float("nan")
        step = {
            "type": "step",
            "tick": tick,
            "time_s": tick * tick_s,
            "decision": decision,
            "ticks_in_step": window_len,
            "obs": obs,
            "mask": mask,
            "reward": reward_tick,
            "done": tick >= num_ticks,
            "revalidated_slots": revalidated,
            "last_action": last_action,
        }
        if mode != "no_facts":
            step["facts"] = _facts(nodes, slots, node_velocities, window_len,
                                   demand_mbps, reward_tick)
        emit(json.dumps(step))
        if decision == 1 and seed in _fail_seeds():
            print(STDERR_MARKER, file=sys.stderr, flush=True)
            return 3
        if tick >= num_ticks:
            return 0
        if mode == "exit3" and decision == 0:
            print(STDERR_MARKER, file=sys.stderr, flush=True)
            return 3
        action = [HOLD] * num_slots
        if not eof:
            line = sys.stdin.readline()
            if not line:
                eof = True
            else:
                action = list(json.loads(line)["action"])
        last_action = list(action)
        velocities, revalidated = _apply(nodes, slots, num_slots, action, mask)
        node_velocities = {index: (velocities[slot][0] * speed,
                                   velocities[slot][1] * speed)
                           for slot, index in enumerate(slots)}

        window_len = min(k, num_ticks - tick)
        for _ in range(window_len):
            _advance(nodes, slots, velocities, speed, tick_s, bounds)
        tick += window_len
        decision += 1


def main() -> int:
    args = parse_args(sys.argv[1:])
    mode = os.environ.get("FAKE_SIM_MODE", "normal")

    ini = configparser.ConfigParser()
    ini.read(args["run-config"])
    duration_s = ini.getfloat("scenario", "duration_s", fallback=0.4)
    tick_s = ini.getfloat("scenario", "tick_s", fallback=0.1)
    seed = int(args["seed"]) if "seed" in args else ini.getint("scenario", "seed", fallback=1)

    write_outputs(Path(args["output-dir"]), seed, args.get("band"))

    if ini.has_option("rl", "controlled_nodes"):
        return run_centralized(args, ini, mode, seed)

    n_messages = int(round(duration_s / tick_s)) + 1
    pos = [0.0, 0.0, 0.0]

    emit(message(0, tick_s, list(pos), n_messages == 1))
    if mode == "exit3":
        print(STDERR_MARKER, file=sys.stderr, flush=True)
        return 3

    for tick in range(1, n_messages):
        line = sys.stdin.readline()
        if not line:
            return 0
        action = int(json.loads(line)["action"])
        if action == 0:
            pos[0] -= STEP_M
        elif action == 1:
            pos[0] += STEP_M
        elif action == 2:
            pos[1] -= STEP_M
        elif action == 3:
            pos[1] += STEP_M
        elif action == 4:
            pos[2] = max(0.0, pos[2] - STEP_M)
        elif action == 5:
            pos[2] += STEP_M

        if mode == "malformed" and tick == 1:
            emit("this is not json {")
            return 0
        emit(message(tick, tick_s, list(pos), tick == n_messages - 1))

    return 0


if __name__ == "__main__":
    sys.exit(main())
