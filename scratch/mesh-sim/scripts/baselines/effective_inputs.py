"""Source snapshots, line-preserving INI edits, nodes.json rewrite, and run-relative identity."""

import configparser
import copy
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from scripts.baselines.artifacts import PLAN_NAME, run_relative, sha256_file, write_plan
from scripts.baselines.config import BaselineConfig, ConfigError, read_ini, scenario_file

SOURCE_DIR = "source-inputs"
EFFECTIVE_DIR = "effective-inputs"
RUN_INI = "run.ini"
NODES_JSON = "nodes.json"
DEFAULT_NODES_FILE = "nodes.json"
_ASSET_KEYS = (("scenario", "buildings_file"), ("scenario", "jammers_file"),
               ("baseline", "mapping_file"))
STAGING_DIR = EFFECTIVE_DIR + ".partial"


class InputError(ValueError):
    """A referenced input is missing, ambiguous, or cannot be rewritten."""


@dataclass(frozen=True)
class ScenarioFiles:
    """The requested INI and every file it references, as absolute existing paths."""

    run_config: Path
    nodes: Path
    buildings: Path | None
    jammers: Path | None
    mapping: Path | None

    def assets(self) -> dict:
        """Referenced files other than nodes, keyed by (section, key)."""
        values = (self.buildings, self.jammers, self.mapping)
        return {key: path for key, path in zip(_ASSET_KEYS, values) if path is not None}


def _unique_names(paths: dict, reserved: tuple = ()) -> None:
    seen = {name: "reserved output name" for name in reserved}
    for label, path in paths.items():
        name = path.name
        if name in seen:
            raise InputError(f"{label} ({path}) has the same basename as {seen[name]}; "
                             "rename one so the copied inputs stay unambiguous")
        seen[name] = f"{label} ({path})"


def scenario_files(ini: configparser.ConfigParser, cfg: BaselineConfig) -> ScenarioFiles:
    """Resolve and check every referenced file; missing files and name clashes fail."""
    run_config = cfg.run_config
    found = {
        "nodes": scenario_file(ini, run_config, "nodes_file", DEFAULT_NODES_FILE),
        "buildings": scenario_file(ini, run_config, "buildings_file"),
        "jammers": scenario_file(ini, run_config, "jammers_file"),
        "mapping": cfg.mapping_file,
    }
    for label, path in found.items():
        if path is not None and not path.is_file():
            raise InputError(f"{label} file not found: {path} (referenced by {run_config})")
    files = ScenarioFiles(run_config=run_config,
                          **{k: (v.resolve() if v else None) for k, v in found.items()})
    labels = {"run config": files.run_config, "nodes file": files.nodes,
              **{f"{s}.{k}": p for (s, k), p in files.assets().items()}}
    _unique_names(labels)
    _unique_names({f"{s}.{k}": p for (s, k), p in files.assets().items()},
                  reserved=(RUN_INI, NODES_JSON, PLAN_NAME))
    return files


def scenario_identity(run_root: str | Path, run_ini: Path, nodes: Path,
                      buildings: Path | None, jammers: Path | None) -> dict:
    """Run-relative run.ini path plus SHA-256 of the run.ini and its scenario files."""
    return {
        "run_config": run_relative(run_ini, run_root),
        "run_ini_sha256": sha256_file(run_ini),
        "nodes_json_sha256": sha256_file(nodes),
        "buildings_json_sha256": sha256_file(buildings) if buildings else None,
        "jammers_json_sha256": sha256_file(jammers) if jammers else None,
    }


def snapshot_sources(files: ScenarioFiles, run_root: str | Path) -> dict:
    """Byte-copy the requested inputs into source-inputs/ and return their identity."""
    dest = Path(run_root) / SOURCE_DIR
    dest.mkdir()
    copies = {}
    for label, path in (("run", files.run_config), ("nodes", files.nodes),
                        ("buildings", files.buildings), ("jammers", files.jammers),
                        ("mapping", files.mapping)):
        if path is not None:
            copies[label] = dest / path.name
            shutil.copyfile(path, copies[label])
    return scenario_identity(run_root, copies["run"], copies["nodes"],
                             copies.get("buildings"), copies.get("jammers"))


def ini_edits(files: ScenarioFiles, mode: str) -> dict:
    """(section, key) -> value edits for the effective run.ini."""
    edits = {("baseline", "algorithm"): "none", ("scenario", "nodes_file"): NODES_JSON}
    if mode == "standalone":
        edits[("rl", "enabled")] = "false"
    for key, path in files.assets().items():
        edits[key] = path.name
    return edits


def _split_comment(body: str) -> tuple[str, str]:
    cut = min((i for i in (body.find("#"), body.find(";")) if i >= 0), default=len(body))
    return body[:cut], body[cut:]


def _header(code: str) -> str | None:
    text = code.strip()
    if len(text) >= 2 and text[0] == "[" and text[-1] == "]":
        return text[1:-1].strip()
    return None


def _replace_value(line: str, value: str) -> str:
    body = line.rstrip("\r\n")
    eol = line[len(body):]
    code, comment = _split_comment(body)
    eq = code.index("=")
    after = code[eq + 1:]
    lead = after[:len(after) - len(after.lstrip())] or " "
    trail = after[len(after.rstrip()):] if comment else ""
    return f"{code[:eq + 1]}{lead}{value}{trail}{comment}{eol}"


def edit_ini_text(text: str, edits: dict) -> str:
    """Change only the edited keys; append missing keys to their section or a new one."""
    lines = text.splitlines(keepends=True)
    newline = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += newline
    pending = dict(edits)
    last_line: dict = {}
    section = None
    for index, line in enumerate(lines):
        code, _ = _split_comment(line.rstrip("\r\n"))
        name = _header(code)
        if name is not None:
            section = name
            last_line[section] = index
            continue
        if not code.strip():
            continue
        if section is not None:
            last_line[section] = index
        if "=" in code and section is not None:
            key = code.split("=", 1)[0].strip()
            if (section, key) in pending:
                lines[index] = _replace_value(line, pending.pop((section, key)))
    inserts: dict = {}
    appended: dict = {}
    for (section_name, key), value in pending.items():
        entry = f"{key} = {value}{newline}"
        if section_name in last_line:
            inserts.setdefault(last_line[section_name], []).append(entry)
        else:
            appended.setdefault(section_name, []).append(entry)
    out = []
    for index, line in enumerate(lines):
        out.append(line)
        out.extend(inserts.get(index, []))
    for section_name, entries in appended.items():
        if out and out[-1].strip():
            out.append(newline)
        out.append(f"[{section_name}]{newline}")
        out.extend(entries)
    return "".join(out)


def start_position(entry: dict) -> tuple[float, float, float]:
    """Waypoint nodes start at waypoints[0], others at position; absent fields are 0."""
    if entry.get("mobility", "fixed") == "waypoint":
        waypoints = entry.get("waypoints") or []
        if not waypoints:
            raise InputError(f"node '{entry.get('id')}' has waypoint mobility but no "
                             "waypoints")
        point = waypoints[0]
    else:
        point = entry.get("position") or {}
    return tuple(float(point.get(axis, 0.0)) for axis in ("x", "y", "z"))


def rewrite_nodes(nodes: list, moved: dict, waypoint_policy: str) -> list:
    """Copy of nodes.json with each moved node's start x/y set; z and other nodes untouched."""
    out = copy.deepcopy(nodes)
    for entry in out:
        node_id = entry.get("id")
        if node_id not in moved:
            continue
        x, y = moved[node_id]
        if not (math.isfinite(x) and math.isfinite(y)):
            raise InputError(f"node '{node_id}' has a non-finite planned position")
        if entry.get("mobility", "fixed") == "waypoint":
            if waypoint_policy != "translate":
                raise InputError(f"node '{node_id}' uses waypoint mobility and was moved; "
                                 "set [baseline] waypoint_policy = translate to shift its "
                                 "whole path, or leave it out of movable_nodes")
            sx, sy, _ = start_position(entry)
            dx, dy = x - sx, y - sy
            for waypoint in entry["waypoints"]:
                waypoint["x"] = float(waypoint.get("x", 0.0)) + dx
                waypoint["y"] = float(waypoint.get("y", 0.0)) + dy
            if isinstance(entry.get("position"), dict):
                entry["position"]["x"], entry["position"]["y"] = x, y
        else:
            position = entry.setdefault("position", {})
            position.setdefault("z", 0.0)
            position["x"], position["y"] = x, y
    return out


def stage_effective_inputs(files: ScenarioFiles, run_root: str | Path, mode: str) -> Path:
    """Write effective-inputs.partial/ with the edited INI, assets and source nodes.json."""
    run_root = Path(run_root)
    final = run_root / EFFECTIVE_DIR
    staging = run_root / STAGING_DIR
    if final.exists() or staging.exists():
        raise InputError(f"{final} already exists; effective inputs are written once")
    staging.mkdir()
    try:
        text = files.run_config.read_bytes().decode("utf-8")
        (staging / RUN_INI).write_bytes(
            edit_ini_text(text, ini_edits(files, mode)).encode("utf-8"))
        try:
            read_ini(staging / RUN_INI)
        except ConfigError as exc:
            raise InputError(f"effective run.ini is not valid: {exc}") from exc
        shutil.copyfile(files.nodes, staging / NODES_JSON)
        for path in files.assets().values():
            shutil.copyfile(path, staging / path.name)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging / RUN_INI


def discard_staged_inputs(run_root: str | Path) -> None:
    """Remove an unfinished effective-inputs.partial/, if any."""
    shutil.rmtree(Path(run_root) / STAGING_DIR, ignore_errors=True)


def finish_effective_inputs(files: ScenarioFiles, run_root: str | Path, nodes: list | None,
                            plan: dict) -> dict:
    """Rewrite nodes (None keeps the source bytes), add the plan, rename once, hash."""
    run_root = Path(run_root)
    final = run_root / EFFECTIVE_DIR
    staging = run_root / STAGING_DIR
    if not staging.is_dir():
        raise InputError(f"{staging} was not staged")
    try:
        if final.exists():
            raise InputError(f"{final} already exists; effective inputs are written once")
        if nodes is not None:
            (staging / NODES_JSON).write_text(json.dumps(nodes, indent=2) + "\n",
                                              encoding="utf-8")
        write_plan(staging / PLAN_NAME, plan)
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return scenario_identity(
        run_root, final / RUN_INI, final / NODES_JSON,
        final / files.buildings.name if files.buildings else None,
        final / files.jammers.name if files.jammers else None)
