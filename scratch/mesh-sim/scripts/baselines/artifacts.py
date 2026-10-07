"""baseline_manifest.json and baseline-plan.json schemas, status transitions, and I/O helpers."""

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from scripts.sim_support import find_mesh_root, simulator_env

MANIFEST_NAME = "baseline_manifest.json"
PLAN_NAME = "baseline-plan.json"
PLANNER_LOG_NAME = "planner.log"
SIM_LOG_NAME = "sim.log"
MANIFEST_VERSION = 2
PLAN_VERSION = 2
FINGERPRINT_VERSION = 2

STATUSES = ("preparing", "prepared", "running", "complete", "failed", "interrupted")
TERMINAL_STATUSES = ("complete", "failed", "interrupted")
_TRANSITIONS = {
    "preparing": {"prepared", "failed", "interrupted"},
    "prepared": {"running", "failed", "interrupted"},
    "running": {"complete", "failed", "interrupted"},
}
SEED_STATUSES = ("complete", "missing", "failed")
PACKAGE_DISTRIBUTIONS = ("numpy",)
IDENTITY_HASH_KEYS = ("run_ini_sha256", "nodes_json_sha256", "buildings_json_sha256",
                      "jammers_json_sha256")
# Adapted planner implementation, relative to scripts/baselines/; order is part of the hash.
PLANNER_CODE_FILES = ("solver.py", "config.py", "adapter.py", "effective_inputs.py",
                      "artifacts.py", "planners/objective.py", "planners/geometric.py",
                      "planners/optimization.py", "planners/channel.py")
CHANNEL_MODULES = ("core", "network", "mobility", "propagation", "buildings")
_NS3_LIBRARY = re.compile(r"^libns3(\.[0-9]+|-dev)?-(?P<module>[a-z0-9-]+?)"
                          r"(-(debug|default|release|optimized))?\.(dylib|so)(\.[0-9.]+)?$")
_MACHO_MAGICS = {bytes.fromhex(m) for m in ("feedface", "feedfacf", "cefaedfe", "cffaedfe",
                                            "cafebabe", "bebafeca")}
_ELF_MAGIC = b"\x7fELF"


class RuntimeIdentityError(RuntimeError):
    """The simulator binary's channel libraries cannot be resolved and hashed."""


def now_iso() -> str:
    """UTC timestamp in the same format as the RL manifests."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, payload) -> None:
    """Write JSON atomically so a reader never sees a half-written file."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    try:
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    os.replace(temp, path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_sha256(payload) -> str:
    """SHA-256 of sorted-key, whitespace-free JSON."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def package_versions() -> dict:
    """Installed versions of the planner dependencies; None when absent."""
    versions = {}
    for name in PACKAGE_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_relative(path: str | Path, root: str | Path) -> str:
    """POSIX path of `path` relative to the run directory `root`."""
    return Path(os.path.relpath(Path(path).resolve(), Path(root).resolve())).as_posix()


def new_manifest(mode: str, executor: str, method: str | None,
                 requested_algorithm: str | None) -> dict:
    """Initial manifest in status preparing; later fields are filled as preparation runs."""
    return {
        "baseline_manifest_version": MANIFEST_VERSION,
        "run_id": str(uuid.uuid4()),
        "mode": mode,
        "requested_algorithm": requested_algorithm,
        "method": method,
        "objective": None,
        "application": None,
        "executor": executor,
        "started_at": now_iso(),
        "ended_at": None,
        "status": "preparing",
        "error": None,
        "planner_seed": None,
        "planner_seed_status": None,
        "max_iterations": None,
        "max_iterations_status": None,
        "package_versions": package_versions(),
        "mapping_sha256": None,
        "geofence": None,
        "sim_binary_sha256": None,
        "channel_runtime_sha256": None,
        "channel_runtime_files": None,
        "planner_code_sha256": None,
        "channel_scoring": None,
        "penalties": None,
        "planner_settings": None,
        "ownership": None,
        "source_run_config_abs": None,
        "source_scenario_identity": None,
        "effective_scenario_identity": None,
        "simulation_seeds": None,
        "fingerprint": None,
        "waypoint_policy": None,
        "initial_displacement_m_total": None,
        "planner_wall_s": None,
        "plan": None,
        "planner_log": None,
        "sim_log": None,
        "source_inputs": None,
        "effective_inputs": None,
        "eval_manifest": None,
        "seeds": [],
    }


def update_manifest(path: str | Path, **fields) -> dict:
    """Merge top-level fields into an existing manifest; status changes go through set_status."""
    if "status" in fields:
        raise ValueError("use set_status to change the manifest status")
    manifest = read_json(path)
    manifest.update(fields)
    write_json(path, manifest)
    return manifest


def set_status(path: str | Path, status: str, error: str | None = None,
               final: bool | None = None, **fields) -> dict:
    """Apply one legal status transition; terminal statuses and `final=True` set ended_at."""
    manifest = read_json(path)
    current = manifest.get("status")
    if status not in _TRANSITIONS.get(current, set()):
        raise ValueError(f"illegal baseline manifest transition {current!r} -> {status!r}")
    manifest.update(fields)
    manifest["status"] = status
    if error is not None:
        manifest["error"] = error
    if (status in TERMINAL_STATUSES) if final is None else final:
        manifest["ended_at"] = now_iso()
    write_json(path, manifest)
    return manifest


def seed_record(seed: int, status: str, summary: str | None) -> dict:
    """One per-seed entry; `summary` is a run-relative path or None."""
    if status not in SEED_STATUSES:
        raise ValueError(f"seed status must be one of {SEED_STATUSES}, got {status!r}")
    return {"seed": int(seed), "status": status, "summary": summary}


def set_seed_records(path: str | Path, records: list[dict]) -> dict:
    return update_manifest(path, seeds=list(records))


def _aggregate(entries: list[tuple[str, str]]) -> str:
    listing = "".join(f"{digest}  {name}\n" for name, digest in entries)
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


def planner_code_identity(package_dir: str | Path | None = None) -> dict:
    """Per-file and aggregate SHA-256 of the adapted planner modules."""
    package_dir = Path(package_dir) if package_dir else Path(__file__).resolve().parent
    missing = [name for name in PLANNER_CODE_FILES if not (package_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"adapted planner modules missing under {package_dir}: "
                                f"{', '.join(missing)}")
    files = {name: sha256_file(package_dir / name) for name in PLANNER_CODE_FILES}
    return {"files": files, "aggregate_sha256": _aggregate(list(files.items()))}


def _object_format(path: Path) -> str | None:
    with open(path, "rb") as handle:
        head = handle.read(4)
    if head == _ELF_MAGIC:
        return "elf"
    return "macho" if head in _MACHO_MAGICS else None


def _run_tool(command: list[str], env: dict | None = None) -> str:
    if shutil.which(command[0]) is None:
        raise RuntimeIdentityError(f"'{command[0]}' is needed to resolve the simulator's "
                                   "shared libraries but is not on PATH")
    result = subprocess.run(command, capture_output=True, text=True, env=env,
                            stdin=subprocess.DEVNULL, timeout=60)
    if result.returncode != 0:
        raise RuntimeIdentityError(f"{' '.join(command)} failed: {result.stderr.strip()}")
    return result.stdout


def _macho_rpaths(image: Path) -> list[str]:
    rpaths, in_rpath = [], False
    for line in _run_tool(["otool", "-l", str(image)]).splitlines():
        text = line.strip()
        if text.startswith("cmd "):
            in_rpath = text == "cmd LC_RPATH"
        elif in_rpath and text.startswith("path "):
            rpaths.append(text[5:].rsplit(" (offset", 1)[0])
    return rpaths


def _macho_libraries(binary: Path, env: dict) -> list[Path]:
    """Transitive ns-3 dylibs of a Mach-O binary, searched like dyld with simulator_env."""
    env_dirs = [d for d in env.get("DYLD_LIBRARY_PATH", "").split(os.pathsep) if d]
    binary_rpaths = _macho_rpaths(binary)
    found: dict[str, Path] = {}
    pending, seen = [binary], set()
    while pending:
        image = pending.pop()
        if image in seen:
            continue
        seen.add(image)
        rpaths = _macho_rpaths(image) + binary_rpaths if image != binary else binary_rpaths
        for line in _run_tool(["otool", "-L", str(image)]).splitlines()[1:]:
            name = line.strip().split(" (compatibility", 1)[0]
            leaf = name.rsplit("/", 1)[-1]
            if not leaf.startswith("libns3"):
                continue
            candidates = [Path(d) / leaf for d in env_dirs]
            for rpath in rpaths:
                rpath = rpath.replace("@loader_path", str(image.parent)).replace(
                    "@executable_path", str(binary.parent))
                candidates.append(Path(rpath) / leaf)
            if not name.startswith("@"):
                candidates.append(Path(name))
            resolved = next((c.resolve() for c in candidates if c.is_file()), None)
            if resolved is None:
                raise RuntimeIdentityError(f"cannot resolve {name} needed by {image}")
            found.setdefault(leaf, resolved)
            pending.append(resolved)
    return list(found.values())


def _elf_libraries(binary: Path, env: dict) -> list[Path]:
    found = []
    for line in _run_tool(["ldd", str(binary)], env).splitlines():
        name, arrow, rest = line.strip().partition(" => ")
        if not arrow or not name.startswith("libns3"):
            continue
        target = rest.split(" (", 1)[0].strip()
        if target == "not found" or not target:
            raise RuntimeIdentityError(f"ldd cannot resolve {name} for {binary}")
        found.append(Path(target).resolve())
    return found


def linked_libraries(binary: str | Path) -> list[Path]:
    """Dynamically linked ns-3 libraries of the simulator binary; empty when none."""
    binary = Path(binary).resolve()
    kind = _object_format(binary)
    if kind is None:
        return []
    env = simulator_env(find_mesh_root())
    return _macho_libraries(binary, env) if kind == "macho" else _elf_libraries(binary, env)


def channel_runtime_identity(binary: str | Path) -> dict:
    """Aggregate SHA-256 of the binary and the ns-3 channel-path libraries it loads."""
    binary = Path(binary).resolve()
    libraries = linked_libraries(binary)
    channel = {}
    for path in libraries:
        match = _NS3_LIBRARY.search(path.name)
        if match and match.group("module") in CHANNEL_MODULES:
            channel[match.group("module")] = path
    if libraries and set(channel) != set(CHANNEL_MODULES):
        missing = sorted(set(CHANNEL_MODULES) - set(channel))
        raise RuntimeIdentityError(f"{binary} links ns-3 dynamically but its "
                                   f"{', '.join(missing)} libraries were not found; the "
                                   "channel identity would omit them")
    entries = [("binary", sha256_file(binary))]
    entries += [(f"ns3-{module}", sha256_file(channel[module]))
                for module in CHANNEL_MODULES if module in channel]
    return {"sha256": _aggregate(entries),
            "files": [str(binary)] + [str(channel[m]) for m in CHANNEL_MODULES if m in channel]}


def fingerprint(manifest: dict) -> str:
    """Hash of everything that defines one reusable plan; no timing or paths."""
    identity = manifest.get("effective_scenario_identity") or {}
    scoring = manifest.get("channel_scoring") or {}
    probe = scoring.get("probe") or {}
    grid = scoring.get("candidate_grid") or {}
    penalties = manifest.get("penalties") or {}
    return canonical_sha256({
        "fingerprint_version": FINGERPRINT_VERSION,
        "method": manifest.get("method"),
        "objective": manifest.get("objective"),
        "planner_seed": manifest.get("planner_seed"),
        "max_iterations": manifest.get("max_iterations"),
        "waypoint_policy": manifest.get("waypoint_policy"),
        "mapping_sha256": manifest.get("mapping_sha256"),
        "penalties": {platform: penalties.get(platform) for platform in ("aerial", "ground")},
        "grid": {key: grid.get(key) for key in ("cells", "min_resolution_m", "cell_m",
                                                "count")},
        "probe": {key: probe.get(key) for key in ("height_m", "rx_gain_dbi", "grid_cells",
                                                  "min_resolution_m", "cell_m", "count")},
        "sinr_threshold_db": scoring.get("sinr_threshold_db"),
        "coverage_sinr_db": scoring.get("coverage_sinr_db"),
        "planning_seed": scoring.get("planning_seed"),
        "planning_run_id": scoring.get("planning_run_id"),
        "jammer_seed": scoring.get("jammer_seed"),
        "mode_flag": scoring.get("mode_flag"),
        "band": scoring.get("band"),
        **{key: identity.get(key) for key in IDENTITY_HASH_KEYS},
        "sim_binary_sha256": manifest.get("sim_binary_sha256"),
        "channel_runtime_sha256": manifest.get("channel_runtime_sha256"),
        "planner_code_sha256": manifest.get("planner_code_sha256"),
        "planner_settings": manifest.get("planner_settings"),
        "ownership": manifest.get("ownership"),
    })


def build_plan(method: str, objective: str | None, nodes: list[dict],
               predictions) -> dict:
    """baseline-plan.json payload; node entries come from the adapter."""
    total = sum(float(node["displacement_m"]) for node in nodes)
    return {
        "baseline_plan_version": PLAN_VERSION,
        "method": method,
        "objective": objective,
        "nodes": nodes,
        "initial_displacement_m_total": total,
        "planner_predictions": predictions,
    }


def write_plan(path: str | Path, plan: dict) -> None:
    """Write the plan once; an existing plan is never replaced."""
    if Path(path).exists():
        raise FileExistsError(f"baseline plan already exists: {path}")
    write_json(path, plan)


def eval_metadata(manifest: dict, manifest_rel: str, plan_rel: str) -> dict:
    """The per-policy `baseline` block stored in eval_manifest.json."""
    identity = manifest.get("effective_scenario_identity") or {}
    return {
        "baseline_manifest_version": manifest["baseline_manifest_version"],
        "method": manifest["method"],
        "requested_algorithm": manifest["requested_algorithm"],
        "objective": manifest["objective"],
        "executor": manifest["executor"],
        "planner_seed": manifest["planner_seed"],
        "max_iterations": manifest["max_iterations"],
        "planning_seed": (manifest.get("channel_scoring") or {}).get("planning_seed"),
        "ownership": manifest.get("ownership"),
        "fingerprint": manifest["fingerprint"],
        "initial_displacement_m_total": manifest["initial_displacement_m_total"],
        "effective_scenario_identity": {key: identity.get(key) for key in IDENTITY_HASH_KEYS},
        "mapping_sha256": manifest["mapping_sha256"],
        "manifest": manifest_rel,
        "plan": plan_rel,
    }
