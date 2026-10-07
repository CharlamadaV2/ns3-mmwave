"""SLURM adapter: cluster-config validation, argv builders, job scripts, and parsers."""

import json
import re
import secrets
import shlex
import shutil
import subprocess
from pathlib import Path

CLUSTER_CONFIG_VERSION = 1
JOB_NAME_PREFIX = "meshops"
JOB_ROLES = ("tasks", "compare")

_TOP_KEYS = ("cluster_config_version", "name", "venv", "setup_lines", "task",
             "compare", "max_array_size", "max_concurrent_tasks")
_TASK_KEYS = ("partition", "account", "qos", "constraint", "time", "mem",
              "cpus_per_task")
_COMPARE_KEYS = ("time", "mem", "cpus_per_task")
_INHERITED = ("partition", "account", "qos", "constraint")
_TIME_RE = re.compile(r"^(\d+-)?\d{1,2}:[0-5]\d:[0-5]\d$")
_MEM_RE = re.compile(r"^\d+[KMGT]?$")
_ELEMENT_RE = re.compile(r"^(\d+)_\[(.+)\]$")
_JOB_ID_RE = re.compile(r"^(\d+)(;.*)?$")
_CANCELLED_RE = re.compile(r"^CANCELLED\b.*")


def _check_keys(mapping, allowed: tuple[str, ...], where: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{where} must be a JSON object")
    unknown = sorted(key for key in mapping if key not in allowed)
    if unknown:
        raise ValueError(f"{where} has unknown keys {unknown}; valid keys: "
                         f"{list(allowed)}. There is no free-form sbatch option")
    missing = [key for key in allowed if key not in mapping]
    if missing:
        raise ValueError(f"{where} is missing required keys {missing}; every key "
                         "must be present, with no default supplied by this tool")


def _no_placeholder(value, where: str) -> None:
    if isinstance(value, str) and value.startswith("<"):
        raise ValueError(f"{where} is still the example placeholder {value!r}")


def _text(mapping: dict, key: str, where: str, pattern=None, shape="") -> str:
    value, field = mapping[key], f"{where}.{key}"
    _no_placeholder(value, field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string, got {value!r}")
    if pattern is not None and not pattern.match(value):
        raise ValueError(f"{field} must look like {shape}, got {value!r}")
    return value


def _positive_int(mapping: dict, key: str, where: str) -> int:
    value, field = mapping[key], f"{where}.{key}"
    _no_placeholder(value, field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be an integer >= 1, got {value!r}")
    return value


def _nullable_text(mapping: dict, key: str, where: str) -> str | None:
    if mapping[key] is None:
        return None
    return _text(mapping, key, where)


def _resource_block(raw, keys: tuple[str, ...], where: str) -> dict:
    _check_keys(raw, keys, where)
    block = {"time": _text(raw, "time", where, _TIME_RE, "HH:MM:SS"),
             "mem": _text(raw, "mem", where, _MEM_RE, "4G or 4000M"),
             "cpus_per_task": _positive_int(raw, "cpus_per_task", where)}
    for key in keys:
        if key in _INHERITED:
            block[key] = _nullable_text(raw, key, where)
    return block


def load_cluster_config(path) -> dict:
    """Read and fully validate the cluster config; every key is required."""
    data = json.loads(Path(path).read_text())
    _check_keys(data, _TOP_KEYS, "cluster config")
    if data["cluster_config_version"] != CLUSTER_CONFIG_VERSION:
        raise ValueError(f"cluster_config_version must be {CLUSTER_CONFIG_VERSION}, "
                         f"got {data['cluster_config_version']!r}")
    setup = data["setup_lines"]
    if not isinstance(setup, list) or not all(isinstance(l, str) for l in setup):
        raise ValueError("setup_lines must be an array of shell command strings")
    for index, line in enumerate(setup):
        _no_placeholder(line, f"setup_lines[{index}]")

    config = {"cluster_config_version": CLUSTER_CONFIG_VERSION,
              "name": _text(data, "name", "cluster config"),
              "venv": _text(data, "venv", "cluster config"),
              "setup_lines": list(setup),
              "task": _resource_block(data["task"], _TASK_KEYS, "task"),
              "compare": _resource_block(data["compare"], _COMPARE_KEYS, "compare"),
              "max_array_size": _positive_int(data, "max_array_size", "cluster config"),
              "max_concurrent_tasks": None}
    if data["max_concurrent_tasks"] is not None:
        config["max_concurrent_tasks"] = _positive_int(data, "max_concurrent_tasks",
                                                       "cluster config")
    if not Path(config["venv"]).is_absolute():
        raise ValueError(f"venv must be an absolute path, got {config['venv']!r}")
    if not (Path(config["venv"]) / "bin" / "python").exists():
        raise ValueError(f"{config['venv']}/bin/python does not exist; prepare the "
                         "venv with scripts/rl/bootstrap_venv.py --venv <path>")
    return config


def resources(config: dict, role: str) -> dict:
    """Resource flags for one job role; compare inherits placement from task."""
    if role not in JOB_ROLES:
        raise ValueError(f"unknown job role {role!r}; valid: {list(JOB_ROLES)}")
    block = config["task"] if role == "tasks" else config["compare"]
    values = {key: config["task"][key] for key in _INHERITED}
    values.update({key: block[key] for key in ("time", "mem", "cpus_per_task")})
    return values


def job_name(role: str) -> str:
    """Random per-submission job name so an interrupted sbatch can be searched."""
    if role not in JOB_ROLES:
        raise ValueError(f"unknown job role {role!r}; valid: {list(JOB_ROLES)}")
    return f"{JOB_NAME_PREFIX}-{secrets.token_hex(16)}-{role}"


def array_spec(indices, max_concurrent: int | None = None) -> str:
    """Compress sorted task indices into an sbatch array spec, with %N when capped."""
    ordered = sorted(set(indices))
    if not ordered:
        raise ValueError("array spec needs at least one task index")
    parts, start, previous = [], ordered[0], ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        parts.append(f"{start}-{previous}" if previous > start else str(start))
        start = previous = index
    parts.append(f"{start}-{previous}" if previous > start else str(start))
    spec = ",".join(parts)
    return f"{spec}%{max_concurrent}" if max_concurrent else spec


def sbatch_argv(config: dict, role: str, name: str, log_pattern, script,
                array: str | None = None, dependency: str | None = None) -> list[str]:
    """Build the sbatch command; flags whose config value is null are omitted."""
    values = resources(config, role)
    argv = ["sbatch", "--parsable", "--no-requeue", f"--job-name={name}"]
    if array is not None:
        argv.append(f"--array={array}")
    argv.append(f"--output={log_pattern}")
    if dependency is not None:
        argv.append(f"--dependency={dependency}")
    for flag in _INHERITED:
        if values[flag] is not None:
            argv.append(f"--{flag}={values[flag]}")
    argv += [f"--time={values['time']}", f"--mem={values['mem']}",
             f"--cpus-per-task={values['cpus_per_task']}", str(script)]
    return argv


def _preamble(config: dict, role: str, mesh_root) -> list[str]:
    cpus = resources(config, role)["cpus_per_task"]
    python = shlex.quote(str(Path(config["venv"]) / "bin" / "python"))
    venv = shlex.quote(config["venv"])
    return ["#!/bin/bash", "set -euo pipefail", *config["setup_lines"],
            f"export OMP_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus} "
            "CUDA_VISIBLE_DEVICES=",
            f"cd {shlex.quote(str(mesh_root))}",
            f"{python} scripts/rl/bootstrap_venv.py --venv {venv} --check"]


def render_task_script(config: dict, mesh_root, output_root, records_dir) -> str:
    """Job script for one array element: bootstrap check, then run_task."""
    python = shlex.quote(str(Path(config["venv"]) / "bin" / "python"))
    lines = _preamble(config, "tasks", mesh_root)
    lines.append(f"RECORD_DIR={shlex.quote(str(records_dir))}")
    lines.append(
        f"exec {python} -m scripts.rl.ops.run_task "
        f"--output-root {shlex.quote(str(output_root))} "
        '--task-index "$SLURM_ARRAY_TASK_ID" '
        '--record "$RECORD_DIR/task-$(printf \'%04d\' "$SLURM_ARRAY_TASK_ID").json"')
    return "\n".join(lines) + "\n"


def render_compare_script(config: dict, mesh_root, output_root, records_dir) -> str:
    """Job script for the compare step; strict prerequisites, no --allow-incomplete."""
    python = shlex.quote(str(Path(config["venv"]) / "bin" / "python"))
    record = shlex.quote(str(Path(records_dir) / "compare.json"))
    lines = _preamble(config, "compare", mesh_root)
    lines.append(f"exec {python} -m scripts.rl.ops.run_task "
                 f"--output-root {shlex.quote(str(output_root))} --compare "
                 f"--record {record}")
    return "\n".join(lines) + "\n"


def available(binary: str = "sbatch") -> bool:
    """Whether a scheduler binary is on PATH right now."""
    return shutil.which(binary) is not None


def _run(argv: list[str]) -> dict:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True)
    except OSError as exc:
        return {"ok": False, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}",
                "returncode": None, "argv": argv}
    return {"ok": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr,
            "returncode": proc.returncode, "argv": argv}


def parse_job_id(text: str) -> str | None:
    """Read `jobid[;cluster]` from --parsable output; None when unparseable."""
    for line in text.splitlines():
        match = _JOB_ID_RE.match(line.strip())
        if match:
            return match.group(1)
    return None


def submit(argv: list[str]) -> dict:
    """Run sbatch; job_id is set only when the response parsed as a job id."""
    result = _run(argv)
    result["job_id"] = parse_job_id(result["stdout"]) if result["ok"] else None
    return result


def cancel(job_ids) -> dict:
    """Run scancel on exact job or array-element ids."""
    ids = list(job_ids)
    if not ids:
        raise ValueError("scancel needs at least one job id")
    return _run(["scancel", *ids])


def _expand(job_id: str) -> list[str]:
    match = _ELEMENT_RE.match(job_id)
    if not match:
        return [job_id]
    base, body = match.group(1), match.group(2).split("%")[0]
    elements = []
    for token in body.split(","):
        low, separator, high = token.partition("-")
        if not low.strip().isdigit():
            continue
        start = int(low)
        end = int(high) if separator and high.strip().isdigit() else start
        elements.extend(f"{base}_{index}" for index in range(start, end + 1))
    return elements


def parse_squeue(text: str) -> dict:
    """Parse `%i|%T|%r|%S` rows, expanding collapsed array-element ranges."""
    jobs = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|")
        if len(fields) < 4:
            continue
        job_id, state, reason, start = fields[0], fields[1], fields[2], fields[3]
        for element in _expand(job_id.strip()):
            jobs[element] = {"state": state.strip(), "reason": reason.strip(),
                             "start": start.strip()}
    return jobs


def parse_sacct(text: str) -> dict:
    """Parse `JobID|State|ExitCode` rows; step rows are dropped, CANCELLED normalized."""
    jobs = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|")
        if len(fields) < 3:
            continue
        job_id, state, exit_code = (field.strip() for field in fields[:3])
        if "." in job_id:
            continue
        if _CANCELLED_RE.match(state):
            state = "CANCELLED"
        for element in _expand(job_id):
            jobs[element] = {"state": state, "exit": exit_code}
    return jobs


def _base(job_id: str) -> str:
    return job_id.split("_", 1)[0]


def snapshot(job_ids, user: str) -> dict:
    """Queue and accounting view of the receipts' jobs; a failed query sets ok false."""
    ids = sorted({_base(str(job_id)) for job_id in job_ids})
    if not ids:
        return {"queue": {"ok": True, "jobs": {}},
                "accounting": {"ok": True, "jobs": {}}}
    queue = _run(["squeue", "--noheader", "--array", "--states=all",
                  f"--user={user}", "--format=%i|%T|%r|%S"])
    accounting = _run(["sacct", "--noheader", "--parsable2", "--array",
                       f"--jobs={','.join(ids)}", "--format=JobID,State,ExitCode"])
    wanted = set(ids)
    queue_jobs = {key: value for key, value in parse_squeue(queue["stdout"]).items()
                  if _base(key) in wanted} if queue["ok"] else {}
    account_jobs = ({key: value
                     for key, value in parse_sacct(accounting["stdout"]).items()
                     if _base(key) in wanted} if accounting["ok"] else {})
    return {"queue": {"ok": queue["ok"], "jobs": queue_jobs,
                      "error": "" if queue["ok"] else queue["stderr"].strip()},
            "accounting": {"ok": accounting["ok"], "jobs": account_jobs,
                           "error": "" if accounting["ok"]
                                    else accounting["stderr"].strip()}}


def find_jobs_by_name(name: str, user: str) -> dict:
    """Look a no-ID submission up by its exact random job name in queue and accounting."""
    queue = _run(["squeue", "--noheader", "--states=all", f"--name={name}",
                  f"--user={user}", "--format=%i|%j|%T"])
    accounting = _run(["sacct", "--noheader", "--parsable2", f"--name={name}",
                       f"--user={user}", "--format=JobID,JobName,State"])
    found: dict[str, str] = {}
    for result in (queue, accounting):
        if not result["ok"]:
            continue
        for line in result["stdout"].splitlines():
            fields = [field.strip() for field in line.strip().split("|")]
            if len(fields) < 3 or fields[1] != name or "." in fields[0]:
                continue
            found.setdefault(_base(fields[0]), fields[2])
    return {"ok": queue["ok"] and accounting["ok"], "job_ids": sorted(found),
            "states": found,
            "error": "; ".join(result["stderr"].strip() for result in (queue, accounting)
                               if not result["ok"])}
