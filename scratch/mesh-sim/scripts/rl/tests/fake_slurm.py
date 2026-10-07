"""Stand-in sbatch/squeue/sacct/scancel shims sharing one JSON state file."""

import json
import os
import stat
import sys
from pathlib import Path

COMMANDS = ("sbatch", "squeue", "sacct", "scancel")
FIRST_JOB_ID = 1000
ACTIVE = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED",
          "REQUEUED", "RESIZING", "SIGNALING", "STAGE_OUT"}
STATE_ENV = "FAKE_SLURM_STATE"


def _state_path() -> Path:
    return Path(os.environ[STATE_ENV])


def _read() -> dict:
    return json.loads(_state_path().read_text())


def _write(state: dict) -> None:
    _state_path().write_text(json.dumps(state, indent=2))


def _flag(argv: list[str], name: str) -> str | None:
    for token in argv:
        if token.startswith(f"--{name}="):
            return token.split("=", 1)[1]
    return None


def _array_indices(spec: str | None) -> list[int]:
    if spec is None:
        return []
    indices = []
    for token in spec.split("%")[0].split(","):
        low, separator, high = token.partition("-")
        indices.extend(range(int(low), int(high) + 1) if separator
                       else [int(low)])
    return indices


def _sbatch(argv: list[str]) -> int:
    state = _read()
    if state["fail"].get("sbatch"):
        sys.stderr.write("fake-slurm: sbatch refused the job\n")
        return 1
    job_id = str(state["next_job_id"])
    state["next_job_id"] += 1
    indices = _array_indices(_flag(argv, "array"))
    keys = [f"{job_id}_{index}" for index in indices] or [job_id]
    state["jobs"][job_id] = {
        "name": _flag(argv, "job-name"), "indices": indices,
        "dependency": _flag(argv, "dependency"),
        "elements": {key: {"state": "PENDING", "reason": "Priority",
                           "start": "N/A", "exit": "0:0"} for key in keys}}
    state["submissions"].append(argv)
    _write(state)
    if state["die_after_queue"]:
        sys.stderr.write("fake-slurm: connection lost after the job was queued\n")
        return 1
    sys.stdout.write(f"{job_id}\n")
    return 0


def _matching(state: dict, name: str | None, wanted: set[str] | None):
    for job_id, job in state["jobs"].items():
        if name is not None and job["name"] != name:
            continue
        if wanted is not None and job_id not in wanted:
            continue
        yield job_id, job


def _squeue(argv: list[str]) -> int:
    state = _read()
    if state["fail"].get("squeue"):
        sys.stderr.write("fake-slurm: squeue is unavailable\n")
        return 1
    name = _flag(argv, "name")
    by_name = name is not None
    for job_id, job in _matching(state, name, None):
        for key, element in job["elements"].items():
            if element["state"] not in ACTIVE:
                continue
            if by_name:
                sys.stdout.write(f"{key}|{job['name']}|{element['state']}\n")
            else:
                sys.stdout.write(f"{key}|{element['state']}|{element['reason']}"
                                 f"|{element['start']}\n")
    return 0


def _sacct(argv: list[str]) -> int:
    state = _read()
    if state["fail"].get("sacct"):
        sys.stderr.write("fake-slurm: accounting is not enabled\n")
        return 1
    name = _flag(argv, "name")
    jobs = _flag(argv, "jobs")
    wanted = set(jobs.split(",")) if jobs else None
    for job_id, job in _matching(state, name, wanted):
        if name is not None:
            for key in job["elements"]:
                sys.stdout.write(f"{key}|{job['name']}|"
                                 f"{job['elements'][key]['state']}\n")
            continue
        if job["indices"]:
            sys.stdout.write(f"{job_id}|COMPLETED|0:0\n")
        for key, element in job["elements"].items():
            sys.stdout.write(f"{key}|{element['state']}|{element['exit']}\n")
            sys.stdout.write(f"{key}.batch|{element['state']}|{element['exit']}\n")
    return 0


def _scancel(argv: list[str]) -> int:
    state = _read()
    ids = [token for token in argv if not token.startswith("-")]
    state["scancel_calls"].append(ids)
    if state["fail"].get("scancel"):
        _write(state)
        sys.stderr.write("fake-slurm: scancel: invalid job id\n")
        return 1
    for job_id in ids:
        base = job_id.split("_", 1)[0]
        job = state["jobs"].get(base)
        if not job:
            continue
        keys = [job_id] if job_id in job["elements"] else list(job["elements"])
        for key in keys:
            job["elements"][key]["state"] = "CANCELLED"
            job["elements"][key]["exit"] = "0:15"
    _write(state)
    return 0


_DISPATCH = {"sbatch": _sbatch, "squeue": _squeue, "sacct": _sacct,
             "scancel": _scancel}


def main(argv: list[str]) -> int:
    return _DISPATCH[argv[0]](argv[1:])


class FakeSlurm:
    """Handle on the shim directory and its shared state, for use from tests."""

    def __init__(self, bin_dir: Path, state_path: Path):
        self.bin = bin_dir
        self.state_path = state_path

    def read(self) -> dict:
        return json.loads(self.state_path.read_text())

    def write(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state, indent=2))

    def jobs(self) -> dict:
        return self.read()["jobs"]

    def job_ids(self) -> list[str]:
        return sorted(self.jobs(), key=int)

    def last_job_id(self) -> str:
        return self.job_ids()[-1]

    def submissions(self) -> list[list[str]]:
        return self.read()["submissions"]

    def scancel_calls(self) -> list[list[str]]:
        return self.read()["scancel_calls"]

    def set_element(self, job_id: str, index: int | None, state: str,
                    reason: str = "None", start: str = "N/A",
                    exit_code: str = "0:0") -> None:
        data = self.read()
        key = job_id if index is None else f"{job_id}_{index}"
        data["jobs"][job_id]["elements"][key] = {
            "state": state, "reason": reason, "start": start, "exit": exit_code}
        self.write(data)

    def set_job(self, job_id: str, state: str, exit_code: str = "0:0") -> None:
        data = self.read()
        for element in data["jobs"][job_id]["elements"].values():
            element["state"], element["exit"] = state, exit_code
        self.write(data)

    def drop_element(self, job_id: str, index: int) -> None:
        """Remove an element so neither queue nor accounting reports it."""
        data = self.read()
        data["jobs"][job_id]["elements"].pop(f"{job_id}_{index}", None)
        data["jobs"][job_id]["indices"] = [
            i for i in data["jobs"][job_id]["indices"] if i != index]
        self.write(data)

    def drop_job(self, job_id: str) -> None:
        """Remove every element so neither queue nor accounting reports the job."""
        data = self.read()
        data["jobs"][job_id]["elements"] = {}
        data["jobs"][job_id]["indices"] = []
        self.write(data)

    def clone_job(self, job_id: str) -> str:
        """Copy a job under a new id so its job name matches more than one job."""
        data = self.read()
        new_id = str(data["next_job_id"])
        data["next_job_id"] += 1
        job = json.loads(json.dumps(data["jobs"][job_id]))
        job["elements"] = {key.replace(job_id, new_id, 1): element
                           for key, element in job["elements"].items()}
        data["jobs"][new_id] = job
        self.write(data)
        return new_id

    def fail(self, command: str, enabled: bool = True) -> None:
        data = self.read()
        data["fail"][command] = enabled
        self.write(data)

    def die_after_queue(self, enabled: bool = True) -> None:
        data = self.read()
        data["die_after_queue"] = enabled
        self.write(data)


def install(tmp_path: Path) -> FakeSlurm:
    """Create the four shims in tmp_path/slurm-bin over a fresh state file."""
    bin_dir = tmp_path / "slurm-bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    state_path = tmp_path / "fake-slurm-state.json"
    state_path.write_text(json.dumps(
        {"next_job_id": FIRST_JOB_ID, "jobs": {}, "submissions": [],
         "scancel_calls": [], "fail": {}, "die_after_queue": False}, indent=2))
    for command in COMMANDS:
        shim = bin_dir / command
        shim.write_text(f"#!/bin/sh\n{STATE_ENV}={state_path} "
                        f'exec "{sys.executable}" "{Path(__file__).resolve()}" '
                        f'{command} "$@"\n')
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return FakeSlurm(bin_dir, state_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
