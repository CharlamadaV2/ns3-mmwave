"""Stand-in for a non-RL mesh-sim run: checks argv, writes seed-N/summary.json."""

# Environment: FAKE_CHILD_MODE (ok, fail, missing, hang, hang-ignore-term),
# FAKE_CHILD_RECORD (JSON of argv/env/stdin facts), FAKE_CHILD_PID_FILE (hang modes).

import configparser
import json
import os
import signal
import sys
import time
from pathlib import Path

ALLOWED_FLAGS = ("--run-config=", "--output-dir=", "--seeds=", "--band=")
DEFAULT_SEED = 42
FAIL_EXIT_CODE = 7


def _flags(argv: list[str]) -> dict:
    flags = {}
    for arg in argv:
        prefix = next((p for p in ALLOWED_FLAGS if arg.startswith(p)), None)
        if prefix is None or prefix in flags:
            sys.exit(f"fake_child: unexpected or repeated argument {arg!r}")
        flags[prefix] = arg[len(prefix):]
    for required in ALLOWED_FLAGS[:2]:
        if required not in flags:
            sys.exit(f"fake_child: missing {required}<value>")
    return flags


def _value(ini: configparser.ConfigParser, section: str, key: str) -> str | None:
    if not ini.has_option(section, key):
        return None
    return ini.get(section, key).split("#", 1)[0].split(";", 1)[0].strip() or None


def _stdin_is_devnull() -> bool:
    try:
        return os.path.samestat(os.fstat(0), os.stat(os.devnull))
    except OSError:
        return False


def main() -> int:
    flags = _flags(sys.argv[1:])
    run_config = Path(flags["--run-config="])
    out = Path(flags["--output-dir="])
    ini = configparser.ConfigParser(interpolation=None, default_section="\n")
    ini.optionxform = str
    ini.read_string(run_config.read_text(encoding="utf-8"))
    algorithm = _value(ini, "baseline", "algorithm") or "none"
    if algorithm != "none":
        print(f"baseline.algorithm '{algorithm}' is active but this is a direct simulator run",
              file=sys.stderr)
        return 1
    if "--seeds=" in flags:
        seeds = [int(token) for token in flags["--seeds="].split(",")]
    else:
        seeds = [int(_value(ini, "scenario", "seed") or DEFAULT_SEED)]

    record_path = os.environ.get("FAKE_CHILD_RECORD")
    if record_path:
        Path(record_path).write_text(json.dumps({
            "argv": sys.argv[1:],
            "env": {key: os.environ.get(key) for key in
                    ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "FAKE_CHILD_MARKER")},
            "stdin_devnull": _stdin_is_devnull(),
            "manifest_status": json.loads(
                (out / "baseline_manifest.json").read_text())["status"],
            "rl_enabled": _value(ini, "rl", "enabled"),
            "nodes_file": _value(ini, "scenario", "nodes_file"),
        }, indent=2))

    mode = os.environ.get("FAKE_CHILD_MODE", "ok")
    print(f"fake_child: mode={mode} seeds={seeds}", flush=True)
    if mode.startswith("hang"):
        if mode == "hang-ignore-term":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pid_file = os.environ.get("FAKE_CHILD_PID_FILE")
        if pid_file:
            Path(pid_file).write_text(str(os.getpid()))
        while True:
            time.sleep(0.1)

    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "run.log").write_text("fake_child run log\n")
    for index, seed in enumerate(seeds):
        if mode == "fail" and index == 1:
            print("fake_child: simulated failure", file=sys.stderr)
            return FAIL_EXIT_CODE
        if mode == "missing" and index == len(seeds) - 1:
            continue
        seed_dir = out / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "summary.json").write_text(json.dumps({"seed": seed}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
