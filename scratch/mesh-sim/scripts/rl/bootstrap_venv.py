#!/usr/bin/env python3
"""Create mesh-sim's .venv, install requirements.txt, and verify the imports.

    python3 scripts/rl/bootstrap_venv.py [--venv PATH] [--check]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
MESH_SIM_ROOT = Path(__file__).resolve().parents[2]

# Import name -> distribution name in requirements.txt.
DIRECT_DEPS = {
    "gymnasium": "gymnasium",
    "numpy": "numpy",
    "stable_baselines3": "stable-baselines3",
    "sb3_contrib": "sb3-contrib",
    "torch": "torch",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "pytest": "pytest",
}

_VERSION_PROBE = """
import importlib, importlib.metadata, sys
print("python==" + ".".join(str(p) for p in sys.version_info[:3]))
failures = []
for mod, dist in {deps!r}.items():
    try:
        importlib.import_module(mod)
        version = importlib.metadata.version(dist)
        print(dist + "==" + version)
        if version != {expected!r}[dist]:
            failures.append(dist + ": expected " + {expected!r}[dist] + ", got " + version)
    except Exception as exc:
        failures.append(mod + ": " + type(exc).__name__ + ": " + str(exc))
for f in failures:
    print("DEPENDENCY-FAILED " + f, file=sys.stderr)
sys.exit(1 if failures else 0)
"""


def _fail(message: str) -> int:
    print(f"bootstrap_venv: {message}", file=sys.stderr)
    return 1


def _run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        _fail(f"command failed (exit {rc}): {' '.join(cmd)}")
    return rc


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def read_pins(requirements: Path) -> dict[str, str]:
    """Read the project's exact direct-dependency pins."""
    pins = {}
    for line in requirements.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, separator, version = line.partition("==")
        if not separator or not name or not version:
            raise ValueError(f"Expected an exact dependency pin, got: {line}")
        pins[name] = version
    if set(pins) != set(DIRECT_DEPS.values()):
        raise ValueError("requirements.txt and DIRECT_DEPS must name the same dependencies")
    return pins


def main() -> int:
    p = argparse.ArgumentParser(description="Bootstrap the mesh-sim Python venv")
    p.add_argument("--venv", default=str(MESH_SIM_ROOT / ".venv"),
                   help="Virtual environment directory (default: mesh-sim/.venv)")
    p.add_argument("--check", action="store_true",
                   help="Verify an existing venv without installing")
    args = p.parse_args()

    if sys.version_info < MIN_PYTHON:
        return _fail(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, "
            f"got {platform_version()}"
        )

    venv = Path(args.venv).resolve()
    python = _venv_python(venv)
    requirements = MESH_SIM_ROOT / "requirements.txt"
    try:
        expected = read_pins(requirements)
    except (OSError, ValueError) as exc:
        return _fail(str(exc))

    if not python.is_file():
        if args.check:
            return _fail(f"no virtual environment at {venv}; run without --check first")
        if _run([sys.executable, "-m", "venv", str(venv)]) != 0:
            return 1

    if not args.check:
        if not requirements.is_file():
            return _fail(f"missing {requirements}")
        if _run([str(python), "-m", "pip", "install", "-r", str(requirements)]) != 0:
            return 1

    probe = _VERSION_PROBE.format(deps=DIRECT_DEPS, expected=expected)
    result = subprocess.run([str(python), "-c", probe], capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, end="", file=sys.stderr)
        return _fail("dependency import/version check failed; run setup without --check")

    print("\nActivate the environment with:")
    print(f"  POSIX:   source {venv}/bin/activate")
    print(f"  Windows: {venv}\\Scripts\\activate")
    print("mesh-sim CI covers Linux and macOS; this helper does not install "
          "CMake/compiler tools or establish Windows simulator support.")
    return 0


def platform_version() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])


if __name__ == "__main__":
    sys.exit(main())
