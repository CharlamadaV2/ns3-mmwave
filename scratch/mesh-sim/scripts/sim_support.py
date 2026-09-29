"""Shared paths, loader environment, and diagnostics for simulator launchers."""

from collections import deque
import os
from pathlib import Path


## @fn find_mesh_root
# @brief Find the mesh-sim root directory (the one containing `sim.cc`).
#
# @param start  File or directory to start from; defaults to this file.
# @return Path of the nearest ancestor (or `start` itself) containing `sim.cc`.
# @throws RuntimeError if no ancestor contains `sim.cc`.
def find_mesh_root(start: str | Path = __file__) -> Path:
    """Find the ancestor containing mesh-sim's sim.cc."""
    path = Path(start).resolve()
    directory = path.parent if path.is_file() else path
    for candidate in (directory, *directory.parents):
        if (candidate / "sim.cc").is_file():
            return candidate
    raise RuntimeError(f"Could not locate mesh-sim root from {path}")


## @fn find_sim_binary
# @brief Locate the built simulator binary.
#
# @param mesh_root  mesh-sim root; the ns-3 root is assumed two levels above it.
# @return Path string of the first (sorted) match of `<ns3>/build/scratch/mesh-sim/ns3*-sim-*`, or None.
def find_sim_binary(mesh_root: str | Path) -> str | None:
    """Return the first matching binary in the containing ns-3 build tree."""
    build = Path(mesh_root).resolve().parent.parent / "build" / "scratch" / "mesh-sim"
    matches = sorted(build.glob("ns3*-sim-*"))
    return str(matches[0]) if matches else None


## @fn simulator_env
# @brief Build an environment for launching the simulator.
#
# @param mesh_root  mesh-sim root; ns-3's `build/lib` is two levels above it.
# @return Copy of `os.environ` with `<ns3>/build/lib` prepended to `DYLD_LIBRARY_PATH` and `LD_LIBRARY_PATH`.
def simulator_env(mesh_root: str | Path) -> dict[str, str]:
    """Copy the process environment and prepend ns-3's shared-library directory."""
    env = dict(os.environ)
    lib = str(Path(mesh_root).resolve().parent.parent / "build" / "lib")
    for variable in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        parts = [part for part in env.get(variable, "").split(os.pathsep) if part]
        env[variable] = os.pathsep.join([lib, *parts])
    return env


## @fn parse_seed_spec
# @brief Expand a seed list such as `1,3,5-7` into integers.
#
# @param raw       Comma-separated integers and inclusive `A-B` ranges.
# @param distinct  If True (default), duplicate seeds are an error.
# @return List of ints in the order given, e.g. `[1, 3, 5, 6, 7]`.
# @throws ValueError if empty, a token is not an integer or range, `A > B`, or (when `distinct`) a seed repeats.
#
# Negative numbers are not supported because `-` is the range separator.
def parse_seed_spec(raw: str, distinct: bool = True) -> list[int]:
    """Expand a comma list of integers and inclusive `A-B` ranges, order preserved."""
    tokens = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not tokens:
        raise ValueError("seed specification is empty")
    seeds: list[int] = []
    for token in tokens:
        low, separator, high = token.partition("-")
        try:
            start = int(low.strip())
            end = int(high.strip()) if separator else start
        except ValueError as exc:
            raise ValueError(
                f"seed {token!r} is not an integer or an A-B range: {exc}") from exc
        if start > end:
            raise ValueError(f"seed range {token!r} must have A <= B")
        seeds.extend(range(start, end + 1))
    if distinct:
        seen: set[int] = set()
        for seed in seeds:
            if seed in seen:
                raise ValueError(f"seed {seed} is listed more than once")
            seen.add(seed)
    return seeds


## @fn strip_inline_comment
# @brief Remove a trailing `#` or `;` comment and surrounding whitespace.
#
# @param value  Raw INI value.
# @return Cleaned string. Matches the C++ parser, so a `#` or `;` inside a value also truncates it.
def strip_inline_comment(value: str) -> str:
    """Mirror the C++ INI parser's # and ; inline-comment handling."""
    for marker in ("#", ";"):
        value = value.split(marker, 1)[0]
    return value.strip()


## @fn tail_lines
# @brief Return the last lines of a text file, for failure messages.
#
# @param path   File to read (invalid UTF-8 bytes are replaced).
# @param count  Number of lines to keep (default 40).
# @return The lines joined with newlines, without line endings.
# @throws OSError if the file cannot be opened.
#
# Streams the file and keeps only `count` lines in memory.
def tail_lines(path: str | Path, count: int = 40) -> str:
    """Read only the trailing lines into memory for failure diagnostics."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return "\n".join(line.rstrip("\r\n") for line in deque(handle, maxlen=count))
