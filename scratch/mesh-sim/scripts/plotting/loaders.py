"""Single-file data loaders and seed directory discovery for mesh-sim outputs."""

import json
import os
import re

import pandas as pd

_SEED_DIR_RE = re.compile(r"^seed-(\d+)$")


## @fn discover_seed_dirs
# @brief Find the `seed-N/` subdirectories of a run output directory.
#
# @param data_dir  Run output directory (the one that contains `seed-<N>/`).
# @return List of directory paths sorted by numeric N (so seed-2 comes before seed-10);
#         empty list if `data_dir` is not a directory or has no matches.
#
# Only names matching `^seed-(\d+)$` that are directories are kept.
# Nothing is created or modified.
def discover_seed_dirs(data_dir: str) -> list[str]:
    """Find all seed-N/ subdirs in *data_dir*, sorted by seed number."""
    results = []
    if not os.path.isdir(data_dir):
        return results
    for name in os.listdir(data_dir):
        m = _SEED_DIR_RE.match(name)
        if m and os.path.isdir(os.path.join(data_dir, name)):
            results.append((int(m.group(1)), os.path.join(data_dir, name)))
    results.sort(key=lambda x: x[0])
    return [path for _, path in results]


# ---------------------------------------------------------------------------
# CSV loaders — each returns a DataFrame or None if file is missing.
# ---------------------------------------------------------------------------

## @fn load_links_csv
# @brief Load `links.csv` (per-link metrics per tick) into a DataFrame.
#
# @param path  Absolute or relative path to the file.
# @return A pandas DataFrame, or None if `path` is not an existing file.
# @throws pandas.errors.ParserError / EmptyDataError if the file exists but is malformed or empty.
#
# Checks that the file exists, then reads it with `pandas.read_csv`. Lines starting with `#` are skipped.
# No columns are validated here; callers check for the columns they need.
def load_links_csv(path: str) -> pd.DataFrame | None:
    """Load links.csv (comment lines start with #)."""
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, comment="#")


## @fn load_rx_power_csv
# @brief Load `rx-power.csv` into a DataFrame.
#
# @param path  Absolute or relative path to the file.
# @return A pandas DataFrame, or None if `path` is not an existing file.
# @throws pandas.errors.ParserError / EmptyDataError if the file exists but is malformed or empty.
#
# Checks that the file exists, then reads it with `pandas.read_csv`.
# No columns are validated here; callers check for the columns they need.
def load_rx_power_csv(path: str) -> pd.DataFrame | None:
    """Load rx-power.csv."""
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


## @fn load_mcs_csv
# @brief Load `mcs.csv` into a DataFrame.
#
# @param path  Absolute or relative path to the file.
# @return A pandas DataFrame, or None if `path` is not an existing file.
# @throws pandas.errors.ParserError / EmptyDataError if the file exists but is malformed or empty.
#
# Checks that the file exists, then reads it with `pandas.read_csv`.
# No columns are validated here; callers check for the columns they need.
def load_mcs_csv(path: str) -> pd.DataFrame | None:
    """Load mcs.csv."""
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


## @fn load_flows_csv
# @brief Load `flows.csv` into a DataFrame.
#
# @param path  Absolute or relative path to the file.
# @return A pandas DataFrame, or None if `path` is not an existing file.
# @throws pandas.errors.ParserError / EmptyDataError if the file exists but is malformed or empty.
#
# Checks that the file exists, then reads it with `pandas.read_csv`.
# No columns are validated here; callers check for the columns they need.
def load_flows_csv(path: str) -> pd.DataFrame | None:
    """Load flows.csv."""
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


## @fn load_routes_csv
# @brief Load `routes.csv` into a DataFrame.
#
# @param path  Absolute or relative path to the file.
# @return A pandas DataFrame, or None if `path` is not an existing file.
# @throws pandas.errors.ParserError / EmptyDataError if the file exists but is malformed or empty.
#
# Checks that the file exists, then reads it with `pandas.read_csv`.
# No columns are validated here; callers check for the columns they need.
def load_routes_csv(path: str) -> pd.DataFrame | None:
    """Load routes.csv."""
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


## @fn load_positions_csv
# @brief Load `positions.csv` (node positions per tick) into a DataFrame.
#
# @param path  Absolute or relative path to the file.
# @return A pandas DataFrame, or None if `path` is not an existing file.
# @throws pandas.errors.ParserError / EmptyDataError if the file exists but is malformed or empty.
#
# Checks that the file exists, then reads it with `pandas.read_csv`. Lines starting with `#` are skipped.
# No columns are validated here; callers check for the columns they need.
def load_positions_csv(path: str) -> pd.DataFrame | None:
    """Load positions.csv (comment lines start with #)."""
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, comment="#")


## @fn load_summary
# @brief Load a per-seed `summary.json`.
#
# @param path  Path to the JSON file.
# @return The parsed JSON as a dict, or None if `path` is not an existing file.
# @throws json.JSONDecodeError if the file exists but is not valid JSON.
#
# Checks that the file exists, then parses it with `json.load`.
def load_summary(path: str) -> dict | None:
    """Load summary.json."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)
