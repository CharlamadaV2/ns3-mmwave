"""Single-file data loaders and seed directory discovery for mesh-sim outputs."""

import json
import os
import re

import pandas as pd

_SEED_DIR_RE = re.compile(r"^seed-(\d+)$")


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

def load_links_csv(path: str) -> pd.DataFrame | None:
    """Load links.csv (comment lines start with #)."""
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, comment="#")


def load_rx_power_csv(path: str) -> pd.DataFrame | None:
    """Load rx-power.csv."""
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def load_mcs_csv(path: str) -> pd.DataFrame | None:
    """Load mcs.csv."""
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def load_flows_csv(path: str) -> pd.DataFrame | None:
    """Load flows.csv."""
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def load_routes_csv(path: str) -> pd.DataFrame | None:
    """Load routes.csv."""
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def load_positions_csv(path: str) -> pd.DataFrame | None:
    """Load positions.csv (comment lines start with #)."""
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, comment="#")


def load_summary(path: str) -> dict | None:
    """Load summary.json."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)
