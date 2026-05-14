"""Helpers shared across bh2 and GPS plots."""

import pandas as pd

from ..paths import KNOWN_BAD_SCENARIOS


def scenario_caption(df: pd.DataFrame) -> str:
    """e.g. ``3 rabs, 9.7 ks``."""
    n_nodes = df["__node__"].nunique() if "__node__" in df.columns else 0
    if "__sec__" in df.columns and not df["__sec__"].empty:
        duration_s = float(df["__sec__"].max())
    elif "__t__" in df.columns and not df["__t__"].empty:
        duration_s = (df["__t__"].max() - df["__t__"].min()).total_seconds()
    else:
        duration_s = 0.0
    dur = f"{duration_s / 1000:.1f} ks" if duration_s >= 1000 else f"{duration_s:.0f} s"
    label = "rabs" if n_nodes != 1 else "rab"
    return f"{n_nodes} {label}, {dur}"


def crashed_suffix(scenario: str) -> str:
    return "  [CRASHED -- see event log]" if scenario in KNOWN_BAD_SCENARIOS else ""


def concat_trace(rows: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
