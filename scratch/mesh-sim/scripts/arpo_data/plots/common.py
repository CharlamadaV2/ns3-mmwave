##@file common.py
# @brief Helpers shared across bh2 and GPS plots.
#
#
##

import pandas as pd

from ..paths import KNOWN_BAD_SCENARIOS

## @brief Generates a short human-readable caption summarising a scenario DataFrame.
#
# Counts the number of unique nodes and computes the total duration from either
# ``__sec__`` (float seconds) or ``__t__`` (datetime). Duration is formatted as
# kiloseconds when >= 1000 s, otherwise as whole seconds.
#
# @param df   Loaded scenario DataFrame. Must contain ``__node__`` and either
#             ``__sec__`` or ``__t__`` columns for a non-trivial caption.
# @return     Caption string, e.g. ``"3 rabs, 9.7 ks"`` or ``"1 rab, 120 s"``.
def scenario_caption(df: pd.DataFrame) -> str:
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

## @brief Returns a crash-warning suffix for a scenario name if it is known to be bad.
#
# Looks up @p scenario in @ref KNOWN_BAD_SCENARIOS. Used to annotate plot
# titles so readers know the underlying data may be contaminated or partial.
#
# @param scenario   Scenario directory name to check.
# @return           Warning string ``"  [CRASHED -- see event log]"`` if the
#                   scenario is known bad, otherwise an empty string.
def crashed_suffix(scenario: str) -> str:
    return "  [CRASHED -- see event log]" if scenario in KNOWN_BAD_SCENARIOS else ""

## @brief Concatenates a list of DataFrames into a single DataFrame.
#
# Resets the index so the result has a clean integer index regardless of the
# input indices. Returns an empty DataFrame when @p rows is empty, which lets
# callers test with ``df.empty`` rather than checking the list length first.
#
# @param rows   List of DataFrames to concatenate. May be empty.
# @return       Single concatenated DataFrame, or an empty DataFrame if
#               @p rows is empty.
def concat_trace(rows: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()