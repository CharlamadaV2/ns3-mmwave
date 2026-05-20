## @package compare_runs
# @brief Cross-batch summary table: one row per validation batch.
#
# Reads the ``validation_summary.csv`` produced by @ref compare for each batch
# and prints ranked comparison tables useful for tuning simulation parameters
# (channel model, TX power, antenna gain).
#
# **Tables printed**
# | Table                  | Contents                                                   |
# |------------------------|------------------------------------------------------------|
# | Batch summary          | Per-run mean |Δmed| and mean K-S across all links.         |
# | Link-class breakdown   | Separate |Δmed| for "close" (rab1↔rab2) vs "far" (↔rab3). |
# | Per-scenario |Δmed|    | Rows=scenario, cols=run — which scenario drives error.     |
# | Per-scenario K-S       | Same layout with K-S statistic.                            |

from __future__ import annotations

import argparse
import configparser
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT    = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = REPO_ROOT / "outputs"

## @brief Default TX power used when ``run.ini`` does not specify it (dBm).
_DEFAULT_TX_POWER_DBM = 10.0

## @brief Default array gain used when ``run.ini`` does not specify it (dBi).
#
# Matches the pre-knob hardcoded value of ``BF_ELEMENTS_PER_NODE=16``.
_DEFAULT_GAIN_DBI = 12.0

## @brief Hostname of the "far" rab; any link involving this node is classified as "far".
_RAB3 = "rab3"


## @brief Aggregated validation results for one batch run.
#
# Produced by @ref _summarize_batch and consumed by the rendering helpers.
@dataclass(frozen=True)
class BatchRow:
    run_dir:            str          ##< Repo-relative path to the batch directory.
    label:              str          ##< Short run label derived from the directory name.
    channel:            str          ##< Channel model name from ``run.ini``.
    scenario:           str          ##< Scenario identifier from ``run.ini``.
    tx_power_dbm:       float        ##< TX power in dBm.
    tx_gain_dbi:        float        ##< TX array gain in dBi.
    rx_gain_dbi:        float        ##< RX array gain in dBi.
    n_pairs:            int          ##< Number of (link, metric) pairs with data.
    mean_abs_dmed:      float        ##< Mean |Δmedian| across all pairs.
    mean_ks:            float        ##< Mean K-S statistic across all pairs.
    close_dmed:         float        ##< Mean |Δmedian| for close links (rab1↔rab2).
    far_dmed:           float        ##< Mean |Δmedian| for far links (↔rab3).
    per_scen_dmed:      dict[str, float]  ##< Per field-scenario mean |Δmedian|.
    per_scen_ks:        dict[str, float]  ##< Per field-scenario mean K-S.
    per_scen_close_dmed: dict[str, float] ##< Per-scenario mean |Δmedian| for close links.
    per_scen_far_dmed:  dict[str, float]  ##< Per-scenario mean |Δmedian| for far links.


## @brief Parse a ``run.ini`` file and return its key-value pairs as a flat dict.
#
# Only the ``[scenario]`` and ``[channel]`` sections are read.
#
# @param ini_path Path to the ``run.ini`` file.
# @return Dict of ``{key: value_string}`` for all keys in the two sections.
def _read_run_ini(ini_path: Path) -> dict[str, str]:
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    cp.read(ini_path)
    out: dict[str, str] = {}
    for section in ("scenario", "channel"):
        if cp.has_section(section):
            for k, v in cp.items(section):
                out[k] = v.strip()
    return out


## @brief Sniff channel model and gain/power settings from the first scenario's ``run.ini``.
#
# Walks the batch directory and reads the first snapshotted ``inputs/run.ini``
# it finds. Falls back to sentinel values if no ini is present.
#
# @param batch_dir Batch output directory.
# @return Dict with keys ``channel``, ``scenario``, ``tx_power_dbm``,
#         ``tx_gain_dbi``, ``rx_gain_dbi``.
def _sniff_batch_config(batch_dir: Path) -> dict[str, float | str]:
    for scen_dir in sorted(batch_dir.iterdir()):
        ini = scen_dir / "inputs" / "run.ini"
        if ini.is_file():
            cfg = _read_run_ini(ini)
            return {
                "channel":      cfg.get("channel_model", "?"),
                "scenario":     cfg.get("scenario", "?"),
                "tx_power_dbm": float(cfg.get("tx_power_dbm", _DEFAULT_TX_POWER_DBM)),
                "tx_gain_dbi":  float(cfg.get("tx_array_gain_dbi", _DEFAULT_GAIN_DBI)),
                "rx_gain_dbi":  float(cfg.get("rx_array_gain_dbi", _DEFAULT_GAIN_DBI)),
            }
    return {
        "channel": "?", "scenario": "?",
        "tx_power_dbm": float("nan"),
        "tx_gain_dbi":  float("nan"),
        "rx_gain_dbi":  float("nan"),
    }


## @brief Classify a link as ``"close"`` or ``"far"`` based on whether rab3 is involved.
#
# Links involving @ref _RAB3 are "far" (physically more distant in the Spring Lake
# deployment); all other links are "close".
#
# @param src  Source rab hostname.
# @param peer Peer rab hostname.
# @return ``"far"`` or ``"close"``.
def _classify(src: str, peer: str) -> str:
    if src == _RAB3 or peer == _RAB3:
        return "far"
    return "close"


## @brief Derive a compact run label from the batch directory name.
#
# Strips the ``-validation`` suffix so the label fits neatly in narrow table columns.
#
# @param batch_dir Batch directory path.
# @return Label string such as ``"14-01-40"`` from ``"14-01-40-validation"``.
def _batch_label(batch_dir: Path) -> str:
    return batch_dir.name.removesuffix("-validation")


## @brief Summarise one batch's validation results into a @ref BatchRow.
#
# Reads ``validation_summary.csv``, filters to the requested metric, and
# computes overall and per-scenario/per-link-class aggregates.
#
# @param batch_dir Batch output directory containing ``validation_summary.csv``.
# @param metric    Short metric name to filter on (e.g. ``"snr"``).
# @return Populated @ref BatchRow, or ``None`` if the CSV is absent or has no
#         matching rows.
def _summarize_batch(batch_dir: Path, metric: str) -> BatchRow | None:
    summary_csv = batch_dir / "validation_summary.csv"
    if not summary_csv.is_file():
        return None
    df = pd.read_csv(summary_csv)
    df = df[df["metric"] == metric].copy()
    if df.empty:
        return None

    df["link_class"] = [_classify(s, p) for s, p in zip(df["src_rab"], df["peer_rab"])]
    cfg = _sniff_batch_config(batch_dir)

    mean_dmed  = float(df["abs_diff_medians"].mean(skipna=True))
    mean_ks    = float(df["ks_statistic"].mean(skipna=True))
    close_dmed = float(df.loc[df["link_class"] == "close",
                              "abs_diff_medians"].mean(skipna=True))
    far_dmed   = float(df.loc[df["link_class"] == "far",
                              "abs_diff_medians"].mean(skipna=True))

    by_scen           = df.groupby("field_scenario", dropna=False)
    per_scen_dmed     = {str(k): float(v) for k, v
                         in by_scen["abs_diff_medians"].mean().items()}
    per_scen_ks       = {str(k): float(v) for k, v
                         in by_scen["ks_statistic"].mean().items()}
    close             = df[df["link_class"] == "close"].groupby("field_scenario",
                                                                 dropna=False)
    far               = df[df["link_class"] == "far"  ].groupby("field_scenario",
                                                                  dropna=False)
    per_scen_close_dmed = {str(k): float(v) for k, v
                           in close["abs_diff_medians"].mean().items()}
    per_scen_far_dmed   = {str(k): float(v) for k, v
                           in far  ["abs_diff_medians"].mean().items()}

    try:
        run_dir = str(batch_dir.resolve().relative_to(REPO_ROOT))
    except ValueError:
        run_dir = str(batch_dir)

    return BatchRow(
        run_dir=run_dir,
        label=_batch_label(batch_dir),
        channel=str(cfg["channel"]),
        scenario=str(cfg["scenario"]),
        tx_power_dbm=float(cfg["tx_power_dbm"]),
        tx_gain_dbi=float(cfg["tx_gain_dbi"]),
        rx_gain_dbi=float(cfg["rx_gain_dbi"]),
        n_pairs=int(df["abs_diff_medians"].notna().sum()),
        mean_abs_dmed=mean_dmed,
        mean_ks=mean_ks,
        close_dmed=close_dmed,
        far_dmed=far_dmed,
        per_scen_dmed=per_scen_dmed,
        per_scen_ks=per_scen_ks,
        per_scen_close_dmed=per_scen_close_dmed,
        per_scen_far_dmed=per_scen_far_dmed,
    )


## @brief Recursively discover all batch directories under a root.
#
# A directory qualifies if it contains a ``validation_summary.csv``.
#
# @param root Directory to search.
# @return Sorted list of batch directory paths.
def _discover_batches(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.rglob("validation_summary.csv"))


## @brief Format a float for table display, returning ``"—"`` for non-finite values.
#
# @param v      Value to format.
# @param digits Decimal places (default 2).
# @return Formatted string.
def _fmt(v: float, digits: int = 2) -> str:
    return "—" if not np.isfinite(v) else f"{v:.{digits}f}"


## @brief Render a plain-text fixed-width table.
#
# @param header Tuple of column header strings.
# @param body   List of row tuples, each matching the header length.
# @return Multi-line string with header, separator, and rows.
def _render_table(header: tuple[str, ...], body: list[tuple[str, ...]]) -> str:
    if not body:
        return "  ".join(header)
    widths = [max(len(h), *(len(row[i]) for row in body)) for i, h in enumerate(header)]
    sep    = "  ".join("-" * w for w in widths)
    lines  = ["  ".join(h.ljust(w) for h, w in zip(header, widths)), sep]
    for row in body:
        lines.append("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
    return "\n".join(lines)


## @brief Render the top-level batch summary table.
#
# Columns: run label, run directory, channel model, scenario, gain per side,
# total gain, pair count, mean |Δmedian|, mean K-S.
#
# @param rows List of @ref BatchRow objects (one per batch).
# @return Formatted table string.
def _render_main(rows: list[BatchRow]) -> str:
    header = ("run", "run dir", "channel", "scenario", "gain/side (Tx,Rx)",
              "tot gain", "n pairs", "mean |Δmed|", "mean K-S")
    body = []
    for r in rows:
        tot = r.tx_gain_dbi + r.rx_gain_dbi
        body.append((
            r.label, r.run_dir, r.channel, r.scenario,
            f"{r.tx_gain_dbi:.1f} / {r.rx_gain_dbi:.1f}",
            _fmt(tot, 1) + " dB",
            str(r.n_pairs),
            _fmt(r.mean_abs_dmed) + " dB",
            _fmt(r.mean_ks, 3),
        ))
    return _render_table(header, body)


## @brief Render the close-vs-far link-class breakdown table.
#
# @param rows List of @ref BatchRow objects.
# @return Formatted table string.
def _render_link_class(rows: list[BatchRow]) -> str:
    header = ("run", "close (rab1↔rab2) mean dB off",
              "far (rab*↔rab3) mean dB off")
    body = [(r.label,
             _fmt(r.close_dmed) + " dB",
             _fmt(r.far_dmed)   + " dB") for r in rows]
    return _render_table(header, body)


## @brief Collect the union of all field-scenario names across all batches, sorted.
#
# @param rows List of @ref BatchRow objects.
# @return Sorted list of unique scenario name strings.
def _scenario_order(rows: list[BatchRow]) -> list[str]:
    seen: dict[str, None] = {}
    for r in rows:
        for s in r.per_scen_dmed:
            seen.setdefault(s, None)
    return sorted(seen)


## @brief Render a per-scenario cross-batch comparison table.
#
# Rows are scenarios; columns are batch runs. The ``picker`` callable selects
# which per-scenario dict to read from each @ref BatchRow.
#
# @param rows         List of @ref BatchRow objects.
# @param picker       Callable ``(BatchRow) -> dict[str, float]``.
# @param value_suffix String appended to each formatted cell (e.g. ``" dB"``).
# @param digits       Decimal places for each value (default 2).
# @return Formatted table string.
def _render_per_scenario(rows: list[BatchRow], picker, value_suffix: str,
                         digits: int = 2) -> str:
    scenarios = _scenario_order(rows)
    header    = ("scenario", *(r.label for r in rows))
    body: list[tuple[str, ...]] = []
    for scen in scenarios:
        cells = [scen]
        for r in rows:
            v = picker(r).get(scen, float("nan"))
            cells.append(_fmt(v, digits) + value_suffix)
        body.append(tuple(cells))
    return _render_table(header, body)


## @brief CLI entry point for the cross-batch comparison tool.
#
# Auto-discovers batches under @ref OUTPUTS_ROOT when no explicit paths are
# given. Prints several comparison tables and optionally writes CSV exports.
#
# @param argv Argument list; defaults to ``sys.argv[1:]`` when ``None``.
# @return 0 on success, 1 on error.
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Cross-batch validation summary table (sim vs ARPO field).")
    p.add_argument("batches", nargs="*",
                   help="batch dirs to compare; if omitted, auto-discovers under outputs/")
    p.add_argument("--metric", default="snr", choices=("snr", "rcpi", "mcs"),
                   help="metric for the summary columns (default: snr)")
    p.add_argument("--csv", default=None,
                   help="optional path to write the per-batch table as CSV")
    p.add_argument("--per-scenario-csv", default=None,
                   help="optional path to write the per-(batch, scenario) long-form CSV")
    p.add_argument("--no-per-scenario", action="store_true",
                   help="suppress the per-scenario tables (overall + close/far only)")
    args = p.parse_args(argv)

    if args.batches:
        batch_dirs = [Path(b).resolve() for b in args.batches]
    else:
        if not OUTPUTS_ROOT.is_dir():
            print(f"no outputs root at {OUTPUTS_ROOT}", file=sys.stderr)
            return 1
        batch_dirs = _discover_batches(OUTPUTS_ROOT)
    if not batch_dirs:
        print("no batches with validation_summary.csv found", file=sys.stderr)
        return 1

    rows: list[BatchRow] = []
    for bd in batch_dirs:
        row = _summarize_batch(bd, args.metric)
        if row is None:
            print(f"  skipping {bd} (no validation_summary.csv or no {args.metric} rows)",
                  file=sys.stderr)
            continue
        rows.append(row)
    if not rows:
        return 1

    rows.sort(key=lambda r: r.run_dir)
    metric = args.metric

    print(f"== batch summary  (metric = {metric}) ==\n")
    print(_render_main(rows))
    print(f"\n== link-class breakdown  (|Δmed| {metric}, dB) ==\n")
    print(_render_link_class(rows))

    if not args.no_per_scenario:
        print(f"\n== per-scenario |Δmed| {metric} (dB)  --  rows=scenario, cols=run ==\n")
        print(_render_per_scenario(rows, lambda r: r.per_scen_dmed, " dB"))
        print(f"\n== per-scenario close (rab1↔rab2) |Δmed| {metric} (dB) ==\n")
        print(_render_per_scenario(rows, lambda r: r.per_scen_close_dmed, " dB"))
        print(f"\n== per-scenario far (rab*↔rab3) |Δmed| {metric} (dB) ==\n")
        print(_render_per_scenario(rows, lambda r: r.per_scen_far_dmed, " dB"))
        print(f"\n== per-scenario K-S {metric} ==\n")
        print(_render_per_scenario(rows, lambda r: r.per_scen_ks, "", digits=3))

    print("\nrun-label legend:")
    for r in rows:
        print(f"  {r.label}  =  {r.run_dir}")

    if args.csv:
        out = Path(args.csv).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        flat = [{k: v for k, v in r.__dict__.items() if not k.startswith("per_scen_")}
                for r in rows]
        pd.DataFrame(flat).to_csv(out, index=False)
        print(f"\nwrote {out}")

    if args.per_scenario_csv:
        out = Path(args.per_scenario_csv).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        long_rows = []
        for r in rows:
            for scen in sorted(set(r.per_scen_dmed) | set(r.per_scen_ks)):
                long_rows.append({
                    "run_label":      r.label,
                    "run_dir":        r.run_dir,
                    "channel":        r.channel,
                    "scenario":       scen,
                    "metric":         metric,
                    "abs_dmed":       r.per_scen_dmed.get(scen,       float("nan")),
                    "abs_dmed_close": r.per_scen_close_dmed.get(scen, float("nan")),
                    "abs_dmed_far":   r.per_scen_far_dmed.get(scen,   float("nan")),
                    "ks_statistic":   r.per_scen_ks.get(scen,         float("nan")),
                })
        pd.DataFrame(long_rows).to_csv(out, index=False)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())