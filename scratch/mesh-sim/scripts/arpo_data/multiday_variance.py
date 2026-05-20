## @file multiday_variance.py
# @brief Day-vs-day variance table from arpo_data's pairwise K-S CSV.
# 
# A command line interface that creates day-vs-day variance.
# The command line interface can take parameters change the metric, 
# file paths, or to only show specific families

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KS_CSV = (REPO_ROOT / "data" / "arpo_extracted" / "_plots"
                  / "multi_day" / "_pairwise_ks.csv")

_UNIT_BY_METRIC = {"snr": "dB", "rcpi": "dB", "mcs": "", "per": "", "throughput": "Mbps"}

## @brief Reformats the date of the scenario to month/day/year
# @param date   The given day/month/year of the scenario
# @return string    The scenarios date formatted in month/day/year
def _fmt_day(day: str) -> str:
    if isinstance(day, str) and len(day) == 8 and day.isdigit():
        return f"{day[0:2]}/{day[2:4]}/{day[4:8]}"
    return str(day)

## @brief Joins header and each row in body together
# @param header     Row of column headers to be linked to body
# @param body   Rows of data to be linked together
# @return string    Result of the assembled table
def _render_table(header: tuple[str, ...], body: list[tuple[str, ...]]) -> str:
    if not body:
        return "  ".join(header) + "\n(no rows)"
    widths = [max(len(h), *(len(row[i]) for row in body)) for i, h in enumerate(header)]
    sep = "  ".join("-" * w for w in widths)
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths)), sep]
    for row in body:
        lines.append("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))
    return "\n".join(lines)

## @brief One row per (family, link, day-pair), sorted by |Δmed| desc.##
# @param df dataframe of scenario of the two compared days
# @param metric unit of the scenarios data
# @return _render_table Result of rendered table with params from formatted header and body
def _summary_table(df: pd.DataFrame, unit: str) -> str:
    unit_suffix = f" {unit}" if unit else ""
    df = df.assign(abs_delta=df["median_delta"].abs()) \
           .sort_values("abs_delta", ascending=False)
    header = ("family", "link", "day_a → day_b", "n_a", "n_b",
              "med_a", "med_b", "Δmed (b−a)", "|Δmed|", "K-S")
    body = []
    for _, r in df.iterrows():
        body.append((
            str(r["family"]),
            f"{r['src_rab']} ↔ {r['peer_rab']}",
            f"{_fmt_day(r['day_a'])} → {_fmt_day(r['day_b'])}",
            f"{int(r['n_a']):,}",
            f"{int(r['n_b']):,}",
            f"{float(r['median_a']):.2f}{unit_suffix}",
            f"{float(r['median_b']):.2f}{unit_suffix}",
            f"{float(r['median_delta']):+.2f}{unit_suffix}",
            f"{float(r['abs_delta']):.2f}{unit_suffix}",
            f"{float(r['ks_statistic']):.3f}",
        ))
    return _render_table(header, body)

## @brief per-link summary across families: how unstable is each link, on average?
# @param df dataframe of scenario of the two compared days
# @param metric unit of the scenarios data
# @return _render_table Result of rendered table with params from formatted header and body
def _link_rollup(df: pd.DataFrame, unit: str) -> str:
    unit_suffix = f" {unit}" if unit else ""
    grp = df.assign(abs_delta=df["median_delta"].abs()).groupby(
        ["src_rab", "peer_rab"], dropna=False
    )
    header = ("link", "n family-pairs", "mean |Δmed|", "max |Δmed|", "mean K-S")
    body = []
    rows = []
    for (s, p), g in grp:
        rows.append((
            f"{s} ↔ {p}",
            len(g),
            float(g["abs_delta"].mean()),
            float(g["abs_delta"].max()),
            float(g["ks_statistic"].mean()),
        ))
    rows.sort(key=lambda r: -r[2])  # by mean |Δmed| desc
    for link, n, mean_d, max_d, mean_ks in rows:
        body.append((
            link, str(n),
            f"{mean_d:.2f}{unit_suffix}",
            f"{max_d:.2f}{unit_suffix}",
            f"{mean_ks:.3f}",
        ))
    return _render_table(header, body)

## @brief Per-family summary: which scenarios show the biggest day-to-day swings?
# @param df dataframe of scenario of the two compared days
# @param metric unit of the scenarios data
# @return _render_table Result of rendered table with params from formatted header and body
def _family_rollup(df: pd.DataFrame, unit: str) -> str:
    unit_suffix = f" {unit}" if unit else ""
    grp = df.assign(abs_delta=df["median_delta"].abs()).groupby("family", dropna=False)
    header = ("family", "n link-pairs", "mean |Δmed|", "max |Δmed|", "worst link")
    body = []
    rows = []
    for fam, g in grp:
        worst = g.loc[g["abs_delta"].idxmax()]
        rows.append((
            str(fam),
            len(g),
            float(g["abs_delta"].mean()),
            float(g["abs_delta"].max()),
            f"{worst['src_rab']} ↔ {worst['peer_rab']}",
        ))
    rows.sort(key=lambda r: -r[2])
    for fam, n, mean_d, max_d, worst in rows:
        body.append((
            fam, str(n),
            f"{mean_d:.2f}{unit_suffix}",
            f"{max_d:.2f}{unit_suffix}",
            worst,
        ))
    return _render_table(header, body)

## @brief Manages command line interface parameters for creating tables
# @param argv List of parameters to be taken from the command line
# @return On success returns 0, 1 if df is empty or csv is not found
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Day-vs-day variance table from arpo_data's pairwise K-S CSV.")
    p.add_argument("--ks-csv", default=str(DEFAULT_KS_CSV),
                   help=f"path to _pairwise_ks.csv (default: {DEFAULT_KS_CSV})")
    p.add_argument("--metric", default="snr",
                   choices=("snr", "rcpi", "mcs", "per", "throughput"),
                   help="metric to filter on (default: snr)")
    p.add_argument("--family", default=None,
                   help="restrict to one family (e.g. 1-1_static)")
    p.add_argument("--top", type=int, default=None,
                   help="show only the top-N worst rows in the detail table")
    p.add_argument("--csv", default=None,
                   help="optional path to write the filtered+sorted table as CSV")
    args = p.parse_args(argv)

    ks_csv = Path(args.ks_csv).resolve()
    if not ks_csv.is_file():
        print(f"pairwise K-S CSV not found: {ks_csv}", file=sys.stderr)
        print("  (run `python -m scripts.arpo_data.cli multi-day` to generate it)",
              file=sys.stderr)
        return 1

    df = pd.read_csv(ks_csv, dtype={"day_a": str, "day_b": str})
    df["day_a"] = df["day_a"].str.zfill(8)
    df["day_b"] = df["day_b"].str.zfill(8)
    df = df[df["metric"] == args.metric].copy()
    if args.family:
        df = df[df["family"] == args.family]
    if df.empty:
        print(f"no rows after filtering (metric={args.metric}, family={args.family})",
              file=sys.stderr)
        return 1

    unit = _UNIT_BY_METRIC.get(args.metric, "")

    print(f"== multiday variance  (metric = {args.metric}"
          + (f", family = {args.family}" if args.family else "")
          + f", source = {ks_csv.relative_to(REPO_ROOT)}) ==\n")

    print("-- per-family rollup (which scenarios swing most day-to-day) --\n")
    print(_family_rollup(df, unit))
    print("\n-- per-link rollup (which links are most unstable across families) --\n")
    print(_link_rollup(df, unit))

    print("\n-- detail: every (family, link, day-pair), sorted by |Δmed| desc --\n")
    detail = df.assign(abs_delta=df["median_delta"].abs()) \
               .sort_values("abs_delta", ascending=False)
    if args.top is not None:
        detail = detail.head(args.top)
    print(_summary_table(detail.drop(columns="abs_delta"), unit))

    #Turns table in csv format 
    if args.csv:
        out = Path(args.csv).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        df_out = df.assign(abs_delta=df["median_delta"].abs()) \
                   .sort_values("abs_delta", ascending=False)
        df_out.to_csv(out, index=False)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
