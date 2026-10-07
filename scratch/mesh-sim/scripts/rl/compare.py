#!/usr/bin/env python3
"""Compare evaluated policies against their baselines and write the comparison outputs."""

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path

from scripts.rl.cli_common import MANIFEST_NAME, write_json
from scripts.rl.policy.compare import (CSV_COLUMNS, EVAL_MANIFEST_NAME, PRIMARY_METRIC,
                                       ComparisonError, build_comparison, episode_rows,
                                       exit_code, load_evaluations)

CSV_NAME = "episodes.csv"
COMPARISON_NAME = "comparison.json"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare mesh-sim policy evaluations")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--eval-dirs", nargs="+", default=None,
                        help="Evaluation output directories holding eval_manifest.json")
    source.add_argument("--plan", default=None,
                        help="experiment_plan.json whose 'evaluations' list is compared")
    p.add_argument("--output-dir", required=True,
                   help="Directory for episodes.csv and comparison.json")
    p.add_argument("--baselines", default=None,
                   help="Comma-separated baselines; omitted -> every non-model policy")
    p.add_argument("--json", action="store_true",
                   help="Print comparison.json instead of the summary lines")
    return p


def _plan_entries(plan_path: str) -> tuple:
    """Entries and per-label expected run counts from a plan's 'evaluations' list."""
    payload = json.loads(Path(plan_path).read_text())
    listed = payload.get("evaluations") if isinstance(payload, dict) else None
    if not isinstance(listed, list) or not listed:
        raise ComparisonError(f"{plan_path} has no 'evaluations' list")
    entries = [{"eval_dir": entry["eval_dir"], "label": entry.get("label"),
                "training_seed": entry.get("training_seed")} for entry in listed]
    expected: dict = {}
    for entry in entries:
        if entry["label"] is not None:
            expected[entry["label"]] = expected.get(entry["label"], 0) + 1
    return entries, expected


def _check_output_dir(output_dir: str) -> None:
    out = Path(output_dir)
    for name in (MANIFEST_NAME, EVAL_MANIFEST_NAME):
        if (out / name).exists():
            raise ComparisonError(f"Refusing to write into {out}: it contains {name}")


def _parse_baselines(raw: str | None) -> list | None:
    if raw is None:
        return None
    names = [token.strip() for token in raw.split(",") if token.strip()]
    if not names:
        raise ComparisonError("--baselines must list at least one policy")
    if len(set(names)) != len(names):
        raise ComparisonError(f"--baselines must be distinct, got {names}")
    return names


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _write_atomic(path: Path, text: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text)
    os.replace(temp, path)


def _write_csv(path: Path, rows: list) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        writer.writerow([_cell(row[column]) for column in CSV_COLUMNS])
    _write_atomic(path, buffer.getvalue())


def _prefix(label, training_seed, eval_dir: str) -> str:
    name = ("train-seed-" + str(training_seed) if training_seed is not None
            else Path(eval_dir).name)
    return f"{label or '(unlabeled)'} {name}"


def _line(prefix: str, width: int, comparison: dict, tail: str) -> str:
    mean = comparison["mean_difference"]
    value = "n/a" if mean is None else f"{mean:+.4f}"
    interval = comparison["interval"]
    if interval is not None:
        value = f"{value} [{interval['low']:+.4f}, {interval['high']:+.4f}]"
    return (f"{prefix.ljust(width)} vs {comparison['baseline']}  "
            f"{comparison['metric']}  {value}  {tail}")


def _evaluation_lines(block: dict, width: int) -> tuple:
    prefix = _prefix(block["label"], block["training_seed"], block["eval_dir"])
    lines, excluded = [], []
    for entry in block["comparisons"]:
        if entry["metric"] != PRIMARY_METRIC:
            continue
        if entry["interval"] is not None:
            tail = (f"paired t, {entry['n_used']}/{entry['n_expected']} "
                    "evaluation seeds, fixed model")
        elif entry["n_used"] == 1:
            tail = "single seed; no interval"
        else:
            tail = "no usable evaluation seeds"
        lines.append(_line(prefix, width, entry, tail))
        excluded += [f"excluded: {prefix} seed {item['seed']} "
                     f"({', '.join(item['reasons'])})" for item in entry["excluded"]]
    return lines, excluded


def _group_lines(group: dict, width: int) -> list:
    prefix = f"{group['label']} (group)"
    lines = []
    for entry in group["comparisons"]:
        if entry["metric"] != PRIMARY_METRIC:
            continue
        if entry["interval"] is not None:
            tail = (f"t across {group['runs_used']}/{group['runs_expected']} training "
                    f"runs, {len(entry['common_seeds'])} common held-out seeds")
        else:
            tail = f"{group['runs_used']}/{group['runs_expected']} training runs; no interval"
        lines.append(_line(prefix, width, entry, tail))
    return lines


def summary_lines(comparison: dict) -> list:
    """Primary-metric lines for every evaluation and group, then the excluded seeds."""
    prefixes = [_prefix(b["label"], b["training_seed"], b["eval_dir"])
                for b in comparison["evaluations"]]
    prefixes += [f"{g['label']} (group)" for g in comparison["groups"]]
    width = max((len(p) for p in prefixes), default=0)
    lines, excluded = [], []
    for block in comparison["evaluations"]:
        block_lines, block_excluded = _evaluation_lines(block, width)
        lines += block_lines
        excluded += [text for text in block_excluded if text not in excluded]
    for group in comparison["groups"]:
        lines += _group_lines(group, width)
    return lines + excluded


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.plan:
            entries, runs_expected = _plan_entries(args.plan)
        else:
            entries = [{"eval_dir": path} for path in args.eval_dirs]
            runs_expected = None
        _check_output_dir(args.output_dir)
        evaluations, missing = load_evaluations(entries)
        comparison = build_comparison(evaluations, missing,
                                      _parse_baselines(args.baselines), runs_expected)
    except (ComparisonError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / CSV_NAME, episode_rows(evaluations))
    write_json(out_dir / COMPARISON_NAME, comparison)

    if args.json:
        print(json.dumps(comparison, indent=2))
    else:
        for line in summary_lines(comparison):
            print(line)
    return exit_code(comparison)


if __name__ == "__main__":
    sys.exit(main())
