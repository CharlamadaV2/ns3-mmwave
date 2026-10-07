"""Normalized snapshot construction and numeric comparison; standard library only."""
# regression_snapshot.py
# Library used by regression_check.py and smoke_check.py. A "snapshot" is a JSON-able
# dict holding one simulator run's per-seed CSV tables (parsed to numbers), its
# summary.json without wall-clock fields, and SHA-256 digests of the scenario input
# files. This file has no CLI.

import csv
import hashlib
import json
import math
from pathlib import Path

SNAPSHOT_VERSION = 1

# Stable per-seed tables covered by the snapshot. links.csv is required.
TABLE_FILES = [
    "positions.csv",
    "links.csv",
    "rx-power.csv",
    "mcs.csv",
    "flows.csv",
    "routes.csv",
]
REQUIRED_TABLES = ["links.csv"]

# summary.json keys dropped because they are wall-clock, not simulation values.
_VOLATILE_SUMMARY_PREFIX = "wall_"


## @fn rel_to_root
# @brief Express a path relative to the mesh-sim root.
#
# @param path  Any path.
# @param root  Resolved mesh-sim root directory.
# @return POSIX-style relative string, or the absolute POSIX path if `path` is outside `root`.
def rel_to_root(path: Path, root: Path) -> str:
    """Path relative to the mesh-sim root with posix separators, if possible."""
    path = Path(path).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


## @fn normalize_value
# @brief Parse one CSV cell into a comparable value.
#
# @param raw  Raw cell text.
# @return A float when the text parses as one; the strings "nan", "inf" or "-inf"
#         for those special values; otherwise the original string unchanged.
def normalize_value(raw: str):
    """Parse a CSV cell as a float when possible, otherwise keep the string."""
    text = raw.strip()
    try:
        num = float(text)
    except ValueError:
        return raw
    if math.isnan(num):
        return "nan"
    if math.isinf(num):
        return "inf" if num > 0 else "-inf"
    return num


## @fn load_table
# @brief Read one simulator CSV into a plain dict.
#
# @param path  CSV file. Leading lines whose first cell starts with `#` are skipped.
# @return `{"columns": [names], "rows": [{column: value}, ...]}`; both lists are
#         empty if the file has no header. Cells go through `normalize_value`;
#         a row shorter than the header gets `None` for the missing cells; blank
#         rows are dropped.
# @throws OSError if the file cannot be opened.
def load_table(path: Path) -> dict:
    """Read one CSV into {"columns": [...], "rows": [{col: value}, ...]}."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            columns = next(reader)
            while columns and columns[0].startswith("#"):
                columns = next(reader)
        except StopIteration:
            return {"columns": [], "rows": []}
        columns = [c.strip() for c in columns]
        rows = []
        for fields in reader:
            if not fields:
                continue
            row = {}
            for idx, col in enumerate(columns):
                row[col] = normalize_value(fields[idx]) if idx < len(fields) else None
            rows.append(row)
    return {"columns": columns, "rows": rows}


## @fn normalize_summary
# @brief Remove volatile fields from a parsed `summary.json`.
#
# @param summary  Parsed summary dict.
# @return New dict without top-level keys that start with `wall_` (wall-clock timings).
def normalize_summary(summary: dict) -> dict:
    """Drop wall-clock fields from summary.json; keep every other key."""
    return {
        key: value
        for key, value in summary.items()
        if not key.startswith(_VOLATILE_SUMMARY_PREFIX)
    }


## @fn sha256_of
# @brief SHA-256 digest of a file.
#
# @param path  File to hash (read in 64 KiB chunks).
# @return Lowercase hex digest string.
# @throws OSError if the file cannot be read.
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()

## @fn source_digests
# @brief Digest the scenario input files that produced a run.
#
# @param paths  Iterable of file paths; entries that are not files are skipped.
# @param root   mesh-sim root used to make the recorded paths relative.
# @return List of `{"path", "sha256"}` dicts sorted by path.
def source_digests(paths, root: Path) -> list[dict]:
    entries = [
        {"path": rel_to_root(p, root), "sha256": sha256_of(p)}
        for p in paths
        if Path(p).is_file()
    ]
    entries.sort(key=lambda e: e["path"])
    return entries


# ---------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------


## @fn build_snapshot
# @brief Build the normalized snapshot of one completed run.
#
# @param run_dir   Run output directory containing `seed-<seed>/`.
# @param sources   Scenario input files to digest (may be empty).
# @param root      mesh-sim root, for relative source paths.
# @param family    Case family label stored in the snapshot (for example "baseline").
# @param case      Case name stored in the snapshot.
# @param seed      Integer seed; selects the `seed-<seed>` subdirectory.
# @param band      Radio band label stored in the snapshot ("mmwave" or "sub-6").
# @return Snapshot dict with keys `snapshot_version`, `family`, `case`, `seed`,
#         `band`, `sources`, `seed_dir`, `tables`, `tables_missing`, `summary`.
# @throws FileNotFoundError if `summary.json` or the required `links.csv` is absent.
#
# Loads every file in `TABLE_FILES` that exists (optional ones that are absent are
# listed in `tables_missing`) and the normalized summary. Nothing is written.
def build_snapshot(
    run_dir,
    sources,
    root: Path,
    family: str,
    case: str,
    seed: int,
    band: str,
) -> dict:
    """Build the normalized snapshot for one completed run directory."""
    run_dir = Path(run_dir).resolve()
    seed_dir_name = "seed-%d" % seed
    seed_dir = run_dir / seed_dir_name

    summary_path = seed_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Missing required %s" % summary_path)

    tables = {}
    tables_missing = []
    for name in TABLE_FILES:
        path = seed_dir / name
        if path.is_file():
            tables[name] = load_table(path)
        elif name in REQUIRED_TABLES:
            raise FileNotFoundError("Missing required %s" % path)
        else:
            tables_missing.append(name)

    with open(summary_path, encoding="utf-8") as handle:
        summary = json.load(handle)

    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "family": family,
        "case": case,
        "seed": int(seed),
        "band": band,
        "sources": source_digests(sources, root),
        "seed_dir": seed_dir_name,
        "tables": tables,
        "tables_missing": tables_missing,
        "summary": normalize_summary(summary),
    }


## @fn serialize_snapshot
# @brief Serialize a snapshot to deterministic JSON text (2-space indent, sorted keys).
#
# @param snapshot  Snapshot dict from `build_snapshot`.
# @return JSON string without a trailing newline.
def serialize_snapshot(snapshot: dict) -> str:
    return json.dumps(snapshot, indent=2, sort_keys=True)


## @fn check_snapshot_clean
# @brief Reject snapshot text that would leak machine-specific or volatile data.
#
# @param text  Serialized snapshot.
# @param root  mesh-sim root whose absolute path must not appear in `text`.
# @return None.
# @throws ValueError listing what was found: the absolute root path, the text
#         `wall_`, or the names `run.log` / `console.log`.
def check_snapshot_clean(text: str, root: Path) -> None:
    """Refuse snapshots leaking absolute paths, wall-clock fields, or raw logs."""
    problems = []
    if str(root) in text:
        problems.append("absolute mesh-sim root path")
    if _VOLATILE_SUMMARY_PREFIX in text:
        problems.append("wall_ field")
    for name in ("run.log", "console.log"):
        if name in text:
            problems.append(name)
    if problems:
        raise ValueError("Snapshot rejected, contains: " + ", ".join(problems))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


## @brief Compare two scalars; numbers use absolute tolerance `atol`, NaN equals NaN.
def _values_equal(a, b, atol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    a_num = isinstance(a, (int, float))
    b_num = isinstance(b, (int, float))
    if a_num != b_num:
        return False
    if a_num:
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= atol
    return a == b


## @brief Recursively compare dicts/lists/scalars, recording up to `max_diffs` differences.
#
# Keys missing on one side and list-length changes count as one difference each.
def _compare_any(where, base, cand, atol, diffs, max_diffs) -> int:
    """Recursively compare two JSON-ish values; returns the difference count."""
    if isinstance(base, dict) and isinstance(cand, dict):
        count = 0
        for key in sorted(set(base) | set(cand)):
            if key not in base:
                count += _record(diffs, max_diffs, "%s.%s" % (where, key), "<absent>", cand[key])
            elif key not in cand:
                count += _record(diffs, max_diffs, "%s.%s" % (where, key), base[key], "<absent>")
            else:
                count += _compare_any(
                    "%s.%s" % (where, key), base[key], cand[key], atol, diffs, max_diffs
                )
        return count
    if isinstance(base, list) and isinstance(cand, list):
        if len(base) != len(cand):
            return _record(diffs, max_diffs, "%s.length" % where, len(base), len(cand))
        count = 0
        for idx, (b, c) in enumerate(zip(base, cand)):
            count += _compare_any("%s[%d]" % (where, idx), b, c, atol, diffs, max_diffs)
        return count
    if _values_equal(base, cand, atol):
        return 0
    return _record(diffs, max_diffs, where, base, cand)


## @brief Append one difference to `diffs` if there is room; always returns 1.
def _record(diffs, max_diffs, where, baseline, candidate) -> int:
    if len(diffs) < max_diffs:
        diffs.append({"where": where, "baseline": baseline, "candidate": candidate})
    return 1


## @brief Compare two loaded tables (columns, row count, then cells of the baseline's columns).
#
# @return Number of differences found (recording is capped by `max_diffs`).
def _compare_table(name, base, cand, atol, diffs, max_diffs) -> int:
    count = 0
    if base.get("columns") != cand.get("columns"):
        count += _record(
            diffs, max_diffs, "%s.columns" % name, base.get("columns"), cand.get("columns")
        )
    base_rows = base.get("rows", [])
    cand_rows = cand.get("rows", [])
    if len(base_rows) != len(cand_rows):
        count += _record(
            diffs, max_diffs, "%s.row_count" % name, len(base_rows), len(cand_rows)
        )
    local = []
    for idx, (b_row, c_row) in enumerate(zip(base_rows, cand_rows)):
        for col in base.get("columns", []):
            where = "%s[row %d].%s" % (name, idx, col)
            count += _compare_any(where, b_row.get(col), c_row.get(col), atol, local, max_diffs)
    diffs.extend(local[: max(0, max_diffs - len(diffs))])
    return count


## @fn compare_snapshots
# @brief Compare two snapshots table by table and the summary.
#
# @param baseline   Reference snapshot.
# @param candidate  Snapshot under test.
# @param atol       Absolute numeric tolerance (default 1e-9).
# @param max_diffs  Maximum number of differences recorded in the report (default 20).
# @return Dict `{"match": bool, "differences": [{where, baseline, candidate}],
#         "counts": {<table or "summary.json">: n, "total": n}}`. `match` is True
#         only when the total difference count is 0.
#
# Tables present on only one side count as one difference. Source digests are
# not compared here; see `compare_source_digests`.
def compare_snapshots(baseline: dict, candidate: dict, atol: float = 1e-9,
                      max_diffs: int = 20) -> dict:
    """Compare two normalized snapshots; returns a report dict."""
    diffs = []
    counts = {}

    base_tables = baseline.get("tables", {})
    cand_tables = candidate.get("tables", {})
    for name in sorted(set(base_tables) | set(cand_tables)):
        if name not in cand_tables:
            counts[name] = _record(diffs, max_diffs, "%s" % name, "<present>", "<absent>")
            continue
        if name not in base_tables:
            counts[name] = _record(diffs, max_diffs, "%s" % name, "<absent>", "<present>")
            continue
        table_diffs = []
        counts[name] = _compare_table(
            name, base_tables[name], cand_tables[name], atol, table_diffs, max_diffs
        )
        diffs.extend(table_diffs[: max(0, max_diffs - len(diffs))])

    summary_diffs = []
    counts["summary.json"] = _compare_any(
        "summary", baseline.get("summary", {}), candidate.get("summary", {}),
        atol, summary_diffs, max_diffs
    )
    diffs.extend(summary_diffs[: max(0, max_diffs - len(diffs))])

    total = sum(counts.values())
    counts["total"] = total
    return {"match": total == 0, "differences": diffs, "counts": counts}


## @fn compare_source_digests
# @brief Re-hash each input file recorded in a snapshot against the current checkout.
#
# @param baseline  Snapshot dict with a `sources` list.
# @param root      mesh-sim root that the recorded paths are relative to.
# @return List of `{"path", "baseline", "candidate", "match"}`; `candidate` is None
#         when the file no longer exists.
def compare_source_digests(baseline: dict, root: Path) -> list[dict]:
    """Recompute each recorded source digest against the current checkout."""
    results = []
    for entry in baseline.get("sources", []):
        path = root / entry["path"]
        current = sha256_of(path) if path.is_file() else None
        results.append(
            {
                "path": entry["path"],
                "baseline": entry["sha256"],
                "candidate": current,
                "match": current == entry["sha256"],
            }
        )
    return results


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
