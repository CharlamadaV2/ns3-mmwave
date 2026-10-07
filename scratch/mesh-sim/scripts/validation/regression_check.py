#!/usr/bin/env python3
"""Capture and compare compact normalized simulator snapshots for baseline regression.

Standard-library helpers only; no RL/analysis dependencies are needed.

  python3 -m scripts.validation.regression_check capture ...
  python3 -m scripts.validation.regression_check compare ...
  python3 -m scripts.validation.regression_check verify-suite ...
"""
# regression_check.py
# CLI and suite runner for the baseline regression check. `capture` runs the simulator
# on one scenario (optionally saving a snapshot), `compare` diffs a run against a
# saved snapshot, `verify-suite` does both for every case in a manifest.
# Snapshot building and numeric comparison live in regression_snapshot.py.
# Exit codes: 0 pass, 1 mismatch/failure (EXIT_MISMATCH), 2 usage or I/O error (EXIT_USAGE).

from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.sim_support import find_mesh_root, simulator_env, strip_inline_comment, tail_lines
from .regression_snapshot import (
    SNAPSHOT_VERSION, build_snapshot, check_snapshot_clean, compare_snapshots,
    compare_source_digests, load_table, rel_to_root, serialize_snapshot, sha256_of,
)

MANIFEST_VERSION = 1

EXIT_MISMATCH = 1
EXIT_USAGE = 2


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


## @fn mesh_sim_root
# @brief Locate the mesh-sim root directory independent of the working directory.
#
# @return Path of the mesh-sim root (found from this file's location).
def mesh_sim_root() -> Path:
    """Locate mesh-sim without depending on the current working directory."""
    return find_mesh_root(__file__)


## @fn path_under_root
# @brief Resolve a manifest path and keep it inside the mesh-sim root.
#
# @param value  Path string from the manifest; must be relative.
# @param root   Resolved mesh-sim root.
# @return Absolute resolved path.
# @throws ValueError if `value` is absolute or resolves outside `root`.
def path_under_root(value: str, root: Path) -> Path:
    """Resolve a manifest path while rejecting absolute or escaping paths."""
    path = Path(value)
    if path.is_absolute():
        raise ValueError("Manifest path must be relative: %s" % value)
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Manifest path escapes mesh-sim root: %s" % value) from exc
    return resolved


## @fn suite_output_dir
# @brief Validate the `--out` directory of a suite run.
#
# @param value  Path string given to `--out`.
# @param root   Resolved mesh-sim root.
# @return Absolute resolved path beneath `<root>/outputs/`.
# @throws ValueError if the path is outside `outputs/` or is `outputs/` itself.
def suite_output_dir(value: str, root: Path) -> Path:
    """Require suite output beneath outputs/ without allowing outputs/ itself."""
    path = Path(value).resolve()
    outputs = (root / "outputs").resolve()
    try:
        path.relative_to(outputs)
    except ValueError as exc:
        raise ValueError("Suite --out must be beneath %s" % outputs) from exc
    if path == outputs:
        raise ValueError("Suite --out must be a directory beneath %s" % outputs)
    return path


## @fn prepare_suite_output
# @brief Delete products of a previous suite run, then make sure the output directory exists.
#
# @param out_root    Suite output directory.
# @param case_names  Manifest case names (each is a subdirectory of `out_root`).
# @return True if anything from an earlier run was found and removed.
#
# Removes only `suite-report.json` and the named case subdirectories; other
# siblings are untouched. Creates `out_root` if needed.
def prepare_suite_output(out_root: Path, case_names: list[str]) -> bool:
    """Remove only files owned by a previous run of this suite."""
    targets = [out_root / "suite-report.json"]
    targets.extend(out_root / name for name in case_names)
    replaced = any(path.exists() or path.is_symlink() for path in targets)
    for path in targets:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    out_root.mkdir(parents=True, exist_ok=True)
    return replaced


# ---------------------------------------------------------------------------
# Scenario sources
# ---------------------------------------------------------------------------


## @fn scenario_source_files
# @brief List the input files a scenario depends on.
#
# @param run_config  Path to the scenario's `run.ini`.
# @return `run.ini` followed by the existing files named by `nodes_file`,
#         `buildings_file` and `jammers_file` in its `[scenario]` section
#         (relative names resolve against the `run.ini` directory; blank or
#         missing files are skipped).
# @throws OSError if `run_config` cannot be read.
def scenario_source_files(run_config: Path) -> list[Path]:
    """run.ini plus the node/buildings/jammer files it references."""
    run_config = Path(run_config).resolve()
    parser = configparser.ConfigParser(
        allow_no_value=True, interpolation=None, strict=False
    )
    parser.optionxform = str
    with open(run_config, encoding="utf-8") as handle:
        parser.read_file(handle)

    sources = [run_config]
    base = run_config.parent
    for key in ("nodes_file", "buildings_file", "jammers_file"):
        raw = parser.get("scenario", key, fallback="") or ""
        name = strip_inline_comment(raw)
        if not name:
            continue
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = base / candidate
        if candidate.is_file():
            sources.append(candidate.resolve())
    return sources


## @fn cmd_capture
# @brief `capture` subcommand: run the simulator once and optionally write a snapshot.
#
# @param args  Parsed arguments: `sim_binary`, `run_config`, `band`, `seed`,
#              `family`, `case`, `out`, `snapshot`, `force`, and optional `quiet`.
# @return 0 on success; 2 (EXIT_USAGE) for a missing input or an existing output
#         without `--force`; the simulator's own non-zero exit code if it fails.
#
# Runs `<sim> --run-config=... --band=... --seed=... --output-dir=<out>` with
# stdin closed, and writes the combined stdout/stderr to `<out>/console.log`
# (last 40 lines are echoed to stderr on failure). If `--snapshot` is set, builds
# the snapshot, refuses it if `check_snapshot_clean` objects, and writes it there
# (parent directories are created). `--atol` is accepted but unused here.
def cmd_capture(args) -> int:
    root = mesh_sim_root()
    quiet = getattr(args, "quiet", False)
    run_config = Path(args.run_config).resolve()
    if not run_config.is_file():
        print("Error: run-config not found: %s" % run_config, file=sys.stderr)
        return EXIT_USAGE

    sim_binary = Path(args.sim_binary).resolve()
    if not sim_binary.is_file():
        print("Error: sim-binary not found: %s" % sim_binary, file=sys.stderr)
        return EXIT_USAGE

    out_dir = Path(args.out).resolve()
    seed_dir = out_dir / ("seed-%d" % args.seed)
    if (seed_dir / "summary.json").is_file() and not args.force:
        print(
            "Error: %s already exists. Use --force to overwrite."
            % (seed_dir / "summary.json"),
            file=sys.stderr,
        )
        return EXIT_USAGE

    snapshot_path = Path(args.snapshot).resolve() if args.snapshot else None
    if snapshot_path and snapshot_path.exists() and not args.force:
        print(
            "Error: snapshot %s already exists. Use --force to overwrite."
            % snapshot_path,
            file=sys.stderr,
        )
        return EXIT_USAGE

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(sim_binary),
        "--run-config=%s" % run_config,
        "--band=%s" % args.band,
        "--seed=%d" % args.seed,
        "--output-dir=%s" % out_dir,
    ]
    if not quiet:
        print("Command: %s" % " ".join(cmd))

    env = simulator_env(root)

    console_path = out_dir / "console.log"
    with open(console_path, "w", encoding="utf-8") as console:
        console.write("Command: %s\n\n" % " ".join(cmd))
        console.flush()
        result = subprocess.run(
            cmd, stdin=subprocess.DEVNULL, stdout=console,
            stderr=subprocess.STDOUT, env=env
        )

    if result.returncode != 0:
        print("Simulator exited %d. Last 40 lines of %s:"
              % (result.returncode, console_path), file=sys.stderr)
        for line in tail_lines(console_path).splitlines():
            print("  " + line, file=sys.stderr)
        return result.returncode

    if not quiet:
        print("Captured run: %s" % out_dir)

    if snapshot_path:
        snapshot = build_snapshot(
            out_dir,
            scenario_source_files(run_config),
            root,
            args.family,
            args.case,
            args.seed,
            args.band,
        )
        text = serialize_snapshot(snapshot)
        check_snapshot_clean(text, root)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(text + "\n", encoding="utf-8")
        if not quiet:
            print("Wrote snapshot: %s" % snapshot_path)

    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


## @fn cmd_compare
# @brief `compare` subcommand: diff a candidate run against a saved snapshot.
#
# @param args  Parsed arguments: `baseline`, `candidate`, `atol`, `max_diffs`,
#              `report`, and optional `quiet`.
# @return 0 on match; 1 (EXIT_MISMATCH) on any difference; 2 (EXIT_USAGE) if the
#         baseline or the candidate's required files are missing.
#
# Takes seed, family, case and band from the baseline, builds a snapshot of the
# candidate run directory, and compares tables and summary. It also re-hashes the
# scenario input files recorded in the baseline against the current checkout; any
# changed or missing file counts as a difference. Optionally writes a JSON report
# to `--report`. Prints `MATCH` or `MISMATCH` plus up to five differences.
def cmd_compare(args) -> int:
    root = mesh_sim_root()
    quiet = getattr(args, "quiet", False)
    baseline_path = Path(args.baseline).resolve()
    if not baseline_path.is_file():
        print("Error: baseline snapshot not found: %s" % baseline_path, file=sys.stderr)
        return EXIT_USAGE

    with open(baseline_path, encoding="utf-8") as handle:
        baseline = json.load(handle)

    candidate_dir = Path(args.candidate).resolve()
    seed = int(baseline.get("seed", 0))
    try:
        candidate = build_snapshot(
            candidate_dir,
            [],
            root,
            baseline.get("family", ""),
            baseline.get("case", ""),
            seed,
            baseline.get("band", ""),
        )
    except FileNotFoundError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return EXIT_USAGE

    report = compare_snapshots(baseline, candidate, args.atol, args.max_diffs)
    digests = compare_source_digests(baseline, root)
    digest_diffs = [d for d in digests if not d["match"]]
    if digest_diffs:
        report["match"] = False
        report["counts"]["source_digests"] = len(digest_diffs)
        report["counts"]["total"] += len(digest_diffs)
        for entry in digest_diffs:
            report["differences"].append(
                {
                    "where": "sources[%s].sha256" % entry["path"],
                    "baseline": entry["baseline"],
                    "candidate": entry["candidate"],
                }
            )

    report_body = {
        "baseline": rel_to_root(baseline_path, root),
        "candidate": args.candidate,
        "atol": args.atol,
        "match": report["match"],
        "differences": report["differences"],
        "counts": report["counts"],
        "source_digests": digests,
    }

    if args.report:
        report_file = Path(args.report).resolve()
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(report_body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not quiet:
            print("Wrote report: %s" % report_file)

    if report_body["match"]:
        if not quiet:
            print("MATCH: %s == %s (atol=%g)"
                  % (report_body["baseline"], args.candidate, args.atol))
        return 0

    if not quiet:
        print("MISMATCH: %d difference(s) (atol=%g)"
              % (report["counts"]["total"], args.atol))
        for diff in report_body["differences"][:5]:
            print("  %s: baseline=%r candidate=%r"
                  % (diff["where"], diff["baseline"], diff["candidate"]))
    return EXIT_MISMATCH


# ---------------------------------------------------------------------------
# suite verification
# ---------------------------------------------------------------------------


## @fn load_suite_manifest
# @brief Read and validate the suite manifest.
#
# @param path  Manifest JSON file.
# @param root  Resolved mesh-sim root, used to check case paths.
# @return The parsed manifest dict.
# @throws ValueError if the version is not `MANIFEST_VERSION`, `baseline` lacks
#         `git_commit` or `simulator_binary_sha256`, `cases` is empty, a case lacks a
#         required key, has an unsafe or duplicate `name`, a bad `label`,
#         `required` or `band`, a non-integer `seed`, or paths that are absolute
#         or escape the root.
# @throws OSError, json.JSONDecodeError if the file cannot be read or parsed.
def load_suite_manifest(path: Path, root: Path) -> dict:
    """Load and validate the tracked regression-suite manifest."""
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(
            "Unsupported manifest_version %r" % manifest.get("manifest_version")
        )
    baseline = manifest.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("Manifest baseline must be an object")
    for key in ("git_commit", "simulator_binary_sha256"):
        if not isinstance(baseline.get(key), str) or not baseline[key]:
            raise ValueError("Manifest baseline.%s is required" % key)

    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest cases must be a non-empty array")
    names = set()
    required = (
        "name", "label", "required", "family", "case", "seed", "band", "run_config",
        "snapshot", "snapshot_sha256",
    )
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError("Manifest cases[%d] must be an object" % index)
        missing = [key for key in required if key not in case]
        if missing:
            raise ValueError(
                "Manifest cases[%d] missing: %s" % (index, ", ".join(missing))
            )
        if (
            not isinstance(case["name"], str)
            or Path(case["name"]).name != case["name"]
            or case["name"] in (".", "..")
        ):
            raise ValueError("Manifest case name must be one safe path segment")
        if case["name"] in names:
            raise ValueError("Duplicate manifest case name: %s" % case["name"])
        names.add(case["name"])
        if not isinstance(case["label"], str) or not case["label"]:
            raise ValueError("Manifest label is required for %s" % case["name"])
        if not isinstance(case["required"], bool):
            raise ValueError("Manifest required must be boolean for %s" % case["name"])
        if case["band"] not in ("mmwave", "sub-6"):
            raise ValueError("Invalid band for %s: %s" % (case["name"], case["band"]))
        int(case["seed"])
        path_under_root(case["run_config"], root)
        path_under_root(case["snapshot"], root)
    return manifest


## @fn verify_manifest_snapshots
# @brief Check every reference snapshot before any simulator is started.
#
# @param manifest  Validated manifest from `load_suite_manifest`.
# @param root      Resolved mesh-sim root.
# @return List of `{"case", "snapshot", "snapshot_path"}` in manifest order.
# @throws ValueError if a snapshot file is not installed, its SHA-256 differs from
#         `snapshot_sha256`, its metadata (version, family, case, seed, band) differs
#         from the manifest, or it does not record the case's `run_config`.
def verify_manifest_snapshots(manifest: dict, root: Path) -> list[dict]:
    """Verify snapshot bytes and duplicated case metadata before running."""
    verified = []
    for case in manifest["cases"]:
        snapshot_path = path_under_root(case["snapshot"], root)
        if not snapshot_path.is_file():
            raise ValueError(
                "Reference snapshot not installed: %s. Ask the project team for "
                "the approved cloud baseline bundle and unpack it under "
                "tests/fixtures/regression/p0/. Reference snapshots are local-only; "
                "do not recapture them from the changed simulator."
                % case["snapshot"]
            )
        actual_sha = sha256_of(snapshot_path)
        if actual_sha != case["snapshot_sha256"]:
            raise ValueError(
                "Snapshot digest mismatch for %s: expected %s, got %s"
                % (case["name"], case["snapshot_sha256"], actual_sha)
            )
        with open(snapshot_path, encoding="utf-8") as handle:
            snapshot = json.load(handle)
        expected = {
            "snapshot_version": SNAPSHOT_VERSION,
            "family": case["family"],
            "case": case["case"],
            "seed": int(case["seed"]),
            "band": case["band"],
        }
        for key, value in expected.items():
            if snapshot.get(key) != value:
                raise ValueError(
                    "Snapshot metadata mismatch for %s.%s: expected %r, got %r"
                    % (case["name"], key, value, snapshot.get(key))
                )
        source_paths = {entry.get("path") for entry in snapshot.get("sources", [])}
        if case["run_config"] not in source_paths:
            raise ValueError(
                "Snapshot %s does not record run_config %s"
                % (case["name"], case["run_config"])
            )
        verified.append(
            {"case": case, "snapshot": snapshot, "snapshot_path": snapshot_path}
        )
    return verified


## @fn source_issues
# @brief Find scenario input files that are missing or changed since the snapshot.
#
# @param snapshot  Reference snapshot with a `sources` list.
# @param root      Resolved mesh-sim root.
# @return Tuple `(missing_paths, changed_paths)` of relative path strings.
def source_issues(snapshot: dict, root: Path) -> tuple[list[str], list[str]]:
    """Return missing and digest-mismatched scenario source paths."""
    missing = []
    changed = []
    for result in compare_source_digests(snapshot, root):
        if result["candidate"] is None:
            missing.append(result["path"])
        elif not result["match"]:
            changed.append(result["path"])
    return missing, changed


## @fn cmd_verify_suite
# @brief `verify-suite` subcommand: run and compare every case in a manifest.
#
# @param args  Parsed arguments: `manifest`, `sim_binary`, `out`, `atol`,
#              `max_diffs`, `require_all`.
# @return 0 if no case failed (skipped optional cases do not count); 1 otherwise;
#         2 if the manifest or binary file does not exist.
# @throws ValueError for an invalid manifest, snapshot or `--out` (caught in `main`,
#         which returns 2).
#
# Order: validate manifest, verify all snapshot hashes, validate `--out`, clear
# earlier suite products, then per case: if input files are missing the case is
# FAIL (required, or `--require-all`) or SKIP (optional); if inputs changed it is
# FAIL; otherwise runs `cmd_capture` into `<out>/<name>` and `cmd_compare` (report
# at `<out>/<name>/comparison.json`). Prints one row per case and writes
# `<out>/suite-report.json`.
def cmd_verify_suite(args) -> int:
    root = mesh_sim_root()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.is_file():
        print("Error: suite manifest not found: %s" % manifest_path, file=sys.stderr)
        return EXIT_USAGE

    sim_binary = Path(args.sim_binary).resolve()
    if not sim_binary.is_file():
        print("Error: sim-binary not found: %s" % sim_binary, file=sys.stderr)
        return EXIT_USAGE

    manifest = load_suite_manifest(manifest_path, root)
    verified = verify_manifest_snapshots(manifest, root)
    out_root = suite_output_dir(args.out, root)
    suite_report_path = out_root / "suite-report.json"
    replaced = prepare_suite_output(
        out_root, [item["case"]["name"] for item in verified]
    )

    baseline = manifest["baseline"]
    baseline_sha = baseline["simulator_binary_sha256"]
    current_sha = sha256_of(sim_binary)
    build = baseline.get("build", {})
    required_total = sum(1 for item in verified if item["case"]["required"])
    optional_total = len(verified) - required_total

    print("Baseline simulator regression suite")
    print("Purpose: detect unintended changes against fixed pre-change results.")
    print(
        "Reference: commit %s | %s | %s"
        % (
            baseline["git_commit"][:7],
            build.get("profile", "unknown build"),
            build.get("architecture", "unknown architecture"),
        )
    )
    print("Cases: %d required, %d optional" % (required_total, optional_total))
    relation = (
        "matches reference"
        if current_sha == baseline_sha
        else "different binary; results decide compatibility"
    )
    print("Binary: %s... (%s)\n" % (current_sha[:12], relation))
    if replaced:
        print("Previous suite output: replaced\n")

    results = []
    required_passed = 0
    optional_passed = 0
    skipped = 0
    failed = 0
    label_width = max(len(item["case"]["label"]) for item in verified)
    for index, item in enumerate(verified, start=1):
        case = item["case"]
        name = case["name"]
        case_out = out_root / name
        missing, changed = source_issues(item["snapshot"], root)
        status = None
        reason = None
        capture_exit = None
        compare_exit = None

        if missing:
            if case["required"] or args.require_all:
                status = "FAIL"
                failed += 1
                reason = "required source data is missing"
            else:
                status = "SKIP"
                skipped += 1
                reason = "optional source data is not installed"
        elif changed:
            status = "FAIL"
            failed += 1
            reason = "source data differs from the recorded baseline"

        if status is None:
            capture_args = argparse.Namespace(
                sim_binary=str(sim_binary),
                run_config=str(path_under_root(case["run_config"], root)),
                band=case["band"],
                seed=int(case["seed"]),
                family=case["family"],
                case=case["case"],
                out=str(case_out),
                snapshot=None,
                force=False,
                quiet=True,
            )
            capture_exit = cmd_capture(capture_args)
            if capture_exit == 0:
                compare_args = argparse.Namespace(
                    baseline=str(item["snapshot_path"]),
                    candidate=str(case_out),
                    atol=args.atol,
                    max_diffs=args.max_diffs,
                    report=str(case_out / "comparison.json"),
                    quiet=True,
                )
                compare_exit = cmd_compare(compare_args)

            if capture_exit == 0 and compare_exit == 0:
                status = "PASS"
                if case["required"]:
                    required_passed += 1
                else:
                    optional_passed += 1
            else:
                status = "FAIL"
                failed += 1
                reason = "simulation or comparison failed; inspect the case logs"

        display_band = "mmWave" if case["band"] == "mmwave" else "sub-6"
        details = "%s | %s | seed %s" % (
            case["family"], display_band, case["seed"]
        )
        print(
            "[%d/%d] %-*s  %-28s  %s"
            % (index, len(verified), label_width, case["label"], details, status)
        )
        if missing:
            print("      Missing data: %s" % ", ".join(missing))
            print(
                "      Action: %s"
                % case.get(
                    "missing_help",
                    "Ask the project team or data owner for the missing scenario data.",
                )
            )
        elif changed:
            print("      Changed input: %s" % ", ".join(changed))
            print("      Action: restore the recorded input or approve a new baseline.")
        elif status == "FAIL":
            print("      Details: %s" % rel_to_root(case_out, root))

        results.append(
            {
                "name": name,
                "label": case["label"],
                "required": case["required"],
                "status": status,
                "reason": reason,
                "missing_sources": missing,
                "changed_sources": changed,
                "capture_exit": capture_exit,
                "compare_exit": compare_exit,
                "output": rel_to_root(case_out, root) if status != "SKIP" else None,
                "comparison_report": (
                    rel_to_root(case_out / "comparison.json", root)
                    if compare_exit is not None else None
                ),
            }
        )
    passed = required_passed + optional_passed
    suite_status = "PASS" if failed == 0 else "FAIL"
    report = {
        "manifest_version": manifest["manifest_version"],
        "manifest": rel_to_root(manifest_path, root),
        "baseline_git_commit": baseline["git_commit"],
        "baseline_simulator_sha256": baseline_sha,
        "current_simulator_sha256": current_sha,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": len(results),
        "required_passed": required_passed,
        "required_total": required_total,
        "optional_passed": optional_passed,
        "optional_total": optional_total,
        "require_all": args.require_all,
        "status": suite_status,
        "cases": results,
    }
    suite_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "\nRESULT: %s — required %d/%d, optional %d/%d, skipped %d"
        % (
            suite_status,
            required_passed,
            required_total,
            optional_passed,
            optional_total,
            skipped,
        )
    )
    print("Detailed report: %s" % rel_to_root(suite_report_path, root))
    return 0 if suite_status == "PASS" else EXIT_MISMATCH


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


## @fn build_parser
# @brief Build the argparse parser with the `capture`, `compare` and `verify-suite` subcommands.
#
# @return Configured `argparse.ArgumentParser`. The suite's `--force` flag is hidden and unused.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regression_check",
        description="Capture and compare normalized simulator regression snapshots.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="run one scenario and optionally snapshot it")
    cap.add_argument("--sim-binary", required=True)
    cap.add_argument("--run-config", required=True)
    cap.add_argument("--band", required=True, choices=["mmwave", "sub-6"])
    cap.add_argument("--seed", required=True, type=int)
    cap.add_argument("--family", required=True)
    cap.add_argument("--case", required=True)
    cap.add_argument("--out", required=True, help="disposable run output directory")
    cap.add_argument("--snapshot", help="write the normalized snapshot here")
    cap.add_argument("--atol", type=float, default=1e-9, help="unused by capture")
    cap.add_argument("--force", action="store_true",
                     help="overwrite an existing run or snapshot")
    cap.set_defaults(func=cmd_capture)

    cmp_ = sub.add_parser("compare", help="compare a snapshot with a candidate run")
    cmp_.add_argument("--baseline", required=True, help="local reference snapshot JSON")
    cmp_.add_argument("--candidate", required=True, help="candidate run directory")
    cmp_.add_argument("--atol", type=float, default=1e-9)
    cmp_.add_argument("--max-diffs", type=int, default=20,
                      help="max recorded differences per table")
    cmp_.add_argument("--report", help="write a JSON report here")
    cmp_.set_defaults(func=cmd_compare)

    suite = sub.add_parser(
        "verify-suite", help="run and compare every case in a suite manifest"
    )
    suite.add_argument("--sim-binary", required=True)
    suite.add_argument("--manifest", required=True, help="tracked suite manifest JSON")
    suite.add_argument("--out", required=True, help="disposable suite output directory")
    suite.add_argument("--atol", type=float, default=1e-9)
    suite.add_argument("--max-diffs", type=int, default=20,
                       help="max recorded differences per table")
    suite.add_argument(
        "--require-all", action="store_true",
        help="fail when an optional local-data case is unavailable",
    )
    suite.add_argument("--force", action="store_true",
                       help=argparse.SUPPRESS)
    suite.set_defaults(func=cmd_verify_suite)

    return parser


## @fn main
# @brief CLI entry point.
#
# @param argv  Argument list; defaults to `sys.argv[1:]` when None.
# @return The subcommand's exit code; 2 (EXIT_USAGE) if it raised OSError,
#         ValueError or json.JSONDecodeError (message printed to stderr).
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
