"""Reference-copy hashes, adapted-code and channel-runtime identity, and fingerprint v2."""

import copy
import re
import shutil
from pathlib import Path

import pytest

from scripts.baselines import artifacts

MESH_ROOT = Path(__file__).resolve().parents[3]
REFERENCE = MESH_ROOT / "third_party" / "arpo_placement"
BASELINES = MESH_ROOT / "scripts" / "baselines"
_ROW = re.compile(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$")


def _provenance_hashes() -> dict:
    text = (REFERENCE / "PROVENANCE.md").read_text(encoding="utf-8")
    return dict(m.groups() for m in map(_ROW.match, text.splitlines()) if m)


def test_reference_files_match_provenance():
    listed = _provenance_hashes()
    assert len(listed) == 9
    on_disk = sorted(str(p.relative_to(REFERENCE)) for p in REFERENCE.rglob("*.py"))
    assert on_disk == sorted(listed)
    for name, digest in listed.items():
        assert artifacts.sha256_file(REFERENCE / name) == digest, name


def test_reference_copy_is_not_imported_at_runtime():
    for path in BASELINES.rglob("*.py"):
        if "tests" in path.relative_to(BASELINES).parts:
            continue
        assert "arpo_placement" not in path.read_text(encoding="utf-8"), path


def _package_copy(tmp_path) -> Path:
    root = tmp_path / "pkg"
    for name in artifacts.PLANNER_CODE_FILES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(f"# {name}\n")
    return root


def test_planner_code_identity_covers_every_adapted_module(tmp_path):
    assert artifacts.PLANNER_CODE_FILES == (
        "solver.py", "config.py", "adapter.py", "effective_inputs.py", "artifacts.py",
        "planners/objective.py", "planners/geometric.py", "planners/optimization.py",
        "planners/channel.py")
    real = artifacts.planner_code_identity()
    assert list(real["files"]) == list(artifacts.PLANNER_CODE_FILES)
    assert real["files"]["artifacts.py"] == artifacts.sha256_file(
        BASELINES / "artifacts.py")
    root = _package_copy(tmp_path)
    before = artifacts.planner_code_identity(root)["aggregate_sha256"]
    assert artifacts.planner_code_identity(root)["aggregate_sha256"] == before
    for name in artifacts.PLANNER_CODE_FILES:
        original = (root / name).read_text()
        (root / name).write_text(original + "# changed\n")
        assert artifacts.planner_code_identity(root)["aggregate_sha256"] != before, name
        (root / name).write_text(original)
    (root / "planners/geometric.py").unlink()
    with pytest.raises(FileNotFoundError, match="planners/geometric.py"):
        artifacts.planner_code_identity(root)


def _fake_runtime(root: Path, monkeypatch, modules=artifacts.CHANNEL_MODULES) -> Path:
    root.mkdir(parents=True)
    binary = root / "ns3.42-sim-debug"
    binary.write_bytes(b"binary")
    libs = []
    for module in (*modules, "lte"):
        lib = root / f"libns3.42-{module}-debug.dylib"
        lib.write_bytes(f"lib {module}".encode())
        libs.append(lib)
    monkeypatch.setattr(artifacts, "linked_libraries",
                        lambda path: [root / lib.name for lib in libs])
    return binary


def test_channel_runtime_identity(tmp_path, monkeypatch):
    script = tmp_path / "script-bin"
    script.write_text("#!/bin/sh\n")
    assert artifacts.linked_libraries(script) == []
    alone = artifacts.channel_runtime_identity(script)
    assert alone["files"] == [str(script.resolve())]

    binary = _fake_runtime(tmp_path / "a", monkeypatch)
    first = artifacts.channel_runtime_identity(binary)
    assert [Path(f).name for f in first["files"]] == [
        "ns3.42-sim-debug", *(f"libns3.42-{m}-debug.dylib" for m in artifacts.CHANNEL_MODULES)]
    moved = tmp_path / "b"
    shutil.copytree(tmp_path / "a", moved)
    monkeypatch.setattr(artifacts, "linked_libraries",
                        lambda path: sorted(moved.glob("libns3*")))
    assert artifacts.channel_runtime_identity(moved / binary.name)["sha256"] == first["sha256"]
    (moved / "libns3.42-lte-debug.dylib").write_bytes(b"unrelated module rebuilt")
    assert artifacts.channel_runtime_identity(moved / binary.name)["sha256"] == first["sha256"]
    (moved / "libns3.42-propagation-debug.dylib").write_bytes(b"rebuilt")
    assert artifacts.channel_runtime_identity(moved / binary.name)["sha256"] != first["sha256"]


def test_channel_runtime_refuses_a_missing_channel_library(tmp_path, monkeypatch):
    binary = _fake_runtime(tmp_path / "a", monkeypatch, modules=("core", "network"))
    with pytest.raises(artifacts.RuntimeIdentityError, match="buildings, mobility, propagation"):
        artifacts.channel_runtime_identity(binary)


BASE_MANIFEST = {
    "method": "optimization", "objective": "balanced", "planner_seed": 7,
    "max_iterations": 60, "waypoint_policy": "reject", "mapping_sha256": None,
    "penalties": {"aerial": {"fixed_cost_m2": 150000.0, "cost_m2_per_m": 500.0,
                             "max_displacement_m": None},
                  "ground": {"fixed_cost_m2": 150000.0, "cost_m2_per_m": 100.0,
                             "max_displacement_m": None},
                  "aoi_m2": 160000.0, "fixed_cost_over_aoi": {"aerial": 0.9375,
                                                              "ground": 0.9375}},
    "channel_scoring": {
        "contract": "mesh_channel_query_v1", "isolation": "fork_per_layout",
        "planning_seed": 1, "planning_run_id": 1, "jammer_seed": 1, "mode_flag": "--rl-mode",
        "band": "mmwave", "band_source": "default", "sinr_threshold_db": -6.7,
        "coverage_sinr_db": -6.7,
        "probe": {"height_m": 1.5, "rx_gain_dbi": 12.0, "grid_cells": 400,
                  "min_resolution_m": 5.0, "cell_m": 20.0, "count": 400},
        "candidate_grid": {"cells": 400, "min_resolution_m": 5.0, "cell_m": 20.0,
                           "count": 400},
        "queries": {"requests": 3, "layouts": 70, "links": 1050, "probe_links": 1000,
                    "wall_s": 4.2}},
    "effective_scenario_identity": {"run_config": "effective-inputs/run.ini",
                                    "run_ini_sha256": "a" * 64, "nodes_json_sha256": "b" * 64,
                                    "buildings_json_sha256": None,
                                    "jammers_json_sha256": None},
    "sim_binary_sha256": "c" * 64, "channel_runtime_sha256": "d" * 64,
    "channel_runtime_files": ["/build/sim", "/build/lib/libns3.42-core-debug.dylib"],
    "planner_code_sha256": "e" * 64,
    "planner_settings": {"t0_area_frac": 0.02, "move_weights": {"nudge": 0.647}},
    "ownership": {"mode": "evaluation", "movable_resolved": ["a", "b"],
                  "controlled_resolved": ["a", "b"]},
    "planner_wall_s": 12.5, "started_at": "t0", "ended_at": "t1", "run_id": "uuid",
    "source_run_config_abs": "/abs/run.ini", "plan": "effective-inputs/baseline-plan.json",
    "eval_manifest": "../../eval_manifest.json",
}


def _set(manifest: dict, path: str, value) -> dict:
    changed = copy.deepcopy(manifest)
    *parents, leaf = path.split(".")
    target = changed
    for key in parents:
        target = target[key]
    target[leaf] = value
    return changed


@pytest.mark.parametrize("path,value", [
    ("sim_binary_sha256", "f" * 64),
    ("channel_runtime_sha256", "f" * 64),
    ("planner_code_sha256", "f" * 64),
    ("planner_settings.t0_area_frac", 0.03),
    ("ownership.movable_resolved", ["a"]),
    ("ownership.controlled_resolved", ["b", "a"]),
    ("penalties.ground.cost_m2_per_m", 0.0),
    ("penalties.aerial.max_displacement_m", 50.0),
    ("channel_scoring.probe.rx_gain_dbi", 3.0),
    ("channel_scoring.probe.height_m", 2.0),
    ("channel_scoring.candidate_grid.cells", 100),
    ("channel_scoring.coverage_sinr_db", 0.0),
    ("channel_scoring.planning_seed", 2),
    ("channel_scoring.planning_run_id", 2),
    ("channel_scoring.mode_flag", None),
    ("effective_scenario_identity.nodes_json_sha256", "0" * 64),
    ("planner_seed", 8),
    ("mapping_sha256", "1" * 64),
])
def test_fingerprint_changes_with_plan_identity(path, value):
    assert artifacts.fingerprint(_set(BASE_MANIFEST, path, value)) != artifacts.fingerprint(
        BASE_MANIFEST)


@pytest.mark.parametrize("path,value", [
    ("planner_wall_s", 99.0),
    ("channel_scoring.queries", {"requests": 9, "layouts": 1, "links": 1,
                                 "probe_links": 1, "wall_s": 0.1}),
    ("started_at", "t9"), ("ended_at", None), ("run_id", "other"),
    ("source_run_config_abs", "/elsewhere/run.ini"),
    ("channel_runtime_files", ["/moved/sim"]),
    ("effective_scenario_identity.run_config", "x/run.ini"),
    ("plan", "moved/plan.json"), ("eval_manifest", None),
    ("penalties.fixed_cost_over_aoi", {"aerial": 0.0, "ground": 0.0}),
])
def test_fingerprint_ignores_timing_and_paths(path, value):
    assert artifacts.fingerprint(_set(BASE_MANIFEST, path, value)) == artifacts.fingerprint(
        BASE_MANIFEST)
