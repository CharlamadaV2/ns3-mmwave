"""Placement methods as `scripts.rl.evaluate` policies: preparation order, failure, metadata.

Contract: scripts/baselines/README.md "Commands" (evaluation: every requested placement
method is prepared before any episode; a preparation failure exits 1, runs no episode
and writes no eval_manifest.json; the policy name is the method and [baseline] algorithm
is only `requested_algorithm`), "Channel scoring" (the query gets --rl-mode in
evaluation), "Outputs" (<eval-out>/<method>/baseline/ without sim.log or seed-N/;
policies[<method>].baseline fields; eval_manifest_version stays 2) and the manifest
`status` row (evaluation stops at prepared). Episodes run on scripts/rl/tests/fake_sim.py;
channel queries on fake_query.py, through one composite executable.
"""

import json
import sys
from pathlib import Path

import pytest

from scripts.baselines import adapter, artifacts
from scripts.baselines.tests.conftest import FAKE_QUERY, install_stub_planner, write_scenario
from scripts.rl import evaluate as evaluate_cli
from scripts.rl.policy import evaluate as policy_evaluate
from scripts.rl.tests.conftest import FAKE_SIM

SMALL_GRID = {"candidate_grid_cells": "16", "coverage_grid_cells": "16",
              "max_iterations": "10", "aerial_fixed_cost_m2": "0",
              "ground_fixed_cost_m2": "0", "aerial_cost_m2_per_m": "0",
              "ground_cost_m2_per_m": "0"}
BASELINE_BLOCK_KEYS = {"baseline_manifest_version", "method", "requested_algorithm",
                       "objective", "executor", "planner_seed", "max_iterations",
                       "planning_seed", "ownership", "fingerprint",
                       "initial_displacement_m_total", "effective_scenario_identity",
                       "mapping_sha256", "manifest", "plan"}


def _composite(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "mesh-sim-composite"
    path.write_text(
        f"#!{sys.executable}\nimport runpy, sys\n"
        f"target = {str(FAKE_QUERY)!r} if '--channel-query' in sys.argv[1:] "
        f"else {str(FAKE_SIM)!r}\nsys.argv[0] = target\n"
        "runpy.run_path(target, run_name='__main__')\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def composite(tmp_path) -> Path:
    return _composite(tmp_path / "bin")


def _scenario(tmp_path: Path, controlled: str = "uav-a, uav-c", **overrides) -> Path:
    ini = write_scenario(tmp_path / "scenario", overrides={**SMALL_GRID, **overrides})
    ini.write_text(ini.read_text().replace("controlled_nodes = uav-a, uav-b, uav-c, walker",
                                           f"controlled_nodes = {controlled}"))
    return ini


def _evaluate(binary, ini, out, policies, *extra) -> int:
    return evaluate_cli.main(["--sim-binary", str(binary), "--run-config", str(ini),
                              "--output-dir", str(out), "--seeds", "1,2",
                              "--policies", policies, *extra])


def _cpp_value(text: str, section: str, key: str) -> str | None:
    current, value = "", None
    for raw in text.split("\n"):
        line = raw.split("#", 1)[0].split(";", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
        elif "=" in line and current == section and line.split("=", 1)[0].strip() == key:
            value = line.split("=", 1)[1].strip()
    return value


def test_real_planner_through_the_query_in_rl_mode(tmp_path, composite, monkeypatch):
    record = tmp_path / "query.jsonl"
    monkeypatch.setenv("FAKE_QUERY_RECORD", str(record))
    ini = _scenario(tmp_path, algorithm="none")
    out = tmp_path / "eval"
    assert _evaluate(composite, ini, out, "hold,geometric") == 0

    prep_dir = out / "geometric" / "baseline"
    argv = [json.loads(line)["argv"] for line in record.read_text().splitlines()
            if "argv" in json.loads(line)]
    assert argv == [[f"--run-config={prep_dir.resolve()}/effective-inputs.partial/run.ini",
                     "--channel-query", "--seed=1", "--rl-mode"]]

    prep = artifacts.read_json(prep_dir / "baseline_manifest.json")
    assert (prep["mode"], prep["executor"], prep["status"]) == ("evaluation", "hold",
                                                                "prepared")
    assert prep["ended_at"] is not None and prep["error"] is None
    assert (prep["requested_algorithm"], prep["method"]) == ("none", "geometric")
    assert prep["channel_scoring"]["mode_flag"] == "--rl-mode"
    assert prep["channel_scoring"]["contract"] == "mesh_channel_query_v1"
    assert prep["sim_log"] is None and prep["seeds"] == []
    assert prep["ownership"] == {"mode": "evaluation", "movable_resolved": ["uav-a", "uav-c"],
                                 "controlled_resolved": ["uav-a", "uav-c"]}
    assert prep["fingerprint"] == artifacts.fingerprint(prep)
    assert sorted(p.name for p in prep_dir.iterdir()) == [
        "baseline_manifest.json", "effective-inputs", "planner.log", "source-inputs"]
    effective = (prep_dir / "effective-inputs/run.ini").read_text()
    assert _cpp_value(effective, "rl", "enabled") == "true"
    assert _cpp_value(effective, "baseline", "algorithm") == "none"

    manifest = json.loads((out / "eval_manifest.json").read_text())
    assert manifest["eval_manifest_version"] == 2
    assert manifest["status"] == "completed"
    assert "baseline" not in manifest["policies"]["hold"]
    block = manifest["policies"]["geometric"]["baseline"]
    assert set(block) == BASELINE_BLOCK_KEYS
    assert block == artifacts.eval_metadata(prep, "geometric/baseline/baseline_manifest.json",
                                            "geometric/baseline/effective-inputs/"
                                            "baseline-plan.json")
    assert (block["method"], block["requested_algorithm"], block["executor"]) == (
        "geometric", "none", "hold")
    assert block["planning_seed"] == 1
    assert (out / block["manifest"]).is_file() and (out / block["plan"]).is_file()
    plan = artifacts.read_json(out / block["plan"])
    slots = {n["id"]: n["slot"] for n in plan["nodes"]}
    assert (slots["uav-a"], slots["uav-c"], slots["gw"]) == (0, 1, None)
    for episode in manifest["policies"]["geometric"]["episodes"]:
        assert episode["status"] == "completed"
        assert Path(episode["episode_dir"]).parent.name == "geometric"


def test_every_method_is_prepared_before_the_first_episode(tmp_path, monkeypatch):
    install_stub_planner(monkeypatch)
    events = []
    real_prepare, real_episode = adapter.prepare, policy_evaluate.run_episode

    def prepare(run_config, method, *args, **kwargs):
        events.append(("prepare", method))
        return real_prepare(run_config, method, *args, **kwargs)

    def run_episode(env, policy, seed, *args, **kwargs):
        events.append(("episode", seed))
        return real_episode(env, policy, seed, *args, **kwargs)

    monkeypatch.setattr(adapter, "prepare", prepare)
    monkeypatch.setattr(policy_evaluate, "run_episode", run_episode)
    binary = _composite(tmp_path / "bin")
    out = tmp_path / "eval"
    assert _evaluate(binary, _scenario(tmp_path), out, "geometric,hold,optimization") == 0
    assert events[:2] == [("prepare", "geometric"), ("prepare", "optimization")]
    assert [e for e in events[2:] if e[0] == "prepare"] == []
    assert len([e for e in events if e[0] == "episode"]) == 6
    manifest = json.loads((out / "eval_manifest.json").read_text())
    assert list(manifest["policies"]) == ["geometric", "hold", "optimization"]


def _assert_no_episode(out: Path) -> None:
    assert not (out / "eval_manifest.json").exists()
    assert not list(out.rglob("episode-*"))
    assert not list(out.rglob("rl_episode.json"))
    assert not (out / "hold").exists()


def test_query_crash_runs_no_episode_and_stages_nothing(tmp_path, composite, monkeypatch,
                                                         capsys):
    monkeypatch.setenv("FAKE_QUERY_FAULT", "crash")
    out = tmp_path / "eval"
    assert _evaluate(composite, _scenario(tmp_path), out, "hold,geometric") == 1
    assert "BaselinePreparationError" in capsys.readouterr().err
    _assert_no_episode(out)
    prep_dir = out / "geometric/baseline"
    assert artifacts.read_json(prep_dir / "baseline_manifest.json")["status"] == "failed"
    assert not (prep_dir / "effective-inputs").exists()
    assert not (prep_dir / "effective-inputs.partial").exists()
    assert "FAILED:" in (prep_dir / "planner.log").read_text()


def test_policy_name_is_the_method_and_ini_algorithm_is_only_recorded(tmp_path, monkeypatch):
    install_stub_planner(monkeypatch)
    out = tmp_path / "eval"
    ini = _scenario(tmp_path, algorithm="optimization")
    assert _evaluate(_composite(tmp_path / "bin"), ini, out, "geometric",
                     "--band", "sub-6") == 0
    block = json.loads((out / "eval_manifest.json").read_text())["policies"]["geometric"][
        "baseline"]
    assert (block["method"], block["requested_algorithm"]) == ("geometric", "optimization")
    assert (block["planner_seed"], block["max_iterations"]) == (None, None)
    prep = artifacts.read_json(out / block["manifest"])
    assert prep["planner_seed_status"] == "unused"
    assert (prep["channel_scoring"]["band"], prep["channel_scoring"]["band_source"]) == (
        "sub-6", "cli")
    assert not (out / "optimization").exists()
