"""Process-tree sampling, benchmark records, and the arithmetic resource estimate."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.rl import experiment
from scripts.rl.cli_common import write_json
from scripts.rl.env.config import read_scenario_identity
from scripts.rl.ops import benchmark
from scripts.sim_support import find_mesh_root

TRACKED_MATRIX = find_mesh_root() / "inputs/experiments/bypass-smoke-matrix.json"
LEAF = "row-a/train-seed-1"

# pid ppid rss: a grandchild under the measured root plus an unrelated process.
PS_TEXT = """
  100     1  20000
  200   100  30000
  300   200  40000
  400     1  99999
"""


def test_ps_text_maps_to_one_process_tree():
    table = benchmark.parse_ps(PS_TEXT)

    assert table[300] == (200, 40000)
    assert benchmark.tree_usage(table, 100) == {"rss_kb": 90000, "count": 3,
                                                "max_rss_kb": 40000}
    assert benchmark.tree_usage(table, 200) == {"rss_kb": 70000, "count": 2,
                                                "max_rss_kb": 40000}
    assert benchmark.tree_usage(table, 999) == {"rss_kb": 0, "count": 0,
                                                "max_rss_kb": 0}


def test_ps_parsing_skips_unusable_lines():
    assert benchmark.parse_ps("PID PPID RSS\n1 0\n\n7 1 12\n") == {7: (1, 12)}


def _step(kind: str, output_dir: Path) -> dict:
    manifest = "train_manifest.json" if kind == "train" else "eval_manifest.json"
    return {"id": f"{kind}/{LEAF}", "kind": kind, "module": "stub", "args": [],
            "output_dir": str(output_dir), "manifest": manifest, "needs": []}


class _FakeProcess:
    """Stands in for Popen so no step module is ever executed."""

    def __init__(self, pid: int, code: int = 0):
        self.pid = pid
        self._code = code

    def wait(self) -> int:
        return self._code


@pytest.mark.skipif(shutil.which("ps") is None, reason="ps is not on PATH")
def test_a_real_process_tree_is_measured(tmp_path):
    child = "import time; time.sleep(3)"
    parent = (f"import subprocess, sys, time;"
              f"p = subprocess.Popen([sys.executable, '-c', {child!r}]);"
              f"time.sleep(1.0); p.terminate(); p.wait()")

    def launch(step):
        return subprocess.Popen([sys.executable, "-c", parent], start_new_session=True)

    record = benchmark.measure_step(_step("train", tmp_path), 0.05, launch=launch)

    assert record["exit_code"] == 0 and record["tolerated"] is True
    assert record["wall_seconds"] >= 1.0
    assert "measurement_error" not in record
    assert record["samples"] >= 2
    assert record["peak_process_count"] >= 2
    assert record["peak_tree_rss_kb"] >= record["peak_single_process_rss_kb"] > 0


def test_a_ps_failure_nulls_memory_but_keeps_timing(tmp_path):
    def failing_ps():
        raise benchmark.PsError("ps exited 1: not supported here")

    record = benchmark.measure_step(
        _step("evaluate", tmp_path), 0.05,
        launch=lambda step: _FakeProcess(pid=1, code=2), read=failing_ps)

    assert record["exit_code"] == 2 and record["tolerated"] is True
    assert record["wall_seconds"] >= 0.0
    assert record["peak_tree_rss_kb"] is None
    assert record["peak_process_count"] is None
    assert record["peak_single_process_rss_kb"] is None
    assert record["samples"] == 0
    assert "not supported here" in record["measurement_error"]


ROUNDING = [(256, 64, 256), (250, 64, 256), (257, 64, 320), (100, 1024, 1024),
            (0, 64, None), (100, 0, None), (None, 64, None), (100, None, None)]


@pytest.mark.parametrize("requested,n_steps,expected", ROUNDING)
def test_effective_timesteps_round_up_to_whole_rollouts(requested, n_steps, expected):
    assert benchmark.effective_timesteps(requested, n_steps) == expected


HMS = [(0, "00:00:00"), (1.2, "00:00:02"), (59, "00:00:59"), (3600, "01:00:00"),
       (3661, "01:01:01"), (360000, "100:00:00")]


@pytest.mark.parametrize("seconds,expected", HMS)
def test_seconds_format_as_hms(seconds, expected):
    assert benchmark.format_hms(seconds) == expected


def _plan(root: Path) -> dict:
    steps = [_step("train", root / "train" / LEAF), _step("evaluate", root / "eval" / LEAF)]
    steps[1]["needs"] = [steps[0]["id"]]
    steps.append({"id": "compare", "kind": "compare", "module": "stub", "args":
                  ["--plan", str(root / experiment.PLAN_NAME),
                   "--output-dir", str(root / "comparison")],
                  "output_dir": str(root / "comparison"),
                  "manifest": "comparison.json", "needs": []})
    return {"experiment_plan_version": 1,
            "matrix": {"name": "bypass-smoke", "sha256": "abc"}, "steps": steps}


def _written_plan(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    write_json(root / experiment.PLAN_NAME, _plan(root))
    return root


def _train_manifest(row_name: str = "local-delivery") -> dict:
    matrix = experiment.load_matrix(TRACKED_MATRIX)
    row = next(entry for entry in matrix["rows"] if entry["name"] == row_name)
    return {"status": "completed",
            "scenario_identity": read_scenario_identity(row["run_config"]),
            "selection": {"observation_preset": row["observation_preset"],
                          "reward_components": row["reward_components"],
                          "reward_weights": row["reward_weights"],
                          "telemetry": "off", "telemetry_every": 1},
            "hyperparameters": {"total_timesteps": 256, "n_steps": 64,
                                "eval_every_steps": 128, "eval_episodes": 1}}


class _FakeLauncher:
    """Writes each step's manifest and episode records instead of running it."""

    def __init__(self, codes: dict[str, int] | None = None):
        self.codes = codes or {}
        self.launched: list[str] = []

    def __call__(self, step: dict) -> _FakeProcess:
        self.launched.append(step["id"])
        code = self.codes.get(step["id"], 0)
        out_dir = Path(step["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        if step["kind"] == "train":
            payload = _train_manifest()
            self._episode(out_dir / "episode-0000", 0.5)
            self._episode(out_dir / "eval" / "episode-0000", 0.25)
        else:
            payload = {"status": "completed", "episodes_expected": 9,
                       "episodes_completed": 9}
        write_json(out_dir / step["manifest"], payload)
        return _FakeProcess(pid=1, code=code)

    @staticmethod
    def _episode(directory: Path, seconds: float) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / benchmark.EPISODE_MANIFEST,
                   {"started_at": "2026-01-01T00:00:00+00:00",
                    "ended_at": f"2026-01-01T00:00:{seconds:06.3f}+00:00"})


def _ps(pid: int = 1) -> str:
    return f"{pid} 0 5000\n{pid + 1} {pid} 7000\n"


def _run(root: Path, launcher: _FakeLauncher) -> int:
    return benchmark.run_benchmark(root, 0, 0.05, launch=launcher, read=lambda: _ps())


def test_a_measured_task_records_both_steps(tmp_path, capsys):
    root = _written_plan(tmp_path)
    launcher = _FakeLauncher()

    assert _run(root, launcher) == 0
    assert launcher.launched == [f"train/{LEAF}", f"evaluate/{LEAF}"]
    payload = json.loads((root / "benchmark" / "task-0000.json").read_text())
    assert payload["benchmark_version"] == 1
    assert payload["task_index"] == 0 and payload["task_id"] == LEAF
    assert payload["sampling"]["method"] == "ps-process-tree"
    assert payload["sampling"]["interval_s"] == 0.05
    assert payload["plan"]["matrix_name"] == "bypass-smoke"

    train, evaluate = payload["steps"]
    assert train["peak_tree_rss_kb"] == 12000 and train["peak_process_count"] == 2
    assert train["peak_single_process_rss_kb"] == 7000
    assert train["effective_timesteps"] == 256 and train["n_steps"] == 64
    assert train["seconds_per_timestep"] == train["wall_seconds"] / 256
    assert train["training_episodes"] == 1 and train["selection_episodes"] == 1
    assert train["episode_seconds_sum"] == 0.75
    assert train["non_episode_seconds"] == round(train["wall_seconds"] - 0.75, 3)
    assert evaluate["episodes_expected"] == 9 and evaluate["episodes_completed"] == 9
    assert evaluate["seconds_per_episode"] == evaluate["wall_seconds"] / 9
    assert "wrote" in capsys.readouterr().out


def test_a_failed_training_stops_the_benchmark(tmp_path):
    root = _written_plan(tmp_path)
    launcher = _FakeLauncher(codes={f"train/{LEAF}": 1})

    assert _run(root, launcher) == 1
    assert launcher.launched == [f"train/{LEAF}"]
    payload = json.loads((root / "benchmark" / "task-0000.json").read_text())
    assert [step["kind"] for step in payload["steps"]] == ["train"]
    assert payload["steps"][0]["tolerated"] is False


def test_an_existing_record_is_refused(tmp_path, capsys):
    root = _written_plan(tmp_path)
    assert _run(root, _FakeLauncher()) == 0
    launcher = _FakeLauncher()

    assert _run(root, launcher) == 1
    assert launcher.launched == []
    assert "already exists" in capsys.readouterr().err


def test_a_dirty_step_directory_is_refused(tmp_path, capsys):
    root = _written_plan(tmp_path)
    (root / "eval" / LEAF).mkdir(parents=True)
    launcher = _FakeLauncher()

    assert _run(root, launcher) == 1
    assert launcher.launched == []
    assert "both step directories pending" in capsys.readouterr().err
    assert not (root / "benchmark").exists()


def _benchmark_record(per_timestep: float = 0.5, per_episode: float = 2.0,
                      **train_overrides) -> dict:
    manifest = _train_manifest()
    train = {"id": f"train/{LEAF}", "kind": "train", "exit_code": 0, "tolerated": True,
             "wall_seconds": 128.0, "peak_tree_rss_kb": 500_000,
             "peak_process_count": 3, "peak_single_process_rss_kb": 300_000,
             "samples": 10, "requested_timesteps": 256, "n_steps": 64,
             "effective_timesteps": 256, "seconds_per_timestep": per_timestep,
             "scenario_identity": manifest["scenario_identity"],
             "selection": manifest["selection"], "eval_every_steps": 128,
             "eval_episodes": 1}
    train.update(train_overrides)
    evaluate = {"id": f"evaluate/{LEAF}", "kind": "evaluate", "exit_code": 0,
                "tolerated": True, "wall_seconds": 18.0,
                "episodes_expected": 9, "episodes_completed": 9,
                "seconds_per_episode": per_episode}
    return {"benchmark_version": 1, "task_index": 0, "task_id": LEAF,
            "measured_at": "2026-01-01T00:00:00+00:00",
            "host": {"system": "Darwin", "machine": "arm64"},
            "steps": [train, evaluate]}


def test_estimate_arithmetic_scales_the_measured_seconds(tmp_path):
    matrix = experiment.load_matrix(TRACKED_MATRIX)
    estimate = benchmark.build_estimate(_benchmark_record(), matrix, 2.0)

    # 0.5 s/timestep x 256 timesteps + 2.0 s/episode x 3 policies x 3 held-out seeds.
    row = next(entry for entry in estimate["rows"] if entry["name"] == "local-delivery")
    assert row["reason"] is None and row["per_task_seconds"] == 146.0
    assert row["per_task_with_safety_seconds"] == 292.0
    assert row["per_task_with_safety_hms"] == "00:04:52"
    assert row["tasks"] == 2 and row["selection_differs"] is False
    assert row["warning"] is None

    assert estimate["task_count"] == 8 and estimate["estimated_task_count"] == 8
    assert estimate["totals"]["serial_seconds"] == 146.0 * 8
    assert estimate["totals"]["serial_with_safety_hms"] == "00:38:56"
    assert estimate["memory"] == {"peak_tree_rss_kb": 500_000,
                                 "with_safety_kb": 1_000_000.0}
    assert estimate["assumptions"] == {"linear_in_timesteps": True,
                                       "measured_on": "Darwin arm64",
                                       "is_cluster_estimate": False}
    assert estimate["target_matrix"]["name"] == "bypass-smoke"


MEMORY_PEAKS = [("evaluate peaks higher", 500_000, 900_000, 900_000),
                ("train peaks higher", 500_000, 100_000, 500_000),
                ("evaluate unmeasured", 500_000, None, 500_000),
                ("nothing measured", None, None, None)]


@pytest.mark.parametrize("train_peak,evaluate_peak,expected",
                         [case[1:] for case in MEMORY_PEAKS],
                         ids=[case[0] for case in MEMORY_PEAKS])
def test_the_estimate_takes_the_largest_step_peak(train_peak, evaluate_peak, expected):
    matrix = experiment.load_matrix(TRACKED_MATRIX)
    record = _benchmark_record(peak_tree_rss_kb=train_peak)
    record["steps"][1]["peak_tree_rss_kb"] = evaluate_peak

    estimate = benchmark.build_estimate(record, matrix, 2.0)

    assert estimate["memory"]["peak_tree_rss_kb"] == expected
    assert estimate["memory"]["with_safety_kb"] == (None if expected is None
                                                    else expected * 2.0)


def test_a_changed_selection_still_estimates_but_warns():
    matrix = experiment.load_matrix(TRACKED_MATRIX)
    estimate = benchmark.build_estimate(_benchmark_record(), matrix, 1.0)

    row = next(entry for entry in estimate["rows"] if entry["name"] == "raw-delivery")
    assert row["per_task_seconds"] == 146.0
    assert row["selection_differs"] is True
    assert "not measured" in row["warning"]


MISMATCH = [
    ("different scenario",
     {"scenario_identity": {"run_ini_sha256": "0" * 64, "nodes_json_sha256": "1" * 64,
                            "buildings_json_sha256": None, "jammers_json_sha256": None}},
     "different scenario"),
    ("missing scenario identity", {"scenario_identity": None}, "different scenario"),
    ("different n_steps", {"n_steps": 32}, "different cadence"),
    ("different evaluation cadence", {"eval_every_steps": 64}, "different cadence"),
    ("different evaluation episodes", {"eval_episodes": 3}, "different cadence"),
]


@pytest.mark.parametrize("override,reason", [case[1:] for case in MISMATCH],
                         ids=[case[0] for case in MISMATCH])
def test_a_mismatched_row_gets_no_estimate(override, reason):
    matrix = experiment.load_matrix(TRACKED_MATRIX)
    estimate = benchmark.build_estimate(_benchmark_record(**override), matrix, 2.0)

    assert all(row["per_task_seconds"] is None for row in estimate["rows"])
    assert all(row["reason"] == reason for row in estimate["rows"])
    assert all(row["per_task_with_safety_hms"] is None for row in estimate["rows"])
    assert estimate["estimated_task_count"] == 0
    assert estimate["totals"]["serial_seconds"] == 0


def test_an_unmeasurable_benchmark_is_refused():
    matrix = experiment.load_matrix(TRACKED_MATRIX)
    with pytest.raises(ValueError, match="seconds_per_timestep"):
        benchmark.build_estimate(_benchmark_record(seconds_per_timestep=None),
                                 matrix, 2.0)


def test_estimate_writes_a_file_and_records_the_benchmark_hash(tmp_path, capsys):
    record = tmp_path / "task-0000.json"
    write_json(record, _benchmark_record())
    output = tmp_path / "out" / "estimate.json"

    assert benchmark.main(["estimate", "--benchmark", str(record),
                           "--target-matrix", str(TRACKED_MATRIX),
                           "--safety-factor", "2.0", "--output", str(output)]) == 0
    payload = json.loads(output.read_text())
    assert payload["safety_factor"] == 2.0
    assert payload["benchmark"]["path"] == str(record.resolve())
    assert len(payload["benchmark"]["sha256"]) == 64
    assert "not a cluster estimate" in capsys.readouterr().out


def test_a_missing_safety_factor_is_an_argparse_error(tmp_path):
    record = tmp_path / "task-0000.json"
    write_json(record, _benchmark_record())

    with pytest.raises(SystemExit):
        benchmark.main(["estimate", "--benchmark", str(record),
                        "--target-matrix", str(TRACKED_MATRIX),
                        "--output", str(tmp_path / "estimate.json")])
    assert not (tmp_path / "estimate.json").exists()


def test_a_non_positive_safety_factor_is_refused(tmp_path):
    matrix = experiment.load_matrix(TRACKED_MATRIX)
    with pytest.raises(ValueError, match="safety-factor"):
        benchmark.build_estimate(_benchmark_record(), matrix, 0.0)
