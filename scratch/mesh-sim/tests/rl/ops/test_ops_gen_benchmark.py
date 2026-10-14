"""Benchmark: process-tree sampling edges, run refusals and tolerance, estimate math.

Expectations come from scripts/rl/ops/README.md "Benchmark". No real process
is benchmarked here: launches and ps snapshots are injected.
"""

import copy
import json
import shutil


from scripts.rl import experiment
from scripts.rl.env.config import read_scenario_identity
from scripts.rl.ops import benchmark
from scripts.sim_support import find_mesh_root

from ._ops_gen_helpers import written_plan

TRACKED_MATRIX = find_mesh_root() / "inputs/experiments/bypass-smoke-matrix.json"


# --- process tree ------------------------------------------------------------


def test_gen_descendants_still_count_after_the_root_exited():
    table = benchmark.parse_ps("11 10 200\n12 11 300\n")
    assert benchmark.tree_usage(table, 10) == {"rss_kb": 500, "count": 2,
                                               "max_rss_kb": 300}


def _reader(*outcomes):
    queue = list(outcomes)

    def read():
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item
    return read


def test_gen_the_first_sample_is_taken_immediately_after_spawn():
    sampler = benchmark.TreeSampler(10, 3600.0, _reader("10 1 42\n"))
    sampler.start()
    sampler.stop()
    assert sampler.result()["samples"] == 1
    assert sampler.result()["peak_tree_rss_kb"] == 42


# --- run ---------------------------------------------------------------------


class _Process:
    def __init__(self, code):
        self.pid, self._code = 10, code

    def wait(self):
        return self._code


class _Launcher:
    def __init__(self, codes):
        self.codes, self.launched = codes, []

    def __call__(self, step):
        self.launched.append(step["kind"])
        return _Process(self.codes.get(step["kind"], 0))


def _run(root, launcher, interval=0.05, index=0):
    return benchmark.run_benchmark(root, index, interval, launch=launcher,
                                   read=lambda: "10 1 1000\n")


def test_gen_an_evaluation_warning_exit_is_tolerated_like_the_runner(tmp_path):
    root = written_plan(tmp_path, 1)
    launcher = _Launcher({"evaluate": 2})
    assert _run(root, launcher) == 0
    record = json.loads((root / "benchmark" / "task-0000.json").read_text())
    assert [(step["kind"], step["exit_code"], step["tolerated"])
            for step in record["steps"]] == [("train", 0, True), ("evaluate", 2, True)]
    assert record["steps"][0]["peak_tree_rss_kb"] == 1000


def test_gen_a_training_exit_2_stops_the_benchmark(tmp_path):
    root = written_plan(tmp_path, 1)
    launcher = _Launcher({"train": 2})
    assert _run(root, launcher) == 1
    assert launcher.launched == ["train"]
    record = json.loads((root / "benchmark" / "task-0000.json").read_text())
    assert record["steps"][0]["tolerated"] is False
    assert record["sampling"] == {"method": "ps-process-tree", "interval_s": 0.05,
                                  "note": benchmark.SAMPLING_NOTE}
    assert "missed" in record["sampling"]["note"]


# --- estimate ----------------------------------------------------------------


def _record(**train_overrides):
    matrix = experiment.load_matrix(TRACKED_MATRIX)
    row = matrix["rows"][0]
    train = {"kind": "train", "seconds_per_timestep": 0.5, "n_steps": 64,
             "eval_every_steps": 128, "eval_episodes": 1, "peak_tree_rss_kb": 1000,
             "scenario_identity": read_scenario_identity(row["run_config"]),
             "selection": {"observation_preset": row["observation_preset"],
                           "reward_components": row["reward_components"],
                           "reward_weights": row["reward_weights"]}}
    train.update(train_overrides)
    evaluate = {"kind": "evaluate", "seconds_per_episode": 2.0,
                "peak_tree_rss_kb": None}
    return {"task_id": "x", "task_index": 0, "host": {"system": "Linux",
                                                      "machine": "x86_64"},
            "steps": [train, evaluate]}


def _matrix():
    return copy.deepcopy(experiment.load_matrix(TRACKED_MATRIX))


def test_gen_scenarios_match_by_file_hashes_not_by_path(tmp_path, multi_run_config):
    matrix = _matrix()
    moved = tmp_path / "copied-scenario"
    shutil.copytree(find_mesh_root() / "inputs/baselines/building-bypass-smoke", moved)
    matrix["rows"][1]["run_config"] = str(moved / "run.ini")
    matrix["rows"][2]["run_config"] = multi_run_config
    estimate = benchmark.build_estimate(_record(), matrix, 1.0)
    reasons = [row["reason"] for row in estimate["rows"]]
    assert reasons == [None, None, "different scenario", None]
    assert estimate["task_count"] == 8 and estimate["estimated_task_count"] == 6
    assert estimate["totals"]["serial_seconds"] == 146.0 * 6
