"""Gap tests for scripts/rl/experiment.py: matrix validation, plan immutability,
step_state, run_plan exit codes, and the `--rows` CLI.

Contracts (scripts/rl/CLAUDE.md, scripts/rl/README.md "experiment",
src/rl/policy-comparison-tests.md):
- Seed roles training / model_selection / held_out are disjoint; load_matrix enforces it.
- "Invalid matrices fail before execution."
- Step state comes only from the step's dir and manifest: missing dir = pending,
  completed manifest = done, anything else = blocked ("move or delete <dir> to retry").
- A changed matrix, binary, or --rows needs a new --output-root.
- evaluate/compare exit 2 is a warning, not a failure; experiment tolerates it.
No simulator is launched except where noted (a `python -m scripts.rl.evaluate` argparse
failure); executors are stubs.
"""

import copy
import json
from pathlib import Path

import pytest

from scripts.rl import experiment

SIM_BINARY = "/nonexistent/mesh-sim-binary"


def _matrix_data(run_config: str) -> dict:
    return {
        "matrix_version": 1,
        "name": "cli-matrix",
        "description": "Generated matrix for tests/rl/cli.",
        "run_config": run_config,
        "band": None,
        "seeds": {"training": [1], "held_out": "11-12"},
        "training": {"total_timesteps": 16, "n_steps": 16},
        "evaluation": {"model": "final", "policies": ["model", "hold"]},
        "rows": [
            {"name": "row-a", "observation_preset": "local_links_v1",
             "action_profile": "move_2d", "reward_components": ["delivery_ratio"],
             "reward_weights": [1.0]},
            {"name": "row-b", "observation_preset": "raw_links_v1",
             "action_profile": "move_2d", "reward_components": ["connectivity"],
             "reward_weights": [2.0]},
        ],
    }


def _write(tmp_path: Path, data: dict, name: str = "matrix.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


def _mutated(run_config: str, mutate) -> dict:
    data = copy.deepcopy(_matrix_data(run_config))
    mutate(data)
    return data


# 1. Seed roles --------------------------------------------------------------------

def _set(path, value):
    def apply(data):
        target = data
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return apply


def _with_cadence(selection, held_out="11-12", training=None):
    def apply(data):
        data["training"]["eval_every_steps"] = 8
        data["seeds"]["model_selection"] = selection
        data["seeds"]["held_out"] = held_out
        if training is not None:
            data["seeds"]["training"] = training
    return apply


def test_disjoint_ranges_and_selection_are_accepted(multi_run_config, tmp_path):
    data = _mutated(multi_run_config, _with_cadence(10, held_out="11-13",
                                                    training="1-3,5"))
    matrix = experiment.load_matrix(_write(tmp_path, data))
    assert matrix["seeds"] == {"training": [1, 2, 3, 5], "model_selection": 10,
                               "held_out": [11, 12, 13]}


def test_held_out_seeds_never_reach_training_arguments(multi_run_config, tmp_path):
    data = _mutated(multi_run_config, _with_cadence(10, held_out="11-12",
                                                    training="1-2"))
    plan = experiment.build_plan(experiment.load_matrix(_write(tmp_path, data)),
                                 tmp_path / "root", SIM_BINARY)
    held_out = {"11", "12"}
    for step in plan["steps"]:
        if step["kind"] != "train":
            continue
        args = step["args"]
        assert args[args.index("--seed") + 1] not in held_out
        assert args[args.index("--eval-seed") + 1] == "10"
        assert not held_out & set(args)
    evaluate = [s for s in plan["steps"] if s["kind"] == "evaluate"]
    assert all(s["args"][s["args"].index("--seeds") + 1] == "11,12" for s in evaluate)


# 2. Other matrix validation gaps ---------------------------------------------------


def test_top_level_band_reaches_train_arguments(multi_run_config, tmp_path):
    data = _mutated(multi_run_config, _set(("band",), "sub-6"))
    plan = experiment.build_plan(experiment.load_matrix(_write(tmp_path, data)),
                                 tmp_path / "root", SIM_BINARY)
    for step in (s for s in plan["steps"] if s["kind"] == "train"):
        assert step["args"][step["args"].index("--band") + 1] == "sub-6"
        assert step["args"].index("--band") < step["args"].index("m-ppo")


def test_plan_layout_matches_the_documented_tree(multi_run_config, tmp_path):
    root = tmp_path / "root"
    plan = experiment.build_plan(
        experiment.load_matrix(_write(tmp_path, _matrix_data(multi_run_config))),
        root, SIM_BINARY)
    root = root.resolve()
    dirs = {step["id"]: step["output_dir"] for step in plan["steps"]}
    assert dirs["train/row-a/train-seed-1"] == str(root / "train/row-a/train-seed-1")
    assert dirs["evaluate/row-b/train-seed-1"] == str(root / "eval/row-b/train-seed-1")
    assert dirs["compare"] == str(root / "comparison")
    compare = plan["steps"][-1]
    assert compare["args"] == ["--plan", str(root / experiment.PLAN_NAME),
                               "--output-dir", str(root / "comparison")]
    assert plan["experiment_plan_version"] == experiment.PLAN_VERSION


# 3. Plan immutability and --rows ---------------------------------------------------

def _plan_argv(matrix_path: str, root: Path, binary: str = SIM_BINARY, *extra) -> list:
    return ["plan", "--matrix", matrix_path, "--output-root", str(root),
            "--sim-binary", binary, *extra]


def test_changed_binary_is_refused_and_plan_unchanged(multi_run_config, tmp_path, capsys):
    matrix_path = _write(tmp_path, _matrix_data(multi_run_config))
    root = tmp_path / "root"
    assert experiment.main(_plan_argv(matrix_path, root)) == 0
    before = (root / experiment.PLAN_NAME).read_bytes()
    capsys.readouterr()
    assert experiment.main(_plan_argv(matrix_path, root, "/other/binary")) == 1
    assert "use a new --output-root" in capsys.readouterr().err
    assert (root / experiment.PLAN_NAME).read_bytes() == before


def test_changed_matrix_content_is_refused_and_plan_unchanged(multi_run_config, tmp_path,
                                                              capsys):
    data = _matrix_data(multi_run_config)
    matrix_path = _write(tmp_path, data)
    root = tmp_path / "root"
    assert experiment.main(_plan_argv(matrix_path, root)) == 0
    before = (root / experiment.PLAN_NAME).read_bytes()

    data["description"] = "edited after planning"
    _write(tmp_path, data)
    capsys.readouterr()
    assert experiment.main(_plan_argv(matrix_path, root)) == 1
    assert "use a new --output-root" in capsys.readouterr().err
    assert (root / experiment.PLAN_NAME).read_bytes() == before

    # `run` with a changed matrix is refused the same way and executes nothing.
    assert experiment.main(["run", "--matrix", matrix_path, "--output-root", str(root),
                            "--sim-binary", SIM_BINARY]) == 1
    assert not (root / "train").exists()


def test_changed_run_config_alone_keeps_the_plan(multi_run_config, tmp_path):
    """The plan records the matrix digest, not the run.ini digest; the documented
    re-plan triggers are only matrix, binary, and --rows."""
    matrix_path = _write(tmp_path, _matrix_data(multi_run_config))
    root = tmp_path / "root"
    assert experiment.main(_plan_argv(matrix_path, root)) == 0
    Path(multi_run_config).write_text(Path(multi_run_config).read_text() + "\n# edit\n")
    assert experiment.main(_plan_argv(matrix_path, root)) == 0


def test_rows_filter_keeps_matrix_order(multi_run_config, tmp_path):
    matrix = experiment.load_matrix(_write(tmp_path, _matrix_data(multi_run_config)))
    plan = experiment.build_plan(matrix, tmp_path / "root", SIM_BINARY,
                                 ["row-b", "row-a"])
    assert [row["name"] for row in plan["rows"]] == ["row-a", "row-b"]
    assert plan["rows_filter"] == ["row-b", "row-a"]


def test_automatic_label_sanitizes_the_scene_directory(multi_run_config, tmp_path):
    scene = tmp_path / "My_Scene.V2 (copy)"
    scene.mkdir()
    (scene / "run.ini").write_text(Path(multi_run_config).read_text())
    (scene / "nodes.json").write_text(
        (Path(multi_run_config).parent / "nodes.json").read_text())
    data = _matrix_data(str(scene / "run.ini"))
    matrix = experiment.load_matrix(_write(tmp_path, data))
    assert experiment.automatic_run_label(matrix, ["row-a"]) \
        == "cli-matrix-my-scene-v2-copy-row-a"
    assert experiment.automatic_run_label(matrix, ["row-a", "row-b"]) == "cli-matrix"
    with pytest.raises(ValueError, match="not in matrix"):
        experiment.automatic_run_label(matrix, ["nope"])


# 4. step_state --------------------------------------------------------------------

def _steps(multi_run_config, tmp_path) -> dict:
    matrix = experiment.load_matrix(_write(tmp_path, _matrix_data(multi_run_config)))
    plan = experiment.build_plan(matrix, tmp_path / "root", SIM_BINARY)
    return {step["kind"]: step for step in plan["steps"]}


def _prepare(step: dict, manifest) -> None:
    out_dir = Path(step["output_dir"])
    out_dir.mkdir(parents=True)
    if manifest is None:
        return
    path = out_dir / step["manifest"]
    if manifest == "<dir>":
        path.mkdir()
    elif isinstance(manifest, str):
        path.write_text(manifest)
    else:
        path.write_text(json.dumps(manifest))


STEP_STATES = [
    ("train dir without manifest", "train", None, "blocked"),
    ("train manifest is a directory", "train", "<dir>", "blocked"),
    ("train manifest is invalid JSON", "train", "{half", "blocked"),
    ("train manifest is empty", "train", "", "blocked"),
    ("train status running", "train", {"status": "running"}, "blocked"),
    ("train status failed", "train", {"status": "failed"}, "blocked"),
    ("train status partial", "train", {"status": "partial"}, "blocked"),
    ("train manifest without status", "train", {}, "blocked"),
    ("train completed", "train", {"status": "completed"}, "done"),
    ("evaluate failed", "evaluate", {"status": "failed"}, "blocked"),
    ("evaluate partial", "evaluate", {"status": "partial"}, "partial"),
    ("evaluate completed", "evaluate", {"status": "completed"}, "done"),
    ("compare dir without comparison.json", "compare", None, "blocked"),
    ("compare incomplete is still done", "compare", {"status": "incomplete"}, "done"),
    ("compare invalid JSON", "compare", "{", "blocked"),
]


@pytest.mark.parametrize("kind,manifest,expected", [case[1:] for case in STEP_STATES],
                         ids=[case[0] for case in STEP_STATES])
def test_step_state_from_dir_and_manifest(multi_run_config, tmp_path, kind, manifest,
                                          expected):
    step = _steps(multi_run_config, tmp_path)[kind]
    assert experiment.step_state(step) == ("pending", "")
    _prepare(step, manifest)
    state, detail = experiment.step_state(step)
    assert state == expected
    if expected == "blocked":
        assert detail == f"move or delete {step['output_dir']} to retry"


def test_step_state_never_modifies_the_directory(multi_run_config, tmp_path):
    step = _steps(multi_run_config, tmp_path)["train"]
    _prepare(step, {"status": "running"})
    (Path(step["output_dir"]) / "partial.bin").write_bytes(b"x")
    before = sorted(p.name for p in Path(step["output_dir"]).iterdir())
    experiment.step_state(step)
    assert sorted(p.name for p in Path(step["output_dir"]).iterdir()) == before


# 5. run_plan exit codes ------------------------------------------------------------

def _plan(multi_run_config, tmp_path, rows=("row-a",)) -> dict:
    matrix = experiment.load_matrix(_write(tmp_path, _matrix_data(multi_run_config)))
    return experiment.build_plan(matrix, tmp_path / "root", SIM_BINARY, list(rows))


def _executor(codes: dict, executed: list):
    def execute(step):
        executed.append(step["id"])
        code = codes.get(step["kind"], 0)
        if code in (0, 2) and step["kind"] != "compare":
            out_dir = Path(step["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / step["manifest"]).write_text(json.dumps({"status": "completed"}))
        return code
    return execute


def test_train_exit_two_is_a_failure_not_a_warning(multi_run_config, tmp_path, capsys):
    plan = _plan(multi_run_config, tmp_path)
    executed = []
    assert experiment.run_plan(plan, _executor({"train": 2}, executed)) == 1
    out = capsys.readouterr().out
    assert "train/row-a/train-seed-1  failed  exit 2" in out
    assert ("evaluate/row-a/train-seed-1  skipped  "
            "train/row-a/train-seed-1 did not complete") in out
    assert executed == ["train/row-a/train-seed-1", "compare"]


def test_compare_exit_two_is_returned_when_all_work_is_done(multi_run_config, tmp_path):
    plan = _plan(multi_run_config, tmp_path)
    assert experiment.run_plan(plan, _executor({"compare": 2}, [])) == 2


def test_compare_failure_is_reported_and_returned(multi_run_config, tmp_path, capsys):
    plan = _plan(multi_run_config, tmp_path)
    assert experiment.run_plan(plan, _executor({"compare": 1}, [])) == 1
    assert "compare  failed  exit 1" in capsys.readouterr().out


def test_partial_evaluation_from_disk_makes_run_return_one(multi_run_config, tmp_path,
                                                           capsys):
    plan = _plan(multi_run_config, tmp_path)
    train, evaluate = plan["steps"][0], plan["steps"][1]
    _prepare(train, {"status": "completed"})
    _prepare(evaluate, {"status": "partial"})
    executed = []
    assert experiment.run_plan(plan, _executor({}, executed)) == 1
    assert executed == ["compare"]
    assert "partial  some episodes did not complete" in capsys.readouterr().out


def test_run_never_executes_into_an_existing_step_dir(multi_run_config, tmp_path):
    plan = _plan(multi_run_config, tmp_path)
    train = plan["steps"][0]
    _prepare(train, {"status": "failed"})
    marker = Path(train["output_dir"]) / train["manifest"]
    before = marker.read_bytes()
    executed = []
    assert experiment.run_plan(plan, _executor({}, executed)) == 1
    assert train["id"] not in executed
    assert marker.read_bytes() == before


def test_status_prints_one_line_per_step_with_details(multi_run_config, tmp_path, capsys):
    matrix_path = _write(tmp_path, _matrix_data(multi_run_config))
    root = tmp_path / "root"
    assert experiment.main(_plan_argv(matrix_path, root, SIM_BINARY,
                                      "--rows", "row-a")) == 0
    plan = experiment.load_plan(root)
    _prepare(plan["steps"][0], None)
    capsys.readouterr()
    assert experiment.main(["status", "--output-root", str(root)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert lines[0] == (f"train/row-a/train-seed-1  blocked  move or delete "
                        f"{plan['steps'][0]['output_dir']} to retry")
    assert lines[1] == "evaluate/row-a/train-seed-1  pending"
