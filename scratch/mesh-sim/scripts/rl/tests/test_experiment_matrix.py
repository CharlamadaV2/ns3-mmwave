"""Matrix validation, plan expansion, step state, and stub-executed runs."""

import copy
import json
from pathlib import Path

import pytest

from scripts.rl import experiment
from scripts.sim_support import find_mesh_root

TRACKED_MATRIX = find_mesh_root() / "inputs/experiments/bypass-smoke-matrix.json"

RUN_INI = """[scenario]
name = fake-matrix
seed = 1
duration_s = 1.0
tick_s = 0.1
nodes_file = nodes.json

[rl]
enabled = true
controlled_nodes = node-b
max_controlled_nodes = 1
action_profile = move_2d
decision_interval_s = 0.5
action_type = discrete
reward_type = all_links_los
step_size_m = 1.0
"""

SIM_BINARY = "/nonexistent/mesh-sim-binary"


def _run_ini(tmp_path: Path) -> str:
    path = tmp_path / "run.ini"
    path.write_text(RUN_INI)
    return str(path)


def _matrix_data(run_config: str) -> dict:
    return {
        "matrix_version": 1,
        "name": "stub-matrix",
        "description": "Two rows used by the matrix tests.",
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


def _write_matrix(tmp_path: Path, data: dict, name: str = "matrix.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


def _stub_matrix(tmp_path: Path) -> str:
    return _write_matrix(tmp_path, _matrix_data(_run_ini(tmp_path)))


class _StubExecutor:
    """Records executed steps and fakes their manifests; no process is started."""

    def __init__(self, failing: tuple[str, ...] = ()):
        self.failing = set(failing)
        self.executed: list[str] = []

    def __call__(self, step: dict) -> int:
        self.executed.append(step["id"])
        if step["id"] in self.failing:
            return 1
        out_dir = Path(step["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        status = "complete" if step["kind"] == "compare" else "completed"
        (out_dir / step["manifest"]).write_text(json.dumps({"status": status}))
        return 0


def _plan(tmp_path: Path, rows=None) -> dict:
    matrix = experiment.load_matrix(_stub_matrix(tmp_path))
    return experiment.build_plan(matrix, tmp_path / "root", SIM_BINARY, rows)


def test_tracked_matrix_expands_to_full_step_list(tmp_path):
    matrix = experiment.load_matrix(TRACKED_MATRIX)
    plan = experiment.build_plan(matrix, tmp_path / "root", SIM_BINARY)

    assert len(plan["steps"]) == 17
    assert len(plan["evaluations"]) == 8
    kinds = [step["kind"] for step in plan["steps"]]
    assert kinds.count("train") == 8 and kinds.count("evaluate") == 8
    assert kinds[-1] == "compare"
    assert plan["seeds"] == {"training": [101, 102], "model_selection": 201,
                             "held_out": [301, 302, 303]}
    assert plan["rows_filter"] is None


def test_automatic_one_row_name_identifies_matrix_scene_and_row(tmp_path):
    matrix = experiment.load_matrix(_stub_matrix(tmp_path))
    assert experiment.automatic_run_label(matrix, None) == "stub-matrix"
    scene = tmp_path.name.replace("_", "-")
    assert experiment.automatic_run_label(matrix, ["row-a"]) == f"stub-matrix-{scene}-row-a"


def test_train_and_evaluate_arguments_follow_the_cli_shape(tmp_path):
    matrix = experiment.load_matrix(TRACKED_MATRIX)
    plan = experiment.build_plan(matrix, tmp_path / "root", SIM_BINARY)
    train = plan["steps"][0]
    evaluate = plan["steps"][1]

    assert train["module"] == "scripts.rl.train"
    assert train["manifest"] == "train_manifest.json"
    assert train["args"][:2] == ["--sim-binary", SIM_BINARY]
    assert train["args"].index("m-ppo") < train["args"].index("--seed")
    assert train["args"].index("--observation-preset") < train["args"].index("m-ppo")
    for flag, value in (("--reward-components", "delivery_ratio"),
                        ("--reward-weights", "1.0"),
                        ("--total-timesteps", "256"), ("--n-steps", "64"),
                        ("--eval-every-steps", "128"), ("--seed", "101"),
                        ("--eval-episodes", "1"), ("--eval-seed", "201")):
        assert train["args"][train["args"].index(flag) + 1] == value

    assert evaluate["module"] == "scripts.rl.evaluate"
    assert evaluate["needs"] == [train["id"]]
    assert evaluate["manifest"] == "eval_manifest.json"
    args = evaluate["args"]
    assert args[args.index("--seeds") + 1] == "301,302,303"
    assert args[args.index("--policies") + 1] == "model,hold,random_valid"
    assert args[args.index("--label") + 1] == "local-delivery"
    assert args[args.index("--model") + 1] == "best"
    assert args[args.index("--run-dir") + 1] == train["output_dir"]
    assert plan["evaluations"][0] == {"label": "local-delivery", "training_seed": 101,
                                      "eval_dir": evaluate["output_dir"]}


def test_row_filter_selects_named_rows(tmp_path):
    matrix = experiment.load_matrix(TRACKED_MATRIX)
    plan = experiment.build_plan(matrix, tmp_path / "root", SIM_BINARY,
                                 ["local-delivery"])

    assert len(plan["steps"]) == 5
    assert [row["name"] for row in plan["rows"]] == ["local-delivery"]
    assert {ev["label"] for ev in plan["evaluations"]} == {"local-delivery"}
    assert plan["rows_filter"] == ["local-delivery"]


def test_unknown_row_name_is_refused(tmp_path):
    matrix = experiment.load_matrix(TRACKED_MATRIX)
    with pytest.raises(ValueError, match="not in the matrix"):
        experiment.build_plan(matrix, tmp_path / "root", SIM_BINARY, ["nope"])


def _mutate(data: dict, path: tuple, value) -> dict:
    data = copy.deepcopy(data)
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return data


MATRIX_REFUSALS = [
    ("list axis", ("rows", 0, "observation_preset"), ["local_links_v1", "raw_links_v1"],
     "no product expansion"),
    ("list scalar seed", ("seeds", "model_selection"), [1, 2], "no product expansion"),
    ("unknown top key", ("colour",), "red", "unknown keys"),
    ("unknown row key", ("rows", 0, "epochs"), 3, "unknown keys"),
    ("unknown training key", ("training", "learning_rate"), 0.1, "unknown keys"),
    ("bad version", ("matrix_version",), 2, "matrix_version"),
    ("bad name", ("name",), "Stub Matrix", "a-z0-9"),
    ("bad row name", ("rows", 1, "name"), "Row B", "a-z0-9"),
    ("duplicate row", ("rows", 1, "name"), "row-a", "unique"),
    ("training seed repeat", ("seeds", "training"), [1, 1], "distinct"),
    ("training overlap", ("seeds", "held_out"), [1, 11], "disjoint"),
    ("wrong action profile", ("rows", 0, "action_profile"), "move_3d",
     "action_profile"),
    ("unknown preset", ("rows", 0, "observation_preset"), "no_such_preset",
     "not registered"),
    ("unknown component", ("rows", 0, "reward_components"), ["no_such_component"],
     "unknown entries"),
    ("weight count", ("rows", 0, "reward_weights"), [1.0, 2.0], "entries"),
    ("no rows", ("rows",), [], "non-empty"),
    ("policies without model", ("evaluation", "policies"), ["hold", "random_valid"],
     "must contain 'model'"),
    ("policies without baseline", ("evaluation", "policies"), ["model"],
     "at least one baseline"),
    ("best without cadence", ("evaluation", "model"), "best",
     "requires training.eval_every_steps"),
    ("selection without cadence", ("seeds", "model_selection"), 5,
     "requires training.eval_every_steps"),
]


@pytest.mark.parametrize("path,value,message",
                         [case[1:] for case in MATRIX_REFUSALS],
                         ids=[case[0] for case in MATRIX_REFUSALS])
def test_matrix_refusals(tmp_path, path, value, message):
    data = _mutate(_matrix_data(_run_ini(tmp_path)), path, value)
    with pytest.raises(ValueError, match=message):
        experiment.load_matrix(_write_matrix(tmp_path, data))


def test_model_selection_seed_is_required_with_an_evaluation_cadence(tmp_path):
    data = _matrix_data(_run_ini(tmp_path))
    data["training"]["eval_every_steps"] = 8
    with pytest.raises(ValueError, match="model_selection is required"):
        experiment.load_matrix(_write_matrix(tmp_path, data))

    data["seeds"]["model_selection"] = 5
    matrix = experiment.load_matrix(_write_matrix(tmp_path, data, "with-seed.json"))
    assert matrix["seeds"]["model_selection"] == 5


def test_held_out_accepts_a_spec_string_or_a_list(tmp_path):
    run_config = _run_ini(tmp_path)
    data = _matrix_data(run_config)
    assert experiment.load_matrix(_write_matrix(tmp_path, data))["seeds"]["held_out"] \
        == [11, 12]

    data["seeds"]["held_out"] = [21, 22]
    listed = experiment.load_matrix(_write_matrix(tmp_path, data, "listed.json"))
    assert listed["seeds"]["held_out"] == [21, 22]


def test_row_can_override_run_config_and_band(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "run.ini").write_text(RUN_INI)
    data = _matrix_data(_run_ini(tmp_path))
    data["rows"][1]["run_config"] = str(other / "run.ini")
    data["rows"][1]["band"] = "sub-6"

    matrix = experiment.load_matrix(_write_matrix(tmp_path, data))
    plan = experiment.build_plan(matrix, tmp_path / "root", SIM_BINARY)
    row_b_train = next(step for step in plan["steps"] if step["id"].startswith("train/row-b"))
    assert row_b_train["args"][row_b_train["args"].index("--run-config") + 1] \
        == str(other / "run.ini")
    assert "--band" in row_b_train["args"]


def test_relative_run_config_resolves_against_the_mesh_root(tmp_path):
    matrix = experiment.load_matrix(TRACKED_MATRIX)
    expected = find_mesh_root() / "inputs/baselines/building-bypass-smoke/run.ini"
    assert matrix["rows"][0]["run_config"] == str(expected)


def test_plan_is_deterministic_and_refuses_a_changed_filter(tmp_path, capsys):
    matrix_path = _stub_matrix(tmp_path)
    root = tmp_path / "root"
    argv = ["plan", "--matrix", matrix_path, "--output-root", str(root),
            "--sim-binary", SIM_BINARY]

    assert experiment.main(argv) == 0
    first = (root / experiment.PLAN_NAME).read_bytes()
    assert experiment.main(argv) == 0
    assert (root / experiment.PLAN_NAME).read_bytes() == first
    assert len(capsys.readouterr().out.strip().splitlines()) == 2 * 5

    assert experiment.main(argv + ["--rows", "row-a"]) == 1
    assert "use a new --output-root" in capsys.readouterr().err
    assert (root / experiment.PLAN_NAME).read_bytes() == first


def test_status_lists_every_step(tmp_path, capsys):
    root = tmp_path / "root"
    argv = ["plan", "--matrix", _stub_matrix(tmp_path), "--output-root", str(root),
            "--sim-binary", SIM_BINARY]
    assert experiment.main(argv) == 0
    capsys.readouterr()

    assert experiment.main(["status", "--output-root", str(root)]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 5
    assert all(line.split()[1] == "pending" for line in lines)


def test_status_without_a_plan_fails(tmp_path):
    assert experiment.main(["status", "--output-root", str(tmp_path)]) == 1


def test_step_states_come_from_manifests(tmp_path):
    plan = _plan(tmp_path)
    steps = {step["id"]: step for step in plan["steps"]}
    train_a = steps["train/row-a/train-seed-1"]
    eval_a = steps["evaluate/row-a/train-seed-1"]
    train_b = steps["train/row-b/train-seed-1"]
    eval_b = steps["evaluate/row-b/train-seed-1"]

    def _write(step, payload):
        out_dir = Path(step["output_dir"])
        out_dir.mkdir(parents=True)
        if payload is not None:
            (out_dir / step["manifest"]).write_text(json.dumps(payload))

    _write(train_a, {"status": "completed"})
    _write(eval_a, {"status": "partial"})
    _write(train_b, None)
    _write(eval_b, {"status": "running"})

    assert experiment.step_state(train_a) == ("done", "")
    assert experiment.step_state(eval_a)[0] == "partial"
    assert experiment.step_state(steps["compare"]) == ("pending", "")
    for step in (train_b, eval_b):
        state, detail = experiment.step_state(step)
        assert state == "blocked"
        assert detail == f"move or delete {step['output_dir']} to retry"


def test_run_skips_the_evaluation_of_a_failed_training(tmp_path):
    plan = _plan(tmp_path)
    executor = _StubExecutor(failing=("train/row-a/train-seed-1",))

    assert experiment.run_plan(plan, executor) == 1
    assert executor.executed == ["train/row-a/train-seed-1", "train/row-b/train-seed-1",
                                 "evaluate/row-b/train-seed-1", "compare"]
    assert experiment.step_state(plan["steps"][-1]) == ("done", "")


def test_run_executes_only_pending_steps_on_a_second_pass(tmp_path):
    plan = _plan(tmp_path)
    first = _StubExecutor()
    assert experiment.run_plan(plan, first) == 0
    assert len(first.executed) == 5

    second = _StubExecutor()
    assert experiment.run_plan(plan, second) == 0
    assert second.executed == ["compare"]


def test_run_treats_evaluate_exit_two_as_done(tmp_path):
    plan = _plan(tmp_path)

    def execute(step):
        if step["kind"] == "evaluate":
            return 2
        Path(step["output_dir"]).mkdir(parents=True, exist_ok=True)
        return 0

    assert experiment.run_plan(plan, execute) == 0


def test_run_reports_a_blocked_training_directory(tmp_path):
    plan = _plan(tmp_path)
    blocked = plan["steps"][0]
    Path(blocked["output_dir"]).mkdir(parents=True)
    executor = _StubExecutor()

    assert experiment.run_plan(plan, executor) == 1
    assert blocked["id"] not in executor.executed
    assert "evaluate/row-a/train-seed-1" not in executor.executed
    assert "compare" in executor.executed


def test_unknown_evaluation_policy_fails_at_plan_time(tmp_path, capsys):
    data = _matrix_data(_run_ini(tmp_path))
    data["evaluation"]["policies"] = ["model", "hold", "greedy"]
    matrix_path = _write_matrix(tmp_path, data)
    with pytest.raises(ValueError, match=r"unknown names \['greedy'\]"):
        experiment.load_matrix(matrix_path)

    root = tmp_path / "root"
    assert experiment.main(["plan", "--matrix", matrix_path, "--output-root", str(root),
                            "--sim-binary", SIM_BINARY]) == 1
    assert "greedy" in capsys.readouterr().err
    assert not (root / experiment.PLAN_NAME).exists()


def test_every_evaluation_policy_name_is_accepted(tmp_path):
    names = ["model", "hold", "random_valid", "geometric", "optimization"]
    data = _matrix_data(_run_ini(tmp_path))
    data["evaluation"]["policies"] = names
    matrix = experiment.load_matrix(_write_matrix(tmp_path, data))
    assert matrix["evaluation"]["policies"] == names

    plan = experiment.build_plan(matrix, tmp_path / "root", SIM_BINARY)
    evaluate = next(step for step in plan["steps"] if step["kind"] == "evaluate")
    args = evaluate["args"]
    assert args[args.index("--policies") + 1] == ",".join(names)


def test_existing_matrices_keep_their_policies(tmp_path):
    tracked = experiment.load_matrix(TRACKED_MATRIX)
    assert tracked["evaluation"]["policies"] == ["model", "hold", "random_valid"]
    stub = experiment.load_matrix(_stub_matrix(tmp_path))
    assert stub["evaluation"]["policies"] == ["model", "hold"]
