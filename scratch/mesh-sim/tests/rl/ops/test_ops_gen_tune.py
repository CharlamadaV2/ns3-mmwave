"""Tuning smoke paths that need no Optuna: pin checks, spec validation, seed roles.

Optuna is not installed here and must not be; a plain callable sampler is
injected through tune.main(sampler=...). Expectations come from
scripts/rl/ops/README.md "Tuning smoke".
"""

import json
import shlex
import sys
import types
from pathlib import Path

import pytest

from scripts.rl.cli_common import write_json
from scripts.rl.ops import tune
from scripts.sim_support import find_mesh_root

MATRIX = find_mesh_root() / "inputs/experiments/bypass-smoke-matrix.json"
STUDY = find_mesh_root() / "inputs/experiments/bypass-smoke-study.json"
HELD_OUT = {"301", "302", "303"}


def _spec(tmp_path: Path, matrix=None, **overrides) -> Path:
    body = json.loads(STUDY.read_text())
    body["matrix"] = str(matrix or MATRIX)
    body.update(overrides)
    path = tmp_path / "study.json"
    write_json(path, body)
    return path


def _sampler(space, number):
    return {"n_steps": 64, "gamma": 0.95, "ent_coef": 0.01}


@pytest.fixture
def binary(tmp_path):
    path = tmp_path / "mesh-sim"
    path.write_text("#!/bin/sh\nexit 1\n")
    path.chmod(0o755)
    return str(path)


# --- Optuna pin ----------------------------------------------------------------


def test_gen_the_pinned_version_is_accepted_and_returned(tmp_path, monkeypatch):
    fake = types.SimpleNamespace(__version__="9.9.9")
    monkeypatch.setitem(sys.modules, "optuna", fake)
    pin = tmp_path / tune.PIN_NAME
    pin.write_text("# tuning only\noptuna==9.9.9  # tested\n")
    assert tune.load_optuna(pin) is fake


# --- study spec validation -----------------------------------------------------


def test_gen_a_relative_matrix_resolves_against_the_mesh_root(tmp_path, monkeypatch):
    elsewhere = tmp_path / "deep" / "dir"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    spec = tune.load_study(_spec(elsewhere,
                                 matrix="inputs/experiments/bypass-smoke-matrix.json"))
    assert Path(spec["matrix"]["path"]).resolve() == MATRIX.resolve()


# --- output root, binary, dry run --------------------------------------------


def test_gen_an_existing_empty_root_is_fine(tmp_path, binary):
    out = tmp_path / "out"
    out.mkdir()
    calls = []

    def execute(trial):
        calls.append(trial)
        Path(trial["train_dir"]).mkdir(parents=True)
        write_json(Path(trial["train_dir"]) / "train_manifest.json",
                   {"best_mean_reward": 1.0})
        return 0

    assert tune.main(["--study", str(_spec(tmp_path)), "--output-root", str(out),
                      "--sim-binary", binary], sampler=_sampler, execute=execute) == 0
    assert len(calls) == 3


def _dry(tmp_path, capsys, n_trials, sampler=None):
    out = tmp_path / "out"
    sampled = []

    def recording(space, number):
        sampled.append(number)
        return {"n_steps": 32 + 32 * (number % 2), "gamma": 0.9 + number / 1000,
                "ent_coef": 0.001 * (number + 1)}

    assert tune.main(["--study", str(_spec(tmp_path, n_trials=n_trials)),
                      "--output-root", str(out), "--sim-binary", "/no/such/binary",
                      "--dry-run"], sampler=sampler or recording,
                     execute=lambda trial: pytest.fail("dry run executed a trial")) == 0
    assert not out.exists()
    return sampled, capsys.readouterr().out


def test_gen_a_large_dry_run_previews_only_the_startup_trials(tmp_path, capsys):
    sampled, out = _dry(tmp_path, capsys, 12)
    assert sampled == list(range(10))
    assert out.count(" params: ") == 10 and out.count(" command: ") == 10
    assert "the remaining 2 trials depend on earlier objectives" in out


def test_gen_a_small_dry_run_previews_every_trial(tmp_path, capsys):
    sampled, out = _dry(tmp_path, capsys, 3)
    assert sampled == [0, 1, 2] and "remaining" not in out


def test_gen_no_previewed_command_names_a_held_out_seed(tmp_path, capsys):
    _, out = _dry(tmp_path, capsys, 10)
    commands = [shlex.split(line.split(" command: ", 1)[1])
                for line in out.splitlines() if " command: " in line]
    assert len(commands) == 10
    for argv in commands:
        seeds = {argv[i + 1] for i, token in enumerate(argv)
                 if token in ("--seed", "--eval-seed")}
        assert seeds == {"101", "201"}
        assert not HELD_OUT & set(argv)


# --- objectives ----------------------------------------------------------------


@pytest.mark.parametrize("text,objective,failure", [
    ('{"best_mean_reward": 3}', 3.0, None),
    ('{"best_mean_reward": -1.5}', -1.5, None),
    ('{}', None, "None"),
    ("{broken", None, "unreadable"),
    (None, None, "unreadable"),
])
def test_gen_the_objective_is_a_finite_number_or_a_recorded_failure(tmp_path, text,
                                                                    objective, failure):
    if text is not None:
        (tmp_path / "train_manifest.json").write_text(text)
    value, reason = tune._objective(tmp_path)
    assert value == objective
    if failure is None:
        assert reason is None and isinstance(value, float)
    else:
        assert failure in reason
