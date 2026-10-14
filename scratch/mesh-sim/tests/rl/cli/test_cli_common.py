"""Gap tests for scripts/rl/cli_common.py, bootstrap_venv.py pure helpers / --check, and
the Gymnasium registration in scripts/rl/__init__.py.

Sources: cli_common docstrings ("Write JSON atomically so a reader never sees a
half-written manifest"), scripts/rl/README.md (training refuses an output dir with
train_manifest.json or maskable_ppo_mesh.zip; seed falls back to [scenario] seed),
scripts/rl/CLAUDE.md (DIRECT_DEPS recorded in every manifest; registration idempotent).
bootstrap_venv is never run in a creating/installing mode here.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.rl import cli_common
from scripts.rl.bootstrap_venv import DIRECT_DEPS, read_pins
from scripts.rl.cli_common import (MANIFEST_NAME, MODEL_BASENAME, has_previous_run, make_out_dir, resolve_seed, write_json)

MESH_ROOT = Path(__file__).resolve().parents[3]


# 1. write_json -------------------------------------------------------------------

def test_write_json_creates_indented_json_with_trailing_newline(tmp_path):
    path = tmp_path / "out.json"
    write_json(str(path), {"b": [1, 2], "a": None})
    text = path.read_text()
    assert text.endswith("}\n")
    assert json.loads(text) == {"b": [1, 2], "a": None}
    assert '\n  "b"' in text                      # indent=2


def test_write_json_overwrites_an_existing_file(tmp_path):
    path = tmp_path / "out.json"
    path.write_text("old contents that are not json")
    write_json(path, {"new": True})
    assert json.loads(path.read_text()) == {"new": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json"]


def test_write_json_never_exposes_a_partial_file(tmp_path, monkeypatch):
    """A crash mid-dump must leave the old file intact (no truncation in place)."""
    path = tmp_path / "manifest.json"
    write_json(path, {"version": 1})

    real_dump = json.dump

    def exploding_dump(obj, fh, **kwargs):
        fh.write('{"version": ')
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_common.json, "dump", exploding_dump)
    with pytest.raises(KeyboardInterrupt):
        write_json(path, {"version": 2})
    monkeypatch.setattr(cli_common.json, "dump", real_dump)
    assert json.loads(path.read_text()) == {"version": 1}
    assert list(tmp_path.glob("*.tmp")) == []


# 2. has_previous_run / make_out_dir / automatic_output_root -----------------------

def test_has_previous_run_reports_manifest_then_model(tmp_path):
    assert has_previous_run(str(tmp_path)) is None
    (tmp_path / f"{MODEL_BASENAME}.zip").write_bytes(b"zip")
    assert has_previous_run(str(tmp_path)) == "maskable_ppo_mesh.zip"
    (tmp_path / MANIFEST_NAME).write_text("{}")
    assert has_previous_run(str(tmp_path)) == "train_manifest.json"


def test_has_previous_run_ignores_unrelated_files(tmp_path):
    (tmp_path / "eval_manifest.json").write_text("{}")
    (tmp_path / "best_model.zip").write_bytes(b"")
    assert has_previous_run(str(tmp_path)) is None
    assert has_previous_run(str(tmp_path / "missing")) is None


def test_make_out_dir_creates_nested_explicit_dir(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    assert make_out_dir(str(target)) == str(target)
    assert target.is_dir()
    assert make_out_dir(str(target)) == str(target)       # existing is fine


def test_make_out_dir_default_is_timestamped_under_cwd_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = make_out_dir("")
    assert re.fullmatch(r"outputs/\d{4}-\d{2}/\d{2}/\d{2}-\d{2}-\d{2}", out.replace(os.sep, "/"))
    assert (tmp_path / out).is_dir()


def test_automatic_output_root_finds_a_fresh_path(tmp_path, monkeypatch):
    from datetime import datetime

    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 2, 3, 4, 5)

    monkeypatch.setattr(cli_common, "find_mesh_root", lambda: tmp_path)
    monkeypatch.setattr(cli_common, "datetime", _Fixed)
    parent = tmp_path / "outputs" / "2026-01" / "02"

    first = cli_common.automatic_output_root("rl-evaluation")
    assert first == parent / "03-04-05-rl-evaluation"
    assert not first.exists()                    # only finds, never creates

    first.mkdir(parents=True)
    second = cli_common.automatic_output_root("rl-evaluation")
    assert second == parent / "03-04-05-rl-evaluation-2"
    second.mkdir()
    assert cli_common.automatic_output_root("rl-evaluation") \
        == parent / "03-04-05-rl-evaluation-3"


# 3. resolve_seed / sha256_file / package_versions ----------------------------------

def _ini(tmp_path: Path, body: str) -> str:
    path = tmp_path / "run.ini"
    path.write_text(body)
    return str(path)


def test_resolve_seed_prefers_the_cli_even_when_zero(tmp_path):
    run_config = _ini(tmp_path, "[scenario]\nseed = 9\n")
    assert resolve_seed(0, run_config) == (0, "cli")
    assert resolve_seed(5, run_config) == (5, "cli")
    assert resolve_seed(None, run_config) == (9, "run.ini")


def test_resolve_seed_strips_inline_comments(tmp_path):
    run_config = _ini(tmp_path, "[scenario]\nseed = 7   # primary seed\n")
    assert resolve_seed(None, run_config) == (7, "run.ini")


def test_package_versions_covers_every_direct_dependency():
    versions = cli_common.package_versions()
    assert set(versions) == set(DIRECT_DEPS)
    assert all(v is None or isinstance(v, str) for v in versions.values())


def test_scenario_arguments_run_config_optional_when_requested():
    parser = argparse.ArgumentParser()
    cli_common.add_scenario_arguments(parser, run_config_required=False)
    assert parser.parse_args(["--sim-binary", "b"]).run_config is None
    strict = argparse.ArgumentParser()
    cli_common.add_scenario_arguments(strict)
    with pytest.raises(SystemExit):
        strict.parse_args(["--sim-binary", "b"])


# 4. bootstrap_venv pure helpers and --check ---------------------------------------

def _pins_text(**overrides) -> str:
    pins = {dist: "1.0" for dist in DIRECT_DEPS.values()}
    pins.update(overrides)
    return "".join(f"{dist}=={version}\n" for dist, version in pins.items() if version)


def test_read_pins_ignores_comments_and_blank_lines(tmp_path):
    path = tmp_path / "req.txt"
    path.write_text("# header\n\n" + _pins_text().replace("numpy==1.0",
                                                         "numpy==1.0   # inline"))
    pins = read_pins(path)
    assert set(pins) == set(DIRECT_DEPS.values())
    assert pins["numpy"] == "1.0"


def test_tracked_requirements_pin_every_direct_dependency():
    pins = read_pins(MESH_ROOT / "requirements.txt")
    assert set(pins) == set(DIRECT_DEPS.values())
    assert "optuna" not in pins                  # tuning-only, kept separate


def _run_bootstrap_check(venv: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MESH_ROOT / "scripts/rl/bootstrap_venv.py"), "--check",
         "--venv", str(venv)],
        cwd=str(MESH_ROOT), capture_output=True, text=True, timeout=60)


def _shim_venv(tmp_path: Path, exit_code: int) -> tuple[Path, Path]:
    venv = tmp_path / "shim-venv"
    (venv / "bin").mkdir(parents=True)
    log = tmp_path / "invocations.log"
    python = venv / "bin" / "python"
    python.write_text(f'#!/bin/sh\necho "$1" >> "{log}"\n'
                      f'echo "python==3.0.0"\nexit {exit_code}\n')
    python.chmod(0o755)
    return venv, log


def test_check_only_probes_and_never_installs(tmp_path):
    venv, log = _shim_venv(tmp_path, 0)
    result = _run_bootstrap_check(venv)
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines() == ["-c"]       # one probe, no pip, no venv
    assert f"source {venv.resolve()}/bin/activate" in result.stdout
    assert "+ " not in result.stdout                    # _run() never called


# 5. Gymnasium registration ---------------------------------------------------------

def test_env_is_registered_with_the_documented_entry_point():
    import gymnasium

    import scripts.rl as rl

    assert rl.ENV_ID == "mesh_sim/MeshEnv-v0"
    spec = gymnasium.spec(rl.ENV_ID)
    assert spec.entry_point == "scripts.rl.env.mesh_env:MeshRlEnv"
