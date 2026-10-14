"""Fetch: rsync argv construction, source/destination rules, and the manifest.

rsync is always a recording shim under tmp_path (never the system rsync).
Expectations come from scripts/rl/ops/README.md "Fetch".
"""

import itertools
import json
import os
import shlex
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.rl.cli_common import write_json
from scripts.rl.ops import fetch

from ._ops_gen_helpers import leaf, make_plan

REMOTE_ROOT = Path("/remote/run")
SAFE_OPTIONS = {"-a", "--prune-empty-dirs", "--ignore-existing", "--include=*/",
                "--exclude=*"}


def _payload(tmp_path: Path, plan=True, comparison: str | None = None) -> Path:
    payload = tmp_path / "payload"
    (payload / "cluster" / "receipts").mkdir(parents=True)
    if plan:
        write_json(payload / "experiment_plan.json", make_plan(REMOTE_ROOT, 2))
    write_json(payload / "cluster" / "tasks.json", {"tasks_version": 1})
    write_json(payload / "cluster" / "receipts" / "0001.json", {"submission": "0001"})
    if comparison is not None:
        (payload / "comparison").mkdir()
        (payload / "comparison" / "comparison.json").write_text(comparison)
    return payload


def _rsync(tmp_path: Path, monkeypatch, payload: Path | None, code: int = 0) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "rsync-calls.json"
    script = bin_dir / "rsync"
    script.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import json, shutil, sys
        from pathlib import Path
        log = Path({str(log)!r})
        calls = json.loads(log.read_text()) if log.exists() else []
        calls.append(sys.argv[1:])
        log.write_text(json.dumps(calls))
        payload = {str(payload) if payload else None!r}
        if payload and {code} == 0:
            shutil.copytree(payload, sys.argv[-1], dirs_exist_ok=True)
        sys.exit({code})
        """))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def _calls(log: Path) -> list:
    return json.loads(log.read_text()) if log.exists() else []


# --- argv --------------------------------------------------------------------


@pytest.mark.parametrize("size", [1, 2, len(fetch.CATEGORY_INCLUDES)])
def test_gen_no_selection_ever_adds_a_deleting_or_overwriting_option(tmp_path, size):
    for combo in itertools.combinations(sorted(fetch.CATEGORY_INCLUDES), size):
        argv = fetch.build_argv("/r", tmp_path / "d", list(combo))
        options = [token for token in argv[1:-2] if not token.startswith("--include=")
                   or token == "--include=*/"]
        assert set(options) <= SAFE_OPTIONS, combo
        assert argv[-4:-2] == ["--include=*/", "--exclude=*"]
        assert argv.count("--ignore-existing") == 1


@pytest.mark.parametrize("remote,expected", [
    ("u@h:/abs/run", "u@h:/abs/run/"), ("u@h:/abs/run/", "u@h:/abs/run/"),
    ("u@h:/abs/run///", "u@h:/abs/run/"), ("/local/run", "/local/run/"),
    ("rsync://h/mod/run", "rsync://h/mod/run/")])
def test_gen_the_source_always_ends_in_exactly_one_slash(tmp_path, remote, expected):
    argv = fetch.build_argv(remote, tmp_path / "d", [])
    assert argv[-2] == expected and argv[-1] == f"{tmp_path / 'd'}/"


@pytest.mark.parametrize("remote,local", [
    ("user@host:/abs/run", False), ("host:run", False), ("rsync://host/mod", False),
    ("/abs/run", True), ("relative/run", True), ("/abs/with:colon/run", True)])
def test_gen_local_and_remote_sources_are_told_apart(remote, local):
    assert (fetch._local_source(remote) is not None) is local


# --- destination rules -------------------------------------------------------


def test_gen_a_dry_run_needs_no_rsync_on_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    dest = tmp_path / "dest"
    assert fetch.main(["--remote", "u@h:/r/", "--dest", str(dest),
                       "--select", "logs", "--dry-run"]) == 0
    printed = shlex.split(capsys.readouterr().out.strip())
    assert printed == fetch.build_argv("u@h:/r/", dest, ["logs"])
    assert not dest.exists()


def test_gen_missing_destination_parents_are_created(tmp_path, monkeypatch):
    log = _rsync(tmp_path, monkeypatch, _payload(tmp_path))
    dest = tmp_path / "a" / "b" / "c"
    assert fetch.main(["--remote", "u@h:/r", "--dest", str(dest)]) == 0
    assert (dest / fetch.FETCH_MANIFEST_NAME).is_file()
    assert _calls(log)[0][-1] == f"{dest}/"


def test_gen_a_failed_rsync_leaves_no_manifest_even_after_partial_files(tmp_path,
                                                                       monkeypatch,
                                                                       capsys):
    _rsync(tmp_path, monkeypatch, None, code=23)
    dest = tmp_path / "dest"
    assert fetch.main(["--remote", "u@h:/r", "--dest", str(dest)]) == 1
    assert "rsync exited 23" in capsys.readouterr().err
    assert not (dest / fetch.FETCH_MANIFEST_NAME).exists()


# --- the manifest ------------------------------------------------------------


def _fetch(tmp_path, monkeypatch, payload, *extra, dest=None):
    log = _rsync(tmp_path, monkeypatch, payload)
    dest = dest or tmp_path / "dest"
    code = fetch.main(["--remote", "u@h:/remote/run/", "--dest", str(dest), *extra])
    return code, dest, log


def test_gen_repeated_updates_keep_every_earlier_manifest(tmp_path, monkeypatch):
    payload = _payload(tmp_path)
    code, dest, _ = _fetch(tmp_path, monkeypatch, payload)
    assert code == 0
    for expected in (1, 2):
        assert _fetch(tmp_path, monkeypatch, payload, "--update", dest=dest)[0] == 0
        assert (dest / f"fetch_manifest.{expected}.json").is_file()
    names = sorted(path.name for path in dest.glob("fetch_manifest*.json"))
    assert names == ["fetch_manifest.1.json", "fetch_manifest.2.json",
                     "fetch_manifest.json"]
    current = json.loads((dest / fetch.FETCH_MANIFEST_NAME).read_text())
    assert not any(entry["path"].startswith("fetch_manifest")
                   for entry in current["files"])


def _with_steps(payload: Path, states: dict) -> None:
    for index, (train, evaluate) in states.items():
        for part, status, name in (("train", train, "train_manifest.json"),
                                   ("eval", evaluate, "eval_manifest.json")):
            if status is None:
                continue
            out = payload / part / leaf(index)
            out.mkdir(parents=True)
            if status != "blocked":
                write_json(out / name, {"status": status})


@pytest.mark.parametrize("states,expected,incomplete", [
    ({0: ("completed", "completed"), 1: ("completed", "completed")},
     ["completed", "completed"], False),
    ({0: ("completed", "partial"), 1: (None, None)}, ["partial", "missing"], True),
    ({0: ("blocked", None), 1: ("completed", "completed")}, ["failed", "completed"],
     True),
])
def test_gen_task_states_are_read_from_rebased_fetched_manifests(tmp_path, monkeypatch,
                                                                 states, expected,
                                                                 incomplete):
    payload = _payload(tmp_path)
    _with_steps(payload, states)
    code, dest, _ = _fetch(tmp_path, monkeypatch, payload, "--select", "manifests")
    assert code == 0
    manifest = json.loads((dest / fetch.FETCH_MANIFEST_NAME).read_text())
    assert [row["state"] for row in manifest["tasks"]] == expected
    assert [row["id"] for row in manifest["tasks"]] == [leaf(0), leaf(1)]
    assert manifest["snapshot_of_incomplete_run"] is incomplete


@pytest.mark.parametrize("text,expected", [('{"status": "incomplete"}', "incomplete"),
                                           ("{nope", "unreadable")])
def test_gen_the_comparison_state_is_read_from_the_fetched_file(tmp_path, monkeypatch,
                                                                text, expected):
    code, dest, _ = _fetch(tmp_path, monkeypatch, _payload(tmp_path, comparison=text),
                           "--select", "comparison")
    assert code == 0
    assert json.loads((dest / fetch.FETCH_MANIFEST_NAME).read_text())["comparison"] \
        == expected
