"""Fetch include rules, destination safety, and provenance against a fake rsync."""

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.rl.cli_common import write_json
from scripts.rl.ops import fetch
from scripts.sim_support import find_mesh_root

REMOTE_ROOT = Path("/remote/run")
REMOTE = str(REMOTE_ROOT)
LEAVES = ("row-a/train-seed-1", "row-b/train-seed-1")
DONE = {leaf: ("completed", "completed") for leaf in LEAVES}


def _plan(root: Path) -> dict:
    """Plan with the shape build_plan produces, rooted on the remote filesystem."""
    steps = []
    for leaf in LEAVES:
        steps.append({"id": f"train/{leaf}", "kind": "train", "module": "stub.train",
                      "args": [], "output_dir": str(root / "train" / leaf),
                      "manifest": "train_manifest.json", "needs": []})
        steps.append({"id": f"evaluate/{leaf}", "kind": "evaluate",
                      "module": "stub.evaluate", "args": [],
                      "output_dir": str(root / "eval" / leaf),
                      "manifest": "eval_manifest.json", "needs": [f"train/{leaf}"]})
    steps.append({"id": "compare", "kind": "compare", "module": "stub.compare",
                  "args": ["--plan", str(root / "experiment_plan.json"),
                           "--output-dir", str(root / "comparison")],
                  "output_dir": str(root / "comparison"),
                  "manifest": "comparison.json", "needs": []})
    return {"experiment_plan_version": 1, "steps": steps}


def _payload(tmp_path: Path, statuses: dict | None = None,
             comparison: str | None = "complete") -> Path:
    """A fixture tree as it exists on the remote side, for the shim to copy."""
    payload = tmp_path / "payload"
    (payload / "cluster").mkdir(parents=True)
    write_json(payload / "experiment_plan.json", _plan(REMOTE_ROOT))
    write_json(payload / "cluster" / "tasks.json", {"tasks_version": 1, "tasks": []})
    for leaf, (train_status, eval_status) in (statuses or DONE).items():
        for part, status, name in (("train", train_status, "train_manifest.json"),
                                   ("eval", eval_status, "eval_manifest.json")):
            if status is None:
                continue
            out_dir = payload / part / leaf
            out_dir.mkdir(parents=True)
            if status == "blocked":
                (out_dir / "episode-0000").mkdir()
            else:
                write_json(out_dir / name, {"status": status})
    if comparison is not None:
        (payload / "comparison").mkdir()
        write_json(payload / "comparison" / "comparison.json",
                   {"status": comparison})
    return payload


def _fake_rsync(tmp_path: Path, monkeypatch, payload: Path | None = None,
                exit_code: int = 0) -> Path:
    """Install a PATH shim that records argv and copies the fixture tree."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "rsync-argv.json"
    script = bin_dir / "rsync"
    script.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import json, shutil, sys
        from pathlib import Path
        Path({str(log)!r}).write_text(json.dumps(sys.argv))
        payload = {repr(str(payload)) if payload is not None else None}
        if payload is not None and {exit_code} == 0:
            shutil.copytree(payload, sys.argv[-1], dirs_exist_ok=True)
        sys.exit({exit_code})
        """))
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


def _argv(log: Path) -> list[str]:
    return json.loads(log.read_text())


def _includes(argv: list[str]) -> list[str]:
    return [item.removeprefix("--include=") for item in argv
            if item.startswith("--include=") and item != "--include=*/"]


def _manifest(dest: Path) -> dict:
    return json.loads((dest / fetch.FETCH_MANIFEST_NAME).read_text())


CATEGORY_RULES = [
    ("comparison", ["comparison/comparison.json", "comparison/episodes.csv"]),
    ("manifests", ["train/**/train_manifest.json", "eval/**/eval_manifest.json",
                   "train/**/episode-*/rl_episode.json",
                   "eval/**/episode-*/rl_episode.json",
                   "cluster/records/**", "benchmark/**"]),
    ("models", ["train/**/maskable_ppo_mesh.zip", "train/**/best_model.zip"]),
    ("selection-logs", ["train/**/evaluations.npz"]),
    ("inputs", ["train/**/episode-*/inputs/**", "eval/**/episode-*/inputs/**"]),
    ("telemetry", ["train/**/episode-*/steps.jsonl",
                   "eval/**/episode-*/steps.jsonl"]),
    ("episode-data", ["train/**/episode-*/**", "eval/**/episode-*/**"]),
    ("logs", ["cluster/logs/**"]),
]


@pytest.mark.parametrize("category,rules", CATEGORY_RULES,
                         ids=[case[0] for case in CATEGORY_RULES])
def test_each_category_adds_its_include_rules(tmp_path, monkeypatch, category, rules):
    log = _fake_rsync(tmp_path, monkeypatch, _payload(tmp_path))
    dest = tmp_path / "dest"

    assert fetch.main(["--remote", REMOTE, "--dest", str(dest),
                       "--select", category]) == 0
    argv = _argv(log)
    assert _includes(argv) == list(fetch.ALWAYS_INCLUDE) + rules
    assert argv[-4:] == ["--include=*/", "--exclude=*", f"{REMOTE}/", f"{dest}/"]
    assert argv[1:4] == ["-a", "--prune-empty-dirs", "--ignore-existing"]


def test_the_always_included_files_need_no_selection(tmp_path, monkeypatch):
    log = _fake_rsync(tmp_path, monkeypatch, _payload(tmp_path))

    assert fetch.main(["--remote", REMOTE, "--dest", str(tmp_path / "dest")]) == 0
    assert _includes(_argv(log)) == list(fetch.ALWAYS_INCLUDE)


def test_two_categories_are_combined_in_order(tmp_path, monkeypatch):
    log = _fake_rsync(tmp_path, monkeypatch, _payload(tmp_path))

    assert fetch.main(["--remote", REMOTE, "--dest", str(tmp_path / "dest"),
                       "--select", "comparison,manifests"]) == 0
    assert _includes(_argv(log)) == (
        list(fetch.ALWAYS_INCLUDE) + list(fetch.CATEGORY_INCLUDES["comparison"])
        + list(fetch.CATEGORY_INCLUDES["manifests"]))


def test_an_unknown_category_is_refused(tmp_path, monkeypatch, capsys):
    log = _fake_rsync(tmp_path, monkeypatch, _payload(tmp_path))
    dest = tmp_path / "dest"

    assert fetch.main(["--remote", REMOTE, "--dest", str(dest),
                       "--select", "comparison,everything"]) == 1
    assert "unknown --select category 'everything'" in capsys.readouterr().err
    assert not log.exists() and not dest.exists()


@pytest.mark.parametrize("extra,populate", [
    ([], False), (["--select", "episode-data"], False), (["--update"], True)],
    ids=["plain", "episode-data", "update"])
def test_ignore_existing_is_always_passed(tmp_path, monkeypatch, extra, populate):
    log = _fake_rsync(tmp_path, monkeypatch, _payload(tmp_path))
    dest = tmp_path / "dest"
    if populate:
        dest.mkdir()
        (dest / "keep.txt").write_text("local")

    assert fetch.main(["--remote", REMOTE, "--dest", str(dest), *extra]) == 0
    assert "--ignore-existing" in _argv(log)
    if populate:
        assert (dest / "keep.txt").read_text() == "local"


def test_a_non_empty_destination_is_refused_without_update(tmp_path, monkeypatch,
                                                           capsys):
    log = _fake_rsync(tmp_path, monkeypatch, _payload(tmp_path))
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "keep.txt").write_text("local")

    assert fetch.main(["--remote", REMOTE, "--dest", str(dest)]) == 1
    assert "pass --update" in capsys.readouterr().err
    assert not log.exists()
    assert fetch.main(["--remote", REMOTE, "--dest", str(dest), "--update"]) == 0


def _safety_targets() -> dict:
    return {"root": "/", "home": str(Path.home()),
            "mesh root": str(find_mesh_root())}


@pytest.mark.parametrize("label", ["root", "home", "mesh root"])
def test_dangerous_destinations_are_refused(tmp_path, monkeypatch, capsys, label):
    log = _fake_rsync(tmp_path, monkeypatch, _payload(tmp_path))

    assert fetch.main(["--remote", REMOTE, "--dest", _safety_targets()[label],
                       "--update"]) == 1
    assert "refusing to fetch into" in capsys.readouterr().err
    assert not log.exists()


def test_a_destination_symlink_cannot_bypass_the_checks(tmp_path, monkeypatch,
                                                        capsys):
    log = _fake_rsync(tmp_path, monkeypatch, _payload(tmp_path))
    link = tmp_path / "shortcut"
    link.symlink_to(Path.home())

    assert fetch.main(["--remote", REMOTE, "--dest", str(link), "--update"]) == 1
    assert "refusing to fetch into the home directory" in capsys.readouterr().err
    assert not log.exists()


def test_a_local_source_inside_the_destination_is_refused(tmp_path, monkeypatch,
                                                          capsys):
    log = _fake_rsync(tmp_path, monkeypatch, _payload(tmp_path))
    dest = tmp_path / "dest"
    source = dest / "run"
    source.mkdir(parents=True)

    assert fetch.main(["--remote", str(source), "--dest", str(dest),
                       "--update"]) == 1
    assert "is inside the destination" in capsys.readouterr().err
    assert fetch.main(["--remote", str(dest), "--dest", str(dest),
                       "--update"]) == 1
    assert not log.exists()


def test_a_missing_rsync_is_reported_not_crashed(tmp_path, monkeypatch, capsys):
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    dest = tmp_path / "dest"

    assert fetch.main(["--remote", REMOTE, "--dest", str(dest)]) == 1
    assert "rsync was not found on PATH" in capsys.readouterr().err
    assert not dest.exists()


TASK_STATES = [
    ("both done", {"row-a/train-seed-1": ("completed", "completed"),
                   "row-b/train-seed-1": ("completed", "completed")},
     ["completed", "completed"], False),
    ("partial evaluation", {"row-a/train-seed-1": ("completed", "partial"),
                            "row-b/train-seed-1": ("completed", "completed")},
     ["partial", "completed"], True),
    ("blocked training", {"row-a/train-seed-1": ("blocked", None),
                          "row-b/train-seed-1": ("completed", "completed")},
     ["failed", "completed"], True),
    ("nothing fetched", {"row-a/train-seed-1": (None, None),
                         "row-b/train-seed-1": ("completed", "completed")},
     ["missing", "completed"], True),
]


@pytest.mark.parametrize("statuses,states,incomplete",
                         [case[1:] for case in TASK_STATES],
                         ids=[case[0] for case in TASK_STATES])
def test_task_states_come_from_the_rebased_remote_paths(tmp_path, monkeypatch,
                                                        statuses, states,
                                                        incomplete):
    payload = _payload(tmp_path, statuses)
    _fake_rsync(tmp_path, monkeypatch, payload)
    dest = tmp_path / "dest"

    assert fetch.main(["--remote", REMOTE, "--dest", str(dest),
                       "--select", "manifests,comparison"]) == 0
    manifest = _manifest(dest)
    assert [task["state"] for task in manifest["tasks"]] == states
    assert [task["index"] for task in manifest["tasks"]] == [0, 1]
    assert [task["id"] for task in manifest["tasks"]] == list(LEAVES)
    assert manifest["snapshot_of_incomplete_run"] is incomplete
    assert (dest / "experiment_plan.json").read_bytes() == (
        payload / "experiment_plan.json").read_bytes()


def test_unfetched_manifests_are_never_reported_as_missing_tasks(tmp_path,
                                                                 monkeypatch):
    _fake_rsync(tmp_path, monkeypatch, _payload(tmp_path))
    dest = tmp_path / "dest"

    assert fetch.main(["--remote", REMOTE, "--dest", str(dest),
                       "--select", "models"]) == 0
    manifest = _manifest(dest)
    assert [task["state"] for task in manifest["tasks"]] == ["not_fetched"] * 2
    assert manifest["snapshot_of_incomplete_run"] is None
    assert manifest["comparison"] == "not_fetched"
    assert manifest["selection"] == ["models"]


COMPARISONS = [("complete", "complete"), ("incomplete", "incomplete"),
               (None, "absent")]


@pytest.mark.parametrize("status,expected", COMPARISONS,
                         ids=["complete", "incomplete", "absent"])
def test_the_comparison_state_is_read_when_selected(tmp_path, monkeypatch, status,
                                                    expected):
    _fake_rsync(tmp_path, monkeypatch, _payload(tmp_path, comparison=status))
    dest = tmp_path / "dest"

    assert fetch.main(["--remote", REMOTE, "--dest", str(dest),
                       "--select", "comparison"]) == 0
    assert _manifest(dest)["comparison"] == expected


def test_the_manifest_records_remote_argv_and_file_digests(tmp_path, monkeypatch):
    _fake_rsync(tmp_path, monkeypatch, _payload(tmp_path))
    dest = tmp_path / "dest"

    assert fetch.main(["--remote", REMOTE, "--dest", str(dest),
                       "--select", "comparison"]) == 0
    manifest = _manifest(dest)
    assert manifest["fetch_manifest_version"] == 1
    assert manifest["remote"] == REMOTE
    assert manifest["argv"][0] == "rsync" and "--ignore-existing" in manifest["argv"]
    assert "fetched_at" in manifest
    paths = {entry["path"] for entry in manifest["files"]}
    assert "experiment_plan.json" in paths and "cluster/tasks.json" in paths
    assert fetch.FETCH_MANIFEST_NAME not in paths
    for entry in manifest["files"]:
        assert entry["bytes"] == (dest / entry["path"]).stat().st_size
        assert len(entry["sha256"]) == 64


def test_update_keeps_the_earlier_manifest(tmp_path, monkeypatch):
    _fake_rsync(tmp_path, monkeypatch, _payload(tmp_path))
    dest = tmp_path / "dest"

    assert fetch.main(["--remote", REMOTE, "--dest", str(dest)]) == 0
    first = _manifest(dest)
    assert fetch.main(["--remote", REMOTE, "--dest", str(dest), "--update",
                       "--select", "manifests"]) == 0
    assert json.loads((dest / "fetch_manifest.1.json").read_text()) == first
    assert _manifest(dest)["selection"] == ["manifests"]


def test_a_failed_rsync_writes_no_manifest(tmp_path, monkeypatch, capsys):
    log = _fake_rsync(tmp_path, monkeypatch, _payload(tmp_path), exit_code=23)
    dest = tmp_path / "dest"

    assert fetch.main(["--remote", REMOTE, "--dest", str(dest),
                       "--select", "manifests"]) == 1
    assert "rsync exited 23" in capsys.readouterr().err
    assert log.exists()
    assert not (dest / fetch.FETCH_MANIFEST_NAME).exists()


def test_dry_run_prints_the_argv_and_writes_nothing(tmp_path, monkeypatch, capsys):
    log = _fake_rsync(tmp_path, monkeypatch, _payload(tmp_path))
    dest = tmp_path / "dest"

    assert fetch.main(["--remote", REMOTE, "--dest", str(dest),
                       "--select", "comparison", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("rsync -a --prune-empty-dirs --ignore-existing")
    assert f"{dest}/" in out
    assert not log.exists() and not dest.exists()
