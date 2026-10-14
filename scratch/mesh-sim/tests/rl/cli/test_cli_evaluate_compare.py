"""Gap tests for the scripts/rl/evaluate.py and scripts/rl/compare.py CLIs.

Sources: scripts/rl/README.md (evaluate flag + exit tables: 0 clean, 2 counters,
1 error/incomplete; compare flags: exactly one of --eval-dirs/--plan, --output-dir
must not contain a training or evaluation manifest, --baselines default "every
non-model policy"), scripts/rl/CLAUDE.md (exit 2 = warning). Placement policies
(geometric/optimization) are out of scope here. Simulator = fake_sim.py.
"""

import json
from pathlib import Path

import pytest

from scripts.rl import compare as compare_cli
from scripts.rl import evaluate as evaluate_cli
from scripts.rl.tests.test_policy_comparison import SEEDS, write_eval


def _eval_argv(sim_binary, run_config, out_dir, *extra) -> list[str]:
    argv = ["--sim-binary", sim_binary, "--run-config", run_config,
            "--seeds", "1", "--policies", "hold"]
    if out_dir is not None:
        argv += ["--output-dir", str(out_dir)]
    return argv + list(extra)


# 1. evaluate argument errors --------------------------------------------------------


# 2. evaluate automatic output root and policy whitespace --------------------------

def test_automatic_output_root_is_announced_and_used(sim_binary, multi_run_config,
                                                     tmp_path, monkeypatch, capsys):
    auto = tmp_path / "auto-root"
    calls = []

    def fake_root(name):
        calls.append(name)
        return auto

    monkeypatch.setattr(evaluate_cli, "automatic_output_root", fake_root)
    argv = _eval_argv(sim_binary, multi_run_config, None)
    argv[argv.index("--policies") + 1] = " hold , random_valid "
    assert evaluate_cli.main(argv) == 0
    out = capsys.readouterr().out
    assert calls == ["rl-evaluation"]
    assert f"Evaluation output: {auto.resolve()}" in out
    manifest = json.loads((auto / "eval_manifest.json").read_text())
    assert set(manifest["policies"]) == {"hold", "random_valid"}
    assert manifest["status"] == "completed"


# 3. evaluate exit-code helper ------------------------------------------------------

def _episode(status="completed", revalidated=0, violations=0):
    return {"status": status, "revalidated_slots_total": revalidated,
            "mask_violations": violations}


@pytest.mark.parametrize("episodes,expected,code", [
    ([_episode(), _episode()], 2, 0),
    ([_episode(revalidated=None, violations=None), _episode()], 2, 0),
    ([_episode(revalidated=1), _episode()], 2, 2),
    ([_episode(violations=3), _episode()], 2, 2),
    ([_episode(), _episode(status="failed")], 2, 1),
    ([_episode(status="not_run", revalidated=4)], 1, 1),     # failure beats warning
    ([_episode()], 2, 1),                                    # a missing record
    ([_episode(), _episode(), _episode()], 2, 1),            # an unexpected record
])
def test_exit_code_rules(episodes, expected, code, capsys):
    manifest = {"policies": {"hold": {"episodes": episodes}}}
    assert evaluate_cli._exit_code(manifest, expected) == code
    if code == 1:
        assert "Not every evaluation episode completed" in capsys.readouterr().err


# 4. compare argument and input errors ----------------------------------------------


def test_compare_absent_baseline_is_incomplete_not_zero(tmp_path, capsys):
    eval_dir = write_eval(tmp_path / "eval", {s: 0.8 for s in SEEDS})
    out = tmp_path / "cmp"
    assert compare_cli.main(["--eval-dirs", eval_dir, "--output-dir", str(out),
                             "--baselines", "random_valid"]) == 1
    payload = json.loads((out / "comparison.json").read_text())
    assert payload["status"] == "incomplete"
    primary = [c for c in payload["evaluations"][0]["comparisons"]
               if c["metric"] == payload["primary_metric"]]
    assert primary and all(c["mean_difference"] is None for c in primary)
    assert "no usable evaluation seeds" in capsys.readouterr().out


def test_compare_seed_overlap_exits_two(tmp_path):
    eval_dir = write_eval(tmp_path / "eval", {s: 0.8 for s in SEEDS}, held_out=False)
    out = tmp_path / "cmp"
    assert compare_cli.main(["--eval-dirs", eval_dir, "--output-dir", str(out)]) == 2
    assert (out / "comparison.json").is_file()


def test_compare_writes_no_tmp_files(tmp_path):
    eval_dir = write_eval(tmp_path / "eval", {s: 0.8 for s in SEEDS})
    out = tmp_path / "cmp"
    assert compare_cli.main(["--eval-dirs", eval_dir, "--output-dir", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["comparison.json", "episodes.csv"]
    header = (out / "episodes.csv").read_text().splitlines()[0]
    assert header.split(",")[0]
