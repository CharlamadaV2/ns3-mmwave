"""Receipt bookkeeping: lock and allocation contention, corrupt files, state transitions.

Expectations come from scripts/rl/ops/README.md "Receipts and the submission
protocol" and "Layout under the output root".
"""

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path


from scripts.rl.ops import receipts
from scripts.sim_support import find_mesh_root

N_PROCS = 8


def _race(tmp_path: Path, body: str) -> list[str]:
    """Start N processes that wait on a go-file, then run `body`; return stdout lines."""
    go = tmp_path / "go"
    script = tmp_path / "racer.py"
    script.write_text(textwrap.dedent(f"""\
        import sys, time
        from pathlib import Path
        sys.path.insert(0, {str(find_mesh_root())!r})
        from scripts.rl.ops import receipts
        go = Path({str(go)!r})
        while not go.exists():
            time.sleep(0.001)
        root = Path({str(tmp_path / "run")!r})
        """) + textwrap.dedent(body))
    procs = [subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
             for _ in range(N_PROCS)]
    time.sleep(0.3)
    go.write_text("go")
    outputs = []
    for proc in procs:
        out, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err
        outputs.append(out.strip())
    return outputs


def test_gen_exactly_one_of_many_concurrent_submitters_gets_the_lock(tmp_path):
    outputs = _race(tmp_path, """\
        try:
            receipts.acquire_lock(root)
            print("acquired")
        except ValueError as exc:
            print("refused" if "Remove it by hand" in str(exc) else f"odd {exc}")
        """)
    assert sorted(outputs) == ["acquired"] + ["refused"] * (N_PROCS - 1)
    holder = json.loads((tmp_path / "run" / "cluster" / "submit.lock").read_text())
    assert set(holder) == {"pid", "host", "created_at"}


def test_gen_concurrent_allocations_never_share_a_number(tmp_path):
    outputs = _race(tmp_path, """\
        submission, path = receipts.allocate(root)
        print(submission)
        """)
    assert sorted(outputs) == [f"{n:04d}" for n in range(1, N_PROCS + 1)]


# --- loading -----------------------------------------------------------------


def _receipts_dir(tmp_path: Path) -> Path:
    path = receipts.layout(tmp_path)["receipts"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_gen_load_orders_by_submission_and_ignores_temp_and_foreign_files(tmp_path):
    directory = _receipts_dir(tmp_path)
    for number in ("0003", "0001", "0002"):
        (directory / f"{number}.json").write_text(json.dumps({"submission": number}))
    (directory / "0004.json.tmp").write_text("{half")
    (directory / "notes.txt").write_text("hello")
    assert [entry["submission"] for entry in receipts.load(tmp_path)] \
        == ["0001", "0002", "0003"]


# --- allocation --------------------------------------------------------------


# --- receipt bodies and transitions ------------------------------------------


# --- task table --------------------------------------------------------------
