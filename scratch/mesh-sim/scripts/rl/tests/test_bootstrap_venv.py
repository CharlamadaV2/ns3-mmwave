"""Dependency checks reject mismatched versions without installing packages."""

import subprocess
import sys

from scripts.rl.bootstrap_venv import DIRECT_DEPS, MESH_SIM_ROOT, _VERSION_PROBE, read_pins


def test_direct_pins_match_probe_dependencies():
    assert set(read_pins(MESH_SIM_ROOT / "requirements.txt")) == set(DIRECT_DEPS.values())


def test_version_probe_rejects_mismatch():
    probe = _VERSION_PROBE.format(deps={"json": "pytest"}, expected={"pytest": "0.0.invalid"})
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 1
    assert "expected 0.0.invalid" in result.stderr
