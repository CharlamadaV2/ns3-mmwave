"""Shared CLI options and provenance helpers for the mesh-sim RL lifecycle tools."""

import hashlib
import importlib.metadata
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from scripts.rl.bootstrap_venv import DIRECT_DEPS
from scripts.sim_support import find_mesh_root
from scripts.rl.env.config import read_scenario_seed
from scripts.rl.env.decisions import (OBS_VECTOR_MODES, PREFERENCE_MODES,
                                      DecisionRecordSettings, resolve_decision_records)
from scripts.rl.env.selection import TELEMETRY_MODES, RlSelection, resolve_selection

MANIFEST_NAME = "train_manifest.json"
MODEL_BASENAME = "maskable_ppo_mesh"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_versions() -> dict:
    versions = {}
    for mod, distribution in DIRECT_DEPS.items():
        try:
            versions[mod] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[mod] = None
    return versions


def make_out_dir(output_dir: str) -> str:
    if output_dir:
        out_dir = output_dir
    else:
        now = datetime.now()
        out_dir = os.path.join("outputs", now.strftime("%Y-%m"),
                               now.strftime("%d"), now.strftime("%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def automatic_output_root(name: str) -> Path:
    """Find a fresh path in the simulator's standard timestamped output tree."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise ValueError(f"automatic output name must be a lowercase slug, got {name!r}")
    now = datetime.now()
    parent = find_mesh_root() / "outputs" / now.strftime("%Y-%m") / now.strftime("%d")
    stem = f"{now:%H-%M-%S}-{name}"
    candidate = parent / stem
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{stem}-{suffix}"
        suffix += 1
    return candidate


def has_previous_run(out_dir: str) -> str | None:
    for name in (MANIFEST_NAME, f"{MODEL_BASENAME}.zip"):
        if os.path.exists(os.path.join(out_dir, name)):
            return name
    return None


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload) -> None:
    """Write JSON atomically so a reader never sees a half-written manifest."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    try:
        with open(temp, "w") as fh:
            json.dump(payload, fh, indent=2, allow_nan=False)
            fh.write("\n")
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    os.replace(temp, path)


def add_scenario_arguments(parser, run_config_required: bool = True) -> None:
    """Add the scenario options every lifecycle CLI shares."""
    parser.add_argument("--sim-binary", required=True,
                        help="Path to mesh-sim executable")
    parser.add_argument("--run-config", required=run_config_required,
                        help="run.ini with an [rl] section")
    parser.add_argument("--band", choices=["mmwave", "sub-6"], default=None,
                        help="Override the scenario band; omitted -> scenario decides")


def add_selection_arguments(parser) -> None:
    """Add the observation/reward/telemetry options resolved by env/selection.py."""
    parser.add_argument("--observation-preset", default=None,
                        help="Named observation preset; omitted -> run.ini or raw_links_v1")
    parser.add_argument("--reward-components", default=None,
                        help="Comma-separated reward components; omitted -> the C++ reward")
    parser.add_argument("--reward-weights", default=None,
                        help="Comma-separated weights, one per reward component")
    parser.add_argument("--telemetry", default=None, choices=list(TELEMETRY_MODES),
                        help="steps writes <episode-dir>/steps.jsonl")
    parser.add_argument("--telemetry-every", default=None,
                        help="Save every kth policy decision (requires --telemetry steps)")


def selection_from_args(args) -> RlSelection:
    """Resolve the policy selection from parsed CLI arguments; raises ValueError."""
    return resolve_selection(
        args.run_config,
        observation_preset=args.observation_preset,
        reward_components=args.reward_components,
        reward_weights=args.reward_weights,
        telemetry=args.telemetry,
        telemetry_every=args.telemetry_every,
    )


def add_decision_record_arguments(parser) -> None:
    """Add the opt-in decision-record options resolved by env/decisions.py."""
    parser.add_argument("--decision-records", action="store_true", default=None,
                        help="Write <episode-dir>/policy_decisions.jsonl")
    parser.add_argument("--decision-records-every", type=int, default=None,
                        help="Save every kth decision (requires --decision-records)")
    parser.add_argument("--decision-records-max-bytes", type=int, default=None,
                        help="Per-episode decision-record file cap in bytes; 0 = no cap")
    parser.add_argument("--decision-records-obs-vector", default=None,
                        choices=list(OBS_VECTOR_MODES),
                        help="always stores the observation vector in every record")
    parser.add_argument("--decision-records-preferences", default=None,
                        choices=list(PREFERENCE_MODES),
                        help="off disables model action-preference capture")


def decision_records_from_args(args) -> DecisionRecordSettings:
    """Resolve decision-record settings from parsed CLI arguments; raises ValueError."""
    return resolve_decision_records(
        enabled=args.decision_records,
        obs_vector=args.decision_records_obs_vector,
        preferences=args.decision_records_preferences,
        record_every=args.decision_records_every,
        max_bytes_per_episode=args.decision_records_max_bytes,
    )


def resolve_seed(cli_seed: int | None, run_config: str) -> tuple[int, str]:
    """Return the seed and where it came from: the CLI flag or [scenario] seed."""
    if cli_seed is not None:
        return int(cli_seed), "cli"
    ini_seed = read_scenario_seed(run_config)
    if ini_seed is None:
        raise ValueError("No seed available: pass m-ppo --seed <int> or set "
                         f"[scenario] seed in {run_config}")
    return ini_seed, "run.ini"
