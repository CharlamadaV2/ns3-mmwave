#!/usr/bin/env python3
"""Summarize a training run: manifest provenance, model files, and their digests."""

import argparse
import json
import sys
from pathlib import Path

from scripts.rl.cli_common import MANIFEST_NAME, package_versions, sha256_file

# raw_links_v1 records bounds that check_schema does not treat as structural.
_UNBOUNDED_PRESET = "raw_links_v1"


def _model_entry(run_dir: Path, role: str, path, recorded: str | None,
                 num_timesteps=None) -> dict:
    entry = {"role": role, "path": path, "sha256_recorded": recorded,
             "num_timesteps": num_timesteps, "exists": False, "digest_ok": None}
    if not path:
        return entry
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = run_dir / resolved
    entry["path"] = str(resolved)
    entry["exists"] = resolved.is_file()
    if entry["exists"] and recorded:
        entry["digest_ok"] = sha256_file(resolved) == recorded
    return entry


def _models(run_dir: Path, manifest: dict) -> list[dict]:
    entries = [
        _model_entry(run_dir, "final", manifest.get("model_path"),
                     manifest.get("model_sha256")),
    ]
    if manifest.get("best_model_path"):
        entries.append(_model_entry(run_dir, "best", manifest.get("best_model_path"),
                                    manifest.get("best_model_sha256")))
    for checkpoint in manifest.get("checkpoints") or []:
        entries.append(_model_entry(run_dir, "checkpoint", checkpoint.get("path"),
                                    checkpoint.get("sha256"),
                                    checkpoint.get("num_timesteps")))
    return entries


def _versions(manifest: dict) -> list[dict]:
    recorded = manifest.get("package_versions") or {}
    installed = package_versions()
    rows = []
    for name in sorted(set(recorded) | set(installed)):
        was, now = recorded.get(name), installed.get(name)
        rows.append({"package": name, "recorded": was, "installed": now,
                     "match": was == now})
    return rows


def build_report(run_dir: Path, manifest: dict) -> dict:
    """Everything inspect prints, in one JSON-serializable dict."""
    contract = manifest.get("contract") or {}
    observation = manifest.get("observation_schema") or {}
    reward = manifest.get("reward_schema") or {}
    note = None
    if observation.get("schema_id") == _UNBOUNDED_PRESET:
        note = f"bounds not structural for {_UNBOUNDED_PRESET}"
    models = _models(run_dir, manifest)
    return {
        "run_dir": str(run_dir),
        "manifest_path": str(run_dir / MANIFEST_NAME),
        "manifest_version": manifest.get("manifest_version"),
        "status": manifest.get("status"),
        "error": manifest.get("error"),
        "algorithm": manifest.get("algorithm"),
        "seed": manifest.get("seed"),
        "seed_source": manifest.get("seed_source"),
        "control_mode": manifest.get("control_mode"),
        "band": manifest.get("band"),
        "contract": {
            "contract": contract.get("contract"),
            "num_mesh_nodes": contract.get("num_mesh_nodes"),
            "max_controlled_nodes": contract.get("max_controlled_nodes"),
            "obs_dim": contract.get("obs_dim"),
            "mask_dim": contract.get("mask_dim"),
            "num_decisions": contract.get("num_decisions"),
            "reward_type": contract.get("reward_type"),
            "reward_window": contract.get("reward_window"),
            "slot_node_ids": contract.get("slot_node_ids"),
        },
        "selection": manifest.get("selection"),
        "observation_schema": {
            "schema_id": observation.get("schema_id"),
            "dtype": observation.get("dtype"),
            "obs_dim": observation.get("obs_dim"),
            "sha256": observation.get("sha256"),
            "note": note,
        },
        "reward_schema": {
            "authority": reward.get("authority"),
            "components": reward.get("components"),
            "weights": reward.get("weights"),
            "reward_type": reward.get("reward_type"),
            "reward_window": reward.get("reward_window"),
            "sha256": reward.get("sha256"),
        },
        "scenario_identity": manifest.get("scenario_identity"),
        "models": models,
        "evaluation": manifest.get("evaluation"),
        "best_mean_reward": manifest.get("best_mean_reward"),
        "hyperparameters": manifest.get("hyperparameters"),
        "package_versions": _versions(manifest),
        "python_version": manifest.get("python_version"),
        "platform": manifest.get("platform"),
    }


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "[]"
    return str(value)


def print_report(report: dict) -> None:
    contract = report["contract"]
    observation = report["observation_schema"]
    reward = report["reward_schema"]

    print(f"run dir         {report['run_dir']}")
    print(f"manifest        version {_fmt(report['manifest_version'])}, "
          f"status {_fmt(report['status'])}")
    if report["error"]:
        print(f"error           {report['error']}")
    print(f"algorithm       {_fmt(report['algorithm'])}")
    print(f"seed            {_fmt(report['seed'])} (source: {_fmt(report['seed_source'])})")
    print(f"control mode    {_fmt(report['control_mode'])}, band {_fmt(report['band'])}")
    print(f"contract        {_fmt(contract['contract'])}")
    print(f"  nodes N       {_fmt(contract['num_mesh_nodes'])}   "
          f"slots M {_fmt(contract['max_controlled_nodes'])}")
    print(f"  obs_dim       {_fmt(contract['obs_dim'])}   "
          f"mask_dim {_fmt(contract['mask_dim'])}   "
          f"num_decisions {_fmt(contract['num_decisions'])}")

    selection = report["selection"] or {}
    sources = selection.get("source", {}) if isinstance(selection, dict) else {}
    print("selection")
    for key in ("observation_preset", "reward_components", "reward_weights",
                "telemetry", "telemetry_every"):
        if key in selection:
            print(f"  {key:<18}{_fmt(selection[key])} "
                  f"(source: {_fmt(sources.get(key))})")

    print(f"observation     {_fmt(observation['schema_id'])} "
          f"dtype {_fmt(observation['dtype'])} dim {_fmt(observation['obs_dim'])}")
    print(f"  sha256        {_fmt(observation['sha256'])}")
    if observation["note"]:
        print(f"  note          {observation['note']}")
    print(f"reward          authority {_fmt(reward['authority'])}")
    if reward["components"]:
        print(f"  components    {_fmt(reward['components'])}")
        print(f"  weights       {_fmt(reward['weights'])}")
    print(f"  cpp reward    {_fmt(reward['reward_type'] or contract['reward_type'])} "
          f"window {_fmt(reward['reward_window'] or contract['reward_window'])}")
    print(f"  sha256        {_fmt(reward['sha256'])}")

    print("scenario identity")
    for key, value in (report["scenario_identity"] or {}).items():
        print(f"  {key:<22}{_fmt(value)}")

    print("models")
    for entry in report["models"]:
        digest = {True: "ok", False: "MISMATCH", None: "-"}[entry["digest_ok"]]
        steps = ("" if entry["num_timesteps"] is None
                 else f" steps={entry['num_timesteps']}")
        print(f"  {entry['role']:<11} exists={str(entry['exists']).lower()} "
              f"digest={digest}{steps} {_fmt(entry['path'])}")

    evaluation = report["evaluation"]
    if evaluation:
        print("evaluation      " + ", ".join(
            f"{key}={_fmt(value)}" for key, value in evaluation.items()))
    else:
        print("evaluation      off")
    print(f"best_mean_reward {_fmt(report['best_mean_reward'])}")

    print("package versions (recorded vs installed)")
    for row in report["package_versions"]:
        flag = "" if row["match"] else "   <- differs"
        print(f"  {row['package']:<20}{_fmt(row['recorded'])} / "
              f"{_fmt(row['installed'])}{flag}")
    plat = report["platform"] or {}
    print(f"python          {_fmt(report['python_version'])}")
    print(f"platform        {_fmt(plat.get('system'))} {_fmt(plat.get('machine'))}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Summarize a mesh-sim RL training run and its saved models")
    p.add_argument("--run-dir", required=True,
                   help="Training output directory containing train_manifest.json")
    p.add_argument("--json", action="store_true", help="Print the report as JSON")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        print(f"Cannot read {manifest_path}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1
    if not isinstance(manifest, dict):
        print(f"Cannot read {manifest_path}: manifest is not a JSON object",
              file=sys.stderr)
        return 1

    report = build_report(run_dir, manifest)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    for entry in report["models"]:
        if entry["path"] and (not entry["exists"] or entry["digest_ok"] is False):
            print(f"Model problem: {entry['role']} {entry['path']} "
                  f"(exists={entry['exists']}, digest_ok={entry['digest_ok']})",
                  file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
