#!/usr/bin/env python3
"""Evaluate a saved policy and its baselines on mesh-sim scenarios."""

import argparse
import json
import os
import sys
from pathlib import Path

from scripts.rl.cli_common import (MANIFEST_NAME, add_decision_record_arguments,
                                   add_scenario_arguments, add_selection_arguments,
                                   automatic_output_root,
                                   decision_records_from_args, package_versions,
                                   selection_from_args, sha256_file)
from scripts.rl.env.config import read_scenario_identity
from scripts.rl.env.decisions import DecisionContext, DecisionRecording
from scripts.rl.env.mesh_env import MeshRlEnv
from scripts.rl.policy.bundle import (eval_selection, load_model, policy_weights_sha256,
                                      read_bundle, seed_roles, selection_from_manifest,
                                      training_provenance)
from scripts.rl.policy.compat import check_compatibility
from scripts.rl.policy.evaluate import (EVAL_MANIFEST_NAME, HoldPolicy, ModelPolicy,
                                        PolicySpec, Prepared, RandomValidPolicy,
                                        evaluate)
from scripts.rl.policy.preferences import PreferenceCapture
from scripts.sim_support import parse_seed_spec

DEFAULT_POLICIES = ("model", "hold", "random_valid")
PLACEMENT_POLICIES = ("geometric", "optimization")
POLICY_NAMES = DEFAULT_POLICIES + PLACEMENT_POLICIES
SELECTION_FLAGS = ("observation_preset", "reward_components", "reward_weights",
                   "telemetry", "telemetry_every")


def mask_fn(env):
    return env.unwrapped.action_masks()


def _parse_policies(raw: str) -> list[str]:
    names = [token.strip() for token in raw.split(",") if token.strip()]
    if not names:
        raise ValueError("--policies must list at least one policy")
    unknown = [name for name in names if name not in POLICY_NAMES]
    if unknown:
        raise ValueError(f"--policies has unknown entries {unknown}; "
                         f"valid choices: {list(POLICY_NAMES)}")
    if len(set(names)) != len(names):
        raise ValueError(f"--policies must be distinct, got {names}")
    return names


def _check_output_dir(output_dir: str, run_dir: str | None) -> None:
    out = Path(output_dir).resolve()
    if run_dir:
        run = Path(run_dir).resolve()
        if out == run or run in out.parents:
            raise ValueError(
                f"--output-dir {out} is inside the training run {run}; "
                "evaluation never writes into a training directory")
    for name in (MANIFEST_NAME, EVAL_MANIFEST_NAME):
        if (out / name).exists():
            raise ValueError(f"Refusing to start: {out} already contains {name}")


def _baseline_spec(name: str) -> PolicySpec:
    policy = HoldPolicy() if name == "hold" else RandomValidPolicy()
    return PolicySpec(name, lambda env, first_seed: Prepared(policy))


def _placement_spec(prepared) -> PolicySpec:
    """Hold the prepared layout; the plan is applied through the effective run.ini."""
    policy = HoldPolicy()
    return PolicySpec(prepared.method, lambda env, first_seed: Prepared(policy),
                      metadata=prepared.metadata)


def _prepare_placements(args, policies: list[str], run_config: str, band: str | None,
                        seeds: list[int]) -> dict:
    """Plan every requested placement method before any episode runs."""
    methods = [name for name in policies if name in PLACEMENT_POLICIES]
    if not methods:
        return {}
    from scripts.baselines import adapter

    return {method: adapter.prepare(
                run_config, method, Path(args.output_dir) / method / "baseline",
                mode="evaluation", eval_root=args.output_dir, band=band,
                simulation_seeds=seeds, sim_binary=args.sim_binary)
            for method in methods}


def _model_spec(bundle, live_identity: dict, band: str | None,
                allow_different_scenario: bool,
                capture: PreferenceCapture | None = None) -> PolicySpec:
    def build(env, first_seed: int) -> Prepared:
        # The compatibility reset is the first model episode; no extra episode is left.
        initial = env.reset(seed=first_seed, options={"seed_source": "eval"})
        report = check_compatibility(
            bundle.manifest, env, live_identity=live_identity, live_band=band,
            allow_different_scenario=allow_different_scenario)
        model = load_model(bundle, env, mask_fn)
        return Prepared(ModelPolicy(model, capture), initial,
                        {"compatibility": report.describe()})

    return PolicySpec("model", build)


def _model_identity(bundle) -> dict:
    """Decision-record identity of the evaluated model file."""
    path = Path(bundle.model_path)
    run_dir = Path(bundle.run_dir)
    try:
        recorded = str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        recorded = str(path.resolve())
    versions = package_versions()
    return {
        "model_sha256": bundle.model_sha256,
        "policy_weights_sha256": policy_weights_sha256(path),
        "model_selection": bundle.selection,
        "model_path_recorded": recorded,
        "num_timesteps": bundle.num_timesteps,
        "train_manifest_sha256": sha256_file(run_dir / MANIFEST_NAME),
        "inference": {"deterministic": True, "device": "cpu",
                      "stable_baselines3": versions["stable_baselines3"],
                      "sb3_contrib": versions["sb3_contrib"],
                      "torch": versions["torch"]},
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate mesh-sim policies")
    add_scenario_arguments(p, run_config_required=False)
    add_selection_arguments(p)
    add_decision_record_arguments(p)
    p.add_argument("--run-dir", default=None,
                   help="Training output directory holding train_manifest.json")
    p.add_argument("--model", default="final",
                   help="final, best, or checkpoints/<file>.zip inside --run-dir")
    p.add_argument("--output-dir", default=None,
                   help="Override the automatic timestamped evaluation output root")
    p.add_argument("--seeds", required=True,
                   help="Comma-separated distinct episode seeds; A-B is an inclusive range")
    p.add_argument("--label", default=None,
                   help="Name shared by evaluations of independently trained models")
    p.add_argument("--allow-seed-overlap", action="store_true",
                   help="Evaluate on a training or model-selection seed; not held out")
    p.add_argument("--policies", default=",".join(DEFAULT_POLICIES),
                   help=f"Comma-separated subset of {list(POLICY_NAMES)}")
    p.add_argument("--allow-different-scenario", action="store_true",
                   help="Record a scenario mismatch instead of refusing; behavior unproven")
    p.add_argument("--json", action="store_true", help="Print the eval manifest as JSON")
    return p


def _resolve_run(args, policies: list[str]) -> tuple:
    """Return (bundle, run_config, band, selection) for the requested mode."""
    given_selection = [flag for flag in SELECTION_FLAGS if getattr(args, flag) is not None]
    if args.run_dir:
        if given_selection:
            raise ValueError(
                f"selection flags {given_selection} are not allowed with --run-dir; "
                "the saved manifest decides the observation, reward, and telemetry")
        bundle = read_bundle(args.run_dir, args.model)
        run_config = args.run_config or bundle.manifest["run_config"]
        band = args.band if args.band is not None else bundle.manifest.get("band")
        return bundle, run_config, band, selection_from_manifest(bundle.manifest)
    if "model" in policies:
        raise ValueError("--policies model requires --run-dir")
    if not args.run_config:
        raise ValueError("--run-config is required without --run-dir")
    return None, args.run_config, args.band, selection_from_args(args)


def _overlap_message(roles: dict, run_dir: str) -> str:
    """Name every held-out seed that is really a training or model-selection seed."""
    parts = []
    for seed in roles["overlap"]:
        which = []
        if seed == roles["training_seed"]:
            which.append("the training seed")
        if seed == roles["model_selection_seed"]:
            which.append("the model-selection seed")
        parts.append(f"held-out seed {seed} is {' and '.join(which)} of {run_dir}")
    return "; ".join(parts)


def _print_summary(manifest: dict) -> None:
    for name, block in manifest["policies"].items():
        summary = block["summary"]
        print(f"{name}: mean_return={summary['mean_return']} "
              f"episodes={summary['completed_episodes']}/{summary['expected_episodes']} "
              f"revalidated={summary['revalidated_slots_total']} "
              f"mask_violations={summary['mask_violations_total']}")


def _exit_code(manifest: dict, expected_episodes: int) -> int:
    episodes = [episode for block in manifest["policies"].values()
                for episode in block["episodes"]]
    if len(episodes) != expected_episodes or any(
            episode["status"] != "completed" for episode in episodes):
        print("Not every evaluation episode completed", file=sys.stderr)
        return 1
    counters = sum((episode["revalidated_slots_total"] or 0)
                   + (episode["mask_violations"] or 0) for episode in episodes)
    return 2 if counters else 0


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    automatic = args.output_dir is None
    if automatic:
        args.output_dir = str(automatic_output_root("rl-evaluation"))
    try:
        seeds = parse_seed_spec(args.seeds)
        policies = _parse_policies(args.policies)
        _check_output_dir(args.output_dir, args.run_dir)
        records = decision_records_from_args(args)
    except ValueError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if automatic:
        print(f"Evaluation output: {Path(args.output_dir).resolve()}", flush=True)

    try:
        bundle, run_config, band, selection = _resolve_run(args, policies)
        train_manifest = bundle.manifest if bundle is not None else None
        roles = seed_roles(train_manifest, args.model if bundle is not None else None,
                           seeds)
        roles["overlap_allowed"] = bool(args.allow_seed_overlap)
        if roles["overlap"] and "model" in policies:
            message = _overlap_message(roles, args.run_dir)
            if not args.allow_seed_overlap:
                print(f"{message}; choose other seeds or pass --allow-seed-overlap",
                      file=sys.stderr)
                return 1
            print(f"WARNING: {message}; these results are not held out",
                  file=sys.stderr)
        selection = eval_selection(selection)
        identity = read_scenario_identity(run_config)
        base = {
            "sim_binary": os.path.abspath(args.sim_binary),
            "run_config": os.path.abspath(run_config),
            "band": band,
            "label": args.label,
            "seed_roles": roles,
            "training": (training_provenance(train_manifest)
                         if train_manifest is not None else None),
            "scenario_identity": identity,
            "bundle": bundle.describe() if bundle is not None else None,
        }
        placements = _prepare_placements(args, policies, run_config, band, seeds)
        capture = (PreferenceCapture() if "model" in policies and records.enabled
                   and records.preferences != "off" else None)
        model_identity = (_model_identity(bundle) if "model" in policies and records.enabled
                          else None)
        specs = [_model_spec(bundle, identity, band, args.allow_different_scenario, capture)
                 if name == "model"
                 else _placement_spec(placements[name]) if name in placements
                 else _baseline_spec(name) for name in policies]
        configs = {name: str(prepared.effective_run_config)
                   for name, prepared in placements.items()}

        def make_env(name: str) -> MeshRlEnv:
            is_model = name == "model"
            context = DecisionContext(
                mode="evaluation", source="evaluate", policy=name,
                model=model_identity if is_model else None,
                preference_source=capture.take if is_model and capture is not None else None)
            return MeshRlEnv(args.sim_binary, configs.get(name, run_config), seed=seeds[0],
                             output_dir=os.path.join(args.output_dir, name),
                             band=band, selection=selection,
                             decision_records=DecisionRecording(records, context))

        manifest = evaluate(make_env, specs, seeds, args.output_dir, base)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        _print_summary(manifest)
    return _exit_code(manifest, len(policies) * len(seeds))


if __name__ == "__main__":
    sys.exit(main())
