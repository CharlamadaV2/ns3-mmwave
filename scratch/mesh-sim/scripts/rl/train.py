#!/usr/bin/env python3
"""Train a MaskablePPO policy on mesh-sim scenarios."""

import argparse
import math
import os
import platform
import sys
from dataclasses import dataclass

from scripts.rl.agents.callbacks import CHECKPOINT_DIR, build_callbacks, list_checkpoints
from scripts.rl.agents.mask_ppo import MaskablePPOConfig, MaskablePpoTrainer
from scripts.rl.cli_common import (MANIFEST_NAME, MODEL_BASENAME,
                                   add_scenario_arguments, add_selection_arguments,
                                   has_previous_run, make_out_dir, now_iso,
                                   package_versions, resolve_seed, selection_from_args,
                                   sha256_file, write_json)
from scripts.rl.env.config import read_scenario_identity
from scripts.rl.env.mesh_env import MeshRlEnv

MANIFEST_VERSION = 4
BEST_MODEL_NAME = "best_model.zip"
EVAL_LOG_NAME = "evaluations.npz"
EVAL_DIR = "eval"

_MAX_ERROR_CHARS = 1000


@dataclass
class Cadence:
    """Checkpoint/evaluation knobs; units are SB3 timesteps."""
    checkpoint_every: int = 0
    keep_checkpoints: int = 3
    eval_every: int = 0
    eval_episodes: int = 1
    eval_seed: int = 0


def mask_fn(env):
    """Action-mask accessor used by ActionMasker."""
    return env.unwrapped.action_masks()


def _write_manifest(out_dir: str, manifest: dict) -> None:
    write_json(os.path.join(out_dir, MANIFEST_NAME), manifest)


def _evaluation_block(out_dir: str, cadence: Cadence) -> dict | None:
    if cadence.eval_every <= 0:
        return None
    return {
        "every_steps": cadence.eval_every,
        "episodes": cadence.eval_episodes,
        "seed": cadence.eval_seed,
        "seed_source": "eval",
        "output_dir": os.path.abspath(os.path.join(out_dir, EVAL_DIR)),
        "log_path": os.path.abspath(os.path.join(out_dir, EVAL_LOG_NAME)),
    }


def _checkpoint_entries(out_dir: str) -> list[dict]:
    return [{"path": os.path.abspath(str(path)),
             "sha256": sha256_file(path),
             "num_timesteps": steps}
            for steps, path in list_checkpoints(os.path.join(out_dir, CHECKPOINT_DIR))]


def _best_model_entries(out_dir: str, eval_callback) -> dict:
    """Best-model provenance from the eval callback; all null when no evaluation ran."""
    if eval_callback is None:
        return {}
    best_path = os.path.join(out_dir, BEST_MODEL_NAME)
    if not os.path.isfile(best_path):
        return {}
    mean_reward = float(eval_callback.best_mean_reward)
    return {
        "best_model_path": os.path.abspath(best_path),
        "best_model_sha256": sha256_file(best_path),
        "best_mean_reward": mean_reward if math.isfinite(mean_reward) else None,
    }


def train_mppo(cfg: MaskablePPOConfig, sim_binary: str, run_config: str,
               out_dir: str, band: str | None, seed_source: str, selection,
               cadence: Cadence) -> str:
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "status": "running",
        "started_at": now_iso(),
        "ended_at": None,
        "sim_binary": os.path.abspath(sim_binary),
        "run_config": os.path.abspath(run_config),
        "scenario_identity": read_scenario_identity(run_config),
        "control_mode": None,
        "contract": None,
        "algorithm": "MaskablePPO",
        "seed": cfg.seed,
        "seed_source": seed_source,
        "band": band,
        "selection": None,
        "observation_schema": None,
        "reward_schema": None,
        "telemetry": None,
        "hyperparameters": {
            "total_timesteps": cfg.total_timesteps,
            "n_steps": cfg.n_steps,
            "gamma": cfg.gamma,
            "ent_coef": cfg.ent_coef,
            "verbose": cfg.verbose,
            "tensorboard_log": cfg.tensorboard_log,
            "checkpoint_every_steps": cadence.checkpoint_every,
            "keep_checkpoints": cadence.keep_checkpoints,
            "eval_every_steps": cadence.eval_every,
            "eval_episodes": cadence.eval_episodes,
        },
        "evaluation": _evaluation_block(out_dir, cadence),
        "output_dir": os.path.abspath(out_dir),
        "python_version": platform.python_version(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "package_versions": package_versions(),
        "model_path": None,
        "model_sha256": None,
        "best_model_path": None,
        "best_model_sha256": None,
        "best_mean_reward": None,
        "checkpoints": [],
    }
    _write_manifest(out_dir, manifest)

    model_path = os.path.abspath(os.path.join(out_dir, f"{MODEL_BASENAME}.zip"))
    env = eval_env = eval_callback = None
    try:
        # Let the env resolve the run.ini seed itself so episode manifests report
        # the same seed_source as this training manifest.
        env_seed = cfg.seed if seed_source == "cli" else None
        env = MeshRlEnv(sim_binary, run_config, seed=env_seed,
                        output_dir=out_dir, band=band, selection=selection)
        env.reset()                   # populate dynamic obs/action spaces before wrapping
        manifest["control_mode"] = env.control_mode
        manifest["contract"] = env.contract
        if env.control_mode == "centralized":
            manifest.update({
                "selection": selection.describe(),
                "observation_schema": env.observation_schema,
                "reward_schema": env.reward_schema,
                "telemetry": {"mode": selection.telemetry,
                              "every": selection.telemetry_every},
            })
        _write_manifest(out_dir, manifest)

        if cadence.eval_every > 0:
            eval_env = MeshRlEnv(sim_binary, run_config, seed=cadence.eval_seed,
                                 output_dir=os.path.join(out_dir, EVAL_DIR), band=band,
                                 selection=selection)
            eval_env.reset(seed=cadence.eval_seed, options={"seed_source": "eval"})

        callbacks, eval_callback = build_callbacks(
            out_dir, cadence.checkpoint_every, cadence.keep_checkpoints,
            eval_env, cadence.eval_every, cadence.eval_episodes, cfg.verbose)

        trainer = MaskablePpoTrainer(cfg, env, mask_fn)
        trainer.train(callback=callbacks or None)
        trainer.save(os.path.join(out_dir, MODEL_BASENAME))
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["ended_at"] = now_iso()
        manifest["error"] = f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS]
        _write_manifest(out_dir, manifest)
        raise
    finally:
        # Always reap the simulators; a recorded failure is never relabelled here.
        for instance in (env, eval_env):
            if instance is not None:
                try:
                    instance.close()
                except Exception:
                    pass

    manifest["status"] = "completed"
    manifest["ended_at"] = now_iso()
    manifest["model_path"] = model_path
    manifest["model_sha256"] = sha256_file(model_path)
    manifest["checkpoints"] = _checkpoint_entries(out_dir)
    manifest.update(_best_model_entries(out_dir, eval_callback))
    _write_manifest(out_dir, manifest)
    return model_path


def _cadence_from_args(args) -> Cadence | str:
    """Validate the cadence flags before anything is launched; a str is the error."""
    if args.checkpoint_every_steps < 0 or args.eval_every_steps < 0:
        return "--checkpoint-every-steps and --eval-every-steps must be >= 0"
    if args.keep_checkpoints < 1:
        return "--keep-checkpoints must be >= 1"
    if args.eval_episodes < 1:
        return "--eval-episodes must be >= 1"
    return Cadence(checkpoint_every=args.checkpoint_every_steps,
                   keep_checkpoints=args.keep_checkpoints,
                   eval_every=args.eval_every_steps,
                   eval_episodes=args.eval_episodes)


def main() -> int:
    p = argparse.ArgumentParser(description="Train an RL agent on mesh-sim")
    add_scenario_arguments(p)
    p.add_argument("--output-dir", default="")
    p.add_argument("--verbose", type=int, default=1, choices=[0, 1],
                   help="0 = quiet, 1 = SB3 training logs")
    add_selection_arguments(p)

    sub = p.add_subparsers(dest="modeltype", required=True,
                           help="Which agent to train")

    # Maskable PPO
    ppo = sub.add_parser("m-ppo", help="Maskable PPO")
    ppo.add_argument("--total-timesteps", type=int, default=100_000)
    ppo.add_argument("--n-steps", type=int, default=1024)
    ppo.add_argument("--gamma", type=float, default=0.95)
    ppo.add_argument("--ent-coef", type=float, default=0.01)
    ppo.add_argument("--seed", type=int, default=None,
                     help="Training seed; defaults to [scenario] seed in run.ini")
    ppo.add_argument("--tensorboard-log", default=None)
    ppo.add_argument("--checkpoint-every-steps", type=int, default=0,
                     help="Save a checkpoint every N timesteps; 0 disables checkpoints")
    ppo.add_argument("--keep-checkpoints", type=int, default=3,
                     help="How many newest checkpoints to retain (>= 1)")
    ppo.add_argument("--eval-every-steps", type=int, default=0,
                     help="Run a masked evaluation every N timesteps; 0 disables it")
    ppo.add_argument("--eval-episodes", type=int, default=1,
                     help="Episodes per evaluation (>= 1)")
    ppo.add_argument("--eval-seed", type=int, default=None,
                     help="Seed for the evaluation env; defaults to the training seed + 1")

    # QR-DQN (disabled for now — kept so the CLI shape is stable)
    qr = sub.add_parser("qr-dqn", help="Quantile-Regression DQN (disabled)")
    qr.add_argument("--seed", type=int, default=None)

    args = p.parse_args()

    if args.modeltype == "qr-dqn":
        print("qr-dqn is disabled in this version", file=sys.stderr)
        return 1

    if args.modeltype != "m-ppo":
        return 1

    cadence = _cadence_from_args(args)
    if isinstance(cadence, str):
        print(cadence, file=sys.stderr)
        return 1

    try:
        seed, seed_source = resolve_seed(args.seed, args.run_config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        selection = selection_from_args(args)
    except ValueError as exc:
        print(f"Invalid RL selection: {exc}", file=sys.stderr)
        return 1
    cadence.eval_seed = args.eval_seed if args.eval_seed is not None else seed + 1

    out_dir = make_out_dir(args.output_dir)
    existing = has_previous_run(out_dir)
    if existing:
        print(f"Refusing to start: {out_dir} already contains {existing}. "
              "Choose a new --output-dir.", file=sys.stderr)
        return 1

    print("Creating Maskable-PPO model ...")
    cfg = MaskablePPOConfig(
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        seed=seed,
        verbose=args.verbose,
        tensorboard_log=args.tensorboard_log,
    )
    try:
        train_mppo(cfg, args.sim_binary, args.run_config, out_dir,
                   args.band, seed_source, selection, cadence)
    except Exception as exc:
        print(f"Training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"\nDone. Model + logs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
