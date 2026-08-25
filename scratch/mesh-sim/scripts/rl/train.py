#!/usr/bin/env python3
"""Training entry point for the mesh-sim RL agent (MaskablePPO).

Usage
-----
    python -m scripts.rl.train m-ppo \
        --sim-binary build/scratch/mesh-sim/ns3*-sim-* \
        --run-config scratch/mesh-sim/inputs/baselines/rl-test/run.ini \
        --total-timesteps 100000
"""

import argparse
import os
import sys
from datetime import datetime

from scripts.rl.agents.mask_ppo import MaskablePPOConfig, MaskablePpoTrainer
from scripts.rl.env.mesh_env import MeshRlEnv


## @brief Adapter ActionMasker calls each step to fetch the current mask.
def mask_fn(env):
    return env.unwrapped.action_masks()


## @brief Build the output directory (explicit, or timestamped).
def _make_out_dir(output_dir: str) -> str:
    if output_dir:
        out_dir = output_dir
    else:
        now = datetime.now()
        out_dir = os.path.join("outputs", now.strftime("%Y-%m"),
                               now.strftime("%d"), now.strftime("%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


## @brief Train a MaskablePPO agent on the mesh sim.
def train_mppo(cfg: MaskablePPOConfig, sim_binary: str, run_config: str,
               out_dir: str) -> None:
    env = MeshRlEnv(sim_binary, run_config, seed=cfg.seed, output_dir=out_dir)
    env.reset(seed=cfg.seed)          # populate dynamic obs/action spaces before wrapping

    trainer = MaskablePpoTrainer(cfg, env, mask_fn)
    trainer.train()
    trainer.save(os.path.join(out_dir, "maskable_ppo_mesh"))
    env.close()
    print(f"\nDone. Model + logs in {out_dir}")


def main() -> int:
    p = argparse.ArgumentParser(description="Train an RL agent on mesh-sim")
    p.add_argument("--sim-binary", required=True, help="Path to mesh-sim executable")
    p.add_argument("--run-config", required=True, help="run.ini with an [rl] section")
    p.add_argument("--output-dir", default="")
    p.add_argument("--verbose", type=int, default=1, choices=[0, 1],
                   help="0 = quiet, 1 = SB3 training logs")

    sub = p.add_subparsers(dest="modeltype", required=True,
                           help="Which agent to train")

    # Maskable PPO
    ppo = sub.add_parser("m-ppo", help="Maskable PPO")
    ppo.add_argument("--total-timesteps", type=int, default=100_000)
    ppo.add_argument("--n-steps", type=int, default=1024)
    ppo.add_argument("--gamma", type=float, default=0.95)
    ppo.add_argument("--ent-coef", type=float, default=0.01)
    ppo.add_argument("--seed", type=int, default=42)
    ppo.add_argument("--tensorboard-log", default=None)

    # QR-DQN (disabled for now — kept so the CLI shape is stable)
    qr = sub.add_parser("qr-dqn", help="Quantile-Regression DQN (disabled)")
    qr.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    out_dir = _make_out_dir(args.output_dir)

    if args.modeltype == "qr-dqn":
        print("qr-dqn is disabled in this version", file=sys.stderr)
        return 1

    if args.modeltype == "m-ppo":
        print("Creating Maskable-PPO model ...")
        cfg = MaskablePPOConfig(
            total_timesteps=args.total_timesteps,
            n_steps=args.n_steps,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            seed=args.seed,
            verbose=args.verbose,
            tensorboard_log=args.tensorboard_log,
        )
        train_mppo(cfg, args.sim_binary, args.run_config, out_dir)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
