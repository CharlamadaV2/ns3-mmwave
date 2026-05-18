#!/usr/bin/env python3
"""Training entry point for the mesh-sim RL agent.

Usage
-----
    python -m scripts.rl.train \
        --sim-binary build/scratch/mesh-sim/ns3*-sim-* \
        --run-config scratch/mesh-sim/inputs/baselines/rl-test/run.ini \
        --num-episodes 50
"""

import argparse
import configparser
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np

from scripts.rl.agent import TabularQLearning
from scripts.rl.config import RlTrainConfig
from scripts.rl.env import MeshRlEnv


def parse_args() -> RlTrainConfig:
    p = argparse.ArgumentParser(description="Train an RL agent on mesh-sim")
    p.add_argument("--sim-binary", required=True, help="Path to mesh-sim executable")
    p.add_argument("--run-config", required=True, help="Path to run.ini with [rl] section")
    p.add_argument("--num-episodes", type=int, default=200)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--fixed-seed", action="store_true",
                   help="Use base-seed for every episode (no increment)")
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-decay", type=float, default=0.95)
    p.add_argument("--epsilon-min", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--bin-size", type=float, default=25.0,
                   help="Spatial bin size in meters for Q-table discretization")
    p.add_argument("--output-dir", default="")
    args = p.parse_args()

    return RlTrainConfig(
        num_episodes=args.num_episodes,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        lr=args.lr,
        gamma=args.gamma,
        bin_size=args.bin_size,
        sim_binary=args.sim_binary,
        run_config=args.run_config,
        base_seed=args.base_seed,
        fixed_seed=args.fixed_seed,
        output_dir=args.output_dir,
    )


def _read_rl_bounds(run_config: str) -> dict:
    """Read [rl] section bounds from run.ini for Q-table discretization."""
    ini = configparser.ConfigParser()
    ini.read(run_config)
    return {
        "x_range": (
            float(ini.get("rl", "x_min", fallback="0.0")),
            float(ini.get("rl", "x_max", fallback="500.0")),
        ),
        "y_range": (
            float(ini.get("rl", "y_min", fallback="-250.0")),
            float(ini.get("rl", "y_max", fallback="250.0")),
        ),
    }


def train(cfg: RlTrainConfig) -> None:
    if cfg.output_dir:
        out_dir = cfg.output_dir
    else:
        now = datetime.now()
        out_dir = os.path.join(
            "outputs",
            now.strftime("%Y-%m"),
            now.strftime("%d"),
            now.strftime("%H-%M-%S"),
        )
    os.makedirs(out_dir, exist_ok=True)

    bounds = _read_rl_bounds(cfg.run_config)

    agent = TabularQLearning(
        n_actions=5,
        x_range=bounds["x_range"],
        y_range=bounds["y_range"],
        bin_size=cfg.bin_size,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon,
        lr=cfg.lr,
        epsilon_decay=cfg.epsilon_decay,
        epsilon_min=cfg.epsilon_min,
    )

    all_rewards: list[float] = []

    # CSV writer for per-episode rewards
    rewards_path = os.path.join(out_dir, "rewards.csv")
    rewards_file = open(rewards_path, "w", newline="")
    rewards_writer = csv.writer(rewards_file)
    rewards_writer.writerow(
        ["episode", "seed", "total_reward", "mean_reward_per_step", "num_steps", "epsilon"]
    )

    # Read INI seed for display when using --fixed-seed
    ini_seed = None
    if cfg.fixed_seed:
        ini = configparser.ConfigParser()
        ini.read(cfg.run_config)
        ini_seed = int(ini.get("scenario", "seed", fallback="42"))

    print(f"Training {cfg.num_episodes} episodes → {out_dir}")

    for ep in range(cfg.num_episodes):
        seed = None if cfg.fixed_seed else cfg.base_seed + ep
        env = MeshRlEnv(cfg.sim_binary, cfg.run_config, seed=seed,
                        output_dir=out_dir)

        try:
            obs, info = env.reset()
        except Exception as e:
            print(f"Episode {ep}: failed to start sim: {e}", file=sys.stderr)
            env.close()
            continue

        episode_reward = 0.0
        num_steps = 0
        done = False

        while not done:
            action, _ = agent.predict(obs)
            prev_obs = obs
            obs, reward, done, truncated, info = env.step(action)
            next_obs = None if done else obs
            agent.update(prev_obs, action, reward, next_obs)
            episode_reward += reward
            num_steps += 1

        agent.decay_epsilon()
        all_rewards.append(episode_reward)
        env.close()

        mean_r = episode_reward / max(num_steps, 1)
        seed_label = seed if seed is not None else ini_seed
        rewards_writer.writerow([ep, seed_label, f"{episode_reward:.2f}", f"{mean_r:.2f}", num_steps, f"{agent.epsilon:.4f}"])
        rewards_file.flush()

        print(
            f"  ep={ep:3d}  seed={seed_label}  reward={episode_reward:10.1f}"
            f"  mean={mean_r:8.1f}  eps={agent.epsilon:.3f}"
            f"  states={len(agent.q_table)}"
        )

    rewards_file.close()

    # Save Q-table summary: aggregate action counts and mean Q per action
    q_path = os.path.join(out_dir, "q_values.csv")
    action_names = ["stay", "-x", "+x", "-y", "+y"]
    with open(q_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["action", "mean_q", "total_count"])
        total_counts = np.zeros(agent.n_actions, dtype=int)
        q_sums = np.zeros(agent.n_actions)
        for state, counts in agent.visit_counts.items():
            total_counts += counts
            q_sums += agent.q_table[state] * (counts > 0)
        state_counts = np.array([
            sum(1 for counts in agent.visit_counts.values() if counts[a] > 0)
            for a in range(agent.n_actions)
        ])
        mean_q = np.divide(q_sums, np.maximum(state_counts, 1))
        for i in range(agent.n_actions):
            name = action_names[i] if i < len(action_names) else str(i)
            w.writerow([name, f"{mean_q[i]:.4f}", int(total_counts[i])])

    # Save per-state Q-values for detailed analysis
    q_detail_path = os.path.join(out_dir, "q_table.csv")
    with open(q_detail_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_bin", "y_bin", "sinr_bin", "best_action", "visits"]
                   + [f"q_{name}" for name in action_names])
        for state in sorted(agent.q_table.keys()):
            q = agent.q_table[state]
            counts = agent.visit_counts[state]
            best = action_names[int(np.argmax(q))] if np.any(counts > 0) else "none"
            total_visits = int(np.sum(counts))
            w.writerow(
                [state[0], state[1], state[2], best, total_visits]
                + [f"{q[i]:.4f}" for i in range(agent.n_actions)]
            )

    # Save hyperparameters
    hp_path = os.path.join(out_dir, "hyperparams.json")
    with open(hp_path, "w") as f:
        json.dump(
            {
                "agent": {
                    "type": "tabular_q_learning",
                    "n_actions": agent.n_actions,
                    "gamma": cfg.gamma,
                    "bin_size": cfg.bin_size,
                    "x_range": list(bounds["x_range"]),
                    "y_range": list(bounds["y_range"]),
                    "epsilon_initial": cfg.epsilon,
                    "epsilon_decay": cfg.epsilon_decay,
                    "epsilon_min": cfg.epsilon_min,
                    "lr": cfg.lr,
                },
                "training": {
                    "num_episodes": cfg.num_episodes,
                    "base_seed": cfg.base_seed,
                    "run_config": cfg.run_config,
                },
                "results": {
                    "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
                    "std_reward": float(np.std(all_rewards)) if all_rewards else 0.0,
                    "num_states_visited": len(agent.q_table),
                },
            },
            f,
            indent=2,
        )

    print(f"\nDone. Output: {out_dir}")
    if all_rewards:
        print(f"  Mean reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
        print(f"  States visited: {len(agent.q_table)}")


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
