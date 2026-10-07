"""MaskablePPO trainer wrapper + its config for the mesh-sim RL agent."""

from dataclasses import dataclass
from typing import Callable

from gymnasium import Env
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO


@dataclass
class MaskablePPOConfig:
    total_timesteps: int = 100_000   # budget for .learn(); mesh env is slow -> start low
    n_steps: int = 1024              # SB3 default 2048; smaller = more frequent updates
    gamma: float = 0.95              # SB3 default 0.99
    ent_coef: float = 0.01           # SB3 default 0.0; keeps exploring under masking
    seed: int = 42
    verbose: int = 1
    tensorboard_log: str | None = None


class MaskablePpoTrainer:
    """Wrap the SB3 MaskablePPO policy and its action-mask adapter."""

    def __init__(self, cfg: MaskablePPOConfig, env: Env, mask_fn: Callable):
        self.cfg = cfg
        self.env = ActionMasker(env, mask_fn)   # enable masking
        self.model = MaskablePPO(
            MaskableActorCriticPolicy,          # policy (positional)
            self.env,
            seed=cfg.seed,
            n_steps=cfg.n_steps,
            gamma=cfg.gamma,
            ent_coef=cfg.ent_coef,
            verbose=cfg.verbose,
            tensorboard_log=cfg.tensorboard_log,
        )

    def train(self, callback=None):
        self.model.learn(total_timesteps=self.cfg.total_timesteps, callback=callback)
        return self.model

    @staticmethod
    def load(path: str, env: Env, mask_fn: Callable) -> MaskablePPO:
        """Reload a saved policy onto a fresh masked env; CPU keeps replay deterministic."""
        return MaskablePPO.load(path, env=ActionMasker(env, mask_fn), device="cpu")

    def save(self, path: str) -> None:
        self.model.save(path)
