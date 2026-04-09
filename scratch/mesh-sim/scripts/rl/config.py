"""RL training configuration dataclass."""

from dataclasses import dataclass, field


@dataclass
class RlTrainConfig:
    """All hyperparameters for an RL training run.

    Agent-level parameters live here (not in run.ini) because they are
    Python-side concerns.  Sim-level RL parameters (step_size_m, reward_type,
    etc.) live in the ``[rl]`` section of run.ini.
    """

    # Agent
    num_episodes: int = 200
    epsilon: float = 1.0
    epsilon_decay: float = 0.95
    epsilon_min: float = 0.01
    lr: float = 0.2
    gamma: float = 0.9
    bin_size: float = 25.0

    # Sim
    sim_binary: str = ""
    run_config: str = ""
    base_seed: int = 42
    fixed_seed: bool = False

    # Output
    output_dir: str = ""
