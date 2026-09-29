"""Exploratory tabular Q-learning agent over (x_bin, y_bin, sinr_bin) states."""

from collections import defaultdict

import numpy as np

class TabularQLearning:
    """Tabular Q-learning with discretized state; predict() mirrors SB3's signature."""

    def __init__(
        self,
        n_actions: int = 5,
        x_range: tuple[float, float] = (0.0, 500.0),
        y_range: tuple[float, float] = (-250.0, 250.0),
        bin_size: float = 10.0,
        sinr_edges: list[float] | None = None,
        gamma: float = 0.99,
        epsilon: float = 0.3,
        lr: float = 0.1,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.01,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self._x_min, self._x_max = x_range
        self._y_min, self._y_max = y_range
        self._bin_size = bin_size
        self._sinr_edges = np.array(
            sinr_edges if sinr_edges is not None else [-10.0, 0.0, 10.0, 20.0, 30.0]
        )

        self.q_table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions)
        )
        self.visit_counts: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(n_actions, dtype=int)
        )

    def _discretize(self, obs: np.ndarray) -> tuple:
        """Convert continuous obs [ctrl_x, ctrl_y, sinr_0, cap_0, ...] to state tuple."""
        ctrl_x = obs[0]
        ctrl_y = obs[1]
        # Mean SINR across all links (obs layout: [x, y, sinr0, cap0, sinr1, cap1, ...])
        sinrs = obs[2::2]
        mean_sinr = np.mean(sinrs) if len(sinrs) > 0 else 0.0

        x_bin = int(np.clip((ctrl_x - self._x_min) / self._bin_size, 0,
                            (self._x_max - self._x_min) / self._bin_size - 1))
        y_bin = int(np.clip((ctrl_y - self._y_min) / self._bin_size, 0,
                            (self._y_max - self._y_min) / self._bin_size - 1))
        sinr_bin = int(np.searchsorted(self._sinr_edges, mean_sinr))

        return (x_bin, y_bin, sinr_bin)

    def predict(self, obs: np.ndarray, deterministic: bool = False):
        """Epsilon-greedy action over Q[state]; returns (action, None)."""
        state = self._discretize(obs)
        if not deterministic and np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = int(np.argmax(self.q_table[state]))
        return action, None

    def update(
        self,
        prev_obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray | None,
    ) -> None:
        """Q-learning update: Q(s,a) += lr * (r + gamma * max Q(s',:) - Q(s,a))."""
        s = self._discretize(prev_obs)
        self.visit_counts[s][action] += 1

        if next_obs is None:
            # Terminal state — no future reward
            max_future_q = 0.0
        else:
            s_next = self._discretize(next_obs)
            max_future_q = np.max(self.q_table[s_next])

        td_target = reward + self.gamma * max_future_q
        self.q_table[s][action] += self.lr * (td_target - self.q_table[s][action])

    def decay_epsilon(self) -> None:
        """Decay epsilon after an episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
