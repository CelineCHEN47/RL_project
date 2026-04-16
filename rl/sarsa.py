"""Tabular SARSA algorithm for the Tag game."""

import os
import pickle
import random

from rl.base_algorithm import BaseRLAlgorithm


class SARSA(BaseRLAlgorithm):
    def __init__(self):
        # Q-table: key=(discretized_state, action), value=Q(s, a)
        self.q_table: dict[tuple[tuple, int], float] = {}

        # Hyperparameters
        self.alpha = 0.10
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99995

    def _bin_value(self, value: float, bins: int) -> int:
        v = max(-1.0, min(1.0, value))
        return int((v + 1.0) * 0.5 * (bins - 1))

    def _state_key(self, obs: dict) -> tuple:
        self_pos = obs.get("self_pos", (0.0, 0.0))
        self_vel = obs.get("self_vel", (0.0, 0.0))
        is_tagger = 1 if obs.get("is_tagger", False) else 0
        tagger_rel = obs.get("tagger_rel", (0.0, 0.0))
        tagger_dist = obs.get("tagger_dist", 0.0)
        nearest_runner_rel = obs.get("nearest_runner_rel", (0.0, 0.0))
        nearest_runner_dist = obs.get("nearest_runner_dist", 0.0)
        wall_rays = obs.get("wall_rays", [1.0] * 8)

        return (
            int(self_pos[0] * 9),
            int(self_pos[1] * 9),
            self._bin_value(self_vel[0] / 3.0, 5),
            self._bin_value(self_vel[1] / 3.0, 5),
            is_tagger,
            self._bin_value(tagger_rel[0], 9),
            self._bin_value(tagger_rel[1], 9),
            int(max(0.0, min(1.0, tagger_dist)) * 9),
            self._bin_value(nearest_runner_rel[0], 9),
            self._bin_value(nearest_runner_rel[1], 9),
            int(max(0.0, min(1.0, nearest_runner_dist)) * 9),
            *(int(max(0.0, min(1.0, r)) * 4) for r in wall_rays[:8]),
        )

    def _q(self, state_key: tuple, action: int) -> float:
        return self.q_table.get((state_key, action), 0.0)

    def _epsilon_greedy(self, state_key: tuple) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.ACTION_SPACE_SIZE)

        q_values = [self._q(state_key, a) for a in range(self.ACTION_SPACE_SIZE)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def select_action(self, observation: dict) -> int:
        state_key = self._state_key(observation)
        return self._epsilon_greedy(state_key)

    def learn(self, state: dict, action: int, reward: float,
              next_state: dict, done: bool):
        if self.eval_mode:
            return
        state_key = self._state_key(state)
        next_state_key = self._state_key(next_state)

        a = int(action)
        if a < 0 or a >= self.ACTION_SPACE_SIZE:
            a = 0

        current_q = self._q(state_key, a)
        if done:
            td_target = reward
        else:
            next_action = self._epsilon_greedy(next_state_key)
            td_target = reward + self.gamma * self._q(next_state_key, next_action)

        new_q = current_q + self.alpha * (td_target - current_q)
        self.q_table[(state_key, a)] = new_q

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "q_table": self.q_table,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.q_table = payload.get("q_table", {})
        self.alpha = payload.get("alpha", self.alpha)
        self.gamma = payload.get("gamma", self.gamma)
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.epsilon_min = payload.get("epsilon_min", self.epsilon_min)
        self.epsilon_decay = payload.get("epsilon_decay", self.epsilon_decay)
