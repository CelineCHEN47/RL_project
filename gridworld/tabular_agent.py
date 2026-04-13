"""Tabular Q-Learning and SARSA agents for the gridworld."""

import random
import pickle
import os
from gridworld.env import NUM_ACTIONS


class QLearningAgent:
    """Tabular Q-Learning with epsilon-greedy exploration."""

    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.9995):
        self.q_table: dict[tuple, list[float]] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def _get_q(self, state: tuple) -> list[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * NUM_ACTIONS
        return self.q_table[state]

    def select_action(self, state: tuple) -> int:
        if random.random() < self.epsilon:
            return random.randrange(NUM_ACTIONS)
        q_vals = self._get_q(state)
        max_q = max(q_vals)
        best = [a for a, q in enumerate(q_vals) if q == max_q]
        return random.choice(best)

    def learn(self, state: tuple, action: int, reward: float,
              next_state: tuple, done: bool):
        q_vals = self._get_q(state)
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * max(self._get_q(next_state))
        q_vals[action] += self.alpha * (td_target - q_vals[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
            }, f)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data.get("epsilon", self.epsilon_min)


class SARSAAgent:
    """Tabular SARSA with epsilon-greedy exploration."""

    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.9995):
        self.q_table: dict[tuple, list[float]] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def _get_q(self, state: tuple) -> list[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * NUM_ACTIONS
        return self.q_table[state]

    def select_action(self, state: tuple) -> int:
        if random.random() < self.epsilon:
            return random.randrange(NUM_ACTIONS)
        q_vals = self._get_q(state)
        max_q = max(q_vals)
        best = [a for a, q in enumerate(q_vals) if q == max_q]
        return random.choice(best)

    def learn(self, state: tuple, action: int, reward: float,
              next_state: tuple, done: bool, next_action: int = 0):
        q_vals = self._get_q(state)
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self._get_q(next_state)[next_action]
        q_vals[action] += self.alpha * (td_target - q_vals[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
            }, f)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data.get("epsilon", self.epsilon_min)


class RandomAgent:
    """Always picks a random action. Used as the untrained opponent."""

    def select_action(self, state: tuple) -> int:
        return random.randrange(NUM_ACTIONS)
