"""SARSA algorithm placeholder.

TODO: Implement SARSA (State-Action-Reward-State-Action).

Key implementation notes:
    - Similar to Q-Learning but on-policy: uses the actual next action taken
    - Q-table: dict mapping (discretized_state, action) -> Q-value
    - Discretize continuous observations into grid cells
    - Hyperparameters: learning_rate (alpha), discount_factor (gamma), epsilon
    - Update rule: Q(s,a) += alpha * (reward + gamma * Q(s',a') - Q(s,a))
      where a' is the actual next action (not the max)
    - Need to store the next action for the update

See base_algorithm.py for the full interface specification.
"""

from rl.base_algorithm import BaseRLAlgorithm


class SARSA(BaseRLAlgorithm):
    def __init__(self):
        # TODO: Initialize Q-table, hyperparameters
        # self.q_table = {}
        # self.alpha = 0.1
        # self.gamma = 0.99
        # self.epsilon = 0.1
        # self.next_action = None
        pass

    def select_action(self, observation: dict) -> int:
        # TODO: Implement epsilon-greedy action selection
        return 0  # no-op placeholder

    def learn(self, state: dict, action: int, reward: float,
              next_state: dict, done: bool):
        # TODO: SARSA update (on-policy)
        pass

    def save(self, path: str):
        # TODO: Save Q-table to disk
        pass

    def load(self, path: str):
        # TODO: Load Q-table from disk
        pass
