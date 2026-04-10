"""Dual-role algorithm wrapper: separate tagger and runner models.

Instead of one model learning conflicting chase/flee behaviors,
this wrapper holds two algorithm instances — one trained exclusively
on tagger experiences, the other on runner experiences. Routing
happens automatically based on observation["is_tagger"].

Usage:
    algo = DualRoleAlgorithm(PPO)  # creates PPO for tagger + PPO for runner
    action = algo.select_action(obs)  # routes to correct model
    algo.learn(state, action, reward, next_state, done)  # feeds correct model
"""

import os
from rl.base_algorithm import BaseRLAlgorithm


class DualRoleAlgorithm(BaseRLAlgorithm):
    """Wrapper that holds a tagger model and a runner model."""

    def __init__(self, algo_class, shared_tagger=None, shared_runner=None):
        """
        Args:
            algo_class: The RL algorithm class (e.g., PPO, DQN, QLearning).
            shared_tagger: If provided, share weights with this tagger algo.
            shared_runner: If provided, share weights with this runner algo.
        """
        self.algo_class = algo_class
        self.tagger_algo = algo_class()
        self.runner_algo = algo_class()

        if shared_tagger is not None:
            _share_weights(self.tagger_algo, shared_tagger)
        if shared_runner is not None:
            _share_weights(self.runner_algo, shared_runner)

        # Track which role produced the last action (for routing learn())
        self._last_role_is_tagger = False

    def _active_algo(self, is_tagger: bool) -> BaseRLAlgorithm:
        return self.tagger_algo if is_tagger else self.runner_algo

    def select_action(self, observation: dict) -> int:
        """Route to tagger or runner model based on observation."""
        is_tagger = bool(observation.get("is_tagger", False))
        self._last_role_is_tagger = is_tagger
        return self._active_algo(is_tagger).select_action(observation)

    def learn(self, state: dict, action: int, reward: float,
              next_state: dict, done: bool):
        """Feed experience to the model that produced the action."""
        self._active_algo(self._last_role_is_tagger).learn(
            state, action, reward, next_state, done)

    def save(self, path: str):
        """Save both models with _tagger/_runner suffixes."""
        tagger_path, runner_path = _dual_paths(path)
        self.tagger_algo.save(tagger_path)
        self.runner_algo.save(runner_path)

    def load(self, path: str):
        """Load both models."""
        tagger_path, runner_path = _dual_paths(path)
        self.tagger_algo.load(tagger_path)
        self.runner_algo.load(runner_path)

    def reset(self):
        """Forward reset to both models."""
        self.tagger_algo.reset()
        self.runner_algo.reset()


def _dual_paths(path: str) -> tuple[str, str]:
    """Convert 'models/ppo_model.pt' -> ('models/ppo_tagger_model.pt',
                                          'models/ppo_runner_model.pt')."""
    base, ext = os.path.splitext(path)
    return f"{base}_tagger{ext}", f"{base}_runner{ext}"


def dual_model_exists(path: str) -> bool:
    """Check if both tagger and runner model files exist."""
    tagger_path, runner_path = _dual_paths(path)
    return os.path.exists(tagger_path) and os.path.exists(runner_path)


def _share_weights(target, source):
    """Make target algo share networks/tables/optimizer with source.
    Buffers remain per-instance (intentional — avoids concurrent access)."""
    # PyTorch-based (PPO)
    if hasattr(source, "network") and hasattr(source, "optimizer"):
        target.network = source.network
        target.optimizer = source.optimizer

    # PyTorch-based (DQN)
    if hasattr(source, "q_net"):
        target.q_net = source.q_net
        target.target_net = source.target_net
        target.optimizer = source.optimizer

    # Tabular (Q-Learning, SARSA)
    if hasattr(source, "q_table"):
        target.q_table = source.q_table
