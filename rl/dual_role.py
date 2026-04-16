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
        # Pass role= so role-aware algorithms (PPO) can use slim, role-specific
        # observations. Algorithms that don't accept the kwarg (DQN, tabular)
        # fall back to their default constructor.
        self.tagger_algo = _construct_with_role(algo_class, "tagger")
        self.runner_algo = _construct_with_role(algo_class, "runner")

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

    def load_tagger_only(self, path: str):
        """Load only the tagger model, leave runner untrained."""
        tagger_path, _ = _dual_paths(path)
        self.tagger_algo.load(tagger_path)

    def load_runner_only(self, path: str):
        """Load only the runner model, leave tagger untrained."""
        _, runner_path = _dual_paths(path)
        self.runner_algo.load(runner_path)

    def reset(self):
        """Forward reset to both models."""
        self.tagger_algo.reset()
        self.runner_algo.reset()

    def set_eval(self, enabled: bool) -> None:
        """Forward eval-mode toggle to both sub-algorithms."""
        self.eval_mode = enabled
        self.tagger_algo.set_eval(enabled)
        self.runner_algo.set_eval(enabled)


def _construct_with_role(algo_class, role: str):
    """Call algo_class(role=role) if the constructor accepts it, else algo_class().

    Role-aware algos (e.g. PPO) use the kwarg to select a slimmer,
    role-specific observation encoding. Role-agnostic algos (DQN, Q-Learning,
    SARSA) accept no kwargs, so we fall back to the default constructor.
    """
    try:
        return algo_class(role=role)
    except TypeError:
        return algo_class()


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
    """Make target algo share networks, optimizers, AND experience buffers
    with source so that all agents of the same role contribute to a single
    buffer and trigger exactly one synchronized gradient update.

    Per-agent temporary state (_last_state, _last_log_prob, etc.) remains
    per-instance because each agent's PPO/DQN object is still a separate
    Python object — only the heavy shared resources are aliased.
    """
    # PyTorch-based (PPO): share network + optimizer + rollout buffer
    if hasattr(source, "network") and hasattr(source, "optimizer"):
        target.network = source.network
        target.optimizer = source.optimizer
    if hasattr(source, "buffer"):
        target.buffer = source.buffer

    # PyTorch-based (DQN): share networks + optimizer + replay buffer
    if hasattr(source, "q_net"):
        target.q_net = source.q_net
        target.target_net = source.target_net
        target.optimizer = source.optimizer
    if hasattr(source, "replay_buffer"):
        target.replay_buffer = source.replay_buffer

    # Tabular (Q-Learning, SARSA): share Q-table + exploration state
    if hasattr(source, "q_table"):
        target.q_table = source.q_table
