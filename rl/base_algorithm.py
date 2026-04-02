"""Abstract base class for all RL algorithms.

This defines the interface that every RL algorithm must implement.
Teammates: subclass this and fill in the abstract methods.

Observation format (dict):
    - "self_pos": tuple(float, float) - normalized position [0, 1]
    - "self_vel": tuple(float, float) - current velocity
    - "is_tagger": bool - whether this agent is "it"
    - "other_agents": list of dicts with "pos" and "is_tagger"
    - "tagger_pos": tuple(float, float) - position of current tagger
    - "nearby_walls": list of tuple(int, int) - grid coords of nearby walls
    - "nearby_crates": list of tuple(int, int) - pixel coords of nearby crates

Action space: discrete, 5 actions
    0 = no-op (stand still)
    1 = move up
    2 = move down
    3 = move left
    4 = move right
"""

from abc import ABC, abstractmethod


class BaseRLAlgorithm(ABC):
    ACTION_SPACE_SIZE = 5
    ACTION_MAP = {
        0: (0, 0),    # no-op
        1: (0, -1),   # up
        2: (0, 1),    # down
        3: (-1, 0),   # left
        4: (1, 0),    # right
    }

    @abstractmethod
    def select_action(self, observation: dict) -> int:
        """Given current observation, return action index 0-4."""

    @abstractmethod
    def learn(self, state: dict, action: int, reward: float,
              next_state: dict, done: bool):
        """Update internal model from one transition."""

    @abstractmethod
    def save(self, path: str):
        """Save model/weights to disk."""

    @abstractmethod
    def load(self, path: str):
        """Load model/weights from disk."""

    def reset(self):
        """Called at episode boundaries (e.g., when tag transfers).
        Override if your algorithm needs episode-level resets."""
        pass
