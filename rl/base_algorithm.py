"""Abstract base class for all RL algorithms.

This defines the interface that every RL algorithm must implement.
Teammates: subclass this and fill in the abstract methods.

Observation format (dict, ego-centric):
    - "self_pos": tuple(float, float) - normalized absolute position [0, 1]
    - "self_vel": tuple(float, float) - current velocity
    - "is_tagger": bool - whether this agent is "it"
    - "tagger_rel": tuple(float, float) - relative direction to tagger
    - "tagger_dist": float - normalized distance to tagger
    - "nearest_runner_rel": tuple(float, float) - relative direction to nearest runner
    - "nearest_runner_dist": float - normalized distance to nearest runner
    - "wall_rays": list[float] - 8 raycasts (N,NE,E,SE,S,SW,W,NW), normalized distance
    - "other_agents": list of dicts with "rel_pos", "distance", "is_tagger"
    - "nearby_crates": list of tuple(float, float) - relative crate positions

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
