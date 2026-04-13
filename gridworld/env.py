"""Simple 2-agent gridworld for tabular RL (Q-Learning / SARSA).

A 10x10 grid with one tagger and one runner.
- Turn-based: tagger moves, then runner moves.
- Episode ends when tagger reaches the runner's cell.
- State: (tagger_x, tagger_y, runner_x, runner_y) — 10,000 possible states.
- Actions: 0=stay, 1=up, 2=down, 3=left, 4=right.
"""

import random

GRID_SIZE = 10

# Actions
STAY  = 0
UP    = 1
DOWN  = 2
LEFT  = 3
RIGHT = 4
NUM_ACTIONS = 5

ACTION_NAMES = {STAY: "stay", UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}
ACTION_DELTAS = {
    STAY:  (0, 0),
    UP:    (0, -1),
    DOWN:  (0, 1),
    LEFT:  (-1, 0),
    RIGHT: (1, 0),
}


class TagGridWorld:
    """Turn-based tag game on a grid."""

    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid_size = grid_size
        self.tagger_pos = (0, 0)
        self.runner_pos = (0, 0)
        self.done = False
        self.steps = 0
        self.max_steps = 200  # episode timeout
        self.reset()

    def reset(self) -> tuple:
        """Reset to random non-overlapping positions. Returns initial state."""
        self.tagger_pos = (random.randint(0, self.grid_size - 1),
                           random.randint(0, self.grid_size - 1))
        # Ensure runner starts at a different position
        while True:
            self.runner_pos = (random.randint(0, self.grid_size - 1),
                               random.randint(0, self.grid_size - 1))
            if self.runner_pos != self.tagger_pos:
                break
        self.done = False
        self.steps = 0
        return self.get_state()

    def get_state(self) -> tuple:
        """State: (tagger_x, tagger_y, runner_x, runner_y)."""
        return (*self.tagger_pos, *self.runner_pos)

    def _move(self, pos: tuple, action: int) -> tuple:
        """Apply action to position, clamp to grid bounds."""
        dx, dy = ACTION_DELTAS[action]
        nx = max(0, min(self.grid_size - 1, pos[0] + dx))
        ny = max(0, min(self.grid_size - 1, pos[1] + dy))
        return (nx, ny)

    def manhattan_distance(self) -> int:
        return (abs(self.tagger_pos[0] - self.runner_pos[0]) +
                abs(self.tagger_pos[1] - self.runner_pos[1]))

    def step_tagger(self, action: int) -> tuple:
        """Tagger takes an action.
        Returns: (next_state, reward, done)

        Tagger rewards:
            +10  caught the runner
            -0.1 per step (urgency)
            +distance_bonus: closer = better (0 to +0.5)
        """
        old_dist = self.manhattan_distance()
        self.tagger_pos = self._move(self.tagger_pos, action)
        self.steps += 1
        new_dist = self.manhattan_distance()

        if self.tagger_pos == self.runner_pos:
            self.done = True
            return self.get_state(), 10.0, True

        if self.steps >= self.max_steps:
            self.done = True
            return self.get_state(), -1.0, True

        # Distance shaping: reward for getting closer
        reward = -0.1 + 0.5 * (old_dist - new_dist) / self.grid_size
        return self.get_state(), reward, False

    def step_runner(self, action: int) -> tuple:
        """Runner takes an action.
        Returns: (next_state, reward, done)

        Runner rewards:
            -10  got caught
            +0.1 per step (survival)
            +distance_bonus: farther = better (0 to +0.5)
        """
        old_dist = self.manhattan_distance()
        self.runner_pos = self._move(self.runner_pos, action)

        if self.tagger_pos == self.runner_pos:
            self.done = True
            return self.get_state(), -10.0, True

        new_dist = self.manhattan_distance()

        # Distance shaping: reward for getting farther
        reward = 0.1 + 0.5 * (new_dist - old_dist) / self.grid_size
        return self.get_state(), reward, False
