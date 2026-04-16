"""RL-controlled agent entity."""

from entities.entity import Entity
from rl.base_algorithm import BaseRLAlgorithm
import config


class Agent(Entity):
    def __init__(self, x: float, y: float, entity_id: int,
                 algorithm: BaseRLAlgorithm):
        super().__init__(x, y, config.AGENT_SPEED, entity_id)
        self.algorithm = algorithm
        self.is_human = False
        self.last_action = 0
        self.last_observation = None
        self.frame_counter = 0
        self._pending_reward = 0.0  # accumulate reward between decisions
        self._decided_this_frame = False
        self._pending_done = False

    def decide_action(self, observation: dict):
        """Query the RL algorithm for an action and set velocity.

        If this is a decision frame, flush the previous action's accumulated
        reward BEFORE selecting the next action. This keeps rewards aligned
        with the action that actually produced them.
        """
        self.frame_counter += 1
        self._decided_this_frame = False

        # Only decide every DECISION_INTERVAL frames
        if self.frame_counter % config.DECISION_INTERVAL != 0:
            return

        # Flush previous action transition first (old action -> current obs)
        if self.last_observation is not None:
            self.algorithm.learn(
                self.last_observation,
                self.last_action,
                self._pending_reward,
                observation,
                self._pending_done,
            )
            self._pending_reward = 0.0
            self._pending_done = False

        self._decided_this_frame = True
        self.last_observation = observation
        action = self.algorithm.select_action(observation)
        self.last_action = action

        # Convert action to direction
        direction = BaseRLAlgorithm.ACTION_MAP.get(action, (0, 0))
        dx, dy = direction
        self.set_velocity(float(dx), float(dy))

    def learn(self, reward: float, next_observation: dict, done: bool = False):
        """Feed experience to the RL algorithm.

        Rewards received between decision frames are accumulated and
        delivered as a single transition on the next decision frame.
        This prevents the buffer from being filled with duplicate
        (state, action) pairs that only differ in reward.
        """
        self._pending_reward += reward
        if done:
            self._pending_done = True

        if not self._decided_this_frame:
            # Not a decision frame — accumulate reward and wait.
            # Exception: if done is True (e.g. agent got tagged), flush
            # immediately so the terminal signal is not lost.
            if done and self.last_observation is not None:
                self.algorithm.learn(
                    self.last_observation,
                    self.last_action,
                    self._pending_reward,
                    next_observation,
                    True,
                )
                self._pending_reward = 0.0
                self._pending_done = False
                # Clear last_observation so the next decision frame does NOT
                # re-flush a spurious (pre-tag → post-respawn) transition.
                self.last_observation = None
            return

        # Decision frame rewards belong to the *new* action selected this frame.
        # Accumulation/flush for the previous action already happened in
        # decide_action(), so do not emit another transition here.
        if self.last_observation is not None:
            return
