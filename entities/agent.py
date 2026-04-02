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

    def decide_action(self, observation: dict):
        """Query the RL algorithm for an action and set velocity."""
        self.frame_counter += 1

        # Only decide every DECISION_INTERVAL frames
        if self.frame_counter % config.DECISION_INTERVAL != 0:
            return

        self.last_observation = observation
        action = self.algorithm.select_action(observation)
        self.last_action = action

        # Convert action to direction
        direction = BaseRLAlgorithm.ACTION_MAP.get(action, (0, 0))
        dx, dy = direction
        self.set_velocity(float(dx), float(dy))

    def learn(self, reward: float, next_observation: dict, done: bool = False):
        """Feed experience to the RL algorithm."""
        if self.last_observation is not None:
            self.algorithm.learn(
                self.last_observation,
                self.last_action,
                reward,
                next_observation,
                done,
            )
