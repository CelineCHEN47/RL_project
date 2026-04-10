"""Headless game simulation for fast RL training.

Runs the full game logic (movement, collision, tagging) without any
pygame rendering. This allows training at thousands of steps per second.
"""

import random
import importlib
import pygame
import config
from world.level import Level
from entities.agent import Agent
from entities.movable_object import MovableObject
from game.tag_logic import TagLogic
from physics.collision import resolve_entity_walls, resolve_entity_crates
from rl.environment import TagEnvironment
from rl.dual_role import DualRoleAlgorithm


class HeadlessSimulation:
    """One instance of the tag game running without rendering."""

    def __init__(self, algo_class, shared_tagger=None, shared_runner=None):
        """
        Args:
            algo_class: The RL algorithm class to instantiate.
            shared_tagger: If provided, share tagger model weights with this.
            shared_runner: If provided, share runner model weights with this.
        """
        self.level = Level("level_01.txt")
        self.entities = []
        self.agents = []
        self.movable_objects = []

        spawn_points = list(self.level.spawn_points)
        random.shuffle(spawn_points)

        for i, sp in enumerate(spawn_points):
            if config.DUAL_ROLE_ENABLED:
                algorithm = DualRoleAlgorithm(algo_class,
                                              shared_tagger=shared_tagger,
                                              shared_runner=shared_runner)
                # After first agent, share weights with it
                if i == 0 and shared_tagger is None:
                    shared_tagger = algorithm.tagger_algo
                    shared_runner = algorithm.runner_algo
            else:
                algorithm = algo_class()

            agent = Agent(sp[0], sp[1], i, algorithm)
            self.agents.append(agent)
            self.entities.append(agent)

        for cp in self.level.crate_spawns:
            self.movable_objects.append(MovableObject(cp[0], cp[1]))

        self.tag_logic = TagLogic(self.entities)
        tagger = random.choice(self.entities)
        self.tag_logic.set_tagger(tagger.entity_id)

        self.rl_env = TagEnvironment(self.level, self.entities,
                                     self.movable_objects)
        self.total_tags = 0
        self.steps = 0

    def get_shared_algos(self):
        """Return (shared_tagger, shared_runner) from the first agent."""
        if self.agents and isinstance(self.agents[0].algorithm, DualRoleAlgorithm):
            return self.agents[0].algorithm.tagger_algo, self.agents[0].algorithm.runner_algo
        return None, None

    def step(self) -> dict | None:
        """Run one game step. Returns tag_event if a tag occurred."""
        for agent in self.agents:
            obs = self.rl_env.get_observation(agent)
            agent.decide_action(obs)

        for entity in self.entities:
            entity.apply_velocity()
            resolve_entity_walls(entity, self.level.wall_rects)

        for entity in self.entities:
            resolve_entity_crates(entity, self.movable_objects,
                                  self.level.wall_rects)

        tag_event = self.tag_logic.update()

        rewards = self.rl_env.get_all_rewards(tag_event)
        for agent in self.agents:
            next_obs = self.rl_env.get_observation(agent)
            reward = rewards.get(agent.entity_id, 0.0)
            done = (tag_event is not None and
                    tag_event.get("tagged_id") == agent.entity_id)
            agent.learn(reward, next_obs, done)

        if tag_event:
            self.total_tags += 1
            for agent in self.agents:
                if agent.entity_id == tag_event.get("tagged_id"):
                    agent.algorithm.reset()

        self.steps += 1
        return tag_event

    def reset(self):
        """Reset positions and tagger for a new episode."""
        spawn_points = list(self.level.spawn_points)
        random.shuffle(spawn_points)

        for i, agent in enumerate(self.agents):
            if i < len(spawn_points):
                agent.x = float(spawn_points[i][0])
                agent.y = float(spawn_points[i][1])
                agent.vx = 0.0
                agent.vy = 0.0
                half = agent.ENTITY_SIZE // 2
                agent.rect.x = int(agent.x) - half
                agent.rect.y = int(agent.y) - half

        self.movable_objects.clear()
        for cp in self.level.crate_spawns:
            self.movable_objects.append(MovableObject(cp[0], cp[1]))

        tagger = random.choice(self.entities)
        self.tag_logic.set_tagger(tagger.entity_id)

        self.rl_env = TagEnvironment(self.level, self.entities,
                                     self.movable_objects)
