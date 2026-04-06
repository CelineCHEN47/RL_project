"""Gym-like environment wrapper for the Tag game.

This module provides observations and computes rewards for RL agents.
It is a passive observer - it reads game state but does not control it.

Observation design (ego-centric):
    Everything is relative to the observing agent. The agent is always at
    the center of its own observation. Other agents are represented by
    their relative (dx, dy) offset and distance. Walls are sensed via
    raycasts in 8 directions (how far until a wall in each direction).
"""

import math
from entities.entity import Entity
from entities.movable_object import MovableObject
from world.level import Level
import config

# 8 raycast directions: N, NE, E, SE, S, SW, W, NW
_RAY_DIRS = [
    (0, -1), (1, -1), (1, 0), (1, 1),
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
]
_MAX_RAY_DIST = 8  # max tiles to raycast


class TagEnvironment:
    VISION_RADIUS = 300  # pixels - how far agents can "see" other entities

    def __init__(self, level: Level, entities: list[Entity],
                 movable_objects: list[MovableObject]):
        self.level = level
        self.entities = entities
        self.movable_objects = movable_objects
        self.level_w, self.level_h = level.get_pixel_dimensions()
        self._max_dist = math.sqrt(self.level_w ** 2 + self.level_h ** 2)
        self.steps_since_tag = 0  # tracks time since last tag event

    def _raycast_walls(self, entity: Entity) -> list[float]:
        """Cast 8 rays from entity position, return normalized distance
        to nearest wall in each direction (0 = touching wall, 1 = no wall in range)."""
        gx, gy = self.level.pixel_to_grid(entity.x, entity.y)
        distances = []

        for dx, dy in _RAY_DIRS:
            dist = 0.0
            for step in range(1, _MAX_RAY_DIST + 1):
                wx = gx + dx * step
                wy = gy + dy * step
                if self.level.is_wall(wx, wy):
                    dist = step
                    break
            if dist == 0:
                dist = _MAX_RAY_DIST  # no wall found within range
            distances.append(dist / _MAX_RAY_DIST)  # normalize to [0, 1]

        return distances

    def get_observation(self, entity: Entity) -> dict:
        """Build ego-centric observation dict for a specific entity.

        All positions are relative to the observing entity and normalized.
        """
        # Self state
        norm_x = entity.x / max(self.level_w, 1)
        norm_y = entity.y / max(self.level_h, 1)

        # Other agents: relative position, distance, is_tagger
        other_agents = []
        tagger_rel = (0.0, 0.0)
        tagger_dist = 0.0
        nearest_runner_rel = (0.0, 0.0)
        nearest_runner_dist = 1.0

        for other in self.entities:
            if other.entity_id == entity.entity_id:
                continue

            # Relative position (normalized by max map distance)
            rel_x = (other.x - entity.x) / self._max_dist
            rel_y = (other.y - entity.y) / self._max_dist
            dist = entity.distance_to(other) / self._max_dist

            other_agents.append({
                "rel_pos": (rel_x, rel_y),
                "distance": dist,
                "is_tagger": other.is_tagger,
            })

            if other.is_tagger:
                tagger_rel = (rel_x, rel_y)
                tagger_dist = dist

            if not other.is_tagger and dist < nearest_runner_dist:
                nearest_runner_dist = dist
                nearest_runner_rel = (rel_x, rel_y)

        # Sort other agents by distance (closest first)
        other_agents.sort(key=lambda a: a["distance"])

        # Wall raycasts (8 directions)
        wall_rays = self._raycast_walls(entity)

        # Nearby crates (relative positions)
        nearby_crates = []
        for crate in self.movable_objects:
            cx, cy = crate.get_center()
            dist_sq = (entity.x - cx) ** 2 + (entity.y - cy) ** 2
            if dist_sq <= self.VISION_RADIUS ** 2:
                rel_cx = (cx - entity.x) / self._max_dist
                rel_cy = (cy - entity.y) / self._max_dist
                nearby_crates.append((rel_cx, rel_cy))

        return {
            "self_pos": (norm_x, norm_y),
            "self_vel": (entity.vx, entity.vy),
            "is_tagger": entity.is_tagger,
            "other_agents": other_agents,
            "tagger_rel": tagger_rel,
            "tagger_dist": tagger_dist,
            "nearest_runner_rel": nearest_runner_rel,
            "nearest_runner_dist": nearest_runner_dist,
            "wall_rays": wall_rays,
            "nearby_crates": nearby_crates,
        }

    def compute_reward(self, entity: Entity, tag_event: dict | None) -> float:
        """Compute reward for a single entity this frame.

        Reward design:
            Tagger:
                +20.0  for successful tag
                -time_penalty: starts at -0.02, grows to -0.2 over 500 steps
                +distance_reward: closer to nearest runner = positive
            Runner:
                -20.0  for being tagged
                +time_bonus: starts at +0.02, grows to +0.2 over 500 steps
                +distance_reward: farther from tagger = positive
        """
        reward = 0.0

        # Scaling factor: ramps from 1.0 to 10.0 over 500 steps since last tag
        # This makes tagger increasingly desperate and rewards runners more
        time_scale = min(1.0 + self.steps_since_tag / 55.0, 10.0)

        if entity.is_tagger:
            # Escalating time penalty (gets worse the longer you go without tagging)
            reward -= 0.02 * time_scale

            if tag_event and tag_event.get("tagger_id") == entity.entity_id:
                reward += 20.0
            else:
                min_dist = float("inf")
                for other in self.entities:
                    if other.entity_id == entity.entity_id:
                        continue
                    if not other.is_tagger:
                        d = entity.distance_to(other)
                        min_dist = min(min_dist, d)
                if min_dist < float("inf"):
                    max_dist = max(self.level_w, self.level_h)
                    reward += 0.5 * (1.0 - min_dist / max_dist)
        else:
            # Escalating survival bonus (grows the longer you stay alive)
            reward += 0.02 * time_scale

            if tag_event and tag_event.get("tagged_id") == entity.entity_id:
                reward -= 20.0
            else:
                for other in self.entities:
                    if other.is_tagger:
                        d = entity.distance_to(other)
                        max_dist = max(self.level_w, self.level_h)
                        reward += 0.3 * (d / max_dist)
                        break

        return reward

    def get_all_rewards(self, tag_event: dict | None) -> dict[int, float]:
        """Return {entity_id: reward} for all entities this frame."""
        rewards = {
            entity.entity_id: self.compute_reward(entity, tag_event)
            for entity in self.entities
        }

        # Update step counter
        if tag_event:
            self.steps_since_tag = 0  # reset on tag
        else:
            self.steps_since_tag += 1

        return rewards
