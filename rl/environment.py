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
        """Optimized reward function.

        Key design changes vs original:
        1. Distance CHANGE reward (delta-based) instead of absolute distance
        → prevents "stand still at comfortable distance" exploits
        2. Movement bonus for tagger → encourages active pursuit
        3. Capped, gentler time scaling → prevents reward saturation
        4. Larger tag event rewards relative to per-step rewards
        → makes tag events the dominant learning signal
        5. Wall penalty → discourages corner camping
        """
        reward = 0.0

        # --- Milder time scaling: ramps from 1.0 to 3.0 over 300 steps ---
        time_scale = min(1.0 + self.steps_since_tag / 300.0, 3.0)

        if entity.is_tagger:
            # 1. Tag success: dominant reward signal
            if tag_event and tag_event.get("tagger_id") == entity.entity_id:
                reward += 50.0
                return reward  # no need for shaping on tag frame

            # 2. Time penalty (mild, escalating)
            reward -= 0.01 * time_scale

            # 3. Distance CHANGE to nearest runner (delta-based)
            #    Reward getting closer, punish getting farther
            min_dist = float("inf")
            for other in self.entities:
                if other.entity_id == entity.entity_id or other.is_tagger:
                    continue
                d = entity.distance_to(other)
                min_dist = min(min_dist, d)

            if min_dist < float("inf"):
                # Store last distance for delta computation
                prev_dist = getattr(entity, '_prev_tagger_dist', min_dist)
                delta = prev_dist - min_dist  # positive = getting closer
                max_dist = max(self.level_w, self.level_h)
                reward += 1.0 * (delta / max_dist) * time_scale
                entity._prev_tagger_dist = min_dist

            # 4. Small movement bonus (prevent standing still)
            speed = math.sqrt(entity.vx ** 2 + entity.vy ** 2)
            if speed > 0.5:
                reward += 0.005

        else:  # Runner
            # 1. Got tagged: dominant punishment
            if tag_event and tag_event.get("tagged_id") == entity.entity_id:
                reward -= 50.0
                return reward

            # 2. Survival bonus (mild, escalating)
            reward += 0.01 * time_scale

            # 3. Distance CHANGE from tagger (delta-based)
            for other in self.entities:
                if other.is_tagger:
                    d = entity.distance_to(other)
                    prev_dist = getattr(entity, '_prev_runner_dist', d)
                    delta = d - prev_dist  # positive = getting farther
                    max_dist = max(self.level_w, self.level_h)
                    reward += 0.5 * (delta / max_dist) * time_scale
                    entity._prev_runner_dist = d
                    break

            # 4. Proximity danger penalty (close to tagger = bad)
            #    Only activates within a danger zone, not map-wide
            for other in self.entities:
                if other.is_tagger:
                    d = entity.distance_to(other)
                    danger_zone = config.TAG_RADIUS * 4  # ~96 pixels
                    if d < danger_zone:
                        reward -= 0.1 * (1.0 - d / danger_zone)
                    break

            # 5. Wall proximity penalty (anti-corner-camping)
            wall_rays = self._raycast_walls(entity)
            num_close_walls = sum(1 for r in wall_rays if r < 0.2)
            if num_close_walls >= 3:
                reward -= 0.02 * num_close_walls

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
