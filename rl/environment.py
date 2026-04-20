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
            if other.is_eliminated:
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

    # Proximity reward parameters (Plan C — pure proximity, no delta)
    # Fixed best PPO ablation parameters (portable across devices)
    _PROXIMITY_RADIUS = 140.0   # pixels, ~4.4 tiles — gradient active inside this
    _PROXIMITY_COEF = 0.27      # peak reward at d=0
    _TIME_PENALTY = 0.05        # tagger per non-tag frame
    _SURVIVAL_BONUS = 0.10      # runner per non-tag frame
    _TAG_REWARD = 50.0          # tagger on tag
    _TAG_PENALTY = 80.0         # runner on being tagged
    _PROXIMITY_AGG = "sum"      # sum or mean

    def compute_reward(self, entity: Entity, tag_event: dict | None) -> float:
        """Pure-proximity reward (Plan C).

        Design:
        - NO distance-delta shaping (too noisy, target-switching artifacts)
        - NO time_scale amplifier (unnecessary variance in V(s))
        - NO movement / wall / round-win auxiliary terms (simplify)
        - Quadratic proximity with R=100 makes the gradient sharpest near the
          tag zone where it actually matters; far away the signal is flat so
          the policy must rely on observation (nearest_runner_rel) to navigate.
        - Tagger's proximity BONUS and runner's DANGER penalty are perfectly
          symmetric (same function, same radius, same coefficient), so both
          agents face equally-shaped learning signals.

        Per-frame reward shape (non-tag frames):
            proximity(d) = 0.15 * max(0, 1 - d/100)**2
            tagger: -0.05 + sum_over_runners( proximity(d_i) )
            runner: +0.05 - proximity(d_tagger)

        Terminal (tag frame):
            tagger: +50 (shaping skipped — keeps tag signal clean)
            runner: -50 (same)
        """
        # Eliminated: preserve single terminal penalty, otherwise silent.
        # (In the respawn game mechanic runners are not actually eliminated,
        # but we keep this branch for safety.)
        if entity.is_eliminated:
            if tag_event and tag_event.get("tagged_id") == entity.entity_id:
                return -self._TAG_PENALTY
            return 0.0

        if entity.is_tagger:
            if tag_event and tag_event.get("tagger_id") == entity.entity_id:
                return self._TAG_REWARD

            # Non-tag frame: constant time pressure + summed proximity bonus
            reward = -self._TIME_PENALTY
            r_inv = 1.0 / self._PROXIMITY_RADIUS
            proximity_terms = []
            for other in self.entities:
                if (other.entity_id == entity.entity_id
                        or other.is_tagger or other.is_eliminated):
                    continue
                d = entity.distance_to(other)
                if d < self._PROXIMITY_RADIUS:
                    t = 1.0 - d * r_inv
                    proximity_terms.append(self._PROXIMITY_COEF * t * t)
            if proximity_terms:
                if self._PROXIMITY_AGG == "mean":
                    reward += sum(proximity_terms) / len(proximity_terms)
                else:
                    reward += sum(proximity_terms)
            return reward

        # Runner
        if tag_event and tag_event.get("tagged_id") == entity.entity_id:
            return -self._TAG_PENALTY

        # Non-tag frame: constant survival + danger penalty from tagger
        reward = self._SURVIVAL_BONUS
        for other in self.entities:
            if other.is_tagger and not other.is_eliminated:
                d = entity.distance_to(other)
                if d < self._PROXIMITY_RADIUS:
                    t = 1.0 - d / self._PROXIMITY_RADIUS
                    reward -= self._PROXIMITY_COEF * t * t
                break  # only one tagger
        return reward


    def get_all_rewards(self, tag_event: dict | None) -> dict[int, float]:
        """Return {entity_id: reward} for all entities this frame."""
        rewards = {
            entity.entity_id: self.compute_reward(entity, tag_event)
            for entity in self.entities
            if not entity.is_eliminated
            or (tag_event and tag_event.get("tagged_id") == entity.entity_id)
        }

        # Update step counter
        if tag_event:
            self.steps_since_tag = 0  # reset on tag
        else:
            self.steps_since_tag += 1

        return rewards
