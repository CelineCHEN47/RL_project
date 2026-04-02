"""Gym-like environment wrapper for the Tag game.

This module provides observations and computes rewards for RL agents.
It is a passive observer - it reads game state but does not control it.
"""

from entities.entity import Entity
from entities.movable_object import MovableObject
from world.level import Level
import config


class TagEnvironment:
    VISION_RADIUS = 200  # pixels - how far agents can "see"

    def __init__(self, level: Level, entities: list[Entity],
                 movable_objects: list[MovableObject]):
        self.level = level
        self.entities = entities
        self.movable_objects = movable_objects
        self.level_w, self.level_h = level.get_pixel_dimensions()

    def get_observation(self, entity: Entity) -> dict:
        """Build observation dict for a specific entity."""
        # Normalize position to [0, 1]
        norm_x = entity.x / max(self.level_w, 1)
        norm_y = entity.y / max(self.level_h, 1)

        # Find tagger
        tagger_pos = (0.0, 0.0)
        other_agents = []
        for other in self.entities:
            if other.entity_id == entity.entity_id:
                continue
            other_norm_x = other.x / max(self.level_w, 1)
            other_norm_y = other.y / max(self.level_h, 1)
            other_agents.append({
                "pos": (other_norm_x, other_norm_y),
                "is_tagger": other.is_tagger,
            })
            if other.is_tagger:
                tagger_pos = (other_norm_x, other_norm_y)

        if entity.is_tagger:
            tagger_pos = (norm_x, norm_y)

        # Nearby walls (grid coords within vision radius)
        gx, gy = self.level.pixel_to_grid(entity.x, entity.y)
        vision_tiles = int(self.VISION_RADIUS / config.TILE_SIZE)
        nearby_walls = []
        for dy in range(-vision_tiles, vision_tiles + 1):
            for dx in range(-vision_tiles, vision_tiles + 1):
                wx, wy = gx + dx, gy + dy
                if self.level.is_wall(wx, wy):
                    nearby_walls.append((wx, wy))

        # Nearby crates
        nearby_crates = []
        for crate in self.movable_objects:
            cx, cy = crate.get_center()
            dist_sq = (entity.x - cx) ** 2 + (entity.y - cy) ** 2
            if dist_sq <= self.VISION_RADIUS ** 2:
                nearby_crates.append((cx / max(self.level_w, 1),
                                      cy / max(self.level_h, 1)))

        return {
            "self_pos": (norm_x, norm_y),
            "self_vel": (entity.vx, entity.vy),
            "is_tagger": entity.is_tagger,
            "other_agents": other_agents,
            "tagger_pos": tagger_pos,
            "nearby_walls": nearby_walls,
            "nearby_crates": nearby_crates,
        }

    def compute_reward(self, entity: Entity, tag_event: dict | None) -> float:
        """Compute reward for a single entity this frame.

        Reward design:
            Tagger:
                +20.0  for successful tag
                -0.05  per step (pressure to tag quickly)
                +distance_reward: gets closer to nearest runner = positive
            Runner:
                -20.0  for being tagged
                +0.05  per step survived
                +distance_reward: gets farther from tagger = positive
        """
        reward = 0.0

        if entity.is_tagger:
            # --- Tagger rewards ---
            reward -= 0.05  # time penalty

            if tag_event and tag_event.get("tagger_id") == entity.entity_id:
                reward += 20.0  # successful tag
            else:
                # Distance shaping: reward getting closer to nearest runner
                min_dist = float("inf")
                for other in self.entities:
                    if other.entity_id == entity.entity_id:
                        continue
                    if not other.is_tagger:
                        d = entity.distance_to(other)
                        min_dist = min(min_dist, d)
                if min_dist < float("inf"):
                    # Closer = higher reward (max ~0.5 when very close)
                    max_dist = max(self.level_w, self.level_h)
                    reward += 0.5 * (1.0 - min_dist / max_dist)
        else:
            # --- Runner rewards ---
            reward += 0.05  # survival bonus

            if tag_event and tag_event.get("tagged_id") == entity.entity_id:
                reward -= 20.0  # got tagged
            else:
                # Distance shaping: reward being far from tagger
                for other in self.entities:
                    if other.is_tagger:
                        d = entity.distance_to(other)
                        max_dist = max(self.level_w, self.level_h)
                        reward += 0.3 * (d / max_dist)  # farther = better
                        break

        return reward

    def get_all_rewards(self, tag_event: dict | None) -> dict[int, float]:
        """Return {entity_id: reward} for all entities this frame."""
        return {
            entity.entity_id: self.compute_reward(entity, tag_event)
            for entity in self.entities
        }
