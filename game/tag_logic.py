"""Tag game rules: who is 'it', tagging detection, cooldowns."""

import pygame
from entities.entity import Entity
import config


class TagLogic:
    def __init__(self, entities: list[Entity]):
        self.entities = entities
        self.current_tagger_id = -1
        self.last_tag_time = 0

    def set_tagger(self, entity_id: int):
        """Make an entity the tagger."""
        for e in self.entities:
            e.is_tagger = (e.entity_id == entity_id)
        self.current_tagger_id = entity_id
        self.last_tag_time = pygame.time.get_ticks()

    def update(self) -> dict | None:
        """Check if tagger has tagged any runner.
        Returns tag event dict or None."""
        now = pygame.time.get_ticks()
        if now - self.last_tag_time < config.TAG_COOLDOWN_MS:
            return None

        tagger = None
        for e in self.entities:
            if e.entity_id == self.current_tagger_id:
                tagger = e
                break

        if tagger is None:
            return None

        for runner in self.entities:
            if runner.entity_id == self.current_tagger_id:
                continue
            if tagger.distance_to(runner) < config.TAG_RADIUS:
                # Tag!
                old_tagger_id = self.current_tagger_id
                tagger.tag_count += 1
                self.set_tagger(runner.entity_id)
                return {
                    "tagger_id": old_tagger_id,
                    "tagged_id": runner.entity_id,
                }

        return None
