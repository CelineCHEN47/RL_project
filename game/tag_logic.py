"""Tag game rules: who is 'it', tagging detection, cooldowns.

Cooldown is tracked in game frames (not wall-clock time) so that
headless training and rendered play behave identically.
"""

from entities.entity import Entity
import config


class TagLogic:
    def __init__(self, entities: list[Entity]):
        self.entities = entities
        self.current_tagger_id = -1
        self._cooldown_remaining = 0  # frames until next tag is allowed

    def set_tagger(self, entity_id: int):
        """Make an entity the tagger."""
        assigned = False
        for e in self.entities:
            can_be_tagger = (not e.is_eliminated and e.entity_id == entity_id)
            e.is_tagger = can_be_tagger
            if can_be_tagger:
                assigned = True

        if assigned:
            self.current_tagger_id = entity_id
        else:
            self.current_tagger_id = -1
            for e in self.entities:
                if not e.is_eliminated:
                    e.is_tagger = True
                    self.current_tagger_id = e.entity_id
                    break

        # Apply per-role speed so optimal pursuit produces a positive
        # distance delta (tagger gradually closes on an ideally-fleeing runner).
        for e in self.entities:
            if hasattr(e, "is_human") and e.is_human:
                continue  # don't override human player speed
            e.speed = config.TAGGER_SPEED if e.is_tagger else config.RUNNER_SPEED

        self._cooldown_remaining = config.TAG_COOLDOWN_FRAMES

    def update(self) -> dict | None:
        """Check if tagger has tagged any runner.
        Returns tag event dict or None."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return None

        tagger = None
        for e in self.entities:
            if e.entity_id == self.current_tagger_id and not e.is_eliminated:
                tagger = e
                break

        if tagger is None:
            return None

        for runner in self.entities:
            if runner.entity_id == self.current_tagger_id or runner.is_eliminated:
                continue
            if tagger.distance_to(runner) < config.TAG_RADIUS:
                # Tag runner — runner will be respawned by the simulation layer.
                tagger.tag_count += 1
                runner.vx = 0.0
                runner.vy = 0.0
                self._cooldown_remaining = config.TAG_COOLDOWN_FRAMES
                return {
                    "tagger_id": self.current_tagger_id,
                    "tagged_id": runner.entity_id,
                    "tagged_pos": (runner.x, runner.y),
                }

        return None
