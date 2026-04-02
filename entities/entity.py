"""Base entity class for all moving characters."""

import pygame
import math


class Entity:
    ENTITY_SIZE = 20  # diameter in pixels

    def __init__(self, x: float, y: float, speed: float, entity_id: int):
        self.entity_id = entity_id
        self.speed = speed
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.is_tagger = False
        self.tag_count = 0  # how many times this entity tagged someone
        half = self.ENTITY_SIZE // 2
        self.rect = pygame.Rect(int(x) - half, int(y) - half,
                                self.ENTITY_SIZE, self.ENTITY_SIZE)

    def set_velocity(self, dx: float, dy: float):
        """Set movement direction. Will be normalized to speed."""
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            self.vx = (dx / length) * self.speed
            self.vy = (dy / length) * self.speed
        else:
            self.vx = 0.0
            self.vy = 0.0

    def apply_velocity(self):
        """Move by current velocity. Rect is synced by collision resolution."""
        self.x += self.vx
        self.y += self.vy

    def get_position(self) -> tuple[float, float]:
        return self.x, self.y

    def get_center(self) -> tuple[int, int]:
        return int(self.x), int(self.y)

    def distance_to(self, other: "Entity") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def get_state_vector(self) -> list[float]:
        """Return state for RL observation: [x, y, vx, vy, is_tagger]."""
        return [self.x, self.y, self.vx, self.vy, float(self.is_tagger)]
