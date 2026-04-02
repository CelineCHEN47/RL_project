"""Pushable crate / movable object."""

import pygame
import config


class MovableObject:
    def __init__(self, x: float, y: float):
        self.rect = pygame.Rect(int(x), int(y), config.TILE_SIZE, config.TILE_SIZE)

    def push(self, dx: int, dy: int) -> pygame.Rect:
        """Return the rect the crate would occupy if pushed in (dx, dy).
        Does not actually move the crate. Caller must validate and then apply."""
        new_rect = self.rect.copy()
        new_rect.x += dx * config.CRATE_PUSH_SPEED
        new_rect.y += dy * config.CRATE_PUSH_SPEED
        return new_rect

    def apply_push(self, dx: int, dy: int):
        """Actually move the crate."""
        self.rect.x += int(dx * config.CRATE_PUSH_SPEED)
        self.rect.y += int(dy * config.CRATE_PUSH_SPEED)

    def get_center(self) -> tuple[int, int]:
        return self.rect.centerx, self.rect.centery
