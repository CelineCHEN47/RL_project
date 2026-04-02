"""Human-controlled player entity."""

import pygame
from entities.entity import Entity
import config


class Player(Entity):
    def __init__(self, x: float, y: float, entity_id: int):
        super().__init__(x, y, config.PLAYER_SPEED, entity_id)
        self.is_human = True

    def handle_input(self, keys_pressed):
        """Read WASD/arrow keys and set velocity."""
        dx = 0.0
        dy = 0.0
        if keys_pressed[pygame.K_w] or keys_pressed[pygame.K_UP]:
            dy -= 1.0
        if keys_pressed[pygame.K_s] or keys_pressed[pygame.K_DOWN]:
            dy += 1.0
        if keys_pressed[pygame.K_a] or keys_pressed[pygame.K_LEFT]:
            dx -= 1.0
        if keys_pressed[pygame.K_d] or keys_pressed[pygame.K_RIGHT]:
            dx += 1.0
        self.set_velocity(dx, dy)
