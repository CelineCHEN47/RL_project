"""Particle effects system for visual feedback."""

import pygame
import random
import math


class Particle:
    def __init__(self, x: float, y: float, vx: float, vy: float,
                 color: tuple, lifetime: int, size: float = 3.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size
        self.alive = True

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.05  # slight gravity
        self.vx *= 0.98  # friction
        self.vy *= 0.98
        self.lifetime -= 1
        if self.lifetime <= 0:
            self.alive = False

    def draw(self, surface: pygame.Surface, camera_x: int, camera_y: int):
        if not self.alive:
            return
        progress = self.lifetime / self.max_lifetime
        alpha = int(255 * progress)
        current_size = max(1, int(self.size * progress))
        sx = int(self.x - camera_x)
        sy = int(self.y - camera_y)

        # Draw with fading color
        faded = tuple(min(255, int(c * progress)) for c in self.color[:3])
        pygame.draw.circle(surface, faded, (sx, sy), current_size)


class ParticleSystem:
    def __init__(self):
        self.particles: list[Particle] = []

    def emit_tag_burst(self, x: float, y: float):
        """Burst of particles when a tag happens."""
        colors = [
            (255, 100, 100), (255, 180, 50), (255, 255, 100),
            (255, 150, 80), (255, 80, 80),
        ]
        for _ in range(25):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(1.5, 5.0)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            color = random.choice(colors)
            lifetime = random.randint(20, 45)
            size = random.uniform(2.0, 5.0)
            self.particles.append(Particle(x, y, vx, vy, color, lifetime, size))

    def emit_movement_trail(self, x: float, y: float, is_tagger: bool):
        """Small trail particles behind a moving entity."""
        if random.random() > 0.3:
            return
        color = (180, 60, 60) if is_tagger else (60, 120, 180)
        vx = random.uniform(-0.5, 0.5)
        vy = random.uniform(-0.5, 0.5)
        self.particles.append(
            Particle(x, y, vx, vy, color, random.randint(8, 15), 2.0))

    def update(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.alive]

    def draw(self, surface: pygame.Surface, camera_x: int, camera_y: int):
        for p in self.particles:
            p.draw(surface, camera_x, camera_y)
