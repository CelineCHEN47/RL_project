"""Procedural character sprite generation."""

import pygame
import math


def generate_character_sprite(radius: int, body_color: tuple,
                              is_tagger: bool = False,
                              is_player: bool = False) -> pygame.Surface:
    """Generate a top-down character sprite.

    Characters are drawn as a body circle with a face direction indicator,
    eyes, and optional tagger/player markings.
    """
    size = radius * 2 + 12  # extra padding for effects
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx, cy = size // 2, size // 2

    # Shadow (offset slightly down-right)
    shadow_color = (0, 0, 0, 60)
    pygame.draw.circle(surf, shadow_color, (cx + 2, cy + 3), radius)

    # Body outline (darker ring)
    darker = tuple(max(0, c - 40) for c in body_color)
    pygame.draw.circle(surf, darker, (cx, cy), radius)

    # Body fill (gradient-like: lighter center)
    lighter = tuple(min(255, c + 30) for c in body_color)
    pygame.draw.circle(surf, body_color, (cx, cy), radius - 1)
    pygame.draw.circle(surf, lighter, (cx - 2, cy - 2), radius // 2)

    # Eyes (two small dots looking "forward" / up)
    eye_color = (255, 255, 255)
    pupil_color = (30, 30, 40)
    eye_offset_x = radius // 3
    eye_y = cy - radius // 4

    # Left eye
    pygame.draw.circle(surf, eye_color, (cx - eye_offset_x, eye_y), 3)
    pygame.draw.circle(surf, pupil_color, (cx - eye_offset_x, eye_y - 1), 2)
    # Right eye
    pygame.draw.circle(surf, eye_color, (cx + eye_offset_x, eye_y), 3)
    pygame.draw.circle(surf, pupil_color, (cx + eye_offset_x, eye_y - 1), 2)

    if is_tagger:
        # Angry eyebrows for tagger
        brow_color = (180, 40, 40)
        pygame.draw.line(surf, brow_color,
                         (cx - eye_offset_x - 3, eye_y - 5),
                         (cx - eye_offset_x + 2, eye_y - 3), 2)
        pygame.draw.line(surf, brow_color,
                         (cx + eye_offset_x + 3, eye_y - 5),
                         (cx + eye_offset_x - 2, eye_y - 3), 2)

    if is_player:
        # Player gets a small crown/triangle on top
        crown_color = (255, 220, 50)
        crown_points = [
            (cx, cy - radius - 4),
            (cx - 5, cy - radius + 2),
            (cx + 5, cy - radius + 2),
        ]
        pygame.draw.polygon(surf, crown_color, crown_points)
        pygame.draw.polygon(surf, (200, 170, 30), crown_points, 1)

    return surf


def generate_tagger_aura_frame(radius: int, tick: int) -> pygame.Surface:
    """Generate one frame of the pulsing tagger aura."""
    pulse = int(6 * math.sin(tick * 0.08)) + 8
    aura_radius = radius + pulse
    size = aura_radius * 2 + 4
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx, cy = size // 2, size // 2

    # Multiple fading rings
    for i in range(3):
        r = aura_radius - i * 3
        alpha = max(0, 80 - i * 30)
        color = (255, 80, 80, alpha)
        pygame.draw.circle(surf, color, (cx, cy), r, 2)

    return surf


class SpriteCache:
    """Pre-generates character sprites for all entity states."""

    # Body colors
    TAGGER_COLOR = (200, 55, 55)
    RUNNER_COLOR = (55, 140, 210)
    PLAYER_TAGGER_COLOR = (220, 70, 70)
    PLAYER_RUNNER_COLOR = (70, 180, 240)

    def __init__(self, radius: int = 10):
        self.radius = radius
        self.sprites = {
            "runner": generate_character_sprite(radius, self.RUNNER_COLOR),
            "tagger": generate_character_sprite(radius, self.TAGGER_COLOR,
                                                is_tagger=True),
            "player_runner": generate_character_sprite(radius, self.PLAYER_RUNNER_COLOR,
                                                       is_player=True),
            "player_tagger": generate_character_sprite(radius, self.PLAYER_TAGGER_COLOR,
                                                       is_tagger=True, is_player=True),
        }

    def get_sprite(self, is_tagger: bool, is_player: bool) -> pygame.Surface:
        if is_player:
            return self.sprites["player_tagger" if is_tagger else "player_runner"]
        return self.sprites["tagger" if is_tagger else "runner"]
