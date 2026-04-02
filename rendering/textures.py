"""Procedural texture generation for tiles and objects."""

import pygame
import random
import config

# Seed for deterministic textures
_RNG = random.Random(42)


def _vary_color(base: tuple, variance: int = 15) -> tuple:
    """Slightly randomize a color for texture variation."""
    return tuple(max(0, min(255, c + _RNG.randint(-variance, variance))) for c in base)


def generate_wall_texture(size: int = config.TILE_SIZE) -> pygame.Surface:
    """Generate a stone/brick wall texture."""
    surf = pygame.Surface((size, size))

    # Base stone color
    base = (85, 85, 105)
    surf.fill(base)

    # Brick pattern
    brick_h = size // 3
    mortar_color = (60, 58, 70)
    mortar_w = 2

    for row in range(3):
        y = row * brick_h
        # Horizontal mortar line
        pygame.draw.rect(surf, mortar_color, (0, y, size, mortar_w))
        # Vertical mortar (offset every other row)
        offset = (size // 2) * (row % 2)
        brick_w = size // 2
        for col in range(3):
            x = offset + col * brick_w
            pygame.draw.rect(surf, mortar_color, (x, y, mortar_w, brick_h))
            # Shade each brick slightly differently
            brick_rect = pygame.Rect(x + mortar_w, y + mortar_w,
                                     brick_w - mortar_w * 2, brick_h - mortar_w * 2)
            if brick_rect.width > 0 and brick_rect.height > 0:
                brick_color = _vary_color(base, 12)
                pygame.draw.rect(surf, brick_color, brick_rect)

    # Top highlight edge
    pygame.draw.line(surf, (110, 110, 130), (0, 0), (size - 1, 0), 1)
    # Bottom shadow edge
    pygame.draw.line(surf, (55, 55, 70), (0, size - 1), (size - 1, size - 1), 1)

    return surf


def generate_floor_texture(size: int = config.TILE_SIZE) -> pygame.Surface:
    """Generate a stone floor tile texture."""
    surf = pygame.Surface((size, size))

    base = (50, 52, 60)
    surf.fill(base)

    # Subtle noise pattern
    for _ in range(size * 2):
        x = _RNG.randint(0, size - 1)
        y = _RNG.randint(0, size - 1)
        c = _vary_color(base, 8)
        surf.set_at((x, y), c)

    # Tile border (subtle grid)
    border_color = (42, 44, 52)
    pygame.draw.rect(surf, border_color, (0, 0, size, size), 1)

    # Small corner accent
    accent = (58, 60, 68)
    pygame.draw.line(surf, accent, (0, 0), (3, 0), 1)
    pygame.draw.line(surf, accent, (0, 0), (0, 3), 1)

    return surf


def generate_crate_texture(size: int = config.TILE_SIZE) -> pygame.Surface:
    """Generate a wooden crate texture."""
    surf = pygame.Surface((size, size), pygame.SRCALPHA)

    inset = 2
    inner = pygame.Rect(inset, inset, size - inset * 2, size - inset * 2)

    # Wood base
    wood_base = (160, 115, 55)
    wood_dark = (130, 90, 40)
    wood_light = (185, 140, 70)

    pygame.draw.rect(surf, wood_base, inner, border_radius=3)

    # Wood grain lines (horizontal)
    for i in range(4, size - 4, 4):
        grain_color = _vary_color(wood_dark, 10)
        y = inset + i
        if y < size - inset:
            pygame.draw.line(surf, grain_color,
                             (inset + 3, y), (size - inset - 3, y), 1)

    # Cross planks
    cx, cy = size // 2, size // 2
    plank_color = wood_dark

    # Diagonal cross
    pygame.draw.line(surf, plank_color,
                     (inset + 2, inset + 2), (size - inset - 2, size - inset - 2), 2)
    pygame.draw.line(surf, plank_color,
                     (size - inset - 2, inset + 2), (inset + 2, size - inset - 2), 2)

    # Center circle/nail
    pygame.draw.circle(surf, (100, 75, 35), (cx, cy), 3)
    pygame.draw.circle(surf, (80, 60, 30), (cx, cy), 3, 1)

    # Corner nails
    for nx, ny in [(inset + 4, inset + 4), (size - inset - 4, inset + 4),
                   (inset + 4, size - inset - 4), (size - inset - 4, size - inset - 4)]:
        pygame.draw.circle(surf, (90, 70, 35), (nx, ny), 2)

    # Border
    pygame.draw.rect(surf, wood_dark, inner, 2, border_radius=3)

    # Highlight top-left
    pygame.draw.line(surf, wood_light, (inset + 1, inset + 1),
                     (size - inset - 2, inset + 1), 1)
    pygame.draw.line(surf, wood_light, (inset + 1, inset + 1),
                     (inset + 1, size - inset - 2), 1)

    return surf


def generate_spawn_floor_texture(size: int = config.TILE_SIZE) -> pygame.Surface:
    """Floor texture with a subtle spawn marker."""
    surf = generate_floor_texture(size)

    # Subtle diamond marker
    cx, cy = size // 2, size // 2
    marker_color = (65, 70, 80)
    points = [(cx, cy - 5), (cx + 5, cy), (cx, cy + 5), (cx - 5, cy)]
    pygame.draw.polygon(surf, marker_color, points, 1)

    return surf


class TextureCache:
    """Pre-generates and caches all tile textures."""

    def __init__(self):
        self.wall = generate_wall_texture()
        self.floor = generate_floor_texture()
        self.crate = generate_crate_texture()
        self.spawn_floor = generate_spawn_floor_texture()

        # Generate a few floor variants for visual variety
        self.floor_variants = [generate_floor_texture() for _ in range(4)]

    def get_floor(self, col: int, row: int) -> pygame.Surface:
        """Return a floor variant based on position for subtle variety."""
        idx = (col * 7 + row * 13) % len(self.floor_variants)
        return self.floor_variants[idx]
