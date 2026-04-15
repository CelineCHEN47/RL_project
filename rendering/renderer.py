"""All Pygame rendering calls - uses procedural textures and sprites."""

import pygame
import math
from world.level import Level
from world.tile import TileType
from entities.entity import Entity
from entities.movable_object import MovableObject
from rendering.textures import TextureCache
from rendering.sprites import SpriteCache, generate_tagger_aura_frame
from rendering.particles import ParticleSystem
import config


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.camera_x = 0
        self.camera_y = 0
        self.textures = TextureCache()
        self.sprites = SpriteCache(radius=Entity.ENTITY_SIZE // 2)
        self.particles = ParticleSystem()
        self.label_font = self._safe_sys_font("Arial", 11, bold=True)
        self.badge_font = self._safe_sys_font("Arial", 13, bold=True)

        # Pre-render the shadow surface for entities
        shadow_size = Entity.ENTITY_SIZE + 8
        self.entity_shadow = pygame.Surface((shadow_size, shadow_size), pygame.SRCALPHA)
        pygame.draw.ellipse(self.entity_shadow, (0, 0, 0, 50),
                            (0, shadow_size // 4, shadow_size, shadow_size // 2))

    @staticmethod
    def _safe_sys_font(name: str, size: int, bold: bool = False) -> pygame.font.Font:
        """Prefer system font but fall back to pygame default if discovery fails."""
        try:
            return pygame.font.SysFont(name, size, bold=bold)
        except Exception:
            font = pygame.font.Font(None, size)
            font.set_bold(bold)
            return font

    def set_camera(self, target_x: float, target_y: float):
        """Center camera on a target position."""
        sw, sh = self.screen.get_size()
        self.camera_x = int(target_x - sw // 2)
        self.camera_y = int(target_y - sh // 2)

    def clamp_camera(self, level_w: int, level_h: int):
        sw, sh = self.screen.get_size()
        self.camera_x = max(0, min(self.camera_x, level_w - sw))
        self.camera_y = max(0, min(self.camera_y, level_h - sh))

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        return int(x - self.camera_x), int(y - self.camera_y)

    def draw_level(self, level: Level):
        """Draw textured floor and wall tiles."""
        self.screen.fill(config.COLOR_BG)
        sw, sh = self.screen.get_size()

        start_col = max(0, self.camera_x // config.TILE_SIZE)
        start_row = max(0, self.camera_y // config.TILE_SIZE)
        end_col = min(level.width, (self.camera_x + sw) // config.TILE_SIZE + 2)
        end_row = min(level.height, (self.camera_y + sh) // config.TILE_SIZE + 2)

        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                tile = level.grid[row][col]
                px, py = self.world_to_screen(col * config.TILE_SIZE,
                                               row * config.TILE_SIZE)

                if tile == TileType.WALL:
                    self.screen.blit(self.textures.wall, (px, py))
                elif tile == TileType.SPAWN:
                    self.screen.blit(self.textures.spawn_floor, (px, py))
                else:
                    self.screen.blit(self.textures.get_floor(col, row), (px, py))

        # Draw wall shadows (bottom and right edges of walls cast shadows)
        shadow_surf = pygame.Surface((config.TILE_SIZE, 6), pygame.SRCALPHA)
        shadow_surf.fill((0, 0, 0, 35))
        shadow_side = pygame.Surface((6, config.TILE_SIZE), pygame.SRCALPHA)
        shadow_side.fill((0, 0, 0, 25))

        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                if level.grid[row][col] != TileType.WALL:
                    continue
                px, py = self.world_to_screen(col * config.TILE_SIZE,
                                               row * config.TILE_SIZE)
                # Shadow below wall
                if row + 1 < level.height and level.grid[row + 1][col] != TileType.WALL:
                    self.screen.blit(shadow_surf, (px, py + config.TILE_SIZE))
                # Shadow right of wall
                if col + 1 < level.width and level.grid[row][col + 1] != TileType.WALL:
                    self.screen.blit(shadow_side, (px + config.TILE_SIZE, py))

    def draw_entities(self, entities: list[Entity], current_tagger_id: int,
                      tick: int):
        """Draw all entities with sprites, shadows, and effects."""
        for entity in entities:
            if entity.is_eliminated:
                continue
            sx, sy = self.world_to_screen(entity.x, entity.y)
            is_player = getattr(entity, "is_human", False)

            # Movement trail particles
            if abs(entity.vx) > 0.1 or abs(entity.vy) > 0.1:
                self.particles.emit_movement_trail(entity.x, entity.y,
                                                   entity.is_tagger)

            # Ground shadow
            shadow_rect = self.entity_shadow.get_rect(center=(sx + 2, sy + 5))
            self.screen.blit(self.entity_shadow, shadow_rect)

            # Tagger aura
            if entity.is_tagger:
                aura = generate_tagger_aura_frame(Entity.ENTITY_SIZE // 2, tick)
                aura_rect = aura.get_rect(center=(sx, sy))
                self.screen.blit(aura, aura_rect)

            # Character sprite
            sprite = self.sprites.get_sprite(entity.is_tagger, is_player)
            sprite_rect = sprite.get_rect(center=(sx, sy))
            self.screen.blit(sprite, sprite_rect)

            # Name label above character
            if is_player:
                label = "YOU"
                label_color = (255, 255, 100)
            else:
                label = f"Agent {entity.entity_id}"
                label_color = (200, 200, 220)

            text = self.label_font.render(label, True, label_color)
            text_rect = text.get_rect(centerx=sx, bottom=sy - Entity.ENTITY_SIZE // 2 - 6)

            # Label background
            bg_rect = text_rect.inflate(6, 2)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 120))
            self.screen.blit(bg_surf, bg_rect)
            self.screen.blit(text, text_rect)

            # Tagger "IT!" badge
            if entity.is_tagger:
                badge = self.badge_font.render("IT!", True, (255, 255, 200))
                badge_rect = badge.get_rect(centerx=sx,
                                            top=sy + Entity.ENTITY_SIZE // 2 + 4)
                badge_bg = badge_rect.inflate(8, 4)
                pygame.draw.rect(self.screen, (180, 40, 40), badge_bg,
                                 border_radius=4)
                pygame.draw.rect(self.screen, (255, 80, 80), badge_bg, 1,
                                 border_radius=4)
                self.screen.blit(badge, badge_rect)

    def draw_movable_objects(self, objects: list[MovableObject]):
        """Draw crates with textures and shadows."""
        for obj in objects:
            sx, sy = self.world_to_screen(obj.rect.x, obj.rect.y)

            # Crate shadow
            shadow = pygame.Surface((config.TILE_SIZE, 6), pygame.SRCALPHA)
            shadow.fill((0, 0, 0, 40))
            self.screen.blit(shadow, (sx + 3, sy + config.TILE_SIZE - 2))

            # Crate texture
            self.screen.blit(self.textures.crate, (sx, sy))

    def emit_tag_particles(self, x: float, y: float):
        """Called by game manager when a tag event occurs."""
        self.particles.emit_tag_burst(x, y)

    def update_particles(self):
        """Update particle physics."""
        self.particles.update()

    def draw_particles(self):
        """Draw all active particles."""
        self.particles.draw(self.screen, self.camera_x, self.camera_y)
