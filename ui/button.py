"""Reusable button widget with polished styling."""

import pygame
import config


class Button:
    def __init__(self, x: int, y: int, width: int, height: int,
                 text: str, callback=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.callback = callback
        self.hovered = False
        self.selected = False
        self.disabled = False
        self.font = pygame.font.SysFont("Arial", 19, bold=True)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if self.disabled:
            return False
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.callback:
                    self.callback()
                return True
        return False

    def draw(self, surface: pygame.Surface):
        if self.disabled:
            bg_color = (40, 40, 50)
            border_color = (55, 55, 65)
            text_color = (100, 100, 110)
        elif self.selected:
            bg_color = (40, 160, 90)
            border_color = (80, 220, 130)
            text_color = (255, 255, 255)
        elif self.hovered:
            bg_color = (80, 80, 120)
            border_color = (130, 130, 180)
            text_color = (255, 255, 255)
        else:
            bg_color = (55, 55, 80)
            border_color = (85, 85, 115)
            text_color = (210, 210, 230)

        # Shadow
        shadow_rect = self.rect.move(2, 3)
        shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(shadow_surf, (0, 0, 0, 60), shadow_surf.get_rect(),
                         border_radius=8)
        surface.blit(shadow_surf, shadow_rect)

        # Main button body
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=8)

        # Top highlight
        highlight_rect = pygame.Rect(self.rect.x + 2, self.rect.y + 1,
                                     self.rect.width - 4, self.rect.height // 2)
        highlight_surf = pygame.Surface(highlight_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(highlight_surf, (255, 255, 255, 15),
                         highlight_surf.get_rect(), border_radius=6)
        surface.blit(highlight_surf, highlight_rect)

        # Border
        pygame.draw.rect(surface, border_color, self.rect, 2, border_radius=8)

        # Selected check mark
        if self.selected:
            check = self.font.render("✓ ", True, (200, 255, 200))
            text_surf = self.font.render(self.text, True, text_color)
            total_w = check.get_width() + text_surf.get_width()
            x = self.rect.centerx - total_w // 2
            y = self.rect.centery - check.get_height() // 2
            surface.blit(check, (x, y))
            surface.blit(text_surf, (x + check.get_width(), y))
        else:
            text_surf = self.font.render(self.text, True, text_color)
            text_rect = text_surf.get_rect(center=self.rect.center)
            surface.blit(text_surf, text_rect)
