"""In-game HUD overlay with polished styling."""

import pygame
import config


class HUD:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.SysFont("Arial", 16)
        self.font_bold = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 13)

    def draw(self, entities, tag_logic, mode: config.GameMode,
             algorithm: str, fps: float,
             train_mode: config.TrainMode = config.TrainMode.TRAIN_LIVE):
        """Draw HUD overlay."""
        sw = self.screen.get_width()
        sh = self.screen.get_height()

        # Top bar background
        bar_h = 40
        bar_surf = pygame.Surface((sw, bar_h), pygame.SRCALPHA)
        bar_surf.fill((15, 15, 25, 200))
        self.screen.blit(bar_surf, (0, 0))
        # Bottom accent line
        pygame.draw.line(self.screen, (80, 80, 120), (0, bar_h), (sw, bar_h), 1)

        # Tagger indicator (top-left)
        tagger_label = "IT: ???"
        for e in entities:
            if e.entity_id == tag_logic.current_tagger_id and not e.is_eliminated:
                name = "YOU" if getattr(e, "is_human", False) else f"Agent {e.entity_id}"
                tagger_label = f"IT: {name}"
                break

        # Red dot indicator
        pygame.draw.circle(self.screen, (220, 50, 50), (18, bar_h // 2), 6)
        pygame.draw.circle(self.screen, (255, 100, 100), (18, bar_h // 2), 4)
        text = self.font_large.render(tagger_label, True, (255, 120, 120))
        self.screen.blit(text, (30, bar_h // 2 - text.get_height() // 2))

        # Mode + Algorithm + Train mode (center)
        train_label = "Training" if train_mode == config.TrainMode.TRAIN_LIVE else "Trained"
        train_color = (100, 200, 100) if train_mode == config.TrainMode.TRAIN_LIVE else (100, 180, 255)
        info_str = f"{mode.value}  |  {algorithm}  |  "
        info_text = self.font.render(info_str, True, (180, 180, 200))
        train_text = self.font.render(train_label, True, train_color)
        total_w = info_text.get_width() + train_text.get_width()
        info_x = sw // 2 - total_w // 2
        self.screen.blit(info_text, (info_x, bar_h // 2 - info_text.get_height() // 2))
        self.screen.blit(train_text, (info_x + info_text.get_width(),
                                      bar_h // 2 - train_text.get_height() // 2))

        # FPS (top-right)
        fps_color = (100, 200, 100) if fps > 50 else (200, 200, 100) if fps > 30 else (200, 100, 100)
        fps_text = self.font.render(f"FPS: {int(fps)}", True, fps_color)
        self.screen.blit(fps_text, (sw - fps_text.get_width() - 12,
                                    bar_h // 2 - fps_text.get_height() // 2))

        # Scoreboard panel (top-right, below bar)
        panel_w = 150
        panel_x = sw - panel_w - 10
        panel_y = bar_h + 10
        alive_entities = [e for e in entities if not e.is_eliminated]
        panel_h = 28 + len(alive_entities) * 22

        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surf.fill((15, 15, 25, 180))
        self.screen.blit(panel_surf, (panel_x, panel_y))
        pygame.draw.rect(self.screen, (60, 60, 90),
                         (panel_x, panel_y, panel_w, panel_h), 1, border_radius=4)

        title = self.font_bold.render("Scoreboard", True, (200, 200, 220))
        self.screen.blit(title, (panel_x + panel_w // 2 - title.get_width() // 2,
                                 panel_y + 4))

        y_off = panel_y + 28
        for e in alive_entities:
            name = "You" if getattr(e, "is_human", False) else f"Agent {e.entity_id}"
            if e.is_tagger:
                dot_color = (220, 60, 60)
                text_color = (255, 150, 150)
            else:
                dot_color = (60, 140, 220)
                text_color = (180, 200, 230)

            pygame.draw.circle(self.screen, dot_color, (panel_x + 14, y_off + 8), 4)
            entry = self.font.render(f"{name}: {e.tag_count}", True, text_color)
            self.screen.blit(entry, (panel_x + 24, y_off))
            y_off += 22

        # Bottom hint bar
        hint_h = 26
        hint_surf = pygame.Surface((sw, hint_h), pygame.SRCALPHA)
        hint_surf.fill((15, 15, 25, 160))
        self.screen.blit(hint_surf, (0, sh - hint_h))

        if mode == config.GameMode.PLAYER_MODE:
            hint = "WASD / Arrows: Move  |  Walk near agents to tag  |  Push crates  |  ESC: Menu"
        else:
            hint = f"Simulation Mode  |  {train_label}  |  ESC: Menu"
        hint_text = self.font_small.render(hint, True, (140, 140, 160))
        self.screen.blit(hint_text, hint_text.get_rect(centerx=sw // 2,
                                                        centery=sh - hint_h // 2))
