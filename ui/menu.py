"""Main menu screen with polished visuals."""

import pygame
import math
import config
from ui.button import Button


class Menu:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.selected_mode = config.GameMode.PLAYER_MODE
        self.selected_algorithm = "Q-Learning"
        self.title_font = pygame.font.SysFont("Arial", 56, bold=True)
        self.subtitle_font = pygame.font.SysFont("Arial", 22)
        self.label_font = pygame.font.SysFont("Arial", 18, bold=True)
        self.hint_font = pygame.font.SysFont("Arial", 15)
        self.tick = 0
        self._build_buttons()

    def _build_buttons(self):
        """Create all menu buttons."""
        sw, sh = self.screen.get_size()
        cx = sw // 2

        # Mode buttons
        mode_y = 270
        btn_w, btn_h = 210, 48
        gap = 24

        self.mode_buttons = []
        for i, mode in enumerate(config.GameMode):
            x = cx - btn_w - gap // 2 + i * (btn_w + gap)
            btn = Button(x, mode_y, btn_w, btn_h, mode.value)
            btn._mode = mode
            if mode == self.selected_mode:
                btn.selected = True
            self.mode_buttons.append(btn)

        # Algorithm buttons
        algo_y = 400
        algo_names = list(config.RL_ALGORITHMS.keys())
        abtn_w = 145
        total_w = len(algo_names) * (abtn_w + 15) - 15
        start_x = cx - total_w // 2

        self.algo_buttons = []
        for i, name in enumerate(algo_names):
            x = start_x + i * (abtn_w + 15)
            btn = Button(x, algo_y, abtn_w, btn_h, name)
            btn._algo_name = name
            if name == self.selected_algorithm:
                btn.selected = True
            self.algo_buttons.append(btn)

        # Start button
        self.start_button = Button(cx - 110, 520, 220, 58, "START GAME")

    def handle_event(self, event: pygame.event.Event) -> dict | None:
        """Process events. Returns config dict when START is clicked."""
        for btn in self.mode_buttons:
            if btn.handle_event(event):
                self.selected_mode = btn._mode
                for b in self.mode_buttons:
                    b.selected = (b is btn)

        for btn in self.algo_buttons:
            if btn.handle_event(event):
                self.selected_algorithm = btn._algo_name
                for b in self.algo_buttons:
                    b.selected = (b is btn)

        if self.start_button.handle_event(event):
            return {
                "mode": self.selected_mode,
                "algorithm": self.selected_algorithm,
            }
        return None

    def draw(self):
        """Render the menu with animated background."""
        self.tick += 1
        sw, sh = self.screen.get_size()

        # Animated gradient background
        self.screen.fill(config.COLOR_MENU_BG)

        # Floating dots in background
        for i in range(20):
            seed = i * 137
            x = (seed * 3 + int(self.tick * 0.3 * ((i % 3) + 1))) % sw
            y = (seed * 7 + int(self.tick * 0.2 * ((i % 2) + 1))) % sh
            alpha = int(40 + 20 * math.sin(self.tick * 0.02 + i))
            radius = 2 + (i % 3)
            dot_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(dot_surf, (100, 100, 160, alpha),
                               (radius, radius), radius)
            self.screen.blit(dot_surf, (x, y))

        # Title with glow effect
        glow_offset = int(3 * math.sin(self.tick * 0.04))
        title_text = "TAG"
        # Glow layer
        glow = self.title_font.render(title_text, True, (255, 180, 50, 80))
        self.screen.blit(glow, glow.get_rect(centerx=sw // 2 + 2,
                                              y=72 + glow_offset + 2))
        # Main title
        title = self.title_font.render(title_text, True, config.COLOR_MENU_TITLE)
        self.screen.blit(title, title.get_rect(centerx=sw // 2,
                                                y=72 + glow_offset))

        # Subtitle
        subtitle = self.subtitle_font.render(
            "Reinforcement Learning Laboratory", True, (160, 160, 190))
        self.screen.blit(subtitle, subtitle.get_rect(centerx=sw // 2, y=145))

        # Decorative line
        line_w = 300
        line_y = 185
        pygame.draw.line(self.screen, (60, 60, 90),
                         (sw // 2 - line_w // 2, line_y),
                         (sw // 2 + line_w // 2, line_y), 1)
        # Center diamond
        pygame.draw.polygon(self.screen, (100, 100, 150),
                            [(sw // 2, line_y - 4), (sw // 2 + 4, line_y),
                             (sw // 2, line_y + 4), (sw // 2 - 4, line_y)])

        # Section labels
        mode_label = self.label_font.render("Select Mode", True, (180, 180, 200))
        self.screen.blit(mode_label, mode_label.get_rect(centerx=sw // 2, y=240))

        algo_label = self.label_font.render("Select Algorithm", True, (180, 180, 200))
        self.screen.blit(algo_label, algo_label.get_rect(centerx=sw // 2, y=370))

        # Buttons
        for btn in self.mode_buttons:
            btn.draw(self.screen)
        for btn in self.algo_buttons:
            btn.draw(self.screen)
        self.start_button.draw(self.screen)

        # Mode descriptions
        if self.selected_mode == config.GameMode.PLAYER_MODE:
            desc = "You control one character. AI agents use the selected algorithm."
        else:
            desc = "All characters are AI agents. Watch them learn and play."
        desc_text = self.hint_font.render(desc, True, (130, 130, 160))
        self.screen.blit(desc_text, desc_text.get_rect(centerx=sw // 2, y=330))

        # Bottom instructions
        hints = [
            "Player Mode: WASD/Arrows to move | Walk near agents to tag them",
            "Push crates by walking into them | ESC to return to menu",
        ]
        for i, hint in enumerate(hints):
            h = self.hint_font.render(hint, True, (110, 110, 140))
            self.screen.blit(h, h.get_rect(centerx=sw // 2, y=640 + i * 22))

        # Version
        ver = self.hint_font.render("v1.0 - Framework Build", True, (70, 70, 90))
        self.screen.blit(ver, (sw - ver.get_width() - 10, sh - 24))
