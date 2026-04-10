"""Main menu screen with polished visuals."""

import os
import pygame
import math
import config
from ui.button import Button


def _model_exists(algo_name: str) -> bool:
    """Check if trained model files exist for the given algorithm."""
    safe_name = algo_name.lower().replace("-", "_").replace(" ", "_")
    base_path = os.path.join(config.DEFAULT_MODEL_DIR, f"{safe_name}_model.pt")

    if config.DUAL_ROLE_ENABLED:
        # Check for dual-role model files (_tagger + _runner)
        from rl.dual_role import dual_model_exists
        return dual_model_exists(base_path)

    return os.path.exists(base_path)


class Menu:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.selected_mode = config.GameMode.PLAYER_MODE
        self.selected_algorithm = "PPO"
        self.selected_train_mode = config.TrainMode.TRAIN_LIVE
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
        btn_h = 44

        # --- Row 1: Game Mode ---
        mode_y = 230
        btn_w = 200
        gap = 20

        self.mode_buttons = []
        for i, mode in enumerate(config.GameMode):
            x = cx - btn_w - gap // 2 + i * (btn_w + gap)
            btn = Button(x, mode_y, btn_w, btn_h, mode.value)
            btn._mode = mode
            if mode == self.selected_mode:
                btn.selected = True
            self.mode_buttons.append(btn)

        # --- Row 2: Algorithm ---
        algo_y = 330
        algo_names = list(config.RL_ALGORITHMS.keys())
        abtn_w = 140
        total_w = len(algo_names) * (abtn_w + 12) - 12
        start_x = cx - total_w // 2

        self.algo_buttons = []
        for i, name in enumerate(algo_names):
            x = start_x + i * (abtn_w + 12)
            btn = Button(x, algo_y, abtn_w, btn_h, name)
            btn._algo_name = name
            if name == self.selected_algorithm:
                btn.selected = True
            self.algo_buttons.append(btn)

        # --- Row 3: Train Mode ---
        train_y = 430
        tbtn_w = 200

        self.train_buttons = []
        for i, tm in enumerate(config.TrainMode):
            x = cx - tbtn_w - gap // 2 + i * (tbtn_w + gap)
            btn = Button(x, train_y, tbtn_w, btn_h, tm.value)
            btn._train_mode = tm
            if tm == self.selected_train_mode:
                btn.selected = True
            self.train_buttons.append(btn)

        # --- Start button ---
        self.start_button = Button(cx - 110, 540, 220, 55, "START GAME")

    def _update_train_button_state(self):
        """Disable 'Use Trained' if no model exists for selected algorithm."""
        has_model = _model_exists(self.selected_algorithm)
        for btn in self.train_buttons:
            if btn._train_mode == config.TrainMode.USE_TRAINED:
                btn.disabled = not has_model

    def handle_event(self, event: pygame.event.Event) -> dict | None:
        """Process events. Returns config dict when START is clicked."""
        # Mode buttons
        for btn in self.mode_buttons:
            if btn.handle_event(event):
                self.selected_mode = btn._mode
                for b in self.mode_buttons:
                    b.selected = (b is btn)

        # Algorithm buttons
        for btn in self.algo_buttons:
            if btn.handle_event(event):
                self.selected_algorithm = btn._algo_name
                for b in self.algo_buttons:
                    b.selected = (b is btn)
                self._update_train_button_state()
                # If selected algo has no model, force Train Live
                if not _model_exists(self.selected_algorithm):
                    self.selected_train_mode = config.TrainMode.TRAIN_LIVE
                    for b in self.train_buttons:
                        b.selected = (b._train_mode == config.TrainMode.TRAIN_LIVE)

        # Train mode buttons
        for btn in self.train_buttons:
            if getattr(btn, "disabled", False):
                continue
            if btn.handle_event(event):
                self.selected_train_mode = btn._train_mode
                for b in self.train_buttons:
                    b.selected = (b is btn)

        # Start
        if self.start_button.handle_event(event):
            return {
                "mode": self.selected_mode,
                "algorithm": self.selected_algorithm,
                "train_mode": self.selected_train_mode,
            }
        return None

    def draw(self):
        """Render the menu."""
        self.tick += 1
        sw, sh = self.screen.get_size()
        self._update_train_button_state()

        # Background
        self.screen.fill(config.COLOR_MENU_BG)

        # Floating dots
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

        # Title
        glow_offset = int(3 * math.sin(self.tick * 0.04))
        glow = self.title_font.render("TAG", True, (255, 180, 50, 80))
        self.screen.blit(glow, glow.get_rect(centerx=sw // 2 + 2,
                                              y=52 + glow_offset + 2))
        title = self.title_font.render("TAG", True, config.COLOR_MENU_TITLE)
        self.screen.blit(title, title.get_rect(centerx=sw // 2,
                                                y=52 + glow_offset))

        subtitle = self.subtitle_font.render(
            "Reinforcement Learning Laboratory", True, (160, 160, 190))
        self.screen.blit(subtitle, subtitle.get_rect(centerx=sw // 2, y=125))

        # Decorative line
        line_w = 300
        line_y = 160
        pygame.draw.line(self.screen, (60, 60, 90),
                         (sw // 2 - line_w // 2, line_y),
                         (sw // 2 + line_w // 2, line_y), 1)
        pygame.draw.polygon(self.screen, (100, 100, 150),
                            [(sw // 2, line_y - 4), (sw // 2 + 4, line_y),
                             (sw // 2, line_y + 4), (sw // 2 - 4, line_y)])

        # --- Section: Game Mode ---
        lbl = self.label_font.render("Game Mode", True, (180, 180, 200))
        self.screen.blit(lbl, lbl.get_rect(centerx=sw // 2, y=200))

        for btn in self.mode_buttons:
            btn.draw(self.screen)

        # Mode description
        if self.selected_mode == config.GameMode.PLAYER_MODE:
            desc = "You control one character. AI agents use the selected algorithm."
        else:
            desc = "All characters are AI agents. Watch them play."
        dt = self.hint_font.render(desc, True, (130, 130, 160))
        self.screen.blit(dt, dt.get_rect(centerx=sw // 2, y=282))

        # --- Section: Algorithm ---
        lbl = self.label_font.render("Algorithm", True, (180, 180, 200))
        self.screen.blit(lbl, lbl.get_rect(centerx=sw // 2, y=302))

        for btn in self.algo_buttons:
            btn.draw(self.screen)

        # --- Section: Train Mode ---
        lbl = self.label_font.render("Agent Behavior", True, (180, 180, 200))
        self.screen.blit(lbl, lbl.get_rect(centerx=sw // 2, y=400))

        for btn in self.train_buttons:
            btn.draw(self.screen)

        # Train mode description
        has_model = _model_exists(self.selected_algorithm)
        if self.selected_train_mode == config.TrainMode.TRAIN_LIVE:
            td = "Agents learn in real-time as the game runs."
        else:
            td = "Agents use a pre-trained model (no learning during play)."
        tt = self.hint_font.render(td, True, (130, 130, 160))
        self.screen.blit(tt, tt.get_rect(centerx=sw // 2, y=482))

        if not has_model:
            warn = self.hint_font.render(
                f"No trained model found for {self.selected_algorithm}. "
                f"Use: python train.py -a {self.selected_algorithm}",
                True, (200, 150, 80))
            self.screen.blit(warn, warn.get_rect(centerx=sw // 2, y=502))

        # Start button
        self.start_button.draw(self.screen)

        # Bottom hints
        hints = [
            "WASD/Arrows: Move | Walk near to tag | Push crates | ESC: Menu",
            "Train models offline:  python train.py --algorithm PPO --rounds 200",
        ]
        for i, hint in enumerate(hints):
            h = self.hint_font.render(hint, True, (110, 110, 140))
            self.screen.blit(h, h.get_rect(centerx=sw // 2, y=660 + i * 22))

        ver = self.hint_font.render("v1.1", True, (70, 70, 90))
        self.screen.blit(ver, (sw - ver.get_width() - 10, sh - 24))
