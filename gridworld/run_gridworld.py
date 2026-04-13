#!/usr/bin/env python3
"""Train and visualize tabular RL agents in a simple tag gridworld.

Training pipeline (Option A — train one at a time):
  Phase 1: Train TAGGER vs random runner  (learns to chase)
  Phase 2: Train RUNNER vs frozen trained tagger  (learns to evade)
  Phase 3: Watch trained tagger vs trained runner  (live visualization)

Usage:
    # Full pipeline: train + visualize (default: Q-Learning)
    python -m gridworld.run_gridworld

    # Use SARSA instead
    python -m gridworld.run_gridworld --algo sarsa

    # More training episodes
    python -m gridworld.run_gridworld --episodes 20000

    # Skip training, just visualize saved models
    python -m gridworld.run_gridworld --watch-only

    # Train only, no visualization
    python -m gridworld.run_gridworld --train-only
"""

import argparse
import math
import os
import sys

import pygame

from gridworld.env import TagGridWorld, GRID_SIZE, NUM_ACTIONS
from gridworld.tabular_agent import QLearningAgent, SARSAAgent, RandomAgent
from gridworld.recorder import VideoRecorder
from rendering.textures import generate_floor_texture, generate_wall_texture
from rendering.sprites import generate_character_sprite, generate_tagger_aura_frame
from rendering.particles import ParticleSystem


# ======================================================================
# Configuration
# ======================================================================
CELL_SIZE = 60          # pixels per grid cell
PANEL_WIDTH = 320       # side panel for stats
FPS_WATCH = 6           # slow enough to follow during final demo
SAVE_DIR = "gridworld/saved_models"

# Colors (panel / text)
BG          = (25, 25, 35)
GRID_LINE   = (55, 58, 68)
TEXT_CLR    = (210, 210, 220)
DIM_CLR     = (120, 120, 140)
PANEL_BG    = (30, 30, 42)
ACCENT      = (255, 200, 60)
SUCCESS_CLR = (80, 200, 120)
FAIL_CLR    = (200, 80, 80)


# ======================================================================
# Texture / Sprite cache (built once on startup)
# ======================================================================
class GridWorldRenderer:
    """Handles all drawing using the main game's texture and sprite systems."""

    def __init__(self):
        # Floor tile (scaled to CELL_SIZE)
        raw_floor = generate_floor_texture(32)
        self.floor_tile = pygame.transform.scale(raw_floor, (CELL_SIZE, CELL_SIZE))

        # Wall border tile (for grid edge)
        raw_wall = generate_wall_texture(32)
        self.wall_tile = pygame.transform.scale(raw_wall, (CELL_SIZE, CELL_SIZE))

        # Character sprites (radius ~20 for CELL_SIZE 60)
        sprite_radius = CELL_SIZE // 3
        self.tagger_sprite = generate_character_sprite(
            sprite_radius, (200, 55, 55), is_tagger=True)
        self.runner_sprite = generate_character_sprite(
            sprite_radius, (55, 140, 210), is_player=False)

        # Particles
        self.particles = ParticleSystem()

        # Fonts
        self.font_title = pygame.font.SysFont("Arial", 20, bold=True)
        self.font = pygame.font.SysFont("Arial", 16)
        self.font_small = pygame.font.SysFont("Arial", 13)
        self.font_label = pygame.font.SysFont("Arial", 12, bold=True)

        # Pre-render shadow ellipse
        shadow_w = int(CELL_SIZE * 0.6)
        shadow_h = int(CELL_SIZE * 0.25)
        self.shadow = pygame.Surface((shadow_w, shadow_h), pygame.SRCALPHA)
        pygame.draw.ellipse(self.shadow, (0, 0, 0, 50),
                            (0, 0, shadow_w, shadow_h))

        self.tick = 0

    def draw_grid(self, screen, env: TagGridWorld, ox: int, oy: int):
        """Draw floor tiles, characters, particles."""
        gs = env.grid_size
        self.tick += 1

        # Floor tiles
        for y in range(gs):
            for x in range(gs):
                screen.blit(self.floor_tile, (ox + x * CELL_SIZE,
                                              oy + y * CELL_SIZE))
                # Subtle grid line
                rect = pygame.Rect(ox + x * CELL_SIZE, oy + y * CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, GRID_LINE, rect, 1)

        # Border (wall tiles around the edge)
        border_size = 6
        border_rect = pygame.Rect(ox - border_size, oy - border_size,
                                  gs * CELL_SIZE + border_size * 2,
                                  gs * CELL_SIZE + border_size * 2)
        pygame.draw.rect(screen, (85, 85, 105), border_rect, border_size,
                         border_radius=3)

        # Runner (drawn first)
        rx, ry = env.runner_pos
        rcx = ox + rx * CELL_SIZE + CELL_SIZE // 2
        rcy = oy + ry * CELL_SIZE + CELL_SIZE // 2

        # Shadow
        sr = self.shadow.get_rect(center=(rcx + 2, rcy + CELL_SIZE // 4))
        screen.blit(self.shadow, sr)

        # Sprite
        rr = self.runner_sprite.get_rect(center=(rcx, rcy))
        screen.blit(self.runner_sprite, rr)

        # Label
        label = self.font_label.render("Runner", True, (180, 200, 230))
        lr = label.get_rect(centerx=rcx, bottom=rcy - CELL_SIZE // 3 - 2)
        bg = pygame.Surface(lr.inflate(6, 2).size, pygame.SRCALPHA)
        bg.fill((0, 0, 0, 120))
        screen.blit(bg, lr.inflate(6, 2))
        screen.blit(label, lr)

        # Tagger
        tx, ty = env.tagger_pos
        tcx = ox + tx * CELL_SIZE + CELL_SIZE // 2
        tcy = oy + ty * CELL_SIZE + CELL_SIZE // 2

        # Shadow
        st = self.shadow.get_rect(center=(tcx + 2, tcy + CELL_SIZE // 4))
        screen.blit(self.shadow, st)

        # Aura
        aura = generate_tagger_aura_frame(CELL_SIZE // 3, self.tick)
        ar = aura.get_rect(center=(tcx, tcy))
        screen.blit(aura, ar)

        # Sprite
        tr = self.tagger_sprite.get_rect(center=(tcx, tcy))
        screen.blit(self.tagger_sprite, tr)

        # "IT!" badge
        badge = self.font_label.render("IT!", True, (255, 255, 200))
        br = badge.get_rect(centerx=tcx, top=tcy + CELL_SIZE // 3 + 2)
        badge_bg = br.inflate(8, 4)
        pygame.draw.rect(screen, (180, 40, 40), badge_bg, border_radius=4)
        pygame.draw.rect(screen, (255, 80, 80), badge_bg, 1, border_radius=4)
        screen.blit(badge, br)

        # Catch flash + particles
        if env.tagger_pos == env.runner_pos:
            flash = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            flash.fill((255, 255, 100, 80))
            screen.blit(flash, (ox + tx * CELL_SIZE, oy + ty * CELL_SIZE))

        # Particles
        self.particles.update()
        self.particles.draw(screen, -ox, -oy)  # offset particles to grid space

    def emit_catch_particles(self, env, ox, oy):
        """Burst particles at the catch location."""
        tx, ty = env.tagger_pos
        px = ox + tx * CELL_SIZE + CELL_SIZE // 2
        py = oy + ty * CELL_SIZE + CELL_SIZE // 2
        self.particles.emit_tag_burst(px, py)

    def draw_panel(self, screen, panel_x: int, info: dict):
        """Draw the stats panel on the right side."""
        sh = screen.get_height()

        # Panel background
        panel_rect = pygame.Rect(panel_x, 0, PANEL_WIDTH, sh)
        pygame.draw.rect(screen, PANEL_BG, panel_rect)
        pygame.draw.line(screen, (50, 50, 70), (panel_x, 0),
                         (panel_x, sh), 2)

        x = panel_x + 15
        y = 15

        # Title
        title = self.font_title.render(info.get("title", ""), True, ACCENT)
        screen.blit(title, (x, y))
        y += 35

        # Phase
        phase = self.font.render(info.get("phase", ""), True, TEXT_CLR)
        screen.blit(phase, (x, y))
        y += 28

        # Algorithm
        algo = self.font.render(f"Algorithm: {info.get('algo', '')}", True,
                                DIM_CLR)
        screen.blit(algo, (x, y))
        y += 35

        # Separator
        pygame.draw.line(screen, (50, 50, 70), (x, y),
                         (panel_x + PANEL_WIDTH - 15, y))
        y += 15

        # Stats
        for label, value, color in info.get("stats", []):
            lbl = self.font.render(f"{label}:", True, DIM_CLR)
            val = self.font.render(str(value), True, color)
            screen.blit(lbl, (x, y))
            screen.blit(val, (x + 160, y))
            y += 24

        # Separator
        y += 10
        pygame.draw.line(screen, (50, 50, 70), (x, y),
                         (panel_x + PANEL_WIDTH - 15, y))
        y += 15

        # Last episode result
        result = info.get("last_result", "")
        if "Caught" in result:
            rc = SUCCESS_CLR
        elif "Timeout" in result:
            rc = FAIL_CLR
        else:
            rc = TEXT_CLR
        res = self.font.render(result, True, rc)
        screen.blit(res, (x, y))

        # Instructions at bottom
        y = sh - 60
        hint1 = self.font_small.render("SPACE: pause/resume", True, DIM_CLR)
        hint2 = self.font_small.render("ESC: skip to next phase", True, DIM_CLR)
        screen.blit(hint1, (x, y))
        screen.blit(hint2, (x, y + 18))


# ======================================================================
# Training with live visualization
# ======================================================================
def train_phase(screen, clock, renderer: GridWorldRenderer, env, learner,
                opponent, role: str, algo_name: str, num_episodes: int,
                phase_label: str, recorder: VideoRecorder | None = None):
    """Train one role against a fixed opponent with live visualization."""
    grid_w = GRID_SIZE * CELL_SIZE
    offset_x = 15
    offset_y = (screen.get_height() - GRID_SIZE * CELL_SIZE) // 2
    panel_x = offset_x + grid_w + 25

    catches = 0
    timeouts = 0
    total_reward = 0.0
    last_result = ""
    paused = False

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        ep_reward = 0.0

        while not env.done:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return catches, timeouts
                    if event.key == pygame.K_SPACE:
                        paused = not paused

            if paused:
                clock.tick(30)
                continue

            # --- Tagger turn ---
            if role == "tagger":
                action = learner.select_action(state)
            else:
                action = opponent.select_action(state)

            next_state, tagger_reward, done = env.step_tagger(action)

            if role == "tagger":
                learner.learn(state, action, tagger_reward, next_state, done)
                ep_reward += tagger_reward

            state = next_state

            if done:
                if env.tagger_pos == env.runner_pos:
                    renderer.emit_catch_particles(env, offset_x, offset_y)
                break

            # --- Runner turn ---
            if role == "runner":
                action_r = learner.select_action(state)
            else:
                action_r = opponent.select_action(state)

            next_state, runner_reward, done = env.step_runner(action_r)

            if role == "runner":
                if isinstance(learner, SARSAAgent):
                    next_a = learner.select_action(next_state) if not done else 0
                    learner.learn(state, action_r, runner_reward, next_state,
                                  done, next_action=next_a)
                else:
                    learner.learn(state, action_r, runner_reward, next_state,
                                  done)
                ep_reward += runner_reward

            state = next_state

            if done and env.tagger_pos == env.runner_pos:
                renderer.emit_catch_particles(env, offset_x, offset_y)

            # --- Draw ---
            screen.fill(BG)
            renderer.draw_grid(screen, env, offset_x, offset_y)

            catch_rate = (catches / max(ep - 1, 1)) * 100
            eps_val = getattr(learner, "epsilon", 0)
            q_size = len(getattr(learner, "q_table", {}))

            renderer.draw_panel(screen, panel_x, {
                "title": "Tag Gridworld",
                "phase": phase_label,
                "algo": algo_name,
                "stats": [
                    ("Episode", f"{ep}/{num_episodes}", TEXT_CLR),
                    ("Step", f"{env.steps}/{env.max_steps}", TEXT_CLR),
                    ("Catches", f"{catches}", SUCCESS_CLR),
                    ("Timeouts", f"{timeouts}", FAIL_CLR),
                    ("Catch rate", f"{catch_rate:.1f}%", ACCENT),
                    ("Epsilon", f"{eps_val:.4f}", DIM_CLR),
                    ("Q-table size", f"{q_size:,}", DIM_CLR),
                    ("Total reward", f"{total_reward:.1f}", TEXT_CLR),
                    ("Distance", f"{env.manhattan_distance()}", ACCENT),
                ],
                "last_result": last_result,
            })
            pygame.display.flip()
            if recorder:
                recorder.capture(screen)

            if ep < num_episodes * 0.8:
                clock.tick(0)
            else:
                clock.tick(30)

        # Episode finished
        if env.tagger_pos == env.runner_pos:
            catches += 1
            last_result = f"Caught! (step {env.steps})"
        else:
            timeouts += 1
            last_result = f"Timeout ({env.max_steps} steps)"

        total_reward += ep_reward
        learner.decay_epsilon()

    return catches, timeouts


# ======================================================================
# Watch mode — trained tagger vs trained runner, slow
# ======================================================================
def watch_phase(screen, clock, renderer: GridWorldRenderer, env,
                tagger_agent, runner_agent, algo_name: str,
                num_episodes: int = 20, recorder: VideoRecorder | None = None):
    """Watch trained agents play at slow speed."""
    offset_x = 15
    offset_y = (screen.get_height() - GRID_SIZE * CELL_SIZE) // 2
    panel_x = offset_x + GRID_SIZE * CELL_SIZE + 25

    catches = 0
    last_result = ""
    paused = False

    for ep in range(1, num_episodes + 1):
        state = env.reset()

        while not env.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_SPACE:
                        paused = not paused

            if paused:
                clock.tick(30)
                continue

            # Tagger move
            t_action = tagger_agent.select_action(state)
            state, _, done = env.step_tagger(t_action)

            if done and env.tagger_pos == env.runner_pos:
                renderer.emit_catch_particles(env, offset_x, offset_y)

            # Draw
            screen.fill(BG)
            renderer.draw_grid(screen, env, offset_x, offset_y)
            renderer.draw_panel(screen, panel_x, {
                "title": "Tag Gridworld",
                "phase": "WATCH: Trained vs Trained",
                "algo": algo_name,
                "stats": [
                    ("Episode", f"{ep}/{num_episodes}", TEXT_CLR),
                    ("Step", f"{env.steps}/{env.max_steps}", TEXT_CLR),
                    ("Catches", f"{catches}", SUCCESS_CLR),
                    ("Distance", f"{env.manhattan_distance()}", ACCENT),
                ],
                "last_result": last_result,
            })
            pygame.display.flip()
            if recorder:
                recorder.capture(screen)
            clock.tick(FPS_WATCH)

            if done:
                break

            # Runner move
            r_action = runner_agent.select_action(state)
            state, _, done = env.step_runner(r_action)

            if done and env.tagger_pos == env.runner_pos:
                renderer.emit_catch_particles(env, offset_x, offset_y)

            # Draw
            screen.fill(BG)
            renderer.draw_grid(screen, env, offset_x, offset_y)
            renderer.draw_panel(screen, panel_x, {
                "title": "Tag Gridworld",
                "phase": "WATCH: Trained vs Trained",
                "algo": algo_name,
                "stats": [
                    ("Episode", f"{ep}/{num_episodes}", TEXT_CLR),
                    ("Step", f"{env.steps}/{env.max_steps}", TEXT_CLR),
                    ("Catches", f"{catches}", SUCCESS_CLR),
                    ("Distance", f"{env.manhattan_distance()}", ACCENT),
                ],
                "last_result": last_result,
            })
            pygame.display.flip()
            if recorder:
                recorder.capture(screen)
            clock.tick(FPS_WATCH)

        # Episode result
        if env.tagger_pos == env.runner_pos:
            catches += 1
            last_result = f"Caught! (step {env.steps})"
        else:
            last_result = f"Timeout ({env.max_steps} steps)"

        pygame.time.wait(500)


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train & visualize tabular RL in tag gridworld")
    parser.add_argument("--algo", choices=["qlearning", "sarsa"],
                        default="qlearning", help="Algorithm (default: qlearning)")
    parser.add_argument("--episodes", type=int, default=5000,
                        help="Training episodes per phase (default: 5000)")
    parser.add_argument("--watch-only", action="store_true",
                        help="Skip training, load saved models and watch")
    parser.add_argument("--train-only", action="store_true",
                        help="Train only, no watch phase")
    parser.add_argument("--watch-episodes", type=int, default=20,
                        help="Episodes to watch (default: 20)")
    parser.add_argument("--record", action="store_true",
                        help="Record training & watch phases to MP4 videos")
    args = parser.parse_args()

    # Init pygame
    pygame.init()
    grid_px = GRID_SIZE * CELL_SIZE + 40
    screen_w = grid_px + PANEL_WIDTH
    screen_h = GRID_SIZE * CELL_SIZE + 30
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption(f"Tag Gridworld — {args.algo.upper()}")
    clock = pygame.time.Clock()

    algo_name = args.algo.upper()
    if algo_name == "QLEARNING":
        algo_name = "Q-Learning"

    renderer = GridWorldRenderer()

    def make_agent():
        if args.algo == "sarsa":
            return SARSAAgent()
        return QLearningAgent()

    tagger_path = os.path.join(SAVE_DIR, f"{args.algo}_tagger.pkl")
    runner_path = os.path.join(SAVE_DIR, f"{args.algo}_runner.pkl")

    env = TagGridWorld()
    video_dir = os.path.join("gridworld", "videos")

    if not args.watch_only:
        # ---- Phase 1: Train tagger vs random runner ----
        print(f"\n{'='*50}")
        print(f"  Phase 1: Training TAGGER ({algo_name})")
        print(f"  Opponent: random runner")
        print(f"  Episodes: {args.episodes}")
        if args.record:
            print(f"  Recording to: {video_dir}/")
        print(f"{'='*50}")

        tagger_agent = make_agent()
        random_runner = RandomAgent()

        rec1 = None
        if args.record:
            rec1 = VideoRecorder(
                os.path.join(video_dir, f"{args.algo}_phase1_tagger_training.mp4"),
                screen, fps=30, sample_every=5)

        catches, timeouts = train_phase(
            screen, clock, renderer, env, tagger_agent, random_runner,
            role="tagger", algo_name=algo_name,
            num_episodes=args.episodes,
            phase_label="Phase 1: Train TAGGER vs random",
            recorder=rec1,
        )
        if rec1:
            rec1.finish()

        tagger_agent.save(tagger_path)
        total = catches + timeouts
        print(f"  Tagger trained: {catches} catches, {timeouts} timeouts "
              f"({catches / max(total, 1) * 100:.1f}% catch rate)")
        print(f"  Q-table size: {len(tagger_agent.q_table):,}")
        print(f"  Saved: {tagger_path}")

        # ---- Phase 2: Train runner vs trained tagger ----
        print(f"\n{'='*50}")
        print(f"  Phase 2: Training RUNNER ({algo_name})")
        print(f"  Opponent: trained tagger (frozen)")
        print(f"  Episodes: {args.episodes}")
        print(f"{'='*50}")

        runner_agent = make_agent()
        tagger_agent.epsilon = 0.0

        rec2 = None
        if args.record:
            rec2 = VideoRecorder(
                os.path.join(video_dir, f"{args.algo}_phase2_runner_training.mp4"),
                screen, fps=30, sample_every=5)

        catches, timeouts = train_phase(
            screen, clock, renderer, env, runner_agent, tagger_agent,
            role="runner", algo_name=algo_name,
            num_episodes=args.episodes,
            phase_label="Phase 2: Train RUNNER vs trained tagger",
            recorder=rec2,
        )
        if rec2:
            rec2.finish()

        runner_agent.save(runner_path)
        total = catches + timeouts
        survival_rate = timeouts / max(total, 1) * 100
        print(f"  Runner trained: {catches} catches, {timeouts} survivals "
              f"({survival_rate:.1f}% survival rate)")
        print(f"  Q-table size: {len(runner_agent.q_table):,}")
        print(f"  Saved: {runner_path}")
    else:
        tagger_agent = make_agent()
        runner_agent = make_agent()
        tagger_agent.load(tagger_path)
        runner_agent.load(runner_path)
        tagger_agent.epsilon = 0.0
        runner_agent.epsilon = 0.0
        print(f"  Loaded tagger: {tagger_path}")
        print(f"  Loaded runner: {runner_path}")

    # ---- Phase 3: Watch trained vs trained ----
    if not args.train_only:
        print(f"\n{'='*50}")
        print(f"  Phase 3: WATCH — trained tagger vs trained runner")
        print(f"  Episodes: {args.watch_episodes}")
        print(f"{'='*50}")

        tagger_agent.epsilon = 0.0
        runner_agent.epsilon = 0.0

        rec3 = None
        if args.record:
            rec3 = VideoRecorder(
                os.path.join(video_dir, f"{args.algo}_phase3_watch.mp4"),
                screen, fps=FPS_WATCH)

        watch_phase(screen, clock, renderer, env, tagger_agent, runner_agent,
                    algo_name, args.watch_episodes, recorder=rec3)

        if rec3:
            rec3.finish()

    pygame.quit()
    print("\nDone.")


if __name__ == "__main__":
    main()
