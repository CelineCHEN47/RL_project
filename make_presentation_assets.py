#!/usr/bin/env python3
"""Generate all diagrams, tables, and visuals for the project presentation.

Outputs PNG files into presentation_assets/ — ready to be screenshotted
or dropped directly into slides.

Usage:
    python make_presentation_assets.py
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle


def stroked(text_obj, stroke_color="#1c1f2a", stroke_width=3):
    """Add a dark outline around text so it stays readable on any background."""
    text_obj.set_path_effects([
        pe.Stroke(linewidth=stroke_width, foreground=stroke_color),
        pe.Normal(),
    ])
    return text_obj


# ==========================================================================
# Style
# ==========================================================================
BG          = "#1c1f2a"
PANEL       = "#2a2e3d"
PANEL_LIGHT = "#3a3f52"
TEXT        = "#f2f4fa"
TEXT_DIM    = "#b4bacc"
ACCENT      = "#ffc94a"

# Foreground colors (for text/lines on dark backgrounds — bright)
TAGGER      = "#ff5e5e"
RUNNER      = "#5fb8ff"
SUCCESS     = "#6dd88e"
WARN        = "#ffae5e"

# Dark variants (for box backgrounds — high contrast against white text)
TAGGER_BG   = "#7a1f1f"
RUNNER_BG   = "#1a4a78"
SUCCESS_BG  = "#1f5a30"
EDGE        = "#5a607a"

OUT_DIR = "presentation_assets"


def setup_axes(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def box(ax, x, y, w, h, text, color=PANEL, edge=EDGE, text_color=TEXT,
        fontsize=11, bold=True):
    rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          linewidth=1.5, edgecolor=edge, facecolor=color)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", color=text_color,
            fontsize=fontsize, fontweight=weight)


def arrow(ax, x1, y1, x2, y2, color=EDGE, width=1.5):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=15, linewidth=width,
                        color=color, zorder=2)
    ax.add_patch(a)


# ==========================================================================
# Slide 3: Architecture diagram
# ==========================================================================
def make_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    setup_axes(fig, ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    # Top: main.py
    box(ax, 6, 7.2, 2.0, 0.6, "main.py", color=ACCENT, text_color="#1c1f2a")

    # GameManager (the orchestrator)
    box(ax, 6, 6.0, 2.6, 0.7, "GameManager", color=PANEL_LIGHT, fontsize=12)
    arrow(ax, 6, 6.9, 6, 6.4)

    # Two children: Menu / Game Loop
    box(ax, 3.5, 4.7, 2.2, 0.6, "Menu", color=PANEL)
    box(ax, 8.5, 4.7, 2.2, 0.6, "Game Loop", color=PANEL)
    arrow(ax, 5.4, 5.7, 4.2, 5.0)
    arrow(ax, 6.6, 5.7, 7.8, 5.0)

    # Game Loop fans into 4 systems
    box(ax, 1.2, 3.2, 2.0, 0.6, "Renderer", color=PANEL)
    box(ax, 4.2, 3.2, 2.0, 0.6, "Physics", color=PANEL)
    box(ax, 7.2, 3.2, 2.0, 0.6, "Entities", color=PANEL)
    box(ax, 10.4, 3.2, 2.0, 0.6, "Level", color=PANEL)

    for sx in [1.2, 4.2, 7.2, 10.4]:
        arrow(ax, 8.5, 4.4, sx, 3.5)

    # Renderer feeds
    box(ax, 1.2, 1.8, 2.4, 0.55, "Textures · Sprites\nParticles",
        color=PANEL, fontsize=9, bold=False)
    arrow(ax, 1.2, 2.9, 1.2, 2.1)

    # RL System (right side, separate column)
    box(ax, 7.2, 1.8, 2.0, 0.55, "RL Environment", color=PANEL)
    arrow(ax, 7.2, 2.9, 7.2, 2.1)

    box(ax, 10.4, 1.8, 2.4, 0.55, "Algorithms\n(PPO · DQN · QL · SARSA)",
        color="#3a4a5a", text_color=TEXT, fontsize=9, bold=False)
    arrow(ax, 8.4, 1.8, 9.2, 1.8)

    ax.text(6, 0.4, "Modular architecture — every layer in its own file",
            ha="center", color=TEXT_DIM, fontsize=10, style="italic")

    plt.savefig(f"{OUT_DIR}/03_architecture.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓ 03_architecture.png")


# ==========================================================================
# Slide 4: Game options tree (modes, algorithms, behavior)
# ==========================================================================
def make_game_options():
    # Taller canvas → more vertical room between rows
    fig, ax = plt.subplots(figsize=(14, 10))
    setup_axes(fig, ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)

    stroked(ax.text(7, 9.5, "Game Configuration Options",
            ha="center", color=ACCENT, fontsize=18, fontweight="bold"))

    # Root
    box(ax, 7, 8.6, 2.8, 0.7, "MAIN MENU", color=ACCENT,
        text_color="#1c1f2a", fontsize=13)

    # === Layer 1: Game Mode ===
    stroked(ax.text(0.5, 7.5, "Game Mode", color=ACCENT, fontsize=12,
                    fontweight="bold"))

    box(ax, 4.5, 7.2, 2.8, 0.7, "Player Mode",
        color="#1f3a55", edge="#5a8ac0", text_color="white", fontsize=12)
    box(ax, 9.5, 7.2, 2.8, 0.7, "Simulation Mode",
        color="#1f3a55", edge="#5a8ac0", text_color="white", fontsize=12)
    arrow(ax, 6.3, 8.25, 4.8, 7.6)
    arrow(ax, 7.7, 8.25, 9.2, 7.6)

    # Mode captions (well below the boxes, no overlap)
    stroked(ax.text(4.5, 6.45, "You + AI agents",
            ha="center", color=TEXT_DIM, fontsize=10, style="italic"))
    stroked(ax.text(9.5, 6.45, "All AI, you watch",
            ha="center", color=TEXT_DIM, fontsize=10, style="italic"))

    # === Layer 2: Algorithm ===
    stroked(ax.text(0.5, 5.4, "Algorithm", color=ACCENT, fontsize=12,
                    fontweight="bold"))

    algo_data = [
        (2.0, "Q-Learning", "tabular · off-policy"),
        (5.4, "SARSA",      "tabular · on-policy"),
        (8.6, "PPO",        "deep RL · policy"),
        (12.0, "DQN",       "deep RL · value"),
    ]
    for x, name, desc in algo_data:
        box(ax, x, 5.1, 2.4, 0.75, name,
            color="#3a2a5a", edge="#9477c4",
            text_color="white", fontsize=12)
        stroked(ax.text(x, 4.4, desc, ha="center", color=TEXT_DIM,
                fontsize=9, style="italic"))
        # Arrows from both modes to each algo
        arrow(ax, 4.5, 6.85, x, 5.5, color=EDGE, width=0.7)
        arrow(ax, 9.5, 6.85, x, 5.5, color=EDGE, width=0.7)

    # === Layer 3: Agent Behavior ===
    stroked(ax.text(0.5, 2.9, "Agent Behavior", color=ACCENT, fontsize=12,
                    fontweight="bold"))

    box(ax, 4.5, 2.6, 3.0, 0.75, "Train Live",
        color=SUCCESS_BG, edge=SUCCESS, text_color="white", fontsize=12)
    box(ax, 9.5, 2.6, 3.0, 0.75, "Use Trained",
        color=RUNNER_BG, edge=RUNNER, text_color="white", fontsize=12)

    # Arrows from algorithms to behavior choices
    for x, _, _ in algo_data:
        arrow(ax, x, 4.7, 4.5, 3.0, color=EDGE, width=0.5)
        arrow(ax, x, 4.7, 9.5, 3.0, color=EDGE, width=0.5)

    stroked(ax.text(4.5, 1.85, "Agents learn while playing",
            ha="center", color=TEXT_DIM, fontsize=10, style="italic"))
    stroked(ax.text(9.5, 1.85, "Pre-trained model · no learning",
            ha="center", color=TEXT_DIM, fontsize=10, style="italic"))

    # Bottom note
    stroked(ax.text(7, 0.7,
            "2 modes  ×  4 algorithms  ×  2 behaviors  =  16 distinct setups",
            ha="center", color=ACCENT, fontsize=12, style="italic",
            fontweight="bold"))

    plt.savefig(f"{OUT_DIR}/04_game_options.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓ 04_game_options.png")


# ==========================================================================
# Slide 7: Dual-role network split
# ==========================================================================
def make_dual_role():
    fig, ax = plt.subplots(figsize=(13, 8))
    setup_axes(fig, ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)

    stroked(ax.text(6.5, 7.4, "DualRoleAlgorithm — One agent, two brains",
            ha="center", color=ACCENT, fontsize=16, fontweight="bold"))

    # Observation box (left)
    box(ax, 1.7, 4, 2.6, 1.4,
        "Observation\n(43 features)\nincl. is_tagger",
        color=PANEL_LIGHT, fontsize=11)

    # Router (center)
    box(ax, 5.5, 4, 2.1, 1.2, "Router\nchecks is_tagger",
        color="#3a4a6a", edge="#7a8ab0", fontsize=11)
    arrow(ax, 3.0, 4, 4.5, 4, width=2)

    # Two networks (right, well-separated vertically)
    box(ax, 10.2, 6.0, 3.0, 1.2, "Tagger Brain\n(chase policy)",
        color=TAGGER_BG, edge=TAGGER, text_color="white", fontsize=12)
    box(ax, 10.2, 2.0, 3.0, 1.2, "Runner Brain\n(flee policy)",
        color=RUNNER_BG, edge=RUNNER, text_color="white", fontsize=12)

    arrow(ax, 6.55, 4.4, 8.7, 5.7, color=TAGGER, width=2)
    arrow(ax, 6.55, 3.6, 8.7, 2.3, color=RUNNER, width=2)

    # Route labels — pulled OFF the arrow line so they don't overlap
    stroked(ax.text(7.5, 5.6, "is_tagger = True",
            color=TAGGER, fontsize=11, fontweight="bold", style="italic",
            ha="center"))
    stroked(ax.text(7.5, 2.4, "is_tagger = False",
            color=RUNNER, fontsize=11, fontweight="bold", style="italic",
            ha="center"))

    # Bottom note (separate lines, well below arrows)
    stroked(ax.text(6.5, 0.95,
            "All agents share the same two networks",
            ha="center", color=TEXT, fontsize=11, fontweight="bold"))
    stroked(ax.text(6.5, 0.55,
            "Tagger experiences → tagger brain   ·   Runner experiences → runner brain",
            ha="center", color=TEXT_DIM, fontsize=10))
    stroked(ax.text(6.5, 0.2,
            "On tag transfer, the agent simply switches which brain it queries",
            ha="center", color=TEXT_DIM, fontsize=10))

    plt.savefig(f"{OUT_DIR}/07_dual_role.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓ 07_dual_role.png")


# ==========================================================================
# Slide 8: Agent observation diagram
# ==========================================================================
def make_observation():
    fig, ax = plt.subplots(figsize=(13, 11))
    setup_axes(fig, ax)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5.5, 5.5)
    ax.set_aspect("equal")

    # Walls — moved further out so raycasts/labels don't overlap
    walls = [
        (-5.5, 4.0, 1.8, 1.0),
        (3.7, 3.5, 1.8, 1.2),
        (-5.5, -2.0, 1.2, 1.2),
        (3.0, -4.0, 2.5, 1.0),
    ]
    for wx, wy, ww, wh in walls:
        ax.add_patch(Rectangle((wx, wy), ww, wh,
                               facecolor="#85859a", edgecolor=EDGE,
                               linewidth=1.5))

    # 8 raycasts from agent
    ray_dirs = [(0, 1), (0.707, 0.707), (1, 0), (0.707, -0.707),
                (0, -1), (-0.707, -0.707), (-1, 0), (-0.707, 0.707)]
    ray_lens = [4.5, 3.8, 5.0, 5.0, 5.0, 5.0, 4.8, 4.0]

    for (dx, dy), L in zip(ray_dirs, ray_lens):
        ax.plot([0, dx * L], [0, dy * L], color=ACCENT,
                linewidth=1.5, alpha=0.65, linestyle=(0, (4, 3)), zorder=2)
        ax.plot(dx * L, dy * L, "o", color=ACCENT, markersize=6, zorder=3)

    # Tagger (top-left)
    tx, ty = -3.2, 2.5
    ax.add_patch(Circle((tx, ty), 0.4, facecolor=TAGGER,
                        edgecolor="white", linewidth=2, zorder=5))
    stroked(ax.text(tx, ty + 0.85, "TAGGER", ha="center", color=TAGGER,
            fontsize=12, fontweight="bold"))

    # Arrow to tagger (starts well outside the SELF circle)
    ax.annotate("", xy=(tx + 0.4, ty - 0.4), xytext=(-0.5, 0.5),
                arrowprops=dict(arrowstyle="->", color=TAGGER,
                                linewidth=2.2, alpha=0.9))
    # Label PLACED ABOVE the arrow midpoint (away from raycasts and self)
    stroked(ax.text(-2.3, 1.95, "tagger_rel  ·  tagger_dist",
            color=TAGGER, fontsize=10, fontweight="bold",
            ha="center", style="italic"))

    # Two runners — placed clear of arrows and rays
    runners = [(3.3, 2.0, "Runner 1"), (1.5, -3.2, "Runner 2")]
    for (rx, ry, label) in runners:
        ax.add_patch(Circle((rx, ry), 0.35, facecolor=RUNNER,
                            edgecolor="white", linewidth=2, zorder=5))
        stroked(ax.text(rx, ry + 0.7, label, ha="center", color=RUNNER,
                fontsize=11, fontweight="bold"))

    # Arrow to nearest runner
    ax.annotate("", xy=(3.0, 1.7), xytext=(0.5, 0.3),
                arrowprops=dict(arrowstyle="->", color=RUNNER,
                                linewidth=2.2, alpha=0.9))
    # Label below the arrow
    stroked(ax.text(2.0, 0.55, "nearest_runner_rel",
            color=RUNNER, fontsize=10, fontweight="bold",
            ha="center", style="italic"))

    # Self (middle)
    ax.add_patch(Circle((0, 0), 0.5, facecolor=SUCCESS,
                        edgecolor="white", linewidth=2.5, zorder=6))
    stroked(ax.text(0, -1.05, "SELF", ha="center", color=SUCCESS,
            fontsize=12, fontweight="bold"))

    # Title and bottom legend (well outside the diagram area)
    stroked(ax.text(0, 5.1, "Ego-Centric Observation (43 features total)",
            ha="center", color=ACCENT, fontsize=15, fontweight="bold"))

    stroked(ax.text(0, -4.9,
            "8 wall raycasts (yellow dashed)  ·  relative tagger position + distance",
            ha="center", color=TEXT_DIM, fontsize=10))
    stroked(ax.text(0, -5.25,
            "nearest runner position + distance  ·  up to 6 other agents sorted by distance",
            ha="center", color=TEXT_DIM, fontsize=10))

    plt.savefig(f"{OUT_DIR}/08_observation.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓ 08_observation.png")


# ==========================================================================
# Slide 9: Reward design table
# ==========================================================================
def make_reward_table():
    fig, ax = plt.subplots(figsize=(12, 4.5))
    setup_axes(fig, ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)

    ax.text(6, 4.0, "Reward Function Design", ha="center",
            color=ACCENT, fontsize=16, fontweight="bold")

    # Header
    headers = ["Role", "Per-step", "Tag Event", "Distance Shaping"]
    col_x = [1.5, 4.0, 6.8, 9.8]
    for x, h in zip(col_x, headers):
        ax.text(x, 3.3, h, ha="center", color=TEXT_DIM,
                fontsize=11, fontweight="bold")

    # Header divider
    ax.plot([0.5, 11.5], [3.0, 3.0], color=EDGE, linewidth=1)

    # Tagger row
    ax.add_patch(Rectangle((0.5, 1.9), 11, 0.9, facecolor=PANEL,
                           edgecolor="none", alpha=0.5))
    ax.text(col_x[0], 2.35, "TAGGER", ha="center", color=TAGGER,
            fontsize=12, fontweight="bold")
    ax.text(col_x[1], 2.35, "−0.05  (escalating)", ha="center",
            color=TEXT, fontsize=11)
    ax.text(col_x[2], 2.35, "+20.0  on catch", ha="center",
            color=SUCCESS, fontsize=11)
    ax.text(col_x[3], 2.35, "Closer to runner = more reward",
            ha="center", color=TEXT, fontsize=11)

    # Runner row
    ax.add_patch(Rectangle((0.5, 0.7), 11, 0.9, facecolor=PANEL,
                           edgecolor="none", alpha=0.5))
    ax.text(col_x[0], 1.15, "RUNNER", ha="center", color=RUNNER,
            fontsize=12, fontweight="bold")
    ax.text(col_x[1], 1.15, "+0.05  (escalating)", ha="center",
            color=TEXT, fontsize=11)
    ax.text(col_x[2], 1.15, "−20.0  on caught", ha="center",
            color=TAGGER, fontsize=11)
    ax.text(col_x[3], 1.15, "Farther from tagger = more reward",
            ha="center", color=TEXT, fontsize=11)

    ax.text(6, 0.2,
            "Escalating: penalty/bonus grows ~10× over 500 steps without a tag",
            ha="center", color=TEXT_DIM, fontsize=10, style="italic")

    plt.savefig(f"{OUT_DIR}/09_reward_table.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓ 09_reward_table.png")


# ==========================================================================
# Slide 12: Why tabular methods failed on the full game
# ==========================================================================
def make_tabular_failure():
    """Top half = side-by-side visuals, bottom half = bullets — no overlap."""
    fig, ax = plt.subplots(figsize=(14, 9))
    setup_axes(fig, ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)

    # Big title
    stroked(ax.text(7, 8.5, "Why Tabular RL Fails on the Full Game",
            ha="center", color=ACCENT, fontsize=17, fontweight="bold"))

    # ---- LEFT visual panel (top half) ----
    panel_y = 4.5
    panel_h = 3.4

    # Panel background
    ax.add_patch(FancyBboxPatch((0.4, panel_y), 6.2, panel_h,
                                 boxstyle="round,pad=0.02,rounding_size=0.05",
                                 facecolor=PANEL, edgecolor=EDGE, linewidth=1.2))
    stroked(ax.text(3.5, panel_y + panel_h - 0.4, "Full Tag Game Environment",
            ha="center", color=ACCENT, fontsize=12, fontweight="bold"))

    # Mini map inside panel (centered, smaller)
    walls_l = [(1.2, 7.0, 1.6, 0.25), (1.2, 5.8, 0.25, 1.2),
               (3.5, 6.0, 0.25, 1.5), (4.2, 5.4, 1.5, 0.25)]
    for wx, wy, ww, wh in walls_l:
        ax.add_patch(Rectangle((wx, wy), ww, wh, facecolor="#85859a",
                               edgecolor=EDGE, linewidth=0.5))

    agents = [(2.0, 6.0, TAGGER), (3.0, 7.2, RUNNER), (4.7, 7.2, RUNNER),
              (5.5, 5.5, RUNNER), (1.7, 7.4, RUNNER), (4.0, 5.0, RUNNER)]
    for ax_, ay_, c in agents:
        ax.add_patch(Circle((ax_, ay_), 0.18, facecolor=c,
                            edgecolor="white", linewidth=1.2, zorder=5))

    # ---- RIGHT visual panel (top half) ----
    ax.add_patch(FancyBboxPatch((7.4, panel_y), 6.2, panel_h,
                                 boxstyle="round,pad=0.02,rounding_size=0.05",
                                 facecolor=PANEL, edgecolor=EDGE, linewidth=1.2))
    stroked(ax.text(10.5, panel_y + panel_h - 0.4,
            "Tabular Q-Table (Q-Learning / SARSA)",
            ha="center", color=ACCENT, fontsize=12, fontweight="bold"))

    # Q-table sketch — placed inside panel cleanly
    tx, ty = 8.4, 5.1
    cell = 0.32
    for r in range(5):
        for c in range(8):
            ax.add_patch(Rectangle((tx + c * cell, ty + r * cell),
                                   cell, cell, facecolor="#1f2230",
                                   edgecolor=EDGE, linewidth=0.4))
            if (r * 8 + c) % 11 == 0:
                ax.text(tx + (c + 0.5) * cell, ty + (r + 0.5) * cell,
                        "0.3", ha="center", va="center",
                        color=TEXT_DIM, fontsize=5.5)

    # Big red X over the table
    x0, y0 = tx - 0.1, ty - 0.1
    x1, y1 = tx + 8 * cell + 0.1, ty + 5 * cell + 0.1
    ax.plot([x0, x1], [y0, y1], color=TAGGER, linewidth=6, alpha=0.85,
            zorder=10)
    ax.plot([x0, x1], [y1, y0], color=TAGGER, linewidth=6, alpha=0.85,
            zorder=10)

    stroked(ax.text(10.5, 4.8, "Q(state, action) ←  unreachable",
            ha="center", color=TEXT_DIM, fontsize=10, style="italic"))

    # ---- LEFT bullet panel (bottom half) ----
    bullet_y = 0.5
    bullet_h = 3.6

    ax.add_patch(FancyBboxPatch((0.4, bullet_y), 6.2, bullet_h,
                                 boxstyle="round,pad=0.02,rounding_size=0.05",
                                 facecolor=PANEL, edgecolor=EDGE, linewidth=1.2,
                                 alpha=0.6))
    stroked(ax.text(3.5, bullet_y + bullet_h - 0.4,
            "What the environment looks like",
            ha="center", color=TEXT_DIM, fontsize=11, fontweight="bold"))

    bullets_l = [
        "Continuous (x, y) positions",
        "6 agents (variable count)",
        "Walls, crates, rooms",
        "43-feature observation vector",
        "Never-ending — no clean episodes",
    ]
    for i, b in enumerate(bullets_l):
        stroked(ax.text(0.85, bullet_y + bullet_h - 0.95 - i * 0.45,
                f"•  {b}", color=TEXT, fontsize=11))

    # ---- RIGHT bullet panel (bottom half) ----
    ax.add_patch(FancyBboxPatch((7.4, bullet_y), 6.2, bullet_h,
                                 boxstyle="round,pad=0.02,rounding_size=0.05",
                                 facecolor=PANEL, edgecolor=EDGE, linewidth=1.2,
                                 alpha=0.6))
    stroked(ax.text(10.5, bullet_y + bullet_h - 0.4,
            "Why tabular methods break down",
            ha="center", color=TEXT_DIM, fontsize=11, fontweight="bold"))

    bullets_r = [
        "State space explodes  (~10⁷+ cells)",
        "Most cells visited 0 – 1 times",
        "No episode boundary  →  no done signal",
        "No generalization  —  every state is unique",
    ]
    for i, b in enumerate(bullets_r):
        stroked(ax.text(7.85, bullet_y + bullet_h - 0.95 - i * 0.5,
                f"✗  {b}", color=TAGGER, fontsize=11, fontweight="bold"))

    plt.savefig(f"{OUT_DIR}/12_tabular_failure.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓ 12_tabular_failure.png")


# ==========================================================================
# Slide 12b: Actual experimental results — SARSA/QL fail, PPO/DQN succeed
# ==========================================================================
def make_failure_results():
    """Real numbers from experiments/ folder showing the failure mode."""
    import json
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color(EDGE)
        ax.tick_params(colors=TEXT_DIM)

    # ---- Left: training tags over rounds ----
    paths = [
        ("Q-Learning", "experiments/q_learning_map01/training_log.json", TAGGER),
        ("SARSA",      "experiments/sarsa_map01/training_log.json",      WARN),
        ("DQN",        "experiments/dqn/training_log.json",              RUNNER),
        ("PPO",        "experiments/ppo/training_log.json",              SUCCESS),
    ]

    for name, path, color in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            log = json.load(f)
        rounds = [e["round"] for e in log]
        tags = [e["total_tags"] for e in log]
        ax1.plot(rounds, tags, color=color, linewidth=2.5,
                 label=name, marker="o", markersize=4, markevery=max(1, len(rounds)//15))

    ax1.set_xlabel("Training Round", color=TEXT, fontsize=12, fontweight="bold")
    ax1.set_ylabel("Total Tags Achieved", color=TEXT, fontsize=12, fontweight="bold")
    stroked(ax1.set_title("Training Progress on the Full Game",
                          color=ACCENT, fontsize=14, fontweight="bold", pad=14))
    ax1.legend(facecolor=PANEL, edgecolor=EDGE, labelcolor=TEXT,
               fontsize=11, loc="upper left")
    ax1.grid(True, alpha=0.2, color=EDGE)

    # Annotation pointing at the flat tabular line
    ax1.annotate("Q-Learning & SARSA:\nflat at 0 tags",
                 xy=(500, 0), xytext=(280, 80),
                 color=TAGGER, fontsize=11, fontweight="bold",
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color=TAGGER, lw=2))

    # ---- Right: evaluation bar chart (mean tags/episode) ----
    eval_data = []
    for name, color in [("Q-Learning", TAGGER), ("SARSA", WARN),
                        ("DQN", RUNNER), ("PPO", SUCCESS)]:
        path_map = {
            "Q-Learning": "experiments/q_learning_map01/results/eval_metrics.json",
            "SARSA":      "experiments/sarsa_map01/results/eval_metrics.json",
            "DQN":        "experiments/dqn/results/eval_metrics.json",
            "PPO":        "experiments/ppo/results/eval_metrics.json",
        }
        path = path_map[name]
        if not os.path.exists(path):
            continue
        with open(path) as f:
            metrics = json.load(f)
        # Use the BEST epoch's mean_tags
        best = max(metrics, key=lambda m: m.get("mean_tags", 0))
        eval_data.append((name, best["mean_tags"], best.get("std_tags", 0), color))

    names = [d[0] for d in eval_data]
    means = [d[1] for d in eval_data]
    stds = [d[2] for d in eval_data]
    colors = [d[3] for d in eval_data]

    bars = ax2.bar(names, means, yerr=stds, capsize=6, color=colors,
                   edgecolor="white", linewidth=1.5, alpha=0.9)

    # Label each bar with its value
    for bar, m in zip(bars, means):
        label = f"{m:.2f}" if m > 0 else "0"
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.15, label,
                 ha="center", color=TEXT, fontsize=12, fontweight="bold")

    ax2.set_ylabel("Mean Tags per Eval Episode (best checkpoint)",
                   color=TEXT, fontsize=11, fontweight="bold")
    stroked(ax2.set_title("Best Evaluation Performance",
                          color=ACCENT, fontsize=14, fontweight="bold", pad=14))
    ax2.tick_params(axis="x", colors=TEXT, labelsize=11)
    ax2.grid(True, alpha=0.2, color=EDGE, axis="y")
    ax2.set_ylim(0, max(max(means) + max(stds) + 0.8, 4))

    # Add a "0 tags" annotation on tabular bars
    for i, (n, m, _, _) in enumerate(eval_data):
        if m == 0:
            ax2.text(i, 0.4, "DID NOT\nLEARN", ha="center", va="bottom",
                     color=TAGGER, fontsize=10, fontweight="bold",
                     style="italic")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/12b_failure_results.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓ 12b_failure_results.png")


# ==========================================================================
# Slide 13: Comparison table — Full game vs Gridworld
# ==========================================================================
def make_comparison_table():
    fig, ax = plt.subplots(figsize=(12, 7))
    setup_axes(fig, ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    ax.text(6, 6.5, "Simplifying the Problem", ha="center",
            color=ACCENT, fontsize=16, fontweight="bold")

    # Headers
    ax.text(2.0, 5.7, "Property", color=TEXT_DIM, fontsize=12,
            fontweight="bold")
    ax.text(6.5, 5.7, "Full Tag Game", color=TAGGER, fontsize=12,
            fontweight="bold", ha="center")
    ax.text(10.0, 5.7, "Gridworld (simplified)", color=SUCCESS,
            fontsize=12, fontweight="bold", ha="center")

    ax.plot([0.5, 11.5], [5.4, 5.4], color=EDGE, linewidth=1)

    rows = [
        ("Position",       "Continuous (x, y) pixels",        "Discrete 10×10 grid"),
        ("Agent count",    "6 agents",                         "1 tagger + 1 runner"),
        ("Episode",        "Never ends",                       "Ends when tagger catches"),
        ("Obstacles",      "Walls · crates · rooms",           "Empty grid"),
        ("Observation",    "43 floats (raycasts, etc.)",       "4 ints: (tx, ty, rx, ry)"),
        ("State space",    "~10,000,000+",                     "Exactly 10,000"),
        ("Movement",       "Real-time (60 FPS)",               "Turn-based"),
    ]

    y = 5.0
    for i, (prop, full, grid) in enumerate(rows):
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.5, y - 0.27), 11, 0.55,
                                   facecolor=PANEL, edgecolor="none",
                                   alpha=0.45))
        ax.text(2.0, y, prop, color=TEXT_DIM, fontsize=11)
        ax.text(6.5, y, full, color=TEXT, fontsize=11, ha="center")
        ax.text(10.0, y, grid, color=TEXT, fontsize=11, ha="center")
        y -= 0.55

    plt.savefig(f"{OUT_DIR}/13_comparison_table.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓ 13_comparison_table.png")


# ==========================================================================
# Slide 14: 3-phase training flowchart
# ==========================================================================
def make_training_phases():
    # Wider + taller canvas, right-side caption now sits below each row, not beside it
    fig, ax = plt.subplots(figsize=(14, 9))
    setup_axes(fig, ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)

    stroked(ax.text(7, 8.4, "Training Strategy: One Role at a Time",
            ha="center", color=ACCENT, fontsize=17, fontweight="bold"))

    def phase_row(y, num, learner, opponent, learner_color, opp_color,
                  caption):
        # Phase label box
        ax.add_patch(FancyBboxPatch(
            (0.5, y - 0.5), 1.6, 1.0,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=PANEL_LIGHT, edgecolor=ACCENT, linewidth=1.5))
        stroked(ax.text(1.3, y, f"Phase\n{num}", ha="center", va="center",
                color=ACCENT, fontsize=13, fontweight="bold"))

        # Learner box
        ax.add_patch(FancyBboxPatch(
            (2.8, y - 0.55), 4.0, 1.1,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=learner_color, edgecolor="white", linewidth=2))
        stroked(ax.text(4.8, y, learner, ha="center", va="center",
                color="white", fontsize=12, fontweight="bold"))

        # VS
        stroked(ax.text(7.3, y, "VS", ha="center", va="center",
                color=TEXT, fontsize=15, fontweight="bold"))

        # Opponent box
        ax.add_patch(FancyBboxPatch(
            (7.8, y - 0.55), 4.0, 1.1,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=opp_color, edgecolor="white", linewidth=2))
        stroked(ax.text(9.8, y, opponent, ha="center", va="center",
                color="white", fontsize=12, fontweight="bold"))

        # Caption — placed BELOW the row, centered (no overlap)
        stroked(ax.text(7, y - 0.95, caption, ha="center", va="center",
                color=ACCENT, fontsize=10, style="italic"))

    phase_row(7.0, "1", "Tagger\n(learning)", "Runner\n(random)",
              TAGGER_BG, "#444858", "→ tagger learns to chase a moving target")
    phase_row(4.7, "2", "Runner\n(learning)", "Tagger\n(frozen, trained)",
              RUNNER_BG, TAGGER_BG, "→ runner learns to evade a real chaser")
    phase_row(2.4, "3", "Tagger\n(trained)", "Runner\n(trained)",
              TAGGER_BG, RUNNER_BG, "→ watch both trained agents play & evaluate")

    # Down arrows between phases (in the phase column)
    arrow(ax, 1.3, 6.3, 1.3, 5.4, color=ACCENT, width=2.5)
    arrow(ax, 1.3, 4.0, 1.3, 3.1, color=ACCENT, width=2.5)

    # Bottom note (well below last row)
    stroked(ax.text(7, 0.85,
            "Each phase has a FIXED opponent, so the learner sees a stationary problem",
            ha="center", color=TEXT, fontsize=11, fontweight="bold"))
    stroked(ax.text(7, 0.4,
            "→ this is what lets Q-Learning and SARSA actually converge",
            ha="center", color=TEXT_DIM, fontsize=10, style="italic"))

    plt.savefig(f"{OUT_DIR}/14_training_phases.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓ 14_training_phases.png")


# ==========================================================================
# Slide 16: Algorithm-to-problem matching
# ==========================================================================
def make_algo_matching():
    fig, ax = plt.subplots(figsize=(12, 6))
    setup_axes(fig, ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    ax.text(6, 5.4, "Match Algorithm to Problem Structure",
            ha="center", color=ACCENT, fontsize=15, fontweight="bold")

    # Two columns: Tabular fits vs Deep RL fits
    # Tabular column
    ax.add_patch(FancyBboxPatch(
        (0.5, 0.5), 5.3, 4.0,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=PANEL, edgecolor=SUCCESS, linewidth=2))
    ax.text(3.15, 4.1, "Tabular (Q-Learning, SARSA)",
            ha="center", color=SUCCESS, fontsize=13, fontweight="bold")

    tabular_pts = [
        "✓ Small, discrete state space",
        "✓ Clear episode boundaries",
        "✓ Fast convergence",
        "✓ Interpretable Q-values",
        "✗ Doesn't scale to high dims",
    ]
    for i, t in enumerate(tabular_pts):
        col = SUCCESS if t.startswith("✓") else TAGGER
        ax.text(0.85, 3.4 - i * 0.5, t, color=col, fontsize=11)

    # Deep RL column
    ax.add_patch(FancyBboxPatch(
        (6.2, 0.5), 5.3, 4.0,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=PANEL, edgecolor=RUNNER, linewidth=2))
    ax.text(8.85, 4.1, "Deep RL (PPO, DQN)",
            ha="center", color=RUNNER, fontsize=13, fontweight="bold")

    deep_pts = [
        "✓ Continuous, high-dim states",
        "✓ Generalizes across states",
        "✓ Handles ongoing tasks",
        "✓ Rich observations OK",
        "✗ Slower training, less interpretable",
    ]
    for i, t in enumerate(deep_pts):
        col = SUCCESS if t.startswith("✓") else TAGGER
        ax.text(6.55, 3.4 - i * 0.5, t, color=col, fontsize=11)

    plt.savefig(f"{OUT_DIR}/16_algo_matching.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓ 16_algo_matching.png")


# ==========================================================================
# Slide 19: Learned vs Future Work
# ==========================================================================
def make_learned_future():
    fig, ax = plt.subplots(figsize=(12, 6))
    setup_axes(fig, ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    # Left: Learned
    ax.add_patch(FancyBboxPatch(
        (0.5, 0.5), 5.3, 4.5,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=PANEL, edgecolor=SUCCESS, linewidth=2))
    ax.text(3.15, 4.5, "What I Learned", ha="center",
            color=SUCCESS, fontsize=14, fontweight="bold")

    learned = [
        "• Reward shaping > algorithm choice",
        "• Ego-centric obs train much faster",
        "• Splitting tagger/runner brains was the",
        "   single biggest improvement",
        "• Match the algorithm to the problem,",
        "   not the other way around",
    ]
    for i, t in enumerate(learned):
        ax.text(0.85, 3.85 - i * 0.45, t, color=TEXT, fontsize=11)

    # Right: Future
    ax.add_patch(FancyBboxPatch(
        (6.2, 0.5), 5.3, 4.5,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=PANEL, edgecolor=ACCENT, linewidth=2))
    ax.text(8.85, 4.5, "Future Work", ha="center",
            color=ACCENT, fontsize=14, fontweight="bold")

    future = [
        "• Cooperative behavior between runners",
        "• Dynamic difficulty adjustment",
        "• Support for >6 agents",
        "• Benchmark PPO vs DQN vs Q-Learning",
        "   on the same maps",
        "• Add freeze tag, team tag variants",
    ]
    for i, t in enumerate(future):
        ax.text(6.55, 3.85 - i * 0.45, t, color=TEXT, fontsize=11)

    plt.savefig(f"{OUT_DIR}/19_learned_future.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓ 19_learned_future.png")


# ==========================================================================
# Slide 18: Technical highlights
# ==========================================================================
def make_highlights():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    setup_axes(fig, ax)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)

    ax.text(5.5, 6.0, "Technical Highlights", ha="center",
            color=ACCENT, fontsize=16, fontweight="bold")

    items = [
        ("01", "No external assets",
         "All textures, sprites, and effects generated procedurally"),
        ("02", "Modular architecture",
         "Adding a new algorithm = 1 file + 1 config line"),
        ("03", "Dual-role models",
         "Separate tagger and runner brains, shared across agents"),
        ("04", "Headless training",
         "~2,000 game steps/sec — much faster than real-time"),
        ("05", "Live visualization + video export",
         "Every training run can be recorded as MP4"),
        ("06", "Three difficulty maps + gridworld",
         "Small / Medium / Large + simplified tabular environment"),
    ]

    y = 5.2
    for num, title, desc in items:
        ax.add_patch(Rectangle((0.5, y - 0.35), 10, 0.7,
                               facecolor=PANEL, edgecolor="none", alpha=0.5))
        # Number badge
        ax.add_patch(Circle((1.0, y), 0.25, facecolor=ACCENT,
                            edgecolor="none"))
        ax.text(1.0, y, num, ha="center", va="center", color="#1c1f2a",
                fontsize=11, fontweight="bold")
        ax.text(1.6, y + 0.12, title, color=ACCENT, fontsize=12,
                fontweight="bold", va="center")
        ax.text(1.6, y - 0.18, desc, color=TEXT_DIM, fontsize=10,
                va="center")
        y -= 0.85

    plt.savefig(f"{OUT_DIR}/18_highlights.png", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  ✓ 18_highlights.png")


# ==========================================================================
# Main
# ==========================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating presentation assets into {OUT_DIR}/\n")

    make_architecture()
    make_game_options()
    make_dual_role()
    make_observation()
    make_reward_table()
    make_tabular_failure()
    make_failure_results()
    make_comparison_table()
    make_training_phases()
    make_algo_matching()
    make_highlights()
    make_learned_future()

    print(f"\nAll assets saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
