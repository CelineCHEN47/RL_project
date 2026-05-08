#!/usr/bin/env python3
"""Generate light/academic versions of the diagrams used in the IEEE paper.

Outputs PNGs into figures/. The two architecture diagrams (dual_role,
observation) are drawn cleanly with no overlapping text on lines or shapes.
Tabular_failure.png is regenerated from real training logs.
"""

import os
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Patch,
)


# ---- Academic light palette ----
BG       = "white"
TEXT     = "#1a1a1a"
TEXT_DIM = "#555555"
EDGE     = "#666666"
ACCENT   = "#2c3e50"

TAGGER   = "#c0392b"   # dark red
RUNNER   = "#2c5d8a"   # dark blue
SUCCESS  = "#1f6f3f"   # dark green
WARN     = "#b87a1d"   # dark amber
NEUTRAL  = "#999999"

TAGGER_FILL  = "#fadbd8"
RUNNER_FILL  = "#d6e6f2"
SUCCESS_FILL = "#d6ecdf"
PANEL_FILL   = "#f5f5f7"

OUT = "figures"


def _setup(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _box(ax, x, y, w, h, text, fill=PANEL_FILL, edge=EDGE,
         text_color=TEXT, fontsize=11, bold=True, lw=1.3):
    rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=lw, edgecolor=edge, facecolor=fill,
                          zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", color=text_color,
            fontsize=fontsize, fontweight=("bold" if bold else "normal"),
            zorder=3)


def _arrow(ax, x1, y1, x2, y2, color=EDGE, width=1.3):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=14, linewidth=width,
                        color=color, zorder=2)
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Fig: Two-brain training architecture
# ---------------------------------------------------------------------------
def make_dual_role():
    fig, ax = plt.subplots(figsize=(10, 5.6))
    _setup(fig, ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)

    # Title
    ax.text(5.0, 5.25, "Two-Brain Training",
            ha="center", color=TEXT, fontsize=15, fontweight="bold")

    # Left: tagger experience -> tagger brain
    _box(ax, 1.7, 4.0, 2.6, 0.85,
         "Tagger experience\n$(s, a, r, s')$ when role = tagger",
         fill=TAGGER_FILL, edge=TAGGER, text_color=TAGGER, fontsize=10)
    _box(ax, 6.4, 4.0, 2.6, 0.85, r"Tagger brain $\pi_T$",
         fill="white", edge=TAGGER, text_color=TAGGER, fontsize=12)
    _arrow(ax, 3.05, 4.0, 5.05, 4.0, color=TAGGER, width=1.6)

    # Right: runner experience -> runner brain
    _box(ax, 1.7, 1.7, 2.6, 0.85,
         "Runner experience\n$(s, a, r, s')$ when role = runner",
         fill=RUNNER_FILL, edge=RUNNER, text_color=RUNNER, fontsize=10)
    _box(ax, 6.4, 1.7, 2.6, 0.85, r"Runner brain $\pi_R$",
         fill="white", edge=RUNNER, text_color=RUNNER, fontsize=12)
    _arrow(ax, 3.05, 1.7, 5.05, 1.7, color=RUNNER, width=1.6)

    # Caption text below (placed safely outside any line)
    ax.text(5.0, 0.45,
            "The two networks share no parameters and no replay buffer.",
            ha="center", color=TEXT, fontsize=10)
    ax.text(5.0, 0.10,
            "An observation never produces gradients on both at the same time.",
            ha="center", color=TEXT_DIM, fontsize=9, style="italic")

    plt.savefig(f"{OUT}/dual_role.png", dpi=200, bbox_inches="tight",
                facecolor=BG)
    plt.close()
    print("  wrote dual_role.png")


# ---------------------------------------------------------------------------
# Fig: Ego-centric observation (clean, square)
# ---------------------------------------------------------------------------
def make_observation():
    """Square view of the agent's egocentric observation.

    Rules of thumb:
      - The dot/agent stays clear of every label.
      - Labels for raycasts and entities live in a legend strip on the right.
      - The 8 raycasts are dashed lines; their endpoints are small dots only.
      - We label *one* tagger and *one* runner using arrows to a panel on the
        right, so no text crosses the rays.
    """
    fig = plt.figure(figsize=(11, 6))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.65], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    side = fig.add_subplot(gs[0, 1])

    # ----- Left panel: square egocentric view -----
    R = 5.0
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfbfd")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(EDGE)
        s.set_linewidth(1.3)

    # Title above the square
    ax.set_title("Ego-centric observation (one agent's view)",
                 color=TEXT, fontsize=13, fontweight="bold", pad=10)

    # A few wall blocks (kept far from the centre and the rays)
    walls = [
        (-4.6, 3.0, 1.4, 0.7),
        ( 3.2, 3.3, 1.4, 0.6),
        (-4.6,-3.6, 1.0, 0.9),
        ( 3.4,-2.6, 1.2, 0.8),
    ]
    for wx, wy, ww, wh in walls:
        ax.add_patch(Rectangle((wx, wy), ww, wh,
                               facecolor="#cccccc", edgecolor="#888888",
                               linewidth=1.0, zorder=1))

    # 8 wall raycasts (dashed amber)
    ray_dirs = [(0,1),(0.707,0.707),(1,0),(0.707,-0.707),
                (0,-1),(-0.707,-0.707),(-1,0),(-0.707,0.707)]
    ray_lens = [4.0, 3.4, 4.6, 3.6, 4.6, 4.0, 4.4, 3.0]
    for (dx, dy), L in zip(ray_dirs, ray_lens):
        ax.plot([0, dx*L], [0, dy*L], color=WARN,
                linewidth=1.1, linestyle=(0,(4,3)), alpha=0.85, zorder=2)
        ax.plot(dx*L, dy*L, "o", color=WARN, markersize=4.0, zorder=3)

    # Tagger entity (top-left quadrant, kept clear of any label)
    tx, ty = -2.6, 2.0
    ax.add_patch(Circle((tx, ty), 0.36, facecolor=TAGGER,
                        edgecolor="white", linewidth=2.0, zorder=5))

    # Runner entity (right side)
    rx, ry = 2.2, -1.6
    ax.add_patch(Circle((rx, ry), 0.32, facecolor=RUNNER,
                        edgecolor="white", linewidth=2.0, zorder=5))

    # Self at centre
    ax.add_patch(Circle((0, 0), 0.42, facecolor=SUCCESS,
                        edgecolor="white", linewidth=2.0, zorder=6))

    # Velocity arrow (short, points up-right)
    ax.annotate("", xy=(1.05, 0.65), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", color=SUCCESS,
                                linewidth=2.0))

    # Markers for the side legend (numbered annotations near each entity)
    def _badge(x, y, n, color):
        ax.add_patch(Circle((x, y), 0.28, facecolor="white",
                            edgecolor=color, linewidth=1.5, zorder=7))
        ax.text(x, y, str(n), ha="center", va="center",
                color=color, fontsize=10, fontweight="bold", zorder=8)

    _badge( 0.85,  0.85, 1, SUCCESS)   # self / velocity
    _badge(tx + 0.6, ty + 0.6, 2, TAGGER)
    _badge(rx + 0.6, ry + 0.6, 3, RUNNER)
    _badge( 3.6,  2.4, 4, WARN)         # raycast endpoint badge

    # ----- Right panel: legend / table -----
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.set_xticks([]); side.set_yticks([])
    for s in side.spines.values():
        s.set_visible(False)
    side.set_facecolor(BG)

    side.text(0.02, 0.96, "What the agent sees (43 dims)",
              color=TEXT, fontsize=12, fontweight="bold", va="top")

    rows = [
        (1, SUCCESS, "Self position & velocity",
         "self_pos (2)  +  self_vel (2)"),
        (2, TAGGER,  "Relative offset to tagger",
         "tagger_rel (2) + tagger_dist (1)"),
        (3, RUNNER,  "Nearest runner",
         "nearest_runner_rel (2) + dist (1)"),
        (4, WARN,    "Eight wall raycasts",
         "wall_rays (8 normalized dists)"),
    ]
    y = 0.85
    for n, color, title, body in rows:
        # Numbered swatch
        side.add_patch(Circle((0.05, y), 0.025, facecolor="white",
                              edgecolor=color, linewidth=1.5,
                              transform=side.transAxes))
        side.text(0.05, y, str(n), color=color, fontsize=9,
                  fontweight="bold", ha="center", va="center",
                  transform=side.transAxes)
        side.text(0.11, y + 0.025, title,
                  color=TEXT, fontsize=10.5, fontweight="bold",
                  va="center", transform=side.transAxes)
        side.text(0.11, y - 0.020, body,
                  color=TEXT_DIM, fontsize=9, va="center",
                  transform=side.transAxes, family="monospace")
        y -= 0.135

    side.text(0.02, y - 0.02,
              "+  is_tagger flag (1)\n"
              "+  up to 6 other agents × (rel\\_pos, dist, role) = 24",
              color=TEXT_DIM, fontsize=9.5, va="top",
              transform=side.transAxes, family="monospace")

    plt.savefig(f"{OUT}/observation.png", dpi=200, bbox_inches="tight",
                facecolor=BG)
    plt.close()
    print("  wrote observation.png")


# ---------------------------------------------------------------------------
# Fig: Tabular failure (real data from experiments/)
# ---------------------------------------------------------------------------
def make_tabular_failure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_color(EDGE)
        ax.tick_params(colors=TEXT)

    paths = [
        ("Q-Learning", "experiments/q_learning_map01/training_log.json", TAGGER, "-"),
        ("SARSA",      "experiments/sarsa_map01/training_log.json",      WARN,   "-"),
        ("DQN",        "experiments/dqn3.0/training_log.json",           RUNNER, "-"),
        ("PPO",        "experiments/ppo/training_log.json",              SUCCESS,"--"),
    ]

    for name, path, color, ls in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            log = json.load(f)
        rounds = [e["round"] for e in log]
        tags = [e["total_tags"] for e in log]
        ax1.plot(rounds, tags, color=color, linewidth=2.0,
                 label=name, linestyle=ls,
                 marker="o", markersize=3, markevery=max(1, len(rounds)//15))

    ax1.set_xlabel("Training round", color=TEXT, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cumulative tags", color=TEXT, fontsize=11, fontweight="bold")
    ax1.set_title("Training progress on the full game",
                  color=TEXT, fontsize=12, fontweight="bold", pad=10)
    leg = ax1.legend(facecolor="white", edgecolor=EDGE, labelcolor=TEXT,
                     fontsize=10, loc="upper left")
    leg.get_frame().set_alpha(0.95)
    ax1.grid(True, alpha=0.3, color="#cccccc")

    ax1.annotate("Q-Learning & SARSA:\nflat at 0",
                 xy=(500, 0), xytext=(280, 100),
                 color=TAGGER, fontsize=10, fontweight="bold",
                 ha="center",
                 arrowprops=dict(arrowstyle="->", color=TAGGER, lw=1.5))

    eval_paths = {
        "Q-Learning": ("experiments/q_learning_map01/results/eval_metrics.json", TAGGER),
        "SARSA":      ("experiments/sarsa_map01/results/eval_metrics.json",      WARN),
        "DQN":        ("experiments/dqn3.0/results/eval_metrics.json",           RUNNER),
        "PPO":        ("experiments/ppo/results/eval_metrics.json",              SUCCESS),
    }
    eval_data = []
    for name, (path, color) in eval_paths.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            metrics = json.load(f)
        best = max(metrics, key=lambda m: m.get("mean_tags", 0))
        eval_data.append((name, best["mean_tags"], best.get("std_tags", 0), color))

    names = [d[0] for d in eval_data]
    means = [d[1] for d in eval_data]
    stds = [d[2] for d in eval_data]
    colors = [d[3] for d in eval_data]

    bars = ax2.bar(names, means, yerr=stds, capsize=5, color=colors,
                   edgecolor=TEXT, linewidth=0.8, alpha=0.85)
    for bar, m in zip(bars, means):
        label = f"{m:.2f}" if m > 0 else "0"
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.2, label,
                 ha="center", color=TEXT, fontsize=11, fontweight="bold")

    ax2.set_ylabel("Mean tags / evaluation episode (best ckpt)",
                   color=TEXT, fontsize=11, fontweight="bold")
    ax2.set_title("Best evaluation performance",
                  color=TEXT, fontsize=12, fontweight="bold", pad=10)
    ax2.tick_params(axis="x", colors=TEXT, labelsize=10)
    ax2.grid(True, alpha=0.3, color="#cccccc", axis="y")
    ax2.set_ylim(0, max(max(means) + max(stds) + 1, 4))

    for i, (n, m, _, _) in enumerate(eval_data):
        if m == 0:
            ax2.text(i, 0.4, "did not\nlearn", ha="center", va="bottom",
                     color=TAGGER, fontsize=9, fontweight="bold",
                     style="italic")

    plt.tight_layout()
    plt.savefig(f"{OUT}/tabular_failure.png", dpi=200, bbox_inches="tight",
                facecolor=BG)
    plt.close()
    print("  wrote tabular_failure.png")


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"Generating academic-style figures into {OUT}/\n")
    make_dual_role()
    make_observation()
    make_tabular_failure()
    print("\nDone.")


if __name__ == "__main__":
    main()
