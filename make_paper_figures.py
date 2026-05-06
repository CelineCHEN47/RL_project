#!/usr/bin/env python3
"""Generate light/academic versions of the diagrams used in the IEEE paper.

Outputs PNGs into figures/ (overwriting the dark slide versions).
"""

import os
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle


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

TAGGER_FILL  = "#fadbd8"  # very light red
RUNNER_FILL  = "#d6e6f2"  # very light blue
SUCCESS_FILL = "#d6ecdf"  # very light green
PANEL_FILL   = "#f5f5f7"

OUT = "figures"


def setup(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def box(ax, x, y, w, h, text, fill=PANEL_FILL, edge=EDGE,
        text_color=TEXT, fontsize=11, bold=True, lw=1.2):
    rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          linewidth=lw, edgecolor=edge, facecolor=fill)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", color=text_color,
            fontsize=fontsize, fontweight=weight)


def arrow(ax, x1, y1, x2, y2, color=EDGE, width=1.2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=14, linewidth=width,
                        color=color, zorder=2)
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Fig: Dual-role architecture
# ---------------------------------------------------------------------------
def make_dual_role():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    setup(fig, ax)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)

    ax.text(5.5, 5.05, "Dual-Role Algorithm Wrapper",
            ha="center", color=TEXT, fontsize=14, fontweight="bold")

    # Observation
    box(ax, 1.5, 2.75, 2.4, 1.3,
        "Observation\n(43-d vector)\nincluding is_tagger",
        fill=PANEL_FILL, edge=EDGE, fontsize=10)

    # Router
    box(ax, 4.7, 2.75, 1.9, 1.0,
        "Router\nchecks is_tagger",
        fill="#e8eaf0", edge=ACCENT, fontsize=10)
    arrow(ax, 2.75, 2.75, 3.75, 2.75, width=1.6)

    # Two networks
    box(ax, 8.6, 4.0, 2.6, 1.0, r"Tagger Brain $\pi_T$" + "\n(chase policy)",
        fill=TAGGER_FILL, edge=TAGGER, text_color=TAGGER, fontsize=11)
    box(ax, 8.6, 1.5, 2.6, 1.0, r"Runner Brain $\pi_R$" + "\n(flee policy)",
        fill=RUNNER_FILL, edge=RUNNER, text_color=RUNNER, fontsize=11)

    arrow(ax, 5.65, 3.05, 7.3, 3.85, color=TAGGER, width=1.6)
    arrow(ax, 5.65, 2.45, 7.3, 1.65, color=RUNNER, width=1.6)

    ax.text(6.5, 3.7, "is_tagger=True",
            color=TAGGER, fontsize=9, fontweight="bold",
            style="italic", ha="center")
    ax.text(6.5, 1.85, "is_tagger=False",
            color=RUNNER, fontsize=9, fontweight="bold",
            style="italic", ha="center")

    ax.text(5.5, 0.55,
            "All agents share the same two networks.   "
            "Tagger experiences $\\rightarrow \\pi_T$.   "
            "Runner experiences $\\rightarrow \\pi_R$.",
            ha="center", color=TEXT_DIM, fontsize=10)

    plt.savefig(f"{OUT}/dual_role.png", dpi=200, bbox_inches="tight",
                facecolor=BG)
    plt.close()
    print("  wrote dual_role.png")


# ---------------------------------------------------------------------------
# Fig: Ego-centric observation
# ---------------------------------------------------------------------------
def make_observation():
    fig, ax = plt.subplots(figsize=(8, 7.5))
    setup(fig, ax)
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")

    ax.text(0, 4.5, "Ego-centric observation (43 features)",
            ha="center", color=TEXT, fontsize=13, fontweight="bold")

    # Walls
    walls = [(-5, 3, 1.5, 0.8), (3, 3, 1.5, 1.0),
             (-5, -1.5, 1.0, 1.0), (2.5, -3, 2.0, 1.0)]
    for wx, wy, ww, wh in walls:
        ax.add_patch(Rectangle((wx, wy), ww, wh,
                               facecolor="#cccccc", edgecolor="#888888",
                               linewidth=1.2))

    # 8 raycasts
    ray_dirs = [(0, 1), (0.707, 0.707), (1, 0), (0.707, -0.707),
                (0, -1), (-0.707, -0.707), (-1, 0), (-0.707, 0.707)]
    ray_lens = [3.5, 2.8, 4.0, 4.0, 4.0, 4.0, 3.8, 3.0]
    for (dx, dy), L in zip(ray_dirs, ray_lens):
        ax.plot([0, dx * L], [0, dy * L], color=WARN,
                linewidth=1.2, alpha=0.85, linestyle=(0, (3, 2)), zorder=2)
        ax.plot(dx * L, dy * L, "o", color=WARN, markersize=4, zorder=3)

    # Tagger
    tx, ty = -2.5, 2.0
    ax.add_patch(Circle((tx, ty), 0.32, facecolor=TAGGER,
                        edgecolor="white", linewidth=2, zorder=5))
    ax.text(tx, ty + 0.6, "TAGGER", ha="center", color=TAGGER,
            fontsize=10, fontweight="bold")

    # Arrow to tagger
    ax.annotate("", xy=(tx + 0.3, ty - 0.3), xytext=(0.3, 0.3),
                arrowprops=dict(arrowstyle="->", color=TAGGER,
                                linewidth=1.6, alpha=0.9))
    ax.text(-1.5, 1.4, "tagger_rel,\ntagger_dist",
            color=TAGGER, fontsize=9, ha="center", style="italic")

    # Runners
    for (rx, ry, label) in [(2.5, 1.5, "runner"), (1.0, -2.5, "runner")]:
        ax.add_patch(Circle((rx, ry), 0.28, facecolor=RUNNER,
                            edgecolor="white", linewidth=2, zorder=5))
        ax.text(rx, ry + 0.55, label, ha="center", color=RUNNER,
                fontsize=9)

    # Arrow to nearest runner
    ax.annotate("", xy=(2.2, 1.3), xytext=(0.3, 0.2),
                arrowprops=dict(arrowstyle="->", color=RUNNER,
                                linewidth=1.6, alpha=0.9))
    ax.text(1.5, 0.55, "nearest_runner",
            color=RUNNER, fontsize=9, ha="center", style="italic")

    # Self
    ax.add_patch(Circle((0, 0), 0.38, facecolor=SUCCESS,
                        edgecolor="white", linewidth=2, zorder=6))
    ax.text(0, -0.85, "SELF", ha="center", color=SUCCESS,
            fontsize=10, fontweight="bold")

    # Velocity
    ax.annotate("", xy=(1.0, 0.5), xytext=(0.4, 0.2),
                arrowprops=dict(arrowstyle="->", color=SUCCESS,
                                linewidth=2.0))
    ax.text(0.95, 0.85, "velocity",
            color=SUCCESS, fontsize=8, style="italic")

    ax.text(0, -4.6,
            "Yellow dashed lines: 8 wall raycasts.   Other agents are "
            "appended sorted by distance.",
            ha="center", color=TEXT_DIM, fontsize=9)

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

    # ---- Left: training tags over rounds ----
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

    # ---- Right: best evaluation ----
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
# Main
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
