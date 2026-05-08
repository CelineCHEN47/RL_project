#!/usr/bin/env python3
"""Generate the two missing PPO figures referenced in the report:

  figures/ppo_training_curve.png   — round vs tags, raw + smoothed
  figures/ppo_role_evaluation.png  — isolated tagger / runner per checkpoint

Pulled from `dqn and ppo/metric/training_log.json` and
`dqn and ppo/metric/role_eval_metrics.json` (the 1000-epoch PPO run).
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BG       = "white"
TEXT     = "#1a1a1a"
TEXT_DIM = "#555555"
EDGE     = "#666666"
TAGGER   = "#c0392b"
RUNNER   = "#2c5d8a"
PPO_LINE = "#2c5d8a"   # blue, matches the DQN curve convention

OUT = "figures"
PPO_TRAINING_LOG = "dqn and ppo/metric/training_log.json"
DQN_TRAINING_LOG = "experiments/dqn3.0/training_log.json"
PPO_ROLE_EVAL    = "dqn and ppo/metric/role_eval_metrics.json"
DQN_ROLE_EVAL    = "experiments/dqn3.0/results/role_eval_metrics.json"


def smooth(y, k=51):
    y = np.asarray(y, dtype=np.float32)
    if len(y) < k:
        return y
    pad = k // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    out = np.convolve(yp, np.ones(k) / k, mode="valid")
    return out[: len(y)]


def _training_curve(log_path, title, out_name):
    """Render a training curve in the unified PPO/DQN style."""
    with open(log_path) as f:
        log = json.load(f)
    rounds = np.array([e["round"] for e in log])
    tags = np.array([e["round_tags"] for e in log], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(rounds, tags, color=PPO_LINE, alpha=0.30, linewidth=0.9,
            label="raw")
    ax.plot(rounds, smooth(tags, 51), color=PPO_LINE, linewidth=2.0,
            label="smoothed (window 51)")

    ax.set_xlabel("Training round (epoch)", color=TEXT, fontsize=11,
                  fontweight="bold")
    ax.set_ylabel("Tags this round", color=TEXT, fontsize=11,
                  fontweight="bold")
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.3, color="#cccccc")
    leg = ax.legend(facecolor="white", edgecolor=EDGE, labelcolor=TEXT,
                    fontsize=9, loc="lower right")
    leg.get_frame().set_alpha(0.95)
    for s in ax.spines.values():
        s.set_color(EDGE)
    ax.tick_params(colors=TEXT)
    ax.set_xlim(rounds.min(), rounds.max())

    plt.tight_layout()
    plt.savefig(f"{OUT}/{out_name}", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  wrote {out_name}")


def make_training_curve():
    _training_curve(PPO_TRAINING_LOG,
                    "PPO training curve  (1000-epoch run)",
                    "ppo_training_curve.png")


def make_dqn_training_curve():
    _training_curve(DQN_TRAINING_LOG,
                    "DQN training curve  (1000-epoch run)",
                    "dqn_training_curve.png")


def _role_eval(role_path, out_name):
    """Render an isolated-role evaluation panel pair in unified style."""
    with open(role_path) as f:
        data = json.load(f)
    # Keep only checkpoints with numeric epochs (drop sentinel rows like "best").
    data = [d for d in data if isinstance(d.get("epoch"), (int, float))]
    epochs = [d["epoch"] for d in data]
    t_mean = [d["tagger"]["mean_tags"] for d in data]
    t_std  = [d["tagger"]["std_tags"]  for d in data]
    r_mean = [d["runner"]["mean_tags"] for d in data]
    r_std  = [d["runner"]["std_tags"]  for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_color(EDGE)
        ax.tick_params(colors=TEXT)
        ax.grid(True, alpha=0.3, color="#cccccc")

    ax = axes[0]
    ax.errorbar(epochs, t_mean, yerr=t_std, color=TAGGER, marker="o",
                capsize=4, linewidth=1.8, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Checkpoint epoch", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean tags / episode", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_title("Trained tagger vs. random runners  (higher = better)",
                 color=TEXT, fontsize=11, fontweight="bold", pad=8)

    ax = axes[1]
    ax.errorbar(epochs, r_mean, yerr=r_std, color=RUNNER, marker="o",
                capsize=4, linewidth=1.8, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Checkpoint epoch", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean tags allowed", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_title("Trained runner vs. random tagger  (lower = better)",
                 color=TEXT, fontsize=11, fontweight="bold", pad=8)

    plt.tight_layout()
    plt.savefig(f"{OUT}/{out_name}", dpi=200,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  wrote {out_name}")


def make_role_eval():
    _role_eval(PPO_ROLE_EVAL, "ppo_role_evaluation.png")


def make_dqn_role_eval():
    _role_eval(DQN_ROLE_EVAL, "dqn_role_evaluation.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    make_training_curve()
    make_dqn_training_curve()
    make_role_eval()
    make_dqn_role_eval()
    print("done.")
