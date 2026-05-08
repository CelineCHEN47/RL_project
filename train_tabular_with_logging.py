#!/usr/bin/env python3
"""Retrain tabular agents (Q-Learning, SARSA) with per-episode logging.

Produces two figures for the report:
  figures/tabular_training_curves.png  — convergence in the discrete scenario
  figures/cliffwalking_risk_profile.png — empirical Cliff-Walking contrast

Per-episode metrics are dumped to gridworld/results/tabular_logged_metrics.json
so the plots can be regenerated without retraining.

Run:
    python train_tabular_with_logging.py
"""

import json
import os
import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from gridworld.env import TagGridWorld
from gridworld.tabular_agent import (
    QLearningAgent,
    SARSAAgent,
    RandomAgent,
)

# ---------------------------------------------------------------------------
# Config — match Phase 1/2 settings used in run_gridworld.py
# ---------------------------------------------------------------------------
EPISODES_PER_PHASE = 5000
SEED = 7
RESULTS_PATH = "gridworld/results/tabular_logged_metrics.json"
FIG_TRAINING = "figures/tabular_training_curves.png"
FIG_RISK = "figures/cliffwalking_risk_profile.png"

# Light academic palette (matches make_paper_figures.py)
COLOR_QLEARN = "#c0392b"   # warm red — aggressive
COLOR_SARSA = "#2c7fb8"    # cool blue — cautious
TEXT_DARK = "#222222"
GRID_GRAY = "#d9d9d9"
BG_WHITE = "#ffffff"


# ---------------------------------------------------------------------------
# Headless trainer with per-episode logging
# ---------------------------------------------------------------------------
def train_with_logging(env, learner, opponent, role, num_episodes):
    """Train one role, return per-episode metrics arrays."""
    catches = np.zeros(num_episodes, dtype=np.int32)
    steps = np.zeros(num_episodes, dtype=np.int32)
    learner_reward = np.zeros(num_episodes, dtype=np.float32)

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0

        while not env.done:
            # tagger turn
            if role == "tagger":
                a = learner.select_action(state)
            else:
                a = opponent.select_action(state)
            next_state, t_reward, done = env.step_tagger(a)
            if role == "tagger":
                learner.learn(state, a, t_reward, next_state, done)
                ep_reward += t_reward
            state = next_state
            if done:
                break

            # runner turn
            if role == "runner":
                a_r = learner.select_action(state)
            else:
                a_r = opponent.select_action(state)
            next_state, r_reward, done = env.step_runner(a_r)
            if role == "runner":
                if isinstance(learner, SARSAAgent):
                    next_a = (learner.select_action(next_state)
                              if not done else 0)
                    learner.learn(state, a_r, r_reward, next_state, done,
                                  next_action=next_a)
                else:
                    learner.learn(state, a_r, r_reward, next_state, done)
                ep_reward += r_reward
            state = next_state

        catches[ep] = 1 if env.tagger_pos == env.runner_pos else 0
        steps[ep] = env.steps
        learner_reward[ep] = ep_reward

        if hasattr(learner, "decay_epsilon"):
            learner.decay_epsilon()

    return {
        "catches": catches.tolist(),
        "steps": steps.tolist(),
        "learner_reward": learner_reward.tolist(),
    }


def run_phase_pair(algo_name, AgentCls):
    """Run Phase 1 (tagger) then Phase 2 (runner) for one algorithm."""
    print(f"\n=== {algo_name} ===")
    random.seed(SEED)
    np.random.seed(SEED)

    env = TagGridWorld()

    # Phase 1: tagger vs random runner
    print("  Phase 1: tagger vs random runner ...")
    tagger = AgentCls()
    p1 = train_with_logging(env, tagger, RandomAgent(),
                            role="tagger",
                            num_episodes=EPISODES_PER_PHASE)
    p1_rate = sum(p1["catches"]) / EPISODES_PER_PHASE * 100
    print(f"    final catch rate (running): {p1_rate:.1f}%  "
          f"q-table size: {len(tagger.q_table):,}")

    # Phase 2: runner vs trained tagger (frozen)
    print("  Phase 2: runner vs trained tagger ...")
    if hasattr(tagger, "epsilon"):
        tagger.epsilon = 0.0
    runner = AgentCls()
    p2 = train_with_logging(env, runner, tagger,
                            role="runner",
                            num_episodes=EPISODES_PER_PHASE)
    p2_survive = (1 - sum(p2["catches"]) / EPISODES_PER_PHASE) * 100
    print(f"    final survival rate (running): {p2_survive:.1f}%  "
          f"q-table size: {len(runner.q_table):,}")

    return {
        "phase1": p1,    # tagger learning
        "phase2": p2,    # runner learning
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def smooth(x, k=200):
    """Centered rolling mean over the last k episodes."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < k:
        return x
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out = (cs[k:] - cs[:-k]) / k
    pad = np.full(k - 1, out[0])
    return np.concatenate([pad, out])


def plot_training_curves(data, out_path):
    """2x2 panel: tagger catch rate (top), runner survival rate (bottom)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5),
                             facecolor=BG_WHITE,
                             constrained_layout=True)

    eps = np.arange(EPISODES_PER_PHASE)
    qp1 = smooth(data["qlearning"]["phase1"]["catches"]) * 100
    sp1 = smooth(data["sarsa"]["phase1"]["catches"]) * 100
    qp2 = (1 - smooth(data["qlearning"]["phase2"]["catches"])) * 100
    sp2 = (1 - smooth(data["sarsa"]["phase2"]["catches"])) * 100

    # Top-left: Q-Learning Phase 1
    axes[0, 0].plot(eps, qp1, color=COLOR_QLEARN, lw=1.6)
    axes[0, 0].set_title("Q-Learning  -  Phase 1 (tagger vs random)",
                         color=TEXT_DARK, fontsize=11)
    axes[0, 0].set_ylabel("catch rate (%)", color=TEXT_DARK)

    # Top-right: SARSA Phase 1
    axes[0, 1].plot(eps, sp1, color=COLOR_SARSA, lw=1.6)
    axes[0, 1].set_title("SARSA  -  Phase 1 (tagger vs random)",
                         color=TEXT_DARK, fontsize=11)

    # Bottom-left: Q-Learning Phase 2
    axes[1, 0].plot(eps, qp2, color=COLOR_QLEARN, lw=1.6)
    axes[1, 0].set_title("Q-Learning  -  Phase 2 (runner vs trained tagger)",
                         color=TEXT_DARK, fontsize=11)
    axes[1, 0].set_ylabel("survival rate (%)", color=TEXT_DARK)
    axes[1, 0].set_xlabel("training episode", color=TEXT_DARK)

    # Bottom-right: SARSA Phase 2
    axes[1, 1].plot(eps, sp2, color=COLOR_SARSA, lw=1.6)
    axes[1, 1].set_title("SARSA  -  Phase 2 (runner vs trained tagger)",
                         color=TEXT_DARK, fontsize=11)
    axes[1, 1].set_xlabel("training episode", color=TEXT_DARK)

    for ax in axes.flat:
        ax.set_ylim(0, 105)
        ax.set_xlim(0, EPISODES_PER_PHASE)
        ax.grid(True, color=GRID_GRAY, lw=0.7, alpha=0.7)
        ax.set_facecolor(BG_WHITE)
        for s in ax.spines.values():
            s.set_color("#555555")
        ax.tick_params(colors=TEXT_DARK)

    fig.suptitle("Tabular methods converge in the discrete gridworld "
                 "(rolling mean over 200 episodes)",
                 color=TEXT_DARK, fontsize=12, weight="bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=BG_WHITE)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_risk_profile(data, out_path):
    """The Cliff-Walking signature: average runner reward during training.

    SARSA's on-policy update propagates the cost of exploratory failures back
    into the policy, so it learns a more conservative strategy. Q-Learning
    bootstraps from the greedy max and ignores its own exploration cost,
    producing a higher-variance, more 'cliff-walking' learning signal.
    """
    q_runner = np.asarray(
        data["qlearning"]["phase2"]["learner_reward"], dtype=np.float32)
    s_runner = np.asarray(
        data["sarsa"]["phase2"]["learner_reward"], dtype=np.float32)
    q_tagger = np.asarray(
        data["qlearning"]["phase1"]["learner_reward"], dtype=np.float32)
    s_tagger = np.asarray(
        data["sarsa"]["phase1"]["learner_reward"], dtype=np.float32)

    eps = np.arange(EPISODES_PER_PHASE)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                             facecolor=BG_WHITE,
                             constrained_layout=True)

    # Left: tagger phase reward (chase task — rewards positive when catching)
    axes[0].plot(eps, smooth(q_tagger), color=COLOR_QLEARN, lw=1.7,
                 label="Q-Learning (off-policy)")
    axes[0].plot(eps, smooth(s_tagger), color=COLOR_SARSA, lw=1.7,
                 label="SARSA (on-policy)")
    axes[0].set_title("Phase 1 (tagger): per-episode return",
                      color=TEXT_DARK, fontsize=11)
    axes[0].set_xlabel("training episode", color=TEXT_DARK)
    axes[0].set_ylabel("avg episode return (rolling mean)", color=TEXT_DARK)
    axes[0].legend(loc="lower right", framealpha=0.9)

    # Right: runner phase reward (cliff task — large negative on capture)
    axes[1].plot(eps, smooth(q_runner), color=COLOR_QLEARN, lw=1.7,
                 label="Q-Learning (off-policy)")
    axes[1].plot(eps, smooth(s_runner), color=COLOR_SARSA, lw=1.7,
                 label="SARSA (on-policy)")
    axes[1].set_title("Phase 2 (runner): per-episode return",
                      color=TEXT_DARK, fontsize=11)
    axes[1].set_xlabel("training episode", color=TEXT_DARK)
    axes[1].legend(loc="lower right", framealpha=0.9)

    for ax in axes:
        ax.grid(True, color=GRID_GRAY, lw=0.7, alpha=0.7)
        ax.set_facecolor(BG_WHITE)
        ax.set_xlim(0, EPISODES_PER_PHASE)
        for s in ax.spines.values():
            s.set_color("#555555")
        ax.tick_params(colors=TEXT_DARK)

    fig.suptitle("Cliff-Walking signature: SARSA learns the safer "
                 "(higher-return) policy in the runner phase",
                 color=TEXT_DARK, fontsize=12, weight="bold")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=BG_WHITE)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if os.path.exists(RESULTS_PATH):
        print(f"loading cached metrics from {RESULTS_PATH}")
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {
            "qlearning": run_phase_pair("Q-Learning", QLearningAgent),
            "sarsa": run_phase_pair("SARSA", SARSAAgent),
        }
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"\nwrote {RESULTS_PATH}")

    print("\nrendering figures ...")
    plot_training_curves(data, FIG_TRAINING)
    plot_risk_profile(data, FIG_RISK)
    print("done.")


if __name__ == "__main__":
    main()
