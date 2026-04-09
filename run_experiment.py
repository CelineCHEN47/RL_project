#!/usr/bin/env python3
"""Automated DPO Experiment Pipeline.

Handles the full experiment workflow:
  1. Train DPO and save checkpoints at specified epoch milestones.
  2. Load each checkpoint and run evaluation episodes (no learning).
  3. Collect quantitative metrics (tags, survival steps).
  4. Record agent trajectories for qualitative behavioral analysis.
  5. Generate plots: learning curves, performance comparison, trajectory maps.

Key flag: --display
  Without it  → headless mode, fast training, works on remote servers.
  With it     → opens a Pygame window so you can watch agents live.
               (Forces --sims 1; throttled to game FPS.)

Usage:
    # Headless full experiment
    python run_dpo_experiment.py

    # Watch training live on your local machine
    python run_dpo_experiment.py --display

    # Evaluate only (skip training, use existing checkpoints)
    python run_dpo_experiment.py --eval-only

    # Custom epochs
    python run_dpo_experiment.py --epochs 10 50 100 200 500 1000

    # Display only the evaluation phase (watch trained agents)
    python run_dpo_experiment.py --eval-only --display
"""

import argparse
import importlib
import json
import os
import sys
import time

# ======================================================================
# Default experiment configuration
# ======================================================================
ALGORITHM_NAME   = "DPO"
DEFAULT_EPOCHS   = [10, 50, 100, 200, 500, 1000]
STEPS_PER_ROUND  = 1000
PARALLEL_SIMS    = 2
EVAL_EPISODES    = 20
EVAL_STEPS       = 2000
SAVE_DIR         = "experiments/dpo"
LOG_INTERVAL     = 5


# ======================================================================
# Pygame init — deferred so --display flag is parsed first
# ======================================================================
def init_pygame(display: bool):
    """Initialize Pygame in either real or dummy display mode."""
    if not display:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    import pygame
    pygame.init()

    if display:
        import config
        screen = pygame.display.set_mode(
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        )
        pygame.display.set_caption("DPO Experiment — Training")
        clock = pygame.time.Clock()
        return screen, clock
    else:
        pygame.display.set_mode((1, 1))
        return None, None


# ======================================================================
# Helpers
# ======================================================================
def get_algo_class(algo_name: str):
    import config
    if algo_name not in config.RL_ALGORITHMS:
        print(f"Error: '{algo_name}' not in config.RL_ALGORITHMS.")
        print(f"Available: {list(config.RL_ALGORITHMS.keys())}")
        print(f"\nMake sure you added this to config.py:\n"
              f'  "{algo_name}": ("rl.dpo", "DPO"),')
        sys.exit(1)
    module_path, class_name = config.RL_ALGORITHMS[algo_name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def checkpoint_path(save_dir: str, epoch: int) -> str:
    return os.path.join(save_dir, "checkpoints", f"dpo_epoch_{epoch}.pt")


# ======================================================================
# Renderer wrapper — only active when --display is on
# ======================================================================
class DisplayManager:
    """Thin wrapper around the project's Renderer for the experiment script.
    Created only when --display is passed."""

    def __init__(self, screen, clock):
        import pygame
        import config
        from rendering.renderer import Renderer

        self.screen = screen
        self.clock = clock
        self.renderer = Renderer(screen)
        self.config = config
        self.pygame = pygame
        self.tick = 0

        self.font = pygame.font.SysFont("Consolas", 14)
        self.hud_lines: list[str] = []

    def set_hud(self, lines: list[str]):
        """Set overlay text lines shown in top-left corner."""
        self.hud_lines = lines

    def render_sim(self, sim):
        """Render one frame of a HeadlessSimulation.
        Returns 'skip' if user pressed ESC, else None."""
        pygame = self.pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "skip"

        # Camera follows the tagger
        for e in sim.entities:
            if e.is_tagger:
                self.renderer.set_camera(e.x, e.y)
                break

        lw, lh = sim.level.get_pixel_dimensions()
        self.renderer.clamp_camera(lw, lh)

        self.renderer.draw_level(sim.level)
        self.renderer.draw_movable_objects(sim.movable_objects)
        self.renderer.draw_entities(
            sim.entities, sim.tag_logic.current_tagger_id, self.tick
        )
        self.renderer.update_particles()
        self.renderer.draw_particles()

        # HUD overlay
        y = 8
        for line in self.hud_lines:
            surf = self.font.render(line, True, (255, 255, 255))
            bg = pygame.Surface(
                (surf.get_width() + 10, surf.get_height() + 4), pygame.SRCALPHA
            )
            bg.fill((0, 0, 0, 160))
            self.screen.blit(bg, (4, y - 2))
            self.screen.blit(surf, (8, y))
            y += surf.get_height() + 4

        pygame.display.flip()
        self.clock.tick(self.config.FPS)
        self.tick += 1
        return None


# ======================================================================
# Phase 1: Training with periodic checkpoint saving
# ======================================================================
def train_with_checkpoints(algo_class, epochs: list[int], save_dir: str,
                           display_mgr: DisplayManager | None,
                           parallel_sims: int):
    from game.simulation import HeadlessSimulation

    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Create shared algorithm instances
    sample_sim = HeadlessSimulation(algo_class)
    num_agents = len(sample_sim.agents)
    shared_algorithms = [agent.algorithm for agent in sample_sim.agents]
    del sample_sim

    # Display mode forces 1 sim (can only render one)
    if display_mgr:
        parallel_sims = 1

    sims = []
    for _ in range(parallel_sims):
        sim = HeadlessSimulation(algo_class, shared_algorithms=shared_algorithms)
        sims.append(sim)

    max_epoch = max(epochs)
    epoch_set = set(epochs)
    training_log = []

    print("=" * 65)
    print(f"  PHASE 1: Training DPO  (max {max_epoch} rounds)")
    print(f"  Checkpoints at: {sorted(epochs)}")
    print(f"  Parallel sims: {parallel_sims}, Steps/round: {STEPS_PER_ROUND}")
    print(f"  Agents per sim: {num_agents}")
    if display_mgr:
        print(f"  Display: ON  (ESC = skip to evaluation)")
    print("=" * 65)

    total_tags = 0
    t_start = time.time()

    for round_num in range(1, max_epoch + 1):
        round_tags = 0

        for sim in sims:
            sim.reset()
            sim.total_tags = 0

        skip = False
        for step in range(STEPS_PER_ROUND):
            for sim in sims:
                tag_event = sim.step()
                if tag_event:
                    round_tags += 1
                    if display_mgr:
                        for e in sim.entities:
                            if e.entity_id == tag_event.get("tagged_id"):
                                display_mgr.renderer.emit_tag_particles(
                                    e.x, e.y
                                )

            if display_mgr:
                next_ckpt = min(
                    (e for e in sorted(epochs) if e >= round_num), default=None
                )
                display_mgr.set_hud([
                    f"TRAINING — Round {round_num}/{max_epoch}  "
                    f"Step {step+1}/{STEPS_PER_ROUND}",
                    f"Round tags: {round_tags}  |  "
                    f"Total tags: {total_tags + round_tags}",
                    f"Next checkpoint: epoch {next_ckpt or '—'}",
                    "ESC = skip to evaluation",
                ])
                if display_mgr.render_sim(sims[0]) == "skip":
                    skip = True
                    break

        total_tags += round_tags
        training_log.append({
            "round": round_num,
            "round_tags": round_tags,
            "total_tags": total_tags,
        })

        if round_num % LOG_INTERVAL == 0 or round_num == 1:
            elapsed = time.time() - t_start
            print(f"  Round {round_num:>5}/{max_epoch}  |  "
                  f"Tags: {round_tags:>4}  |  "
                  f"Total: {total_tags:>6}  |  "
                  f"Elapsed: {elapsed:.1f}s")

        if round_num in epoch_set:
            path = checkpoint_path(save_dir, round_num)
            for algo in shared_algorithms:
                algo.save(path)
            print(f"  >> Checkpoint saved: {path}")

        if skip:
            for ep in sorted(epochs):
                if ep > round_num:
                    path = checkpoint_path(save_dir, ep)
                    for algo in shared_algorithms:
                        algo.save(path)
                    print(f"  >> Early-stop checkpoint: {path} "
                          f"(actual: {round_num} rounds)")
            break

    total_time = time.time() - t_start
    print(f"\n  Training complete. Total time: {total_time:.1f}s\n")

    log_path = os.path.join(save_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"  Training log saved: {log_path}")

    return training_log


# ======================================================================
# Phase 2: Evaluation + Trajectory Recording
# ======================================================================
def evaluate_checkpoint(algo_class, ckpt_path: str, num_episodes: int,
                        steps_per_episode: int,
                        display_mgr: DisplayManager | None,
                        epoch_label: str = ""):
    from game.simulation import HeadlessSimulation
    import numpy as np

    sim = HeadlessSimulation(algo_class)
    num_agents = len(sim.agents)

    for agent in sim.agents:
        agent.algorithm.load(ckpt_path)

    episode_metrics = []
    all_trajectories = []

    for ep in range(num_episodes):
        sim.reset()
        ep_tags = 0
        traj = {i: [] for i in range(num_agents)}

        skip = False
        for step in range(steps_per_episode):
            for i, agent in enumerate(sim.agents):
                traj[i].append((
                    float(agent.rect.centerx),
                    float(agent.rect.centery),
                    agent.is_tagger,
                ))

            tag_event = sim.step()
            if tag_event:
                ep_tags += 1
                if display_mgr:
                    for e in sim.entities:
                        if e.entity_id == tag_event.get("tagged_id"):
                            display_mgr.renderer.emit_tag_particles(e.x, e.y)

            if display_mgr:
                display_mgr.set_hud([
                    f"EVAL — Epoch {epoch_label}  "
                    f"Episode {ep+1}/{num_episodes}  "
                    f"Step {step+1}/{steps_per_episode}",
                    f"Episode tags: {ep_tags}",
                    "ESC = skip to next epoch",
                ])
                if display_mgr.render_sim(sim) == "skip":
                    skip = True
                    break

        episode_metrics.append({"episode": ep, "tags": ep_tags})
        all_trajectories.append(traj)
        if skip:
            break

    tags_list = [m["tags"] for m in episode_metrics]
    metrics = {
        "checkpoint": ckpt_path,
        "num_episodes": len(episode_metrics),
        "mean_tags": float(np.mean(tags_list)) if tags_list else 0,
        "std_tags": float(np.std(tags_list)) if tags_list else 0,
        "min_tags": int(np.min(tags_list)) if tags_list else 0,
        "max_tags": int(np.max(tags_list)) if tags_list else 0,
        "total_tags": int(np.sum(tags_list)) if tags_list else 0,
    }
    return metrics, all_trajectories


def run_evaluations(algo_class, epochs: list[int], save_dir: str,
                    display_mgr: DisplayManager | None,
                    eval_episodes: int):
    results_dir = os.path.join(save_dir, "results")
    traj_dir = os.path.join(save_dir, "trajectories")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    print("=" * 65)
    print(f"  PHASE 2: Evaluating checkpoints")
    print(f"  Epochs: {sorted(epochs)}")
    print(f"  Episodes per checkpoint: {eval_episodes}")
    print(f"  Steps per episode: {EVAL_STEPS}")
    if display_mgr:
        print(f"  Display: ON  (ESC = skip current epoch)")
    print("=" * 65)

    all_metrics = []

    for epoch in sorted(epochs):
        ckpt = checkpoint_path(save_dir, epoch)
        if not os.path.exists(ckpt):
            print(f"  [SKIP] Not found: {ckpt}")
            continue

        print(f"\n  Evaluating epoch {epoch}...")
        if display_mgr:
            import pygame
            pygame.display.set_caption(
                f"DPO Experiment — Eval Epoch {epoch}"
            )

        metrics, trajectories = evaluate_checkpoint(
            algo_class, ckpt, eval_episodes, EVAL_STEPS,
            display_mgr, epoch_label=str(epoch),
        )
        metrics["epoch"] = epoch
        all_metrics.append(metrics)

        print(f"    Tags: {metrics['mean_tags']:.1f} "
              f"± {metrics['std_tags']:.1f}  "
              f"(min={metrics['min_tags']}, max={metrics['max_tags']})")

        traj_path = os.path.join(traj_dir, f"traj_epoch_{epoch}.json")
        serializable = []
        for ep_traj in trajectories[:3]:
            serializable.append({str(k): v for k, v in ep_traj.items()})
        with open(traj_path, "w") as f:
            json.dump(serializable, f)

    metrics_path = os.path.join(results_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Metrics saved: {metrics_path}")

    return all_metrics


# ======================================================================
# Phase 3: Generate Plots
# ======================================================================
def generate_plots(save_dir: str, training_log: list | None,
                   eval_metrics: list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping plots.")
        return

    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- Training curve ---
    if training_log:
        fig, ax = plt.subplots(figsize=(10, 5))
        rounds = [e["round"] for e in training_log]
        tags = [e["round_tags"] for e in training_log]
        window = max(1, len(tags) // 20)
        smoothed = np.convolve(tags, np.ones(window) / window, mode="valid")

        ax.plot(rounds, tags, alpha=0.3, color="steelblue", label="Raw")
        ax.plot(rounds[window - 1:], smoothed, color="steelblue",
                linewidth=2, label=f"Smoothed (w={window})")
        ax.set_xlabel("Training Round")
        ax.set_ylabel("Tags per Round")
        ax.set_title("DPO Training Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "training_curve.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: {plots_dir}/training_curve.png")

    # --- Epoch comparison ---
    if eval_metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        ep_list = [m["epoch"] for m in eval_metrics]
        means = [m["mean_tags"] for m in eval_metrics]
        stds = [m["std_tags"] for m in eval_metrics]

        ax.bar(range(len(ep_list)), means, yerr=stds, capsize=5,
               color="coral", edgecolor="black", alpha=0.8)
        ax.set_xticks(range(len(ep_list)))
        ax.set_xticklabels([f"Epoch\n{e}" for e in ep_list])
        ax.set_ylabel("Mean Tags per Episode")
        ax.set_title("DPO Performance by Training Epoch")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "epoch_comparison.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: {plots_dir}/epoch_comparison.png")

    # --- Trajectory + heatmap per epoch ---
    traj_dir = os.path.join(save_dir, "trajectories")
    if not os.path.exists(traj_dir):
        return
    for fname in sorted(os.listdir(traj_dir)):
        if not fname.endswith(".json"):
            continue
        epoch_label = fname.replace("traj_epoch_", "").replace(".json", "")
        with open(os.path.join(traj_dir, fname)) as f:
            episodes = json.load(f)
        if not episodes:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Agent Trajectories — Epoch {epoch_label}", fontsize=14)
        colors = plt.cm.tab10.colors

        ep = episodes[0]
        for aidx_str, positions in ep.items():
            idx = int(aidx_str)
            if not positions:
                continue
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            c = colors[idx % len(colors)]
            axes[0].plot(xs, ys, color=c, alpha=0.5, linewidth=0.8,
                         label=f"Agent {idx}")
            axes[0].scatter(xs[0], ys[0], color=c, marker="o", s=60, zorder=5)
            axes[0].scatter(xs[-1], ys[-1], color=c, marker="x", s=60, zorder=5)

        axes[0].set_title("Trajectories (○=start, ×=end)")
        axes[0].set_xlabel("X");  axes[0].set_ylabel("Y")
        axes[0].invert_yaxis()
        axes[0].legend(fontsize=8)
        axes[0].set_aspect("equal")

        all_xs, all_ys = [], []
        for ep_data in episodes:
            for positions in ep_data.values():
                all_xs.extend(p[0] for p in positions)
                all_ys.extend(p[1] for p in positions)
        if all_xs:
            axes[1].hist2d(all_xs, all_ys, bins=30, cmap="hot")
            axes[1].set_title("Position Heatmap")
            axes[1].set_xlabel("X");  axes[1].set_ylabel("Y")
            axes[1].invert_yaxis()

        fig.tight_layout()
        fig.savefig(
            os.path.join(plots_dir, f"trajectory_epoch_{epoch_label}.png"),
            dpi=150,
        )
        plt.close(fig)
        print(f"  Saved: {plots_dir}/trajectory_epoch_{epoch_label}.png")


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Automated DPO Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--epochs", nargs="+", type=int,
                        default=DEFAULT_EPOCHS,
                        help=f"Epoch milestones (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; evaluate existing checkpoints")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help=f"Output directory (default: {SAVE_DIR})")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES,
                        help=f"Episodes per eval (default: {EVAL_EPISODES})")
    parser.add_argument("--sims", type=int, default=PARALLEL_SIMS,
                        help=f"Parallel sims for training (default: "
                             f"{PARALLEL_SIMS})")
    parser.add_argument("--display", action="store_true",
                        help="Open Pygame window to watch live "
                             "(local only; forces --sims 1)")
    args = parser.parse_args()

    # --- Init Pygame ---
    screen, clock = init_pygame(args.display)

    algo_class = get_algo_class(ALGORITHM_NAME)

    display_mgr = None
    if args.display and screen is not None:
        display_mgr = DisplayManager(screen, clock)

    # Phase 1
    training_log = None
    if not args.eval_only:
        training_log = train_with_checkpoints(
            algo_class, args.epochs, args.save_dir,
            display_mgr, args.sims,
        )
    else:
        log_path = os.path.join(args.save_dir, "training_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                training_log = json.load(f)

    # Phase 2
    eval_metrics = run_evaluations(
        algo_class, args.epochs, args.save_dir,
        display_mgr, args.eval_episodes,
    )

    # Phase 3
    print("\n" + "=" * 65)
    print("  PHASE 3: Generating plots")
    print("=" * 65)
    generate_plots(args.save_dir, training_log, eval_metrics)

    print("\n" + "=" * 65)
    print("  EXPERIMENT COMPLETE")
    print("=" * 65)
    print(f"  Output: {args.save_dir}/")
    print(f"    checkpoints/     Model files at each epoch")
    print(f"    results/         eval_metrics.json")
    print(f"    trajectories/    Agent position logs")
    print(f"    plots/           All figures")
    print(f"    training_log.json")
    print("=" * 65)

    if args.display:
        import pygame
        pygame.quit()


if __name__ == "__main__":
    main()
