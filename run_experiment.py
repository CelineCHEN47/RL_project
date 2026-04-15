#!/usr/bin/env python3
"""Automated RL Experiment Pipeline.

Handles the full experiment workflow for any algorithm:
  1. Train and save checkpoints at specified epoch milestones.
  2. Load each checkpoint and run evaluation episodes (no learning).
  3. Collect quantitative metrics (tags, survival steps).
  4. Record agent trajectories for qualitative behavioral analysis.
  5. Generate plots: learning curves, performance comparison, trajectory maps.

Key flag: --display
  Without it  -> headless mode, fast training, works on remote servers.
  With it     -> opens a Pygame window so you can watch agents live.
               (Forces --sims 1; throttled to game FPS.)

Usage:
    # Headless full experiment with PPO
    python run_experiment.py --algorithm PPO

    # Watch training live on your local machine
    python run_experiment.py --algorithm PPO --display

    # Evaluate only (skip training, use existing checkpoints)
    python run_experiment.py --algorithm PPO --eval-only

    # Custom epochs and algorithm
    python run_experiment.py --algorithm DQN --epochs 10 50 100 200 500 1000

    # Display only the evaluation phase (watch trained agents)
    python run_experiment.py --algorithm Q-Learning --eval-only --display
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
DEFAULT_EPOCHS   = [10, 50, 100, 200, 500, 1000]
STEPS_PER_ROUND  = 2000
PARALLEL_SIMS    = 2
EVAL_EPISODES    = 20
EVAL_STEPS       = 2000
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
        pygame.display.set_caption("RL Experiment — Training")
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
        sys.exit(1)
    module_path, class_name = config.RL_ALGORITHMS[algo_name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def checkpoint_path(save_dir: str, algo_name: str, epoch: int) -> str:
    safe_name = algo_name.lower().replace("-", "_").replace(" ", "_")
    return os.path.join(save_dir, "checkpoints", f"{safe_name}_epoch_{epoch}.pt")


def get_save_dir(algo_name: str) -> str:
    safe_name = algo_name.lower().replace("-", "_").replace(" ", "_")
    return os.path.join("experiments", safe_name)


# ======================================================================
# Renderer wrapper — only active when --display is on
# ======================================================================
class DisplayManager:
    """Thin wrapper around the project's Renderer for the experiment script."""

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

        self.font = self._safe_sys_font("Consolas", 14)
        self.hud_lines: list[str] = []

    @staticmethod
    def _safe_sys_font(name: str, size: int, bold: bool = False):
        import pygame
        try:
            return pygame.font.SysFont(name, size, bold=bold)
        except Exception:
            font = pygame.font.Font(None, size)
            font.set_bold(bold)
            return font

    def set_hud(self, lines: list[str]):
        self.hud_lines = lines

    def render_sim(self, sim):
        """Render one frame. Returns 'skip' if user pressed ESC."""
        pygame = self.pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "skip"

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
def train_with_checkpoints(algo_class, algo_name: str, epochs: list[int],
                           save_dir: str,
                           display_mgr: DisplayManager | None,
                           parallel_sims: int):
    import config
    from game.simulation import HeadlessSimulation

    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Create first simulation (sets up shared dual-role models)
    first_sim = HeadlessSimulation(algo_class)
    shared_tagger, shared_runner = first_sim.get_shared_algos()
    num_agents = len(first_sim.agents)

    if display_mgr:
        parallel_sims = 1

    sims = [first_sim]
    for _ in range(parallel_sims - 1):
        sim = HeadlessSimulation(algo_class,
                                 shared_tagger=shared_tagger,
                                 shared_runner=shared_runner)
        sims.append(sim)

    max_epoch = max(epochs)
    epoch_set = set(epochs)
    training_log = []

    role_mode = "Dual (tagger + runner)" if config.DUAL_ROLE_ENABLED else "Single"

    print("=" * 65)
    print(f"  PHASE 1: Training {algo_name}  (max {max_epoch} rounds)")
    print(f"  Role mode: {role_mode}")
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
                    f"TRAINING {algo_name} — Round {round_num}/{max_epoch}  "
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
            path = checkpoint_path(save_dir, algo_name, round_num)
            first_sim.agents[0].algorithm.save(path)
            print(f"  >> Checkpoint saved: {path}")

        if skip:
            for ep in sorted(epochs):
                if ep > round_num:
                    path = checkpoint_path(save_dir, algo_name, ep)
                    first_sim.agents[0].algorithm.save(path)
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
def evaluate_checkpoint(algo_class, algo_name: str, ckpt_path: str,
                        num_episodes: int, steps_per_episode: int,
                        display_mgr: DisplayManager | None,
                        epoch_label: str = ""):
    from game.simulation import HeadlessSimulation
    import numpy as np

    sim = HeadlessSimulation(algo_class)
    num_agents = len(sim.agents)

    # Load checkpoint — DualRoleAlgorithm.load handles _tagger/_runner paths
    # sim.agents[0].algorithm.load(ckpt_path)
    for agent in sim.agents:
        agent.algorithm.load(ckpt_path)

    # Freeze policy updates during evaluation while keeping dynamics intact.
    def _eval_no_learn(*_args, **_kwargs):
        return None

    for agent in sim.agents:
        agent.learn = _eval_no_learn

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
                    f"EVAL {algo_name} — Epoch {epoch_label}  "
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


def run_evaluations(algo_class, algo_name: str, epochs: list[int],
                    save_dir: str, display_mgr: DisplayManager | None,
                    eval_episodes: int):
    results_dir = os.path.join(save_dir, "results")
    traj_dir = os.path.join(save_dir, "trajectories")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    print("=" * 65)
    print(f"  PHASE 2: Evaluating {algo_name} checkpoints")
    print(f"  Epochs: {sorted(epochs)}")
    print(f"  Episodes per checkpoint: {eval_episodes}")
    print(f"  Steps per episode: {EVAL_STEPS}")
    if display_mgr:
        print(f"  Display: ON  (ESC = skip current epoch)")
    print("=" * 65)

    all_metrics = []

    for epoch in sorted(epochs):
        ckpt = checkpoint_path(save_dir, algo_name, epoch)
        # Check for both single and dual-role checkpoint files
        from rl.dual_role import dual_model_exists
        if not os.path.exists(ckpt) and not dual_model_exists(ckpt):
            print(f"  [SKIP] Not found: {ckpt}")
            continue

        print(f"\n  Evaluating epoch {epoch}...")
        if display_mgr:
            import pygame
            pygame.display.set_caption(
                f"{algo_name} Experiment — Eval Epoch {epoch}"
            )

        metrics, trajectories = evaluate_checkpoint(
            algo_class, algo_name, ckpt, eval_episodes, EVAL_STEPS,
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
# Phase 2b: Isolated Role Evaluation
# ======================================================================
def evaluate_role_isolated(algo_class, algo_name: str, ckpt_path: str,
                           role: str, num_episodes: int,
                           steps_per_episode: int,
                           display_mgr: DisplayManager | None,
                           epoch_label: str = ""):
    """Evaluate one role in isolation.

    Args:
        role: "tagger" or "runner"
            - "tagger": trained tagger vs untrained runners
              Metric: more tags = better tagger
            - "runner": trained runners vs untrained tagger
              Metric: fewer tags = better runners
    """
    from game.simulation import HeadlessSimulation
    from rl.dual_role import DualRoleAlgorithm
    import numpy as np

    sim = HeadlessSimulation(algo_class)
    num_agents = len(sim.agents)

    # Load only the role being tested
    for agent in sim.agents:
        if isinstance(agent.algorithm, DualRoleAlgorithm):
            if role == "tagger":
                agent.algorithm.load_tagger_only(ckpt_path)
            else:
                agent.algorithm.load_runner_only(ckpt_path)

    # No learning during evaluation
    def _no_learn(*_args, **_kwargs):
        return None
    for agent in sim.agents:
        agent.learn = _no_learn

    episode_metrics = []

    for ep in range(num_episodes):
        sim.reset()
        ep_tags = 0
        steps_between_tags = []
        steps_since_last = 0

        skip = False
        for step in range(steps_per_episode):
            tag_event = sim.step()
            steps_since_last += 1
            if tag_event:
                ep_tags += 1
                steps_between_tags.append(steps_since_last)
                steps_since_last = 0
                if display_mgr:
                    for e in sim.entities:
                        if e.entity_id == tag_event.get("tagged_id"):
                            display_mgr.renderer.emit_tag_particles(e.x, e.y)

            if display_mgr:
                role_label = "TAGGER" if role == "tagger" else "RUNNER"
                display_mgr.set_hud([
                    f"ROLE EVAL ({role_label}) — Epoch {epoch_label}  "
                    f"Ep {ep+1}/{num_episodes}  "
                    f"Step {step+1}/{steps_per_episode}",
                    f"Tags: {ep_tags}  |  "
                    f"Trained: {role_label}, Opponent: untrained",
                    "ESC = skip",
                ])
                if display_mgr.render_sim(sim) == "skip":
                    skip = True
                    break

        avg_steps_between = (float(np.mean(steps_between_tags))
                             if steps_between_tags else float(steps_per_episode))
        episode_metrics.append({
            "episode": ep,
            "tags": ep_tags,
            "avg_steps_between_tags": avg_steps_between,
        })
        if skip:
            break

    tags_list = [m["tags"] for m in episode_metrics]
    avg_gaps = [m["avg_steps_between_tags"] for m in episode_metrics]

    return {
        "role": role,
        "epoch": epoch_label,
        "num_episodes": len(episode_metrics),
        "mean_tags": float(np.mean(tags_list)) if tags_list else 0,
        "std_tags": float(np.std(tags_list)) if tags_list else 0,
        "mean_steps_between_tags": float(np.mean(avg_gaps)) if avg_gaps else 0,
    }


def run_role_evaluations(algo_class, algo_name: str, epochs: list[int],
                         save_dir: str, display_mgr: DisplayManager | None,
                         eval_episodes: int):
    """Run isolated tagger and runner evaluations for each checkpoint."""
    from rl.dual_role import dual_model_exists
    import config

    if not config.DUAL_ROLE_ENABLED:
        print("  [SKIP] Role evaluation requires DUAL_ROLE_ENABLED=True")
        return []

    results_dir = os.path.join(save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 65)
    print(f"  PHASE 2b: Isolated Role Evaluation ({algo_name})")
    print(f"  Epochs: {sorted(epochs)}")
    print(f"  Episodes per test: {eval_episodes}")
    print(f"  Tests per epoch:")
    print(f"    - Trained TAGGER vs untrained runners (more tags = better)")
    print(f"    - Trained RUNNERS vs untrained tagger (fewer tags = better)")
    print("=" * 65)

    all_role_metrics = []

    for epoch in sorted(epochs):
        ckpt = checkpoint_path(save_dir, algo_name, epoch)
        if not dual_model_exists(ckpt):
            print(f"  [SKIP] No dual checkpoint for epoch {epoch}")
            continue

        print(f"\n  Epoch {epoch}:")

        # Test 1: Trained tagger vs untrained runners
        if display_mgr:
            import pygame
            pygame.display.set_caption(
                f"{algo_name} — Tagger Eval Epoch {epoch}")

        tagger_metrics = evaluate_role_isolated(
            algo_class, algo_name, ckpt, "tagger",
            eval_episodes, EVAL_STEPS, display_mgr, str(epoch))
        tagger_metrics["epoch"] = epoch

        print(f"    Trained TAGGER vs random runners: "
              f"{tagger_metrics['mean_tags']:.1f} "
              f"± {tagger_metrics['std_tags']:.1f} tags  "
              f"(avg {tagger_metrics['mean_steps_between_tags']:.0f} "
              f"steps/tag)")

        # Test 2: Trained runners vs untrained tagger
        if display_mgr:
            import pygame
            pygame.display.set_caption(
                f"{algo_name} — Runner Eval Epoch {epoch}")

        runner_metrics = evaluate_role_isolated(
            algo_class, algo_name, ckpt, "runner",
            eval_episodes, EVAL_STEPS, display_mgr, str(epoch))
        runner_metrics["epoch"] = epoch

        print(f"    Trained RUNNERS vs random tagger: "
              f"{runner_metrics['mean_tags']:.1f} "
              f"± {runner_metrics['std_tags']:.1f} tags  "
              f"(avg {runner_metrics['mean_steps_between_tags']:.0f} "
              f"steps/tag)")

        all_role_metrics.append({
            "epoch": epoch,
            "tagger": tagger_metrics,
            "runner": runner_metrics,
        })

    metrics_path = os.path.join(results_dir, "role_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_role_metrics, f, indent=2)
    print(f"\n  Role metrics saved: {metrics_path}")

    return all_role_metrics


# ======================================================================
# Phase 3: Generate Plots
# ======================================================================
def generate_plots(algo_name: str, save_dir: str, training_log: list | None,
                   eval_metrics: list, role_metrics: list | None = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping plots.")
        print("  Install with: pip install matplotlib")
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
        ax.set_title(f"{algo_name} Training Curve")
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
        ax.set_title(f"{algo_name} Performance by Training Epoch")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "epoch_comparison.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: {plots_dir}/epoch_comparison.png")

    # --- Role isolation comparison ---
    if role_metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{algo_name} — Isolated Role Performance", fontsize=14)

        ep_list = [m["epoch"] for m in role_metrics]
        x = range(len(ep_list))

        # Tagger performance (more tags = better)
        tagger_means = [m["tagger"]["mean_tags"] for m in role_metrics]
        tagger_stds = [m["tagger"]["std_tags"] for m in role_metrics]
        ax1.bar(x, tagger_means, yerr=tagger_stds, capsize=5,
                color=(0.85, 0.3, 0.3), edgecolor="black", alpha=0.8)
        ax1.set_xticks(list(x))
        ax1.set_xticklabels([f"Epoch\n{e}" for e in ep_list])
        ax1.set_ylabel("Mean Tags (higher = better tagger)")
        ax1.set_title("Trained Tagger vs Untrained Runners")
        ax1.grid(True, axis="y", alpha=0.3)

        # Runner performance (fewer tags = better)
        runner_means = [m["runner"]["mean_tags"] for m in role_metrics]
        runner_stds = [m["runner"]["std_tags"] for m in role_metrics]
        ax2.bar(x, runner_means, yerr=runner_stds, capsize=5,
                color=(0.3, 0.5, 0.85), edgecolor="black", alpha=0.8)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels([f"Epoch\n{e}" for e in ep_list])
        ax2.set_ylabel("Mean Tags (lower = better runners)")
        ax2.set_title("Untrained Tagger vs Trained Runners")
        ax2.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "role_evaluation.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved: {plots_dir}/role_evaluation.png")

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
        fig.suptitle(f"{algo_name} Trajectories — Epoch {epoch_label}",
                     fontsize=14)
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
            axes[0].scatter(xs[-1], ys[-1], color=c, marker="x", s=60,
                            zorder=5)

        axes[0].set_title("Trajectories (o=start, x=end)")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
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
            axes[1].set_xlabel("X")
            axes[1].set_ylabel("Y")
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
        description="Automated RL Experiment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--algorithm", "-a", type=str, default="PPO",
                        help="Algorithm to experiment with (default: PPO)")
    parser.add_argument("--epochs", nargs="+", type=int,
                        default=DEFAULT_EPOCHS,
                        help=f"Epoch milestones (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; evaluate existing checkpoints")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output directory (default: experiments/<algo>)")
    parser.add_argument("--eval_episodes", type=int, default=EVAL_EPISODES,
                        help=f"Episodes per eval (default: {EVAL_EPISODES})")
    parser.add_argument("--sims", type=int, default=PARALLEL_SIMS,
                        help=f"Parallel sims for training (default: "
                             f"{PARALLEL_SIMS})")
    parser.add_argument("--display", action="store_true",
                        help="Open Pygame window to watch live "
                             "(local only; forces --sims 1)")
    args = parser.parse_args()

    algo_name = args.algorithm
    save_dir = args.save_dir or get_save_dir(algo_name)

    # Init Pygame
    screen, clock = init_pygame(args.display)

    algo_class = get_algo_class(algo_name)

    display_mgr = None
    if args.display and screen is not None:
        display_mgr = DisplayManager(screen, clock)

    # Phase 1: Train
    training_log = None
    if not args.eval_only:
        training_log = train_with_checkpoints(
            algo_class, algo_name, args.epochs, save_dir,
            display_mgr, args.sims,
        )
    else:
        log_path = os.path.join(save_dir, "training_log.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                training_log = json.load(f)

    # Phase 2: Evaluate (both models loaded)
    eval_metrics = run_evaluations(
        algo_class, algo_name, args.epochs, save_dir,
        display_mgr, args.eval_episodes,
    )

    # Phase 2b: Isolated role evaluation
    role_metrics = run_role_evaluations(
        algo_class, algo_name, args.epochs, save_dir,
        display_mgr, args.eval_episodes,
    )

    # Phase 3: Plot
    print("\n" + "=" * 65)
    print("  PHASE 3: Generating plots")
    print("=" * 65)
    generate_plots(algo_name, save_dir, training_log, eval_metrics,
                   role_metrics)

    print("\n" + "=" * 65)
    print("  EXPERIMENT COMPLETE")
    print("=" * 65)
    print(f"  Algorithm:     {algo_name}")
    print(f"  Output:        {save_dir}/")
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
