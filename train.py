#!/usr/bin/env python3
"""Universal training script for Tag RL algorithms.

Usage examples:
    # Train PPO for 100 rounds, 1 simulation
    python train.py --algorithm PPO --rounds 100

    # Train Q-Learning with 4 parallel simulations, 500 rounds
    python train.py --algorithm Q-Learning --rounds 500 --sims 4

    # Train with custom steps per round and save path
    python train.py --algorithm PPO --rounds 200 --steps-per-round 2000 --save-dir models/experiment1

    # Resume training from saved models
    python train.py --algorithm PPO --rounds 100 --load saved_models/ppo_model.pt

    # List available algorithms
    python train.py --list-algorithms
"""

import argparse
import importlib
import os
import sys
import time

# Initialize pygame in headless mode (no display needed)
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
pygame.init()
pygame.display.set_mode((1, 1))

import config
from game.simulation import HeadlessSimulation
from rl.dual_role import DualRoleAlgorithm


def get_algo_class(algo_name: str):
    """Load algorithm class by display name."""
    if algo_name not in config.RL_ALGORITHMS:
        print(f"Error: Unknown algorithm '{algo_name}'")
        print(f"Available: {', '.join(config.RL_ALGORITHMS.keys())}")
        sys.exit(1)

    module_path, class_name = config.RL_ALGORITHMS[algo_name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_model_path(save_dir: str, algo_name: str) -> str:
    """Standard model file path for an algorithm."""
    safe_name = algo_name.lower().replace("-", "_").replace(" ", "_")
    return os.path.join(save_dir, f"{safe_name}_model.pt")


def train(args):
    algo_class = get_algo_class(args.algorithm)
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = get_model_path(args.save_dir, args.algorithm)

    # Create first simulation to extract shared models
    first_sim = HeadlessSimulation(algo_class)
    shared_tagger, shared_runner = first_sim.get_shared_algos()

    # Load existing models if requested
    if args.load and config.DUAL_ROLE_ENABLED:
        print(f"Loading models from: {args.load}")
        # DualRoleAlgorithm.load handles _tagger/_runner paths
        first_sim.agents[0].algorithm.load(args.load)
    elif args.load:
        print(f"Loading model from: {args.load}")
        for agent in first_sim.agents:
            agent.algorithm.load(args.load)

    # Create parallel simulations sharing the same model weights
    sims = [first_sim]
    for _ in range(args.sims - 1):
        sim = HeadlessSimulation(algo_class,
                                 shared_tagger=shared_tagger,
                                 shared_runner=shared_runner)
        sims.append(sim)

    num_agents = len(first_sim.agents)
    role_mode = "Dual (tagger + runner)" if config.DUAL_ROLE_ENABLED else "Single"

    print("=" * 60)
    print(f"  TAG RL Training")
    print("=" * 60)
    print(f"  Algorithm:        {args.algorithm}")
    print(f"  Role mode:        {role_mode}")
    print(f"  Rounds:           {args.rounds}")
    print(f"  Steps/round:      {args.steps_per_round}")
    print(f"  Parallel sims:    {args.sims}")
    print(f"  Agents per sim:   {num_agents}")
    print(f"  Save directory:   {args.save_dir}")
    print(f"  Model path:       {model_path}")
    print("=" * 60)
    print()

    total_tags = 0
    total_steps = 0
    training_start = time.time()

    for round_num in range(1, args.rounds + 1):
        round_start = time.time()
        round_tags = 0

        for sim in sims:
            sim.reset()
            sim.total_tags = 0

        for step in range(args.steps_per_round):
            for sim in sims:
                tag_event = sim.step()
                if tag_event:
                    round_tags += 1

        total_tags += round_tags
        total_steps += args.steps_per_round * args.sims
        round_time = time.time() - round_start
        steps_per_sec = (args.steps_per_round * args.sims) / max(round_time, 0.001)

        if round_num % args.log_interval == 0 or round_num == 1:
            elapsed = time.time() - training_start
            print(f"  Round {round_num:>5}/{args.rounds}  |  "
                  f"Tags: {round_tags:>4}  |  "
                  f"Total tags: {total_tags:>6}  |  "
                  f"Steps/sec: {steps_per_sec:>8,.0f}  |  "
                  f"Elapsed: {elapsed:>6.1f}s")

        if round_num % args.save_interval == 0:
            first_sim.agents[0].algorithm.save(model_path)
            print(f"  >> Checkpoint saved")

    # Final save
    first_sim.agents[0].algorithm.save(model_path)

    total_time = time.time() - training_start
    print()
    print("=" * 60)
    print(f"  Training complete!")
    print(f"  Total time:    {total_time:.1f}s")
    print(f"  Total steps:   {total_steps:,}")
    print(f"  Total tags:    {total_tags:,}")
    if config.DUAL_ROLE_ENABLED:
        from rl.dual_role import _dual_paths
        tp, rp = _dual_paths(model_path)
        print(f"  Tagger model:  {tp}")
        print(f"  Runner model:  {rp}")
    else:
        print(f"  Model saved:   {model_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train RL algorithms for the Tag game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--algorithm", "-a", type=str, default="PPO",
                        help="Algorithm to train (default: PPO)")
    parser.add_argument("--rounds", "-r", type=int, default=100,
                        help="Number of training rounds/episodes (default: 100)")
    parser.add_argument("--steps-per-round", "-s", type=int, default=1000,
                        help="Game steps per round (default: 1000)")
    parser.add_argument("--sims", "-n", type=int, default=1,
                        help="Number of parallel simulations (default: 1)")
    parser.add_argument("--save-dir", "-d", type=str, default="saved_models",
                        help="Directory to save models (default: saved_models)")
    parser.add_argument("--load", "-l", type=str, default=None,
                        help="Path to load existing model before training")
    parser.add_argument("--log-interval", type=int, default=5,
                        help="Print progress every N rounds (default: 5)")
    parser.add_argument("--save-interval", type=int, default=20,
                        help="Save checkpoint every N rounds (default: 20)")
    parser.add_argument("--list-algorithms", action="store_true",
                        help="List available algorithms and exit")

    args = parser.parse_args()

    if args.list_algorithms:
        print("Available algorithms:")
        for name in config.RL_ALGORITHMS:
            module_path, class_name = config.RL_ALGORITHMS[name]
            print(f"  {name:15s}  ->  {module_path}.{class_name}")
        return

    train(args)


if __name__ == "__main__":
    main()
