# run_experiment.py — Automated Experiment Script Guide

## Quick Start

```bash
# Place the script in the project root directory, then:

# 1. Run on remote server (headless, full speed)
python run_experiment.py

# 2. Run on local machine (watch agents live)
python run_experiment.py --display
```

## What Does This Script Do?

It runs three phases in one go:

1. **Train** → Automatically saves checkpoints at specified epoch milestones.
2. **Evaluate** → Loads each checkpoint and runs test episodes to collect quantitative metrics (tag counts, etc.).
3. **Plot** → Generates training curves, per-epoch performance bar charts, agent trajectory plots, and position heatmaps.

All outputs go to `experiments/dpo/` (or whichever directory you specify).

## Adapting It to Your Algorithm

At the top of the script you'll find:

```python
ALGORITHM_NAME = "DPO"
```

**Change this to your algorithm's name** (must match the key in `config.py`'s `RL_ALGORITHMS`). For example:

```python
ALGORITHM_NAME = "PPO"        # or "DQN", "Q-Learning", "SARSA"
```

Also update the output directory to avoid overwriting others' results:

```python
SAVE_DIR = "experiments/ppo"  # use your algorithm name
```

That's it — everything else works as-is.

## Common Commands

```bash
# Custom epoch milestones
python run_experiment.py --epochs 10 50 100 200 500 1000

# Skip training, only re-run evaluation and plotting
python run_experiment.py --eval-only

# Watch pre-trained models locally (no retraining)
python run_experiment.py --eval-only --display

# Speed up training with more parallel simulations (default: 2)
python run_experiment.py --sims 4
```

## All Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 10 50 100 200 500 1000 | Epoch milestones for checkpoint saving |
| `--eval-only` | off | Skip training; evaluate existing checkpoints only |
| `--display` | off | Open a Pygame window to watch live (local machine only) |
| `--sims` | 2 | Parallel simulations for training (forced to 1 with `--display`) |
| `--eval-episodes` | 20 | Number of evaluation episodes per checkpoint |
| `--save-dir` | experiments/dpo | Output directory |

## Output Structure

```
experiments/dpo/
├── checkpoints/          # Model files at each epoch
│   ├── dpo_epoch_10.pt
│   ├── dpo_epoch_50.pt
│   └── ...
├── results/
│   └── eval_metrics.json # Quantitative metrics per epoch
├── trajectories/         # Agent coordinate logs (for behavioral analysis)
├── plots/
│   ├── training_curve.png
│   ├── epoch_comparison.png
│   └── trajectory_epoch_*.png
└── training_log.json
```

## Display Mode Controls

- **ESC** during training → skip to evaluation phase
- **ESC** during evaluation → skip to next epoch
- A real-time HUD in the top-left corner shows current round, tags, and next checkpoint
- Frame rate is locked to 60 FPS, so display mode is much slower than headless — use it for behavioral observation, not large-scale training

## Notes

- Requires `matplotlib` for plotting: `pip install matplotlib`
- No `--display` needed on remote servers — the script runs headless by default
- Checkpoint filenames follow the pattern `dpo_epoch_N.pt`; for cross-algorithm comparison (Pathway 1), simply pick the same epoch from each algorithm's output directory
