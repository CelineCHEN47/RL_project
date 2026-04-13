# Reward Function Design Update

## Overview

This document describes the changes made to the reward function in `rl/environment.py` (`compute_reward()`), explains the problems with the original design, and details the motivation behind each change.

---

## Problems with the Original Reward Design

### 1. Absolute Distance Reward Encourages Standing Still

The original tagger reward included `+0.5 * (1 - min_dist / max_dist)`, which gave a positive reward simply for *being* close to a runner. This allowed the tagger to learn a degenerate strategy: move near a runner and stop. As long as the distance was small, the distance reward easily outweighed the time penalty, so the tagger had no incentive to actually close in and tag.

Similarly, the runner received `+0.3 * (d / max_dist)`, rewarding absolute distance from the tagger. This encouraged runners to move to the farthest corner of the map and stand still — no evasive maneuvering needed.

### 2. Time Scale Grew Too Aggressively

The original `time_scale = min(1.0 + steps / 55.0, 10.0)` reached its 10x cap after roughly 500 steps. In early training, agents act near-randomly and tags are rare, so `steps_since_tag` stays high and the scale stays saturated at 10x. This caused:

- Tagger time penalty to stabilize at -0.2/step — disproportionately large
- Runner survival bonus to stabilize at +0.2/step — making "just stay alive" the dominant strategy with no need for active evasion

### 3. Tag Event Rewards Were Too Small Relative to Cumulative Step Rewards

Tag rewards were ±20. But over a 1000+ step episode, a runner's cumulative survival bonus alone could reach tens to hundreds of reward. A single -20 tag penalty became negligible — the agent learned that getting tagged once didn't matter much compared to the steady stream of survival rewards.

### 4. No Incentive to Move

Neither tagger nor runner received any reward for maintaining movement. Both could (and did) learn to stand still in favorable positions, leading to boring, degenerate behavior.

### 5. No Anti-Camping Mechanism

Runners could exploit the map geometry by hiding in corners where multiple walls provided natural protection. The original reward had no mechanism to discourage this.

---

## Updated Reward Design

### Key Design Principles

1. **Delta-based distance rewards** — Reward *changes* in distance rather than absolute distance, preventing "stand at comfortable distance" exploits.
2. **Dominant tag event signal** — Tag events (±50) are the primary learning signal; per-step shaping rewards are kept small.
3. **Gentle time scaling** — Slower ramp, lower cap, preventing reward saturation.
4. **Movement incentive** — Small bonus for tagger movement to encourage active pursuit.
5. **Anti-camping penalties** — Proximity danger zone and wall-hugging penalties for runners.

### Reward Structure

#### Tagger

| Component | Formula | Purpose |
|-----------|---------|---------|
| Tag success | +50.0 (immediate return) | Dominant positive signal |
| Time penalty | -0.01 × time_scale | Mild urgency, escalates over time |
| Distance delta | +1.0 × (prev_dist - curr_dist) / max_dist × time_scale | Rewards getting closer, punishes retreating |
| Movement bonus | +0.005 if speed > 0.5 | Prevents standing still |

#### Runner

| Component | Formula | Purpose |
|-----------|---------|---------|
| Got tagged | -50.0 (immediate return) | Dominant negative signal |
| Survival bonus | +0.01 × time_scale | Mild reward for staying alive |
| Distance delta | +0.5 × (curr_dist - prev_dist) / max_dist × time_scale | Rewards increasing distance from tagger |
| Danger zone penalty | -0.1 × (1 - d / danger_zone) when d < TAG_RADIUS × 4 | Urgency when tagger is close |
| Wall camping penalty | -0.02 × num_close_walls (when ≥ 3 walls nearby) | Discourages corner hiding |

#### Time Scale

```
time_scale = min(1.0 + steps_since_tag / 300.0, 3.0)
```

Ramps from 1.0 to 3.0 over 600 steps (~30% of a 2000-step episode). Compared to the original (1.0 to 10.0 over ~500 steps), this prevents reward saturation while still providing escalating pressure.

> **Tuning rule:** If `STEPS_PER_ROUND` changes, adjust the denominator as `total_steps × 0.15`.

### Additional Code Change: Reset Cached Distances

The delta-based rewards store `_prev_tagger_dist` and `_prev_runner_dist` on each entity. These must be cleared on episode reset to avoid a spurious large delta when agents teleport to new spawn points.

In `simulation.py` → `reset()`, added inside the agent loop:

```python
for attr in ('_prev_tagger_dist', '_prev_runner_dist'):
    if hasattr(agent, attr):
        delattr(agent, attr)
```

---

## Summary of Changes

| Aspect | Original | Updated | Reason |
|--------|----------|---------|--------|
| Distance reward | Absolute | **Delta-based** | Prevent "stand still" exploits |
| Tag event | ±20 | **±50** | Make tag the dominant signal |
| Time scale | 1→10 (55 steps) | **1→3 (300 steps)** | Prevent saturation |
| Per-step reward | ±0.02 × scale | **±0.01 × scale** | Reduce step reward dominance |
| Movement | None | **+0.005 for tagger** | Encourage active pursuit |
| Danger zone | None | **Penalty near tagger** | Encourage evasion |
| Corner camping | None | **Wall proximity penalty** | Prevent degenerate hiding |

