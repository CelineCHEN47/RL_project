# TAG - RL Laboratory: Presentation Script

## Slide 1 — Title

**Show**: Project title, your name, course name, and a screenshot of the main menu.

**Say**:
> "I built a 2D Tag game in Python that doubles as a sandbox for testing reinforcement learning algorithms. The game has the classic chase-and-tag mechanics, but every non-player character is controlled by an RL agent that learns over time. Today I'll walk you through the game design and how it integrates with the RL framework."

---

## Slide 2 — The Game Concept

**Show**: A short 5-second clip or GIF of the game running with the player chasing or being chased.

**Say**:
> "The rules are simple: one character is 'it', they chase others. When they tag someone, that person becomes 'it'. The game never ends. The challenge is that the world has walls, rooms, doorways, and pushable crates — so it's not just about speed, it's about navigation and prediction."

---

## Slide 3 — Tech Stack & Architecture

**Show**: A clean diagram with these boxes connected:

```
main.py → GameManager → Menu / GameLoop
                         ↓
                       Renderer ← Textures / Sprites / Particles
                         ↓
                       Level (text-based maps)
                         ↓
                       Entities (Player, Agents, Crates)
                         ↓
                       Physics (AABB collision)
                         ↓
                       RL Environment ← Algorithm (PPO/DQN/Q-Learning/SARSA)
```

**Say**:
> "Built entirely in Python with Pygame for rendering and PyTorch for the neural network algorithms. The architecture is modular — each layer is in its own file, so the game logic, rendering, physics, and RL are completely separate. This makes it easy for teammates to plug in new RL algorithms without touching the game code."

---

## Slide 4 — Game Modes

**Show**: A screenshot of the menu with the three selection rows highlighted.

**Say**:
> "From the main menu you pick three things. First — Game Mode: either Player Mode where you control one character with WASD, or Simulation Mode where you watch all AI agents play. Second — Algorithm: Q-Learning, SARSA, PPO, or DQN. Third — Agent Behavior: 'Train Live' means agents learn in real-time while you play, or 'Use Trained' loads a pre-trained model so you can test how well training worked."

---

## Slide 5 — The World

**Show**: A screenshot of the textured map with walls, crates, and entities labeled.

**Say**:
> "The world is built from simple text files. Each character represents a tile — `#` for wall, `.` for floor, `S` for spawn points, `C` for crate spawns. This means anyone can design new maps by editing a text file. The map gets rendered with procedural textures — brick walls, stone floors, wooden crates with grain patterns, all generated in code so there are no external image assets."

---

## Slide 6 — Visuals & Feel

**Show**: Side-by-side screenshots: simple flat colors vs the polished version with character sprites, shadows, particles.

**Say**:
> "I started with simple colored circles, then upgraded to procedural sprites — characters have eyes, the tagger has angry eyebrows and a pulsing red aura, everyone casts ground shadows. When a tag happens, particles explode at the contact point. Movement leaves a subtle colored trail. All of this is generated in code, no sprite sheets needed."

---

## Slide 7 — RL Integration: The Big Idea

**Show**: A diagram showing one network split into "Tagger Brain" + "Runner Brain".

**Say**:
> "The interesting part is how RL connects to the game. Originally I had one neural network learning both how to chase AND how to flee — but that creates conflicting gradients because the same network gets opposite reward signals depending on its role. So I split it: every algorithm now has TWO models, one tagger brain and one runner brain. When an agent gets tagged and becomes 'it', it just switches which brain it queries. All the runner agents share the same runner brain, so their experiences pool together."

---

## Slide 8 — What the Agents See

**Show**: A diagram of an agent with arrows showing the 8 wall raycasts and lines pointing to other entities.

**Say**:
> "Each agent gets an ego-centric observation — everything is relative to itself. It knows where it is, where the tagger is and how far away, where the nearest runner is, and it casts 8 rays in every direction to sense walls. It also gets the relative positions of up to 6 other agents, sorted by distance. This is 43 numbers total fed into the neural network every few frames."

---

## Slide 9 — Reward Design

**Show**: A simple table:

| Role | Per step | Tag event | Distance shaping |
|------|----------|-----------|------------------|
| Tagger | -0.05 (escalating) | +20 catch | Closer = +reward |
| Runner | +0.05 (escalating) | -20 caught | Farther = +reward |

**Say**:
> "Reward shaping is critical. The tagger gets a small penalty every step it doesn't catch someone, and that penalty grows over time — so the longer it goes without a tag, the more desperate it should be. Runners get the opposite: a small bonus that grows the longer they survive. Plus continuous distance-based feedback so they're not just relying on the rare tag event for signal."

---

## Slide 10 — Training Pipeline

**Show**: Terminal screenshot of `python train.py -a PPO -r 200 -n 4` running, with the per-round output.

**Say**:
> "I built a headless training script that runs the game logic without rendering — this lets training run at thousands of steps per second instead of being capped at 60 FPS. You can run multiple parallel simulations that all share the same neural network, so experiences pool together. After training, the models are saved to disk and can be loaded back into the game with one click."

---

## Slide 11 — Experiment & Evaluation

**Show**: One of your generated plots — either the training curve or the role evaluation comparison.

**Say**:
> "There's also an experiment script that trains with checkpoints at specific epochs, then evaluates each checkpoint to measure how performance changes over training. It also does isolated role testing — putting a trained tagger against random runners to measure pure tagger skill, and trained runners against a random tagger to measure pure runner skill. This separates 'is the tagger smart' from 'are the runners just bad'."

---

## Slide 12 — Hitting a Wall with Tabular Methods

**Show**: A side-by-side comparison:
- **Left**: The full game (continuous space, 6 agents, 32×24 tiles, never-ending)
- **Right**: A red ❌ over a Q-table icon

**Say**:
> "When we tried running Q-Learning and SARSA on the full game, they basically didn't learn. The reason is structural: tabular methods need a finite, hashable state space — but our observation has continuous positions, velocities, wall raycasts, and up to 6 other agents. Even after discretizing, the state space explodes into millions of cells, most of which an agent will only visit once or twice in an entire training run. There's also no episode boundary — the game runs forever, so there's no clear 'done' signal for the Bellman update."

---

## Slide 13 — Simplifying the Problem: Gridworld

**Show**: Diagram with these properties listed:

| Full game | Simplified gridworld |
|-----------|---------------------|
| Continuous (x, y) positions | Discrete 10×10 grid |
| 6 agents | 1 tagger + 1 runner |
| Never ends | Episode = until catch |
| Walls, crates, rooms | Empty grid, no obstacles |
| 43-feature observation | 4 numbers: (tx, ty, rx, ry) |
| ~10,000,000+ states | Exactly 10,000 states |

**Say**:
> "So we simplified. We took everything that was breaking the tabular methods and stripped it away. The new environment is a 10×10 grid with one tagger and one runner. Turn-based — tagger moves, then runner moves. Episode ends when the tagger lands on the runner's cell. The state is just four integers: the (x,y) of each agent. That gives exactly 10,000 possible states — perfectly tractable for a Q-table."

---

## Slide 14 — Training Strategy: One Role at a Time

**Show**: A 3-phase flowchart:
- **Phase 1**: Trained Tagger ← vs → Random Runner
- **Phase 2**: Frozen Tagger ← vs → Trained Runner
- **Phase 3**: Trained Tagger ← vs → Trained Runner (watch)

**Say**:
> "Even with the simplified environment, training both agents simultaneously creates a non-stationary problem — the opponent's policy keeps shifting under your feet, so Q-Learning never converges. We solved this by training one role at a time. Phase 1: the tagger trains against a runner that moves randomly, so it learns to chase a moving target. Phase 2: we freeze that trained tagger and now train the runner against a real chaser. Phase 3: we put both trained models against each other and watch."

---

## Slide 15 — Results: It Actually Works

**Show**: A screenshot of the gridworld visualization mid-training, showing:
- The 10×10 grid with the red tagger and blue runner
- The stats panel showing catch rate climbing
- Q-table size, epsilon decay

**Say**:
> "And it works. The tagger's catch rate climbs from random — about 20% — up to 90%+ within a few thousand episodes. You can literally watch the Q-table fill up in real-time, and watch the epsilon decay from full exploration down to mostly exploitation. We exported these training runs as videos so we can show convergence visually, not just as a number."

---

## Slide 16 — What This Tells Us About RL

**Show**: A bullet list:

- **Algorithm choice depends on problem structure**, not just performance ceiling
- **Tabular methods**: small discrete state spaces, clear episodes
- **Deep RL (PPO/DQN)**: continuous, high-dim, no episode required
- **Simplifying the environment to fit the algorithm** is a valid research move

**Say**:
> "The bigger lesson here is about how to think about algorithm selection. PPO and DQN can handle the full game because they generalize — a neural network maps similar states to similar actions even if it's never seen the exact state before. Q-Learning can't do that without function approximation. So the takeaway isn't 'tabular methods are bad' — it's that the problem structure has to match the algorithm. When it does, like in the gridworld, tabular Q-Learning learns clean, interpretable policies very quickly. That's actually a strength."

---

## Slide 17 — Demo

**Show**: Live demo or pre-recorded video clips. Suggested order:

1. Open main menu, show all options
2. Player mode with PPO agents (Train Live) — chase a few agents around
3. Simulation mode with trained PPO — show the agents actually playing intelligently
4. Switch to gridworld training video — Phase 1 (tagger learns) → Phase 3 (trained vs trained)
5. Show one of the experiment plots (training curve or role evaluation)

**Say** (during demo):
> "Let me show you all of this in action. *(open menu)* Here's the menu — game mode, algorithm, train live or use trained. *(start player mode)* Player mode — I control the yellow-crowned character. The other agents are PPO agents currently training. *(switch to simulation)* In simulation mode, I just watch. *(play gridworld phase 1 video)* And here's the simplified gridworld where Q-Learning learns to chase from scratch — this is sped up but you can see the tagger getting smarter. *(play phase 3 video)* And this is both trained agents going head to head."

---

## Slide 18 — Technical Highlights

**Show**: A bullet list:

- Procedural textures and sprites — no external assets
- Modular architecture — drop-in algorithm support
- Dual-role model architecture — separate tagger/runner brains
- Headless training at ~2,000 steps/sec
- Live training visualization with video export
- Three difficulty maps + simplified gridworld

**Say**:
> "Quick technical highlights: everything visual is generated procedurally, so the project is just code, no asset folders. Adding a new algorithm is one file plus one config line. Training runs at thousands of steps per second when headless. And I built a recorder that exports training videos as MP4."

---

## Slide 19 — What I Learned / Future Work

**Show**: Two columns: "Learned" and "Next steps".

**Say**:
> "Key things I learned: reward shaping matters more than algorithm choice, ego-centric observations train way faster than absolute coordinates, splitting roles into separate models was the single biggest improvement, and matching the algorithm to the problem structure matters more than picking the 'best' algorithm. For future work: better cooperative behavior between runners, dynamic difficulty, support for more than 6 agents, and benchmarking PPO vs DQN vs the gridworld Q-Learning baseline."

---

## Slide 20 — Thank You / Questions

**Show**: GitHub link, project name, and a final game screenshot.

**Say**:
> "Thanks for listening. The full code is on GitHub — the framework is set up so anyone can plug in their own RL algorithm. Happy to answer any questions."

---

## Quick Presentation Tips

- **Total time**: ~13–15 minutes if you spend 30–45 seconds per slide
- **Gridworld experiment slides (12–16)**: this is the most "research-y" part — slow down here, it's the strongest narrative arc (problem → simplification → solution → insight)
- **Demo slide**: budget 2–3 minutes — practice it so it doesn't drag
- Use the **compressed videos** from `gridworld/videos_compressed/` for the gridworld demo (15s each)
- For the main game demo, record a short 30-second clip beforehand as backup in case the live demo glitches
- Have the trained PPO model ready (`saved_models/ppo_model_tagger.pt` + `_runner.pt`) so "Use Trained" actually shows interesting behavior, not random movement
