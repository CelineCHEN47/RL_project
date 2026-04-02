# TAG - RL Laboratory

A 2D Tag game built with Pygame as a framework for Reinforcement Learning experiments. One entity is "it" (the tagger) and chases others. When tagged, the tagged entity becomes "it". The game runs continuously.

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the game
python main.py
```

## Game Controls

| Key | Action |
|-----|--------|
| WASD / Arrow Keys | Move player |
| ESC | Return to menu |

## Game Modes

### Player Mode
You control one character. The other characters are RL agents using the algorithm you selected. Move around and try to tag (or avoid being tagged by) the AI agents. Once RL is implemented, agents will learn in real-time as you play.

### Simulation Mode
All characters are RL agents. You watch as they interact and learn. Currently agents are stationary (placeholder) until RL algorithms are implemented.

## Project Structure

```
RL_project/
├── main.py                    # Entry point - run this to start
├── config.py                  # All game constants, colors, enums
├── requirements.txt           # Python dependencies
│
├── game/                      # Core game logic
│   ├── game_manager.py        # Main game loop, state machine (MENU/PLAYING)
│   └── tag_logic.py           # Tag rules: who is "it", cooldowns, tag detection
│
├── entities/                  # Game characters and objects
│   ├── entity.py              # Base Entity class (position, velocity, collision rect)
│   ├── player.py              # Human-controlled player (reads keyboard input)
│   ├── agent.py               # RL-controlled agent (queries algorithm for actions)
│   └── movable_object.py      # Pushable crates/boxes
│
├── world/                     # Level/map system
│   ├── tile.py                # TileType enum (FLOOR, WALL, SPAWN, CRATE_SPAWN)
│   ├── level.py               # Level loader, parses text maps, precomputes wall rects
│   └── maps/
│       └── level_01.txt       # Default map (editable text file)
│
├── physics/                   # Collision detection
│   └── collision.py           # AABB collision: entity-wall, entity-crate, crate-wall
│
├── rl/                        # Reinforcement Learning module
│   ├── base_algorithm.py      # Abstract base class - THE RL INTERFACE
│   ├── environment.py         # Gym-like wrapper (observations, rewards)
│   ├── q_learning.py          # Q-Learning placeholder
│   ├── sarsa.py               # SARSA placeholder
│   ├── ppo.py                 # PPO placeholder
│   └── dqn.py                 # DQN placeholder
│
├── ui/                        # User interface
│   ├── button.py              # Reusable button widget
│   ├── menu.py                # Main menu (mode + algorithm selection)
│   └── hud.py                 # In-game HUD overlay (tagger, scores, FPS)
│
└── rendering/                 # Display
    └── renderer.py            # All Pygame draw calls (level, entities, crates)
```

## Map Format

Maps are plain text files in `world/maps/`. Each character represents one tile:

| Char | Meaning |
|------|---------|
| `#` | Wall |
| `.` | Floor |
| `S` | Spawn point (floor tile where entities start) |
| `C` | Crate spawn (floor tile where a pushable crate starts) |
| `D` | Doorway (treated as floor, for readability) |

To create a new map, copy `level_01.txt` and edit it in any text editor. Update the map filename in `game_manager.py` `_init_game()` to use your new map.

---

## Guide for RL Implementation (For Teammates)

This section explains exactly where and how to implement the RL algorithms.

### The RL Interface

Every algorithm must subclass `BaseRLAlgorithm` in `rl/base_algorithm.py`. This is the contract:

```python
class BaseRLAlgorithm(ABC):
    ACTION_SPACE_SIZE = 5
    ACTION_MAP = {
        0: (0, 0),    # no-op (stand still)
        1: (0, -1),   # up
        2: (0, 1),    # down
        3: (-1, 0),   # left
        4: (1, 0),    # right
    }

    @abstractmethod
    def select_action(self, observation: dict) -> int:
        """Return action index 0-4 given current observation."""

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        """Update model from one (s, a, r, s', done) transition."""

    @abstractmethod
    def save(self, path: str): ...
    @abstractmethod
    def load(self, path: str): ...

    def reset(self):
        """Called when tag transfers (episode boundary)."""
        pass
```

### Observation Format

The `TagEnvironment` (in `rl/environment.py`) builds this observation dict each frame:

```python
{
    "self_pos": (float, float),      # Normalized [0,1] position in the level
    "self_vel": (float, float),      # Current velocity (vx, vy)
    "is_tagger": bool,               # True if this agent is "it"
    "other_agents": [                # List of other entities
        {"pos": (float, float), "is_tagger": bool},
        ...
    ],
    "tagger_pos": (float, float),    # Position of whoever is "it"
    "nearby_walls": [(int, int), ...],  # Grid coords of walls within vision
    "nearby_crates": [(float, float), ...],  # Normalized positions of nearby crates
}
```

### Reward Function

Defined in `rl/environment.py` `compute_reward()`. Current placeholder:

| Role | Event | Reward |
|------|-------|--------|
| Tagger | Per step | -0.01 (time penalty) |
| Tagger | Successful tag | +10.0 |
| Runner | Per step | +0.01 (survival bonus) |
| Runner | Got tagged | -10.0 |

**Feel free to modify these values.** Good reward shaping is critical for learning.

### How to Implement an Algorithm

#### Step 1: Pick your file

- **Q-Learning**: `rl/q_learning.py`
- **SARSA**: `rl/sarsa.py`
- **PPO**: `rl/ppo.py`
- **DQN**: `rl/dqn.py`

#### Step 2: Implement the methods

Each file has a class that already extends `BaseRLAlgorithm` with no-op methods. Replace the placeholder code:

1. **`__init__`**: Initialize your data structures (Q-table, neural networks, replay buffers, hyperparameters)
2. **`select_action(observation)`**: Given the observation dict, return an action (0-4). Implement your exploration strategy here (e.g., epsilon-greedy)
3. **`learn(state, action, reward, next_state, done)`**: This is called every `DECISION_INTERVAL` frames (default: 4). Update your model.
4. **`save(path)` / `load(path)`**: Serialize your model to disk. Use pickle for Q-tables, `torch.save` for neural nets.
5. **`reset()`**: Called when this agent gets tagged (episode boundary). Reset any episode-level state.

#### Step 3: Handle the observation

For **tabular methods** (Q-Learning, SARSA), you need to discretize the continuous observation into a hashable state key. Example:

```python
def _discretize(self, obs):
    """Convert continuous observation to a discrete state tuple."""
    # Bin position into grid cells
    gx = int(obs["self_pos"][0] * 20)  # 20 bins
    gy = int(obs["self_pos"][1] * 20)
    is_tagger = int(obs["is_tagger"])

    # Relative tagger position (binned)
    tx = int(obs["tagger_pos"][0] * 10)
    ty = int(obs["tagger_pos"][1] * 10)

    return (gx, gy, is_tagger, tx, ty)
```

For **neural methods** (PPO, DQN), flatten the observation into a fixed-size vector:

```python
def _obs_to_tensor(self, obs):
    """Convert observation dict to a flat tensor."""
    features = list(obs["self_pos"]) + list(obs["self_vel"])
    features.append(float(obs["is_tagger"]))
    features.extend(obs["tagger_pos"])
    # Add other agent positions (pad to fixed size)
    for i in range(4):  # max 4 other agents
        if i < len(obs["other_agents"]):
            a = obs["other_agents"][i]
            features.extend(a["pos"])
            features.append(float(a["is_tagger"]))
        else:
            features.extend([0.0, 0.0, 0.0])
    return torch.tensor(features, dtype=torch.float32)
```

#### Step 4: Test

Run the game in **Simulation Mode** with your algorithm selected. Watch the agents. If they move, your `select_action` is working. Check the console for any errors.

### Game Loop Integration (How RL Gets Called)

The game manager (`game/game_manager.py`) calls your algorithm every frame in this order:

```
1. obs = environment.get_observation(agent)
2. action = agent.algorithm.select_action(obs)    # YOUR CODE
3. agent moves based on action
4. collisions resolved
5. tag checked
6. reward = environment.compute_reward(agent, tag_event)
7. next_obs = environment.get_observation(agent)
8. agent.algorithm.learn(obs, action, reward, next_obs, done)  # YOUR CODE
```

Agents only decide every `DECISION_INTERVAL` frames (default: 4 in `config.py`). Between decisions, they continue their last action. You can adjust this in `config.py`.

### Adding New Algorithms

1. Create a new file in `rl/` (e.g., `rl/actor_critic.py`)
2. Subclass `BaseRLAlgorithm` and implement all abstract methods
3. Add an entry to `RL_ALGORITHMS` in `config.py`:
   ```python
   RL_ALGORITHMS = {
       ...
       "Actor-Critic": ("rl.actor_critic", "ActorCritic"),
   }
   ```
4. The menu will automatically show the new option

### Tips for RL Development

- **Start simple**: Get Q-Learning working first. It's the easiest to debug.
- **Discretize wisely**: Too many bins = slow learning. Too few = can't distinguish states.
- **Tune rewards**: The placeholder rewards are a starting point. Experiment.
- **Decision interval**: If learning is slow, decrease `DECISION_INTERVAL` in `config.py`. If too noisy, increase it.
- **Save/load models**: Implement `save()` and `load()` early so you don't lose training progress.
- **Add dependencies**: If you need PyTorch, TensorFlow, or other libraries, add them to `requirements.txt`.
- **Watch for convergence**: Add logging in your `learn()` method to track loss, Q-values, or policy entropy.

### Key Configuration Constants (`config.py`)

| Constant | Value | Description |
|----------|-------|-------------|
| `PLAYER_SPEED` | 3.0 | Player movement speed (pixels/frame) |
| `AGENT_SPEED` | 3.0 | Agent movement speed |
| `TAG_RADIUS` | 24 | Distance (pixels) to register a tag |
| `TAG_COOLDOWN_MS` | 1000 | Cooldown after a tag (prevents instant re-tags) |
| `DECISION_INTERVAL` | 4 | Agents decide every N frames |
| `FPS` | 60 | Game frame rate |

---

## Dependencies

- Python 3.10+
- pygame 2.x
- numpy

For neural network algorithms (PPO, DQN), you'll also need:
- PyTorch (`pip install torch`) or TensorFlow
