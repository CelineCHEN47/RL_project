"""Central configuration for the Tag RL game."""

from enum import Enum

# Screen
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
FPS = 60

# Tiles
TILE_SIZE = 32

# Gameplay
PLAYER_SPEED = 3.0
AGENT_SPEED = 3.0
TAG_RADIUS = 24
TAG_COOLDOWN_MS = 1000
CRATE_PUSH_SPEED = 2.0
NUM_AGENTS = 4

# RL
DECISION_INTERVAL = 4  # agents decide every N frames

# Colors
COLOR_BG = (30, 30, 40)
COLOR_FLOOR = (60, 60, 75)
COLOR_WALL = (120, 120, 140)
COLOR_CRATE = (160, 120, 60)
COLOR_CRATE_BORDER = (120, 90, 40)
COLOR_TAGGER = (220, 50, 50)
COLOR_RUNNER = (50, 150, 220)
COLOR_PLAYER_HIGHLIGHT = (255, 255, 100)
COLOR_MENU_BG = (20, 20, 35)
COLOR_MENU_TITLE = (255, 200, 50)
COLOR_BUTTON = (70, 70, 100)
COLOR_BUTTON_HOVER = (90, 90, 130)
COLOR_BUTTON_SELECTED = (50, 180, 100)
COLOR_BUTTON_TEXT = (240, 240, 240)
COLOR_HUD_TEXT = (220, 220, 220)
COLOR_HUD_BG = (0, 0, 0, 150)


class GameState(Enum):
    MENU = "menu"
    PLAYING = "playing"


class GameMode(Enum):
    PLAYER_MODE = "Player Mode"
    SIMULATION_MODE = "Simulation Mode"


class TrainMode(Enum):
    TRAIN_LIVE = "Train Live"        # agents learn while playing
    USE_TRAINED = "Use Trained"      # load pre-trained model, no learning


DEFAULT_MODEL_DIR = "saved_models"


# Algorithm registry: display name -> module path and class name
RL_ALGORITHMS = {
    "Q-Learning": ("rl.q_learning", "QLearning"),
    "SARSA": ("rl.sarsa", "SARSA"),
    "PPO": ("rl.ppo", "PPO"),
    "DQN": ("rl.dqn", "DQN"),
}
