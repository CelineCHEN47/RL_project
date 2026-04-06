"""DQN (Deep Q-Network) for the Tag game.

Architecture
------------
- Online Q-network  Q_theta(s, a): predicts Q-values for all actions at once.
- Target Q-network  Q_theta'(s, a): frozen copy, periodically hard-synced to
  the online network.  Provides stable TD targets and prevents oscillation.
- Replay buffer: stores past transitions (s, a, r, s', done) and feeds random
  mini-batches to the optimizer, breaking temporal correlations.
- Epsilon-greedy exploration: epsilon decays from EPSILON_START to EPSILON_END
  over the first EPSILON_DECAY_STEPS total steps.

Update rule
-----------
At each gradient step we minimize the Bellman error over a mini-batch:

    y_t = r_t  +  gamma * max_{a'} Q_theta'(s_{t+1}, a')  *  (1 - done_t)

    L = mean [ ( Q_theta(s_t, a_t) - y_t )^2 ]

Observation encoding (ego-centric, matches ppo.py)
---------------------------------------------------
    [self_pos(2), self_vel(2), is_tagger(1),
     tagger_rel(2), tagger_dist(1),
     nearest_runner_rel(2), nearest_runner_dist(1),
     wall_rays(8),
     agent_0_rel(2), agent_0_dist(1), agent_0_is_tagger(1), ...,
     agent_N_rel(2), agent_N_dist(1), agent_N_is_tagger(1)]
    Padded to MAX_OTHER_AGENTS = 6 slots.  Total: 43 dims.
"""

import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.base_algorithm import BaseRLAlgorithm


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MAX_OTHER_AGENTS     = 6
OBS_DIM              = 2 + 2 + 1 + 2 + 1 + 2 + 1 + 8 + MAX_OTHER_AGENTS * 4  # = 43
HIDDEN_DIM           = 128
LEARNING_RATE        = 1e-3
GAMMA                = 0.99

# Exploration
EPSILON_START        = 1.0
EPSILON_END          = 0.05
EPSILON_DECAY_STEPS  = 50_000   # linear decay over this many steps

# Replay buffer
REPLAY_BUFFER_SIZE   = 50_000
BATCH_SIZE           = 64
MIN_REPLAY_SIZE      = 500      # don't learn until buffer has this many samples

# Target network
TARGET_UPDATE_FREQ   = 500      # hard-sync every N total steps

MAX_GRAD_NORM        = 10.0


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """MLP that maps an observation vector to Q-values for all actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q(s, .) for every action.  Shape: (B, action_dim)."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Circular buffer storing (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int):
        self._buf = deque(maxlen=capacity)

    def add(self, state: torch.Tensor, action: int, reward: float,
            next_state: torch.Tensor, done: bool):
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Return a random mini-batch as stacked tensors."""
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.stack(states),
            torch.tensor(actions,  dtype=torch.long),
            torch.tensor(rewards,  dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones,    dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# DQN algorithm
# ---------------------------------------------------------------------------
class DQN(BaseRLAlgorithm):
    """
    Deep Q-Network agent for the Tag game.

    Compatible with the BaseRLAlgorithm interface.  The game loop calls:
        action = agent.select_action(obs)
        agent.learn(obs, action, reward, next_obs, done)
    """

    def __init__(self):
        self.device = torch.device("cpu")

        self.q_net      = QNetwork(OBS_DIM, self.ACTION_SPACE_SIZE, HIDDEN_DIM)
        self.target_net = QNetwork(OBS_DIM, self.ACTION_SPACE_SIZE, HIDDEN_DIM)

        # Sync target = online at init; freeze target gradients
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        self.epsilon     = EPSILON_START
        self.total_steps = 0
        self.last_loss   = 0.0

        self._last_state: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Observation encoding (ego-centric, matches ppo.py)
    # ------------------------------------------------------------------
    def _obs_to_tensor(self, obs: dict) -> torch.Tensor:
        features = []

        # Self position (2)
        features.extend(obs["self_pos"])

        # Self velocity (2)
        features.append(obs["self_vel"][0] / 3.0)
        features.append(obs["self_vel"][1] / 3.0)

        # Is tagger (1)
        features.append(1.0 if obs["is_tagger"] else 0.0)

        # Relative tagger direction + distance (3)
        features.extend(obs["tagger_rel"])
        features.append(obs["tagger_dist"])

        # Relative nearest runner direction + distance (3)
        features.extend(obs["nearest_runner_rel"])
        features.append(obs["nearest_runner_dist"])

        # Wall raycasts in 8 directions (8)
        features.extend(obs["wall_rays"])

        # Other agents sorted by distance (MAX_OTHER_AGENTS * 4)
        other = obs.get("other_agents", [])
        for i in range(MAX_OTHER_AGENTS):
            if i < len(other):
                features.extend(other[i]["rel_pos"])
                features.append(other[i]["distance"])
                features.append(1.0 if other[i]["is_tagger"] else 0.0)
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

        return torch.tensor(features, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------
    def _current_epsilon(self) -> float:
        ratio = min(self.total_steps / EPSILON_DECAY_STEPS, 1.0)
        return EPSILON_START + ratio * (EPSILON_END - EPSILON_START)

    # ------------------------------------------------------------------
    # Action selection (epsilon-greedy)
    # ------------------------------------------------------------------
    def select_action(self, observation: dict) -> int:
        state = self._obs_to_tensor(observation)
        self._last_state = state

        self.epsilon = self._current_epsilon()

        if random.random() < self.epsilon:
            return random.randrange(self.ACTION_SPACE_SIZE)

        with torch.no_grad():
            q_values = self.q_net(state.unsqueeze(0)).squeeze(0)
        return int(q_values.argmax().item())

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------
    def learn(self, state: dict, action: int, reward: float,
              next_state: dict, done: bool):
        if self._last_state is None:
            return

        next_state_tensor = self._obs_to_tensor(next_state)
        self.replay_buffer.add(
            self._last_state, action, reward, next_state_tensor, done
        )
        self.total_steps += 1

        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return

        self._update()

        if self.total_steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------------
    # Q-network update
    # ------------------------------------------------------------------
    def _update(self):
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(BATCH_SIZE)

        q_values = self.q_net(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q     = self.target_net(next_states)
            max_next_q = next_q.max(dim=1).values
            td_target  = rewards + GAMMA * max_next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_sa, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()

        self.last_loss = loss.item()

    # ------------------------------------------------------------------
    # Episode boundary
    # ------------------------------------------------------------------
    def reset(self):
        pass

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".",
            exist_ok=True,
        )
        torch.save({
            "q_net":       self.q_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "epsilon":     self.epsilon,
        }, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.epsilon     = ckpt.get("epsilon", EPSILON_END)
