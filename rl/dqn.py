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

Observation encoding
--------------------
Identical to ppo.py / dpo.py: flat 19-dim vector
    [self_pos(2), self_vel(2), is_tagger(1), tagger_pos(2),
     agent_0_pos(2), agent_0_is_tagger(1), ..., agent_3_pos(2), agent_3_is_tagger(1)]
     (padded to MAX_OTHER_AGENTS = 4 slots)
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.base_algorithm import BaseRLAlgorithm


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MAX_OTHER_AGENTS     = 4
OBS_DIM              = 2 + 2 + 1 + 2 + MAX_OTHER_AGENTS * 3   # = 19
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
            torch.stack(states),                                        # (B, obs_dim)
            torch.tensor(actions,  dtype=torch.long),                   # (B,)
            torch.tensor(rewards,  dtype=torch.float32),                # (B,)
            torch.stack(next_states),                                   # (B, obs_dim)
            torch.tensor(dones,    dtype=torch.float32),                # (B,)
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
        action = agent.select_action(obs)           # every DECISION_INTERVAL
        agent.learn(obs, action, reward, next_obs, done)
    """

    def __init__(self):
        self.device = torch.device("cpu")

        # Online and target Q-networks
        self.q_net     = QNetwork(OBS_DIM, self.ACTION_SPACE_SIZE, HIDDEN_DIM)
        self.target_net = QNetwork(OBS_DIM, self.ACTION_SPACE_SIZE, HIDDEN_DIM)
        self.q_net.to(self.device)
        self.target_net.to(self.device)

        # Sync target = online at init; freeze target gradients
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Exploration state
        self.epsilon     = EPSILON_START
        self.total_steps = 0

        # Diagnostics (inspectable from outside)
        self.last_loss   = 0.0

        # Cache from select_action for use in learn()
        self._last_state: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Observation encoding  (matches ppo.py / dpo.py)
    # ------------------------------------------------------------------
    def _obs_to_tensor(self, obs: dict) -> torch.Tensor:
        features = []
        features.extend(obs["self_pos"])
        features.append(obs["self_vel"][0] / 3.0)
        features.append(obs["self_vel"][1] / 3.0)
        features.append(1.0 if obs["is_tagger"] else 0.0)
        features.extend(obs["tagger_pos"])

        other = obs.get("other_agents", [])
        for i in range(MAX_OTHER_AGENTS):
            if i < len(other):
                features.extend(other[i]["pos"])
                features.append(1.0 if other[i]["is_tagger"] else 0.0)
            else:
                features.extend([0.0, 0.0, 0.0])

        return torch.tensor(features, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------
    def _current_epsilon(self) -> float:
        """Linearly decay epsilon from START to END over DECAY_STEPS."""
        ratio   = min(self.total_steps / EPSILON_DECAY_STEPS, 1.0)
        return EPSILON_START + ratio * (EPSILON_END - EPSILON_START)

    # ------------------------------------------------------------------
    # Action selection  (epsilon-greedy)
    # ------------------------------------------------------------------
    def select_action(self, observation: dict) -> int:
        state = self._obs_to_tensor(observation).to(self.device)
        self._last_state = state

        self.epsilon = self._current_epsilon()

        if random.random() < self.epsilon:
            # Explore: uniform random action
            return random.randrange(self.ACTION_SPACE_SIZE)

        # Exploit: pick action with highest Q-value
        with torch.no_grad():
            q_values = self.q_net(state.unsqueeze(0)).squeeze(0)  # (action_dim,)
        return int(q_values.argmax().item())

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------
    def learn(self, state: dict, action: int, reward: float,
              next_state: dict, done: bool):
        """
        Store transition in replay buffer; sample a mini-batch and update
        the online Q-network once the buffer is warm.
        """
        if self._last_state is None:
            return

        next_state_tensor = self._obs_to_tensor(next_state).to(self.device)
        self.replay_buffer.add(
            self._last_state, action, reward, next_state_tensor, done
        )
        self.total_steps += 1

        # Don't start learning until the buffer is warm
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return

        self._update()

        # Hard-sync target network periodically
        if self.total_steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    # ------------------------------------------------------------------
    # Q-network update
    # ------------------------------------------------------------------
    def _update(self):
        """One gradient step of DQN on a random mini-batch."""
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(BATCH_SIZE)

        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Current Q-values for the taken actions: Q_theta(s, a)
        # q_net outputs (B, action_dim); gather selects the action column
        q_values = self.q_net(states)                        # (B, action_dim)
        q_sa     = q_values.gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)                                         # (B,)

        # TD targets: r + gamma * max_{a'} Q_theta'(s', a')
        with torch.no_grad():
            next_q      = self.target_net(next_states)       # (B, action_dim)
            max_next_q  = next_q.max(dim=1).values           # (B,)
            td_target   = rewards + GAMMA * max_next_q * (1.0 - dones)

        # Huber loss (smooth L1) — more robust to outliers than plain MSE
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
        """Called on tag transfer.  No episode-level state to reset for DQN."""
        pass

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save Q-networks, optimizer, and training counters."""
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".",
            exist_ok=True,
        )
        torch.save(
            {
                "q_net":      self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "total_steps": self.total_steps,
                "epsilon":    self.epsilon,
            },
            path,
        )

    def load(self, path: str):
        """Restore Q-networks, optimizer, and training counters."""
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.epsilon     = ckpt.get("epsilon",     EPSILON_END)
