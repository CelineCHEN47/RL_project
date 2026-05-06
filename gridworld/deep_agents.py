"""Deep RL agents (PPO, DQN) for the gridworld tag scenario."""

from __future__ import annotations

import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _state_to_tensor(state: tuple[int, int, int, int], grid_size: int,
                     device: torch.device) -> torch.Tensor:
    """Normalize (tx, ty, rx, ry) to [0, 1] and return as tensor."""
    denom = max(grid_size - 1, 1)
    tx, ty, rx, ry = state
    return torch.tensor([
        tx / denom,
        ty / denom,
        rx / denom,
        ry / denom,
    ], dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

# Match hyperparameters with rl/ppo.py
HIDDEN_DIM = 128
LEARNING_RATE = 3e-4
GAMMA = 0.95
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEFF = 0.01
VALUE_COEFF = 0.5
MAX_GRAD_NORM = 0.5
ROLLOUT_LENGTH = 256
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 64


class _ActorCritic(nn.Module):
    """Shared-backbone actor-critic network."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x: torch.Tensor):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value


class _RolloutBuffer:
    """Rollout buffer for PPO updates."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.next_values = []
        self.dones = []

    def add(self, state, action, log_prob, reward, value, next_value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.next_values.append(next_value)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def compute_gae(self):
        n = len(self.rewards)
        advantages = torch.zeros(n, dtype=torch.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            non_terminal = 1.0 - float(self.dones[t])
            delta = (self.rewards[t]
                     + GAMMA * self.next_values[t] * non_terminal
                     - self.values[t])
            last_gae = delta + GAMMA * GAE_LAMBDA * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + torch.tensor(self.values, dtype=torch.float32)
        return advantages, returns

    def get_batches(self, advantages, returns):
        n = len(self.states)
        indices = torch.randperm(n)

        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        adv_tensor = advantages
        ret_tensor = returns

        for start in range(0, n, MINI_BATCH_SIZE):
            end = min(start + MINI_BATCH_SIZE, n)
            idx = indices[start:end]
            yield (states[idx], actions[idx], old_log_probs[idx],
                   adv_tensor[idx], ret_tensor[idx])

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.next_values.clear()
        self.dones.clear()


class GridPPOAgent:
    """PPO agent for the gridworld environment."""

    def __init__(self, grid_size: int = 10, action_dim: int = 5):
        self.grid_size = grid_size
        self.action_dim = action_dim
        self.obs_dim = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = _ActorCritic(self.obs_dim, self.action_dim,
                                    HIDDEN_DIM).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=LEARNING_RATE)
        self.buffer = _RolloutBuffer()
        self.total_steps = 0
        self.eval_mode = False

    def set_eval(self, enabled: bool) -> None:
        self.eval_mode = enabled

    def _obs_tensor(self, state: tuple[int, int, int, int]) -> torch.Tensor:
        return _state_to_tensor(state, self.grid_size, self.device)

    @torch.no_grad()
    def select_action(self, state: tuple[int, int, int, int]) -> int:
        obs = self._obs_tensor(state)
        logits, value = self.network(obs)

        if self.eval_mode:
            action = int(torch.argmax(logits).item())
            log_prob = torch.log_softmax(logits, dim=-1)[action].item()
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action_t = dist.sample()
            action = int(action_t.item())
            log_prob = dist.log_prob(action_t).item()

        self._last_state = obs
        self._last_log_prob = log_prob
        self._last_value = float(value.item())
        return action

    def learn(self, state, action: int, reward: float,
              next_state, done: bool):
        if self.eval_mode:
            return
        if not hasattr(self, "_last_state"):
            return

        with torch.no_grad():
            next_tensor = self._obs_tensor(next_state)
            _, next_value_t = self.network(next_tensor)
            next_value = 0.0 if done else float(next_value_t.item())

        self.buffer.add(
            state=self._last_state,
            action=action,
            log_prob=self._last_log_prob,
            reward=reward,
            value=self._last_value,
            next_value=next_value,
            done=done,
        )
        self.total_steps += 1

        if len(self.buffer) >= ROLLOUT_LENGTH:
            self._update()

    def _update(self):
        advantages, returns = self.buffer.compute_gae()
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        for _ in range(PPO_EPOCHS):
            for (mb_states, mb_actions, mb_old_log_probs,
                 mb_advantages, mb_returns) in self.buffer.get_batches(
                     advantages, returns):

                new_log_probs, entropy, new_values = \
                    self.network.evaluate_actions(mb_states, mb_actions)

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON,
                                    1.0 + CLIP_EPSILON) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_values, mb_returns)
                entropy_loss = -entropy.mean()

                loss = (policy_loss
                        + VALUE_COEFF * value_loss
                        + ENTROPY_COEFF * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

        self.buffer.clear()

    def flush(self):
        """Force an update at episode end to align training cadence."""
        if len(self.buffer) >= MINI_BATCH_SIZE:
            self._update()
        else:
            self.buffer.clear()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------

# Match hyperparameters with rl/dqn.py
DQN_HIDDEN_DIM = 128
DQN_LEARNING_RATE = 1e-3
DQN_GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995
REPLAY_BUFFER_SIZE = 50_000
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 500
TARGET_UPDATE_FREQ = 500
DQN_MAX_GRAD_NORM = 10.0


class _QNetwork(nn.Module):
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
        return self.net(x)


class _ReplayBuffer:
    def __init__(self, capacity: int):
        self._buf = deque(maxlen=capacity)

    def add(self, state: torch.Tensor, action: int, reward: float,
            next_state: torch.Tensor, done: bool):
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self._buf)


class GridDQNAgent:
    """DQN agent for the gridworld environment."""

    def __init__(self, grid_size: int = 10, action_dim: int = 5):
        self.grid_size = grid_size
        self.action_dim = action_dim
        self.obs_dim = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = _QNetwork(self.obs_dim, self.action_dim, DQN_HIDDEN_DIM).to(
            self.device
        )
        self.target_net = _QNetwork(
            self.obs_dim, self.action_dim, DQN_HIDDEN_DIM
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=DQN_LEARNING_RATE)
        self.replay_buffer = _ReplayBuffer(REPLAY_BUFFER_SIZE)

        self.epsilon = EPSILON_START
        self.total_steps = 0
        self.last_loss = 0.0
        self.eval_mode = False
        self._last_state: torch.Tensor | None = None

    def set_eval(self, enabled: bool) -> None:
        self.eval_mode = enabled

    def _obs_tensor(self, state: tuple[int, int, int, int]) -> torch.Tensor:
        return _state_to_tensor(state, self.grid_size, self.device)

    def select_action(self, state: tuple[int, int, int, int]) -> int:
        obs = self._obs_tensor(state)
        self._last_state = obs

        if self.eval_mode:
            with torch.no_grad():
                q_values = self.q_net(obs.unsqueeze(0)).squeeze(0)
            return int(q_values.argmax().item())

        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            q_values = self.q_net(obs.unsqueeze(0)).squeeze(0)
        return int(q_values.argmax().item())

    def learn(self, state, action: int, reward: float,
              next_state, done: bool):
        if self.eval_mode:
            return
        if self._last_state is None:
            return

        next_tensor = self._obs_tensor(next_state)
        self.replay_buffer.add(self._last_state, action, reward, next_tensor, done)
        self.total_steps += 1

        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return

        self._update()

        if self.total_steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def _update(self):
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(BATCH_SIZE)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.q_net(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states)
            max_next_q = next_q.max(dim=1).values
            td_target = rewards + DQN_GAMMA * max_next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_sa, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), DQN_MAX_GRAD_NORM)
        self.optimizer.step()

        self.last_loss = float(loss.item())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.epsilon = ckpt.get("epsilon", EPSILON_MIN)
    
    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
