"""PPO (Proximal Policy Optimization) for the Tag game.

Architecture:
    - Shared feature extractor -> Actor head (policy) + Critic head (value)
    - Collects trajectories in a rollout buffer, updates in mini-batches
    - Uses GAE (Generalized Advantage Estimation) for variance reduction
    - Clipped surrogate objective to keep updates stable

Ego-centric observation vector (fixed-size):
    [self_pos(2), self_vel(2), is_tagger(1),
     tagger_rel(2), tagger_dist(1),
     nearest_runner_rel(2), nearest_runner_dist(1),
     wall_rays(8),
     agent_0_rel(2), agent_0_dist(1), agent_0_is_tagger(1), ...,
     agent_N_rel(2), agent_N_dist(1), agent_N_is_tagger(1)]
    Padded to MAX_OTHER_AGENTS slots.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from rl.base_algorithm import BaseRLAlgorithm


# --- Hyperparameters ---
MAX_OTHER_AGENTS = 6
# Obs: self_pos(2) + self_vel(2) + is_tagger(1)
#      + tagger_rel(2) + tagger_dist(1)
#      + nearest_runner_rel(2) + nearest_runner_dist(1)
#      + wall_rays(8)
#      + agents(MAX_OTHER_AGENTS * 4: rel_x, rel_y, dist, is_tagger)
OBS_DIM = 2 + 2 + 1 + 2 + 1 + 2 + 1 + 8 + MAX_OTHER_AGENTS * 4  # = 43
HIDDEN_DIM = 128
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEFF = 0.01
VALUE_COEFF = 0.5
MAX_GRAD_NORM = 0.5
ROLLOUT_LENGTH = 256       # transitions before each update
PPO_EPOCHS = 4             # passes over the rollout per update
MINI_BATCH_SIZE = 64


class ActorCritic(nn.Module):
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
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value


class RolloutBuffer:
    """Stores one rollout of transitions for PPO update."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def compute_gae(self, last_value: float):
        """Compute GAE advantages and discounted returns."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            next_non_terminal = 1.0 - float(self.dones[t])
            delta = (self.rewards[t]
                     + GAMMA * next_value * next_non_terminal
                     - self.values[t])
            last_gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def get_batches(self, advantages, returns):
        """Yield random mini-batches from the rollout."""
        n = len(self.states)
        indices = np.random.permutation(n)

        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32)
        ret_tensor = torch.tensor(returns, dtype=torch.float32)

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
        self.dones.clear()


class PPO(BaseRLAlgorithm):
    def __init__(self):
        self.device = torch.device("cpu")
        self.network = ActorCritic(OBS_DIM, self.ACTION_SPACE_SIZE, HIDDEN_DIM)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.buffer = RolloutBuffer()
        self.total_steps = 0

    def _obs_to_tensor(self, obs: dict) -> torch.Tensor:
        """Convert ego-centric observation dict to fixed-size tensor."""
        features = []

        # Self position (2) - absolute, for spatial awareness
        features.extend(obs["self_pos"])

        # Self velocity (2) - normalized by speed
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

    @torch.no_grad()
    def select_action(self, observation: dict) -> int:
        """Sample action from the policy network."""
        state = self._obs_to_tensor(observation)
        action, log_prob, value = self.network.get_action_and_value(state)

        self._last_state = state
        self._last_log_prob = log_prob.item()
        self._last_value = value.item()

        return action.item()

    def learn(self, state: dict, action: int, reward: float,
              next_state: dict, done: bool):
        """Store transition and update when buffer is full."""
        if not hasattr(self, "_last_state"):
            return

        self.buffer.add(
            state=self._last_state,
            action=action,
            log_prob=self._last_log_prob,
            reward=reward,
            value=self._last_value,
            done=done,
        )
        self.total_steps += 1

        if len(self.buffer) >= ROLLOUT_LENGTH:
            self._update(next_state)

    def _update(self, last_obs: dict):
        """Run PPO update on the collected rollout."""
        with torch.no_grad():
            last_state = self._obs_to_tensor(last_obs)
            _, last_value = self.network(last_state)
            last_value = last_value.item()

        advantages, returns = self.buffer.compute_gae(last_value)

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

    def save(self, path: str):
        """Save network weights and optimizer state."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str):
        """Load network weights and optimizer state."""
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)

    def reset(self):
        """Called on episode boundary (tag transfer)."""
        pass
