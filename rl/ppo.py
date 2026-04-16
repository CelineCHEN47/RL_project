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

# Role-specific observation dimensions. When instantiated via DualRoleAlgorithm,
# the tagger/runner networks only see features relevant to their role — cuts
# input dim from 43 to 15 (tagger) or 18 (runner), improving sample efficiency.
#   tagger : self_pos(2) + self_vel(2)
#          + nearest_runner_rel(2) + nearest_runner_dist(1)
#          + wall_rays(8)                                               = 15
#   runner : self_pos(2) + self_vel(2)
#          + tagger_rel(2) + tagger_dist(1)
#          + nearest_runner_rel(2) + nearest_runner_dist(1)
#          + wall_rays(8)                                               = 18
#   unified: legacy 43-dim encoding (backward compat when role not given)
TAGGER_OBS_DIM = 2 + 2 + 2 + 1 + 8
RUNNER_OBS_DIM = 2 + 2 + 2 + 1 + 2 + 1 + 8
UNIFIED_OBS_DIM = 2 + 2 + 1 + 2 + 1 + 2 + 1 + 8 + MAX_OTHER_AGENTS * 4  # = 43

_OBS_DIM_BY_ROLE = {
    "tagger": TAGGER_OBS_DIM,
    "runner": RUNNER_OBS_DIM,
    "unified": UNIFIED_OBS_DIM,
}

# Alias kept so any external code that imported OBS_DIM still works.
OBS_DIM = UNIFIED_OBS_DIM
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
    """Stores one rollout of transitions for PPO update.

    When multiple agents share this buffer their transitions are interleaved
    (e.g. A,B,C,A,B,C,...).  GAE is computed **per-agent** so that
    ``values[t+1]`` always refers to the same agent's next step, not a
    different agent's concurrent step.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.next_values = []  # V(s_{t+1}) stored at learn time, per transition
        self.dones = []
        self.agent_ids = []  # tracks which PPO instance produced each entry

    def add(self, state, action, log_prob, reward, value, next_value, done,
            agent_id=None):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.next_values.append(next_value)
        self.dones.append(done)
        self.agent_ids.append(agent_id)

    def __len__(self):
        return len(self.states)

    def compute_gae(self):
        """Compute GAE advantages and discounted returns.

        Each transition carries ``next_values[t] = V(s_{t+1})``, computed
        at learn time from the next_state argument. This gives every agent
        in a shared buffer a correct bootstrap for its last transition —
        no reliance on a caller-provided ``last_value`` or a biased fallback
        that reused ``V(s_t)``.

        Transitions are still grouped by agent_id so that within-agent
        GAE accumulation stops at each agent's tail instead of bleeding
        across into another agent's concurrent sub-trajectory.
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)

        # Group buffer indices by agent
        agent_groups: dict[object, list[int]] = {}
        for t in range(n):
            aid = self.agent_ids[t]
            if aid not in agent_groups:
                agent_groups[aid] = []
            agent_groups[aid].append(t)

        for aid, indices in agent_groups.items():
            m = len(indices)
            last_gae = 0.0

            for i in reversed(range(m)):
                t = indices[i]
                next_val = self.next_values[t]  # V(s_{t+1}), always available

                non_terminal = 1.0 - float(self.dones[t])
                delta = (self.rewards[t]
                         + GAMMA * next_val * non_terminal
                         - self.values[t])
                last_gae = delta + GAMMA * GAE_LAMBDA * non_terminal * last_gae
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
        self.next_values.clear()
        self.dones.clear()
        self.agent_ids.clear()


class PPO(BaseRLAlgorithm):
    def __init__(self, role: str = "unified"):
        """
        Args:
            role: 'tagger', 'runner', or 'unified'.
                Controls the observation encoding and network input size.
                DualRoleAlgorithm passes 'tagger'/'runner' for its two
                sub-algorithms. When instantiated directly, defaults to
                'unified' (43-dim legacy encoding).
        """
        if role not in _OBS_DIM_BY_ROLE:
            raise ValueError(f"PPO role must be one of {list(_OBS_DIM_BY_ROLE)}, "
                             f"got {role!r}")
        self.role = role
        self.obs_dim = _OBS_DIM_BY_ROLE[role]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCritic(self.obs_dim, self.ACTION_SPACE_SIZE,
                                   HIDDEN_DIM).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.buffer = RolloutBuffer()
        self.total_steps = 0

    def _obs_to_tensor(self, obs: dict) -> torch.Tensor:
        """Convert ego-centric observation dict to role-specific fixed-size tensor."""
        if self.role == "tagger":
            features = [
                obs["self_pos"][0], obs["self_pos"][1],
                obs["self_vel"][0] / 3.0, obs["self_vel"][1] / 3.0,
                obs["nearest_runner_rel"][0], obs["nearest_runner_rel"][1],
                obs["nearest_runner_dist"],
            ]
            features.extend(obs["wall_rays"])
        elif self.role == "runner":
            features = [
                obs["self_pos"][0], obs["self_pos"][1],
                obs["self_vel"][0] / 3.0, obs["self_vel"][1] / 3.0,
                obs["tagger_rel"][0], obs["tagger_rel"][1],
                obs["tagger_dist"],
                obs["nearest_runner_rel"][0], obs["nearest_runner_rel"][1],
                obs["nearest_runner_dist"],
            ]
            features.extend(obs["wall_rays"])
        else:
            # Legacy unified encoding (43 dims)
            features = list(obs["self_pos"])
            features.append(obs["self_vel"][0] / 3.0)
            features.append(obs["self_vel"][1] / 3.0)
            features.append(1.0 if obs["is_tagger"] else 0.0)
            features.extend(obs["tagger_rel"])
            features.append(obs["tagger_dist"])
            features.extend(obs["nearest_runner_rel"])
            features.append(obs["nearest_runner_dist"])
            features.extend(obs["wall_rays"])
            other = obs.get("other_agents", [])
            for i in range(MAX_OTHER_AGENTS):
                if i < len(other):
                    features.extend(other[i]["rel_pos"])
                    features.append(other[i]["distance"])
                    features.append(1.0 if other[i]["is_tagger"] else 0.0)
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])

        return torch.tensor(features, dtype=torch.float32, device=self.device)

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
        if self.eval_mode:
            return
        if not hasattr(self, "_last_state"):
            return

        # Compute V(next_state) now so every transition carries its own
        # bootstrap value. This fixes the shared-buffer bias where
        # non-triggering agents previously reused V(s_t) as the tail bootstrap.
        with torch.no_grad():
            next_tensor = self._obs_to_tensor(next_state)
            _, next_value_t = self.network(next_tensor)
            next_value = 0.0 if done else next_value_t.item()

        self.buffer.add(
            state=self._last_state,
            action=action,
            log_prob=self._last_log_prob,
            reward=reward,
            value=self._last_value,
            next_value=next_value,
            done=done,
            agent_id=id(self),  # identify which PPO instance produced this
        )
        self.total_steps += 1

        if len(self.buffer) >= ROLLOUT_LENGTH:
            self._update()

    def _update(self):
        """Run PPO update on the collected rollout."""
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
