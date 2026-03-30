# exploration/rnd.py
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core import DQN, FrameStack, SharedHyperParams, ReplayBuffer, preprocess_frame


# RND network
class RND(nn.Module):
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Strategy-specific hyperparameters
class HyperParams:
    def __init__(self):
        self.RND_OUTPUT_DIM  = 128    # embedding size for target / predictor
        self.RND_LR          = 1e-4   # predictor learning rate
        self.BETA            = 0.01   # intrinsic reward weight
        self.INT_REWARD_CLIP = 5.0    # clamp normalised intrinsic reward to ±this


# Agent
class RNDAgent:
    """
    DQN agent with Random Network Distillation (RND) intrinsic exploration.

    Intrinsic reward = MSE( predictor(s'), target(s') )
    Combined reward  = r_ext + beta * r_int  (normalised via running stats)
    The target RND network is frozen; only the predictor is trained.
    """

    STRATEGY_NAME = "rnd"

    def __init__(
        self,
        env_id: str,
        shared_hp: SharedHyperParams,
        explore_hp: HyperParams,
        action_dim: int,
        device: torch.device,
        checkpoint: str | None = None,
    ):
        self.hp = shared_hp
        self.explore_hp = explore_hp
        self.env_id = env_id
        self.action_dim = action_dim
        self.device = device

        # DQN networks
        self.q_network = DQN(action_dim).to(device)
        self.target_network = DQN(action_dim).to(device)
        if checkpoint:
            print(f"Loading checkpoint: {checkpoint}")
            state = torch.load(checkpoint, map_location=device)
            self.q_network.load_state_dict(state)
            self.target_network.load_state_dict(state)
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.hp.LR)
        self.loss_fn = nn.HuberLoss()

        # RND networks
        self.rnd_target    = RND(explore_hp.RND_OUTPUT_DIM).to(device)
        self.rnd_predictor = RND(explore_hp.RND_OUTPUT_DIM).to(device)

        # Target is fixed, no gradients
        for p in self.rnd_target.parameters():
            p.requires_grad = False

        self.rnd_optimizer = torch.optim.Adam(
            self.rnd_predictor.parameters(), lr=explore_hp.RND_LR
        )

        # Running statistics for intrinsic reward normalization
        self.rnd_running_mean: float = 0.0
        self.rnd_running_std:  float = 1.0

        # Replay buffer & counters
        self.replay_buffer = ReplayBuffer(self.hp.BUFFER_SIZE, self.hp.SEED)
        self.total_env_steps:  int = 0
        self.total_grad_steps: int = 0

    # Intrinsic reward
    def _compute_intrinsic_reward(self, next_states: torch.Tensor) -> torch.Tensor:
        """Returns a (B, 1) tensor of raw intrinsic rewards."""
        with torch.no_grad():
            target_feat = self.rnd_target(next_states)
        pred_feat = self.rnd_predictor(next_states)
        return F.mse_loss(pred_feat, target_feat, reduction="none").mean(dim=1, keepdim=True)

    def _train_rnd(self, next_states: torch.Tensor) -> None:
        target_feat = self.rnd_target(next_states)
        pred_feat   = self.rnd_predictor(next_states)
        loss = F.mse_loss(pred_feat, target_feat)
        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()

    # Action selection
    def select_action(self, state_tensor: torch.Tensor) -> int:
        return self._greedy_action(state_tensor)

    def _greedy_action(self, state_tensor: torch.Tensor) -> int:
        with torch.no_grad():
            return self.q_network(state_tensor).argmax().item()

    # Training step
    def _train_step(self) -> None:
        batch = self.replay_buffer.sample(self.hp.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.stack(states).float().div(255.0).to(self.device)
        next_states = torch.stack(next_states).float().div(255.0).to(self.device)
        actions     = torch.tensor(actions).long().unsqueeze(1).to(self.device)
        rewards     = torch.tensor(rewards).float().unsqueeze(1).to(self.device)
        dones       = torch.tensor(dones).float().unsqueeze(1).to(self.device)

        # Intrinsic reward
        intrinsic = self._compute_intrinsic_reward(next_states).detach()

        # Running normalization
        mean = intrinsic.mean().item()
        std  = intrinsic.std().item()
        self.rnd_running_mean = 0.99 * self.rnd_running_mean + 0.01 * mean
        self.rnd_running_std  = 0.99 * self.rnd_running_std  + 0.01 * std
        intrinsic = (intrinsic - self.rnd_running_mean) / (self.rnd_running_std + 1e-8)
        intrinsic = intrinsic.clamp(-self.explore_hp.INT_REWARD_CLIP, self.explore_hp.INT_REWARD_CLIP)

        # Combined reward & DQN target
        combined_rewards = rewards + self.explore_hp.BETA * intrinsic

        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q  = self.target_network(next_states).max(1)[0].unsqueeze(1)
            targets = combined_rewards + self.hp.GAMMA * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Train RND predictor
        self._train_rnd(next_states)

        self.total_grad_steps += 1

    # Target network sync
    def _sync_target(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Checkpointing
    def save(self, path: str) -> None:
        torch.save(self.q_network.state_dict(), path)
        print(f"  [save] {path}")

    # Training loop
    def train(self, env, num_episodes: int, log_path: str, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        frame_stack = FrameStack(self.hp.FRAME_STACK)
        log_file = open(log_path, "w")

        for episode in range(1, num_episodes + 1):
            obs, info = env.reset()
            state = preprocess_frame(obs)
            frame_stack.reset()
            for _ in range(self.hp.FRAME_STACK):
                frame_stack.append(state)
            state_stack = frame_stack.get_stack()

            episode_reward = 0.0
            ep_len         = 0
            curr_lives     = info.get("lives", 5)
            life_lost      = False

            while ep_len <= self.hp.MAX_EPISODE_LENGTH:
                if life_lost or ep_len == 0:
                    obs, _, _, _, info = env.step(1)
                    frame_stack.append(preprocess_frame(obs))
                    state_stack = frame_stack.get_stack()
                    life_lost = False

                state_tensor = (
                    state_stack.clone().detach().float().unsqueeze(0).div(255.0).to(self.device)
                )
                action = self.select_action(state_tensor)

                next_obs, reward, done, truncated, info = env.step(action)

                if info.get("lives", curr_lives) < curr_lives:
                    life_lost  = True
                    curr_lives = info["lives"]

                ep_len               += 1
                self.total_env_steps += 1
                episode_reward       += reward

                frame_stack.append(preprocess_frame(next_obs))
                next_state_stack = frame_stack.get_stack()

                self.replay_buffer.push(
                    (state_stack, action, reward, next_state_stack, done or life_lost)
                )

                if len(self.replay_buffer) < self.hp.MIN_BUFFER_SIZE:
                    if self.total_env_steps % 10_000 == 0:
                        print(
                            f"  Filling replay buffer … "
                            f"{len(self.replay_buffer):,}/{self.hp.MIN_BUFFER_SIZE:,} steps"
                        )
                    state_stack = next_state_stack
                    continue

                if self.total_env_steps % self.hp.UPDATE_FREQ == 0:
                    self._train_step()

                if self.total_env_steps % self.hp.TARGET_UPDATE == 0:
                    self._sync_target()
                    print(f"  [target sync] step={self.total_env_steps:,}")

                state_stack = next_state_stack

                if done or truncated:
                    break

            record = {
                "episode":     episode,
                "total_steps": self.total_env_steps,
                "reward":      episode_reward,
                "ep_len":      ep_len,
                "rnd_mean":    round(self.rnd_running_mean, 6),
                "rnd_std":     round(self.rnd_running_std,  6),
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            if episode % 100 == 0:
                print(
                    f"[ep {episode:>5}] "
                    f"steps={self.total_env_steps:>9,}  "
                    f"reward={episode_reward:>6.1f}  "
                    f"ep_len={ep_len:>6}  "
                    f"rnd_mean={self.rnd_running_mean:.4f}"
                )

            if self.total_env_steps > 0 and self.total_env_steps % 2_000_000 == 0:
                self.save(os.path.join(model_dir, f"{self._model_stem()}_step{self.total_env_steps}.pth"))

        final_model = os.path.join(model_dir, f"{self._model_stem()}_final.pth")
        self.save(final_model)
        log_file.close()
        print(f"\nTraining complete. Log → {log_path}  |  Model → {final_model}")

    def _model_stem(self) -> str:
        env_slug = self.env_id.replace("/", "-").replace(" ", "_")
        return f"dqn_{env_slug}_{self.STRATEGY_NAME}"
    
    def evaluate(self, env, num_episodes: int = 1) -> dict:
        frame_stack = FrameStack(self.hp.FRAME_STACK)
        rewards: list[float] = []

        self.q_network.eval()
        for ep in range(1, num_episodes + 1):
            obs, _ = env.reset()
            frame_stack.reset()
            frame = preprocess_frame(obs)
            for _ in range(frame_stack.k):
                frame_stack.append(frame)

            obs, _, _, _, _ = env.step(1)
            frame_stack.append(preprocess_frame(obs))

            total_reward = 0.0
            for _ in range(self.hp.MAX_EPISODE_LENGTH):
                state = frame_stack.get_stack().unsqueeze(0).float().div(255.0).to(self.device)
                with torch.no_grad():
                    action = self.q_network(state).argmax(dim=1).item()

                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                frame_stack.append(preprocess_frame(obs))

                if terminated or truncated:
                    break

            rewards.append(total_reward)

        results = {
            "episodes": [
                {"episode": ep, "total_reward": r}
                for ep, r in enumerate(rewards, start=1)
            ],
            "total_reward": sum(rewards),
        }
        if num_episodes > 1:
            results["mean"] = float(np.mean(rewards))
            results["std"] = float(np.std(rewards))
            results["min"] = float(min(rewards))
            results["max"] = float(max(rewards))

        return results