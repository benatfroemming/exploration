# exploration/boltzmann.py
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import DQN, FrameStack, SharedHyperParams, ReplayBuffer, preprocess_frame


class HyperParams:
    def __init__(self):
        self.TEMP_START = 1.0        # initial temperature (high = more random)
        self.TEMP_MIN   = 0.01       # minimum temperature (low = near-greedy)
        self.TEMP_DECAY_STEPS = 1_000_000  # linear decay over this many env steps

class BoltzmannAgent:

    STRATEGY_NAME = "boltzmann"

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

        # Temperature schedule
        self.temperature: float = self.explore_hp.TEMP_START
        self._temp_schedule = np.linspace(
            self.explore_hp.TEMP_START,
            self.explore_hp.TEMP_MIN,
            self.explore_hp.TEMP_DECAY_STEPS,
        )

        # Networks
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
        self.replay_buffer = ReplayBuffer(self.hp.BUFFER_SIZE, self.hp.SEED)

        self.total_env_steps: int = 0
        self.total_grad_steps: int = 0

    def select_action(self, state_tensor: torch.Tensor) -> int:
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)  # (action_dim,)
            q_values = q_values - q_values.max()                # stability trick
            scaled = (q_values / self.temperature).clamp(-10, 10)
            probs = F.softmax(scaled, dim=-1)
            return torch.multinomial(probs, num_samples=1).item()

    def _greedy_action(self, state_tensor: torch.Tensor) -> int:
        with torch.no_grad():
            return self.q_network(state_tensor).argmax().item()

    def _train_step(self) -> None:
        batch = self.replay_buffer.sample(self.hp.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.stack(states).float().div(255.0).to(self.device)
        next_states = torch.stack(next_states).float().div(255.0).to(self.device)
        actions     = torch.tensor(actions).long().unsqueeze(1).to(self.device)
        rewards     = torch.tensor(rewards).float().unsqueeze(1).to(self.device)
        dones       = torch.tensor(dones).float().unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q  = self.target_network(next_states).max(1)[0].unsqueeze(1)
            targets = rewards + self.hp.GAMMA * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_grad_steps += 1

    def _sync_target(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: str) -> None:
        torch.save(self.q_network.state_dict(), path)
        print(f"  [save] {path}")

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

                idx = min(self.total_env_steps, self.explore_hp.TEMP_DECAY_STEPS - 1)
                self.temperature = float(self._temp_schedule[idx])

                if self.total_env_steps % self.hp.UPDATE_FREQ == 0:
                    self._train_step()

                if self.total_env_steps % self.hp.TARGET_UPDATE == 0:
                    self._sync_target()
                    print(
                        f"  [target sync] step={self.total_env_steps:,}  "
                        f"T={self.temperature:.4f}"
                    )

                state_stack = next_state_stack

                if done or truncated:
                    break

            record = {
                "episode":     episode,
                "total_steps": self.total_env_steps,
                "reward":      episode_reward,
                "ep_len":      ep_len,
                "temperature": round(self.temperature, 6),
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            if episode % 100 == 0:
                print(
                    f"[ep {episode:>5}] "
                    f"steps={self.total_env_steps:>9,}  "
                    f"reward={episode_reward:>6.1f}  "
                    f"ep_len={ep_len:>6}  "
                    f"T={self.temperature:.4f}"
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
    
    def evaluate(self, env, num_episodes: int = 1, record: bool = False) -> dict:
        frame_stack = FrameStack(self.hp.FRAME_STACK)
        rewards: list[float] = []
        all_episodes_frames: list[list] = []

        self.q_network.eval()
        for ep in range(1, num_episodes + 1):
            obs, _ = env.reset()
            frame_stack.reset()
            frame = preprocess_frame(obs)
            for _ in range(frame_stack.k):
                frame_stack.append(frame)

            try:
                obs, _, terminated, truncated, _ = env.step(1)
                if not (terminated or truncated):
                    frame_stack.append(preprocess_frame(obs))
            except Exception:
                pass

            ep_frames: list = []
            total_reward = 0.0
            for _ in range(self.hp.MAX_EPISODE_LENGTH):
                if record:
                    ep_frames.append(env.render())

                state = frame_stack.get_stack().unsqueeze(0).float().div(255.0).to(self.device)
                with torch.no_grad():
                    action = self.q_network(state).argmax(dim=1).item() 

                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                frame_stack.append(preprocess_frame(obs))

                if terminated or truncated:
                    break

            rewards.append(total_reward)
            if record:
                all_episodes_frames.append(ep_frames)

        results = {
            "episodes": [
                {"episode": ep, "total_reward": r}
                for ep, r in enumerate(rewards, start=1)
            ],
        }

        if num_episodes == 1:
            results["total_reward"] = rewards[0]
        else:
            results["total_reward"] = sum(rewards)
            results["mean"] = float(np.mean(rewards))
            results["std"] = float(np.std(rewards))
            results["min"] = float(min(rewards))
            results["max"] = float(max(rewards))

        if all_episodes_frames:
            results["frames"] = all_episodes_frames

        return results