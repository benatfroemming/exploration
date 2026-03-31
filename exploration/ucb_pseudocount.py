# exploration/ucb.py
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

from core import DQN, FrameStack, SharedHyperParams, ReplayBuffer, preprocess_frame

class HyperParams:
    def __init__(self):
        self.BETA = 1.0
        self.HASH_DIM = 18       # 2**18 ~ 262k buckets
        self.GRAD_CLIP = 10.0

class UCBAgent:
    STRATEGY_NAME = "ucb"

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

        self.q_network = DQN(action_dim=action_dim).to(device)
        self.target_network = DQN(action_dim=action_dim).to(device)

        if checkpoint:
            state = torch.load(checkpoint, map_location=device)
            sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
            self.q_network.load_state_dict(sd)
            self.target_network.load_state_dict(sd)
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.hp.LR)
        self.loss_fn = nn.HuberLoss()
        self.replay_buffer = ReplayBuffer(self.hp.BUFFER_SIZE, self.hp.SEED)

        # SimHash count tables
        n_buckets = 2 ** self.explore_hp.HASH_DIM
        self._proj: np.ndarray | None = None          # lazy-init on first hash
        self._state_counts = np.zeros(n_buckets, dtype=np.int32)          # N(s)
        self._sa_counts    = np.zeros((n_buckets, action_dim), dtype=np.int32)  # N(s,a)

        self.total_env_steps = 0
        self.total_grad_steps = 0

    # SimHash
    def _hash_state(self, state_tensor: torch.Tensor) -> int:
        """Project flattened state with a fixed random matrix, binarize, pack to int index."""
        flat = state_tensor.cpu().numpy().ravel().astype(np.float32)
        if self._proj is None:
            rng = np.random.RandomState(42)
            self._proj = rng.randn(self.explore_hp.HASH_DIM, flat.size).astype(np.float32)
        bits = (self._proj @ flat >= 0).astype(np.uint8)
        idx = int(bits.dot(1 << np.arange(self.explore_hp.HASH_DIM, dtype=np.int64)))
        return idx  

    # Action selection
    def select_action(self, state_tensor: torch.Tensor) -> tuple[int, float]:
        h = self._hash_state(state_tensor)
        n_s = max(self._state_counts[h], 1)

        with torch.no_grad():
            q_vals = self.q_network(state_tensor)[0]  # [A]

        bonuses = np.array([
            self.explore_hp.BETA * math.sqrt(math.log(n_s) / max(self._sa_counts[h, a], 1))
            for a in range(self.action_dim)
        ])
        ucb_scores = q_vals.cpu().numpy() + bonuses
        action = int(np.argmax(ucb_scores))

        self._state_counts[h] += 1
        self._sa_counts[h, action] += 1

        return action, float(bonuses[action])

    # Training
    def _train_step(self) -> None:
        batch = self.replay_buffer.sample(self.hp.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.stack(states).float().div(255.0).to(self.device)
        next_states = torch.stack(next_states).float().div(255.0).to(self.device)
        actions     = torch.tensor(actions).long().to(self.device)
        rewards     = torch.tensor(rewards).float().to(self.device)
        dones       = torch.tensor(dones).float().to(self.device)

        q_vals  = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_max = self.target_network(next_states).max(dim=1).values
            targets  = rewards + self.hp.GAMMA * next_max * (1.0 - dones)

        loss = self.loss_fn(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.explore_hp.GRAD_CLIP)
        self.optimizer.step()
        self.total_grad_steps += 1

    def _sync_target(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Save / util
    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.q_network.state_dict(),
                "ucb_beta": self.explore_hp.BETA,
                "hash_dim": self.explore_hp.HASH_DIM,
            },
            path,
        )
        print(f"[save] {path}")

    def _model_stem(self) -> str:
        env_slug = self.env_id.replace("/", "-").replace(" ", "_")
        return f"dqn_{env_slug}_{self.STRATEGY_NAME}"

    # Train loop
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
            ep_len = 0
            curr_lives = info.get("lives", 5)
            life_lost = False
            last_bonus = 0.0

            while ep_len <= self.hp.MAX_EPISODE_LENGTH:
                if life_lost or ep_len == 0:
                    obs, _, _, _, info = env.step(1)  # FIRE
                    frame_stack.append(preprocess_frame(obs))
                    state_stack = frame_stack.get_stack()
                    life_lost = False

                state_tensor = (
                    state_stack.clone().detach()
                    .float().unsqueeze(0).div(255.0).to(self.device)
                )

                action, last_bonus = self.select_action(state_tensor)
                next_obs, reward, done, truncated, info = env.step(action)

                if info.get("lives", curr_lives) < curr_lives:
                    life_lost = True
                    curr_lives = info["lives"]

                ep_len += 1
                self.total_env_steps += 1
                episode_reward += reward

                next_state = preprocess_frame(next_obs)
                frame_stack.append(next_state)
                next_state_stack = frame_stack.get_stack()

                self.replay_buffer.push(
                    (state_stack, action, reward, next_state_stack, done or life_lost)
                )

                if len(self.replay_buffer) >= self.hp.MIN_BUFFER_SIZE:
                    if self.total_env_steps % self.hp.UPDATE_FREQ == 0:
                        self._train_step()
                    if self.total_env_steps % self.hp.TARGET_UPDATE == 0:
                        self._sync_target()

                state_stack = next_state_stack
                if done or truncated:
                    break

            record = {
                "episode": episode,
                "total_steps": self.total_env_steps,
                "reward": episode_reward,
                "ep_len": ep_len,
                "ucb_beta": self.explore_hp.BETA,
                "last_bonus": last_bonus,
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            if episode % 100 == 0:
                print(
                    f"[ep {episode:>5}] "
                    f"steps={self.total_env_steps:>9,} "
                    f"reward={episode_reward:>6.1f} "
                    f"ep_len={ep_len:>6} "
                    f"bonus={last_bonus:>7.4f}"
                )

        self.save(os.path.join(model_dir, f"{self._model_stem()}_final.pth"))
        log_file.close()
        print(f"Training complete. Log saved to: {log_path}")

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

            try:
                obs, _, terminated, truncated, _ = env.step(1)
                if not (terminated or truncated):
                    frame_stack.append(preprocess_frame(obs))
            except Exception:
                pass

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
            "episodes": [{"episode": ep, "total_reward": r} for ep, r in enumerate(rewards, start=1)],
            "total_reward": sum(rewards),
        }
        if num_episodes > 1:
            results["mean"] = float(np.mean(rewards))
            results["std"]  = float(np.std(rewards))
            results["min"]  = float(min(rewards))
            results["max"]  = float(max(rewards))

        return results