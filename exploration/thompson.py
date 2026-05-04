# exploration/thompson.py
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn

from core import BootstrappedDQN, FrameStack, SharedHyperParams, ReplayBuffer, preprocess_frame


class HyperParams:
    def __init__(self):
        self.NUM_HEADS = 10
        self.MASK_PROB = 0.8
        self.GRAD_CLIP = 10.0

class ThompsonAgent:
    STRATEGY_NAME = "thompson"

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

        self.q_network = BootstrappedDQN(
            action_dim=action_dim,
            num_heads=self.explore_hp.NUM_HEADS
        ).to(device)

        self.target_network = BootstrappedDQN(
            action_dim=action_dim,
            num_heads=self.explore_hp.NUM_HEADS
        ).to(device)

        if checkpoint:
            state = torch.load(checkpoint, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                self.q_network.load_state_dict(state["model_state_dict"])
                self.target_network.load_state_dict(state["model_state_dict"])
            else:
                self.q_network.load_state_dict(state)
                self.target_network.load_state_dict(state)
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.hp.LR)
        self.loss_fn = nn.HuberLoss(reduction="none")
        self.replay_buffer = ReplayBuffer(self.hp.BUFFER_SIZE, self.hp.SEED)

        self.total_env_steps = 0
        self.total_grad_steps = 0
        self.active_head = 0

    def _sample_head(self) -> int:
        return random.randint(0, self.explore_hp.NUM_HEADS - 1)

    def select_action(self, state_tensor: torch.Tensor) -> int:
        with torch.no_grad():
            q_heads = self.q_network(state_tensor)   # [1, K, A]
            q = q_heads[0, self.active_head]
            return q.argmax().item()

    def _train_step(self) -> None:
        batch = self.replay_buffer.sample(self.hp.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).float().div(255.0).to(self.device)
        next_states = torch.stack(next_states).float().div(255.0).to(self.device)
        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)

        batch_size = states.shape[0]
        K = self.explore_hp.NUM_HEADS

        q_all = self.q_network(states)                  # [B, K, A]
        next_q_all = self.target_network(next_states)   # [B, K, A]

        actions_expanded = actions.view(batch_size, 1, 1).expand(-1, K, 1)
        chosen_q = q_all.gather(2, actions_expanded).squeeze(-1)  # [B, K]

        with torch.no_grad():
            next_max_q = next_q_all.max(dim=2).values
            targets = rewards.view(-1, 1) + self.hp.GAMMA * next_max_q * (1.0 - dones.view(-1, 1))

        mask = torch.bernoulli(
            torch.full((batch_size, K), self.explore_hp.MASK_PROB, device=self.device)
        )

        if mask.sum().item() == 0:
            mask[0, 0] = 1.0

        loss_per_entry = self.loss_fn(chosen_q, targets)
        loss = (loss_per_entry * mask).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.explore_hp.GRAD_CLIP)
        self.optimizer.step()

        self.total_grad_steps += 1
        
        with torch.no_grad():
            # use active head for regret/entropy, consistent with action selection
            all_q = self.q_network(states)[:, self.active_head, :]  # [B, A]
            q_max = all_q.max(dim=1, keepdim=True).values
            q_taken = all_q.gather(1, actions.unsqueeze(1))
            regret = (q_max - q_taken).mean().item()
            probs = torch.softmax(all_q, dim=1)
            log_probs = torch.log(probs + 1e-8)
            entropy = -(probs * log_probs).sum(dim=1).mean().item()

        return {
            "loss": loss.item(),
            "regret": regret,
            "entropy": entropy,
        }

    def _sync_target(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.q_network.state_dict(),
                "num_heads": self.explore_hp.NUM_HEADS,
            },
            path,
        )
        print(f"[save] {path}")

    def _model_stem(self) -> str:
        if getattr(self.hp, 'MAX_STEPS', None):
            return f"{self.STRATEGY_NAME}_{self.hp.SEED}_s{self.hp.MAX_STEPS}"
        return f"{self.STRATEGY_NAME}_{self.hp.SEED}_{self.hp.NUM_EPISODES}"
    
    def train(self, env, num_episodes: int, log_path: str, model_dir: str, max_steps: int | None = None) -> None:
        os.makedirs(model_dir, exist_ok=True)
        frame_stack = FrameStack(self.hp.FRAME_STACK)
        log_file = open(log_path, "w")

        for episode in range(1, num_episodes + 1):
            
            if max_steps is not None and self.total_env_steps >= max_steps:
                print(f"\n[stop] Step limit {max_steps:,} reached at episode {episode}.")
                break
    
            obs, info = env.reset()

            # Thompson-style: sample one head for the whole episode
            self.active_head = self._sample_head()

            state = preprocess_frame(obs)
            frame_stack.reset()
            for _ in range(self.hp.FRAME_STACK):
                frame_stack.append(state)
            state_stack = frame_stack.get_stack()

            episode_reward = 0.0
            ep_len = 0
            curr_lives = info.get("lives", 5)
            life_lost = False
            
            ep_losses: list[float] = []
            ep_regrets: list[float] = []
            ep_entropies: list[float] = []

            while ep_len <= self.hp.MAX_EPISODE_LENGTH:
                if life_lost or ep_len == 0:
                    obs, _, _, _, info = env.step(1)  # FIRE
                    preprocessed = preprocess_frame(obs)
                    frame_stack.append(preprocessed)
                    state_stack = frame_stack.get_stack()
                    life_lost = False

                state_tensor = (
                    state_stack.clone()
                    .detach()
                    .float()
                    .unsqueeze(0)
                    .div(255.0)
                    .to(self.device)
                )

                action = self.select_action(state_tensor)
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

                if len(self.replay_buffer) < self.hp.MIN_BUFFER_SIZE:
                    state_stack = next_state_stack
                    continue

                if self.total_env_steps % self.hp.UPDATE_FREQ == 0:
                    metrics = self._train_step()
                    ep_losses.append(metrics["loss"])
                    ep_regrets.append(metrics["regret"])
                    ep_entropies.append(metrics["entropy"])

                if self.total_env_steps % self.hp.TARGET_UPDATE == 0:
                    self._sync_target()

                state_stack = next_state_stack

                if done or truncated:
                    break
                
                if max_steps is not None and self.total_env_steps >= max_steps:
                    break 

            record = {
                "episode":     episode,
                "total_steps": self.total_env_steps,
                "reward":      episode_reward,
                "ep_len":      ep_len,
                "head":        self.active_head,
                "num_heads":   self.explore_hp.NUM_HEADS,
                "loss":        float(np.mean(ep_losses)) if ep_losses else None,
                "regret":      float(np.mean(ep_regrets)) if ep_regrets else None,
                "entropy":     float(np.mean(ep_entropies)) if ep_entropies else None,
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            if episode % 100 == 0:
                print(
                    f"[ep {episode:>5}] "
                    f"steps={self.total_env_steps:>9,} "
                    f"reward={episode_reward:>6.1f} "
                    f"ep_len={ep_len:>6} "
                    f"head={self.active_head}"
                )

        final_model = os.path.join(model_dir, f"{self._model_stem()}.pth")
        self.save(final_model)
        log_file.close()
        print(f"Training complete. Log saved to: {log_path}")
    
    def evaluate(self, env, num_episodes: int = 1, record: bool = False) -> dict:
        frame_stack = FrameStack(self.hp.FRAME_STACK)
        rewards: list[float] = []
        steps: list[float] = []
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
            ep_steps = 0
            total_reward = 0.0
            for _ in range(self.hp.MAX_EPISODE_LENGTH):
                if record:
                    ep_frames.append(env.render())

                state = frame_stack.get_stack().unsqueeze(0).float().div(255.0).to(self.device)
                with torch.no_grad():
                    q_heads = self.q_network(state)  # [1, K, A]
                    action = q_heads.mean(dim=1).argmax(dim=1).item()

                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                ep_steps += 1
                frame_stack.append(preprocess_frame(obs))

                if terminated or truncated:
                    break

            rewards.append(total_reward)
            steps.append(ep_steps)
            if record:
                all_episodes_frames.append(ep_frames)
        
        results = {
            "episodes": [
                {"episode": ep, "total_reward": r, "steps": s}
                for ep, (r, s) in enumerate(zip(rewards, steps), start=1)
            ],
        }
        
        if num_episodes == 1:
            results["total_reward"] = rewards[0]
            results["total_steps"] = steps[0]
        else:
            results["total_reward"] = sum(rewards)
            results["total_steps"] = sum(steps)

        if all_episodes_frames:
            results["frames"] = all_episodes_frames

        return results