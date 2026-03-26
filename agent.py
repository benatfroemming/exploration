import os
import json
import random

import numpy as np
import torch
import torch.nn as nn

from core import (
    DQN,
    FrameStack,
    ReplayBuffer,
    preprocess_frame,
    save_stats,
)


class Agent:
    def __init__(self, experiment_name, hyperparams, action_space_size, device, chkpt=None):
        self.hyperparams = hyperparams
        self.experiment_name = experiment_name
        self.device = device
        self.action_space_size = action_space_size

        # Convenience aliases
        hp = hyperparams
        self.buffer_size      = hp.BUFFER_SIZE
        self.lr               = hp.LR
        self.batch_size       = hp.BATCH_SIZE
        self.gamma            = hp.GAMMA
        self.update_freq      = hp.UPDATE_FREQ
        self.target_update_freq = hp.TARGET_UPDATE
        self.min_buffer_size  = hp.MIN_BUFFER_SIZE
        self.max_ep_len       = hp.MAX_EPISODE_LENGTH
        self.num_episodes     = hp.NUM_EPISODES

        # Counters
        self.total_env_steps       = 0
        self.total_grad_update_steps = 0

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, hp.SEED)

        # Networks
        self.q_network     = DQN(action_space_size).to(device)
        self.target_network = DQN(action_space_size).to(device)
        if chkpt:
            print(f"Loading checkpoint: {chkpt}")
            state = torch.load(chkpt, map_location=device)
            self.q_network.load_state_dict(state)
            self.target_network.load_state_dict(state)
        self.update_target_network()

        # Optimiser & loss
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.HuberLoss()

    # Network helpers
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def greedy_action(self, state_tensor):
        """Pure greedy: always pick argmax Q."""
        with torch.no_grad():
            return self.q_network(state_tensor).argmax().item()

    # Core training loop
    def run(self, env, action_fn, logger, output_dir=".", training=True):
        """
        Train (or evaluate) the agent for self.num_episodes episodes.

        Parameters
        ----------
        env        : Gymnasium environment
        action_fn  : callable(agent, state_tensor) -> int Supplied by the exploration method module.
        logger     : logging.Logger
        output_dir : directory for checkpoints and stats file
        training   : if False, skip all buffer/gradient steps

        Returns
        -------
        episode_rewards   : list[float]
        episode_lengths   : list[int]
        sample_rewards    : list[float]  one reward sampled per episode step
        """
        os.makedirs(output_dir, exist_ok=True)
        stats_path = os.path.join(output_dir, "training_stats.jsonl")

        episode_rewards  = []
        episode_lengths  = []
        sample_rewards   = []          # one reward collected per episode
        best_avg_reward  = -float("inf")

        frame_stack = FrameStack(self.hyperparams.FRAME_STACK)

        for episode in range(1, self.num_episodes + 1):
            obs, info = env.reset()
            state = preprocess_frame(obs)
            frame_stack.reset()
            for _ in range(self.hyperparams.FRAME_STACK):
                frame_stack.append(state)
            state_stack = frame_stack.get_stack()

            total_reward = 0.0
            ep_len       = 0
            curr_life    = 5
            life_lost    = False
            done         = False

            while ep_len <= self.max_ep_len:
                # Fire to start / resume after life loss
                if life_lost or ep_len == 0:
                    obs, _, _, _, info = env.step(1)
                    preprocessed = preprocess_frame(obs)
                    frame_stack.append(preprocessed)
                    state_stack = frame_stack.get_stack()
                    life_lost = False

                state_tensor = (state_stack.clone().detach()
                                            .float()
                                            .unsqueeze(0)
                                            .div(255.0)
                                            .to(self.device))

                # exploration strategy injected here
                action = action_fn(self, state_tensor)

                next_obs, reward, done, truncated, info = env.step(action)

                if info["lives"] < curr_life:
                    life_lost = True
                    curr_life = info["lives"]

                ep_len              += 1
                self.total_env_steps += 1

                next_state = preprocess_frame(next_obs)
                frame_stack.append(next_state)
                next_state_stack = frame_stack.get_stack()

                if training:
                    self.replay_buffer.push(
                        (state_stack, action, reward,
                        next_state_stack, done or life_lost)
                    )

                    if self.replay_buffer.size() < self.min_buffer_size:
                        # Still warming up — just fill the buffer
                        if self.replay_buffer.size() % 10_000 == 0:
                            print(f"Filling replay buffer… "
                                f"{self.replay_buffer.size()}"
                                f"/{self.min_buffer_size}")
                        state_stack = next_state_stack
                        continue

                    if self.total_env_steps % self.update_freq == 0:
                        self.train_step()

                    if self.total_env_steps % self.target_update_freq == 0:
                        logger.info(
                            f"[Step {self.total_env_steps}] "
                            "Updating target network"
                        )
                        self.update_target_network()

                state_stack   = next_state_stack
                total_reward += reward
                sample_rewards.append(reward)   # store every step reward

                if done:
                    break

            # end of episode
            episode_rewards.append(total_reward)
            episode_lengths.append(ep_len)

            # Periodic console logging
            if episode % 100 == 0:
                avg_r   = np.mean(episode_rewards[-100:])
                avg_len = np.mean(episode_lengths[-100:])
                logger.info(
                    f"Episode {episode}/{self.num_episodes} | "
                    f"Steps {self.total_env_steps} | "
                    f"Avg100 reward {avg_r:.2f} | "
                    f"Avg100 length {avg_len:.1f}"
                )

            # Best-model checkpoint
            if episode % 500 == 0:
                avg_100 = np.mean(episode_rewards[-100:])
                if avg_100 > best_avg_reward:
                    best_avg_reward = avg_100
                    ckpt_path = os.path.join(
                        output_dir,
                        f"qnetwork_{self.experiment_name}_best.pth"
                    )
                    self.save_model(ckpt_path)
                    logger.info(
                        f"[Episode {episode}] New best avg100 "
                        f"{avg_100:.2f} → saved {ckpt_path}"
                    )

            # Periodic stats save (every 1000 episodes)
            if episode % 1000 == 0:
                save_stats(
                    {
                        "rewards":        episode_rewards,
                        "episode_length": episode_lengths,
                        "sample_rewards": sample_rewards,
                    },
                    stats_path,
                )
                logger.info(f"Stats saved → {stats_path}")

        # Final save at end of training
        save_stats(
            {
                "rewards":        episode_rewards,
                "episode_length": episode_lengths,
                "sample_rewards": sample_rewards,
            },
            stats_path,
        )
        logger.info(f"Training complete. Final stats → {stats_path}")

        return episode_rewards, episode_lengths, sample_rewards

    # Gradient update
    def train_step(self):
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.stack(states).float().div(255.0).to(self.device)
        next_states = torch.stack(next_states).float().div(255.0).to(self.device)
        actions     = torch.tensor(actions).long().unsqueeze(1).to(self.device)
        rewards     = torch.tensor(rewards).float().unsqueeze(1).to(self.device)
        dones       = torch.tensor(dones).float().unsqueeze(1).to(self.device)

        q_values      = self.q_network(states)
        next_q_values = self.target_network(next_states)

        q_value      = q_values.gather(1, actions)
        next_q_value = next_q_values.max(1)[0].unsqueeze(1)
        target       = rewards + self.gamma * next_q_value * (1 - dones)

        loss = self.loss_fn(q_value, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_grad_update_steps += 1

    # Evaluation
    def evaluate(self, env, num_episodes=5, debug=False):
        """Greedy evaluation — no exploration, no training."""
        frame_stack    = FrameStack(self.hyperparams.FRAME_STACK)
        total_rewards  = []

        for _ in range(num_episodes):
            obs, info = env.reset()
            state = preprocess_frame(obs)
            frame_stack.reset()
            for _ in range(self.hyperparams.FRAME_STACK):
                frame_stack.append(state)
            state_stack  = frame_stack.get_stack()
            ep_reward    = 0.0
            done         = False
            step_cnt     = 0
            curr_life    = 5
            ep_len       = 0
            life_lost    = False

            while not done:
                if life_lost or ep_len == 0:
                    obs, _, _, _, info = env.step(1)
                    frame_stack.append(preprocess_frame(obs))
                    state_stack = frame_stack.get_stack()
                    life_lost = False
                ep_len += 1

                state_tensor = (torch.tensor(state_stack, dtype=torch.float32)
                                .unsqueeze(0).div(255.0).to(self.device))
                action = self.greedy_action(state_tensor)

                next_obs, reward, done, truncated, info = env.step(action)
                if info["lives"] < curr_life:
                    life_lost = True
                    curr_life = info["lives"]
                if step_cnt == 30_000:
                    break
                if debug:
                    print(f"Step {step_cnt} | action {action} | "
                          f"reward {reward} | info {info}")

                frame_stack.append(preprocess_frame(next_obs))
                state_stack = frame_stack.get_stack()
                ep_reward  += reward
                step_cnt   += 1

            total_rewards.append(ep_reward)

        avg = np.mean(total_rewards)
        print(f"Eval over {num_episodes} episodes → avg reward: {avg:.2f}")
        return avg

    # Save model
    def save_model(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        self.q_network.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.q_network.eval()