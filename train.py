#!/usr/bin/env python

import argparse
import json
import os
import time

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from dqn_base_model import DQN, FrameStack, HyperParams, ReplayBuffer, preprocess_frame
from exploration import available_strategies, make_strategy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exploration", default="epsilon_greedy",
                   choices=available_strategies())
    p.add_argument("--env",  default="ALE/Breakout-v5")
    p.add_argument("--run",  type=int, default=1)
    return p.parse_args()


def to_tensor(stack, device):
    return stack.clone().detach().float().unsqueeze(0).div(255.0).to(device)


def gradient_update(q_net, target_net, replay_buffer, optimizer, loss_fn, hp, device):
    states, actions, rewards, next_states, dones = zip(*replay_buffer.sample(hp.BATCH_SIZE))

    states      = torch.stack(states).float().div(255.0).to(device)
    next_states = torch.stack(next_states).float().div(255.0).to(device)
    actions     = torch.tensor(actions).long().unsqueeze(1).to(device)
    rewards     = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    dones       = torch.tensor(dones,   dtype=torch.float32).unsqueeze(1).to(device)

    q_val = q_net(states).gather(1, actions)

    with torch.no_grad():
        next_q  = target_net(next_states).max(1)[0].unsqueeze(1)
        target  = rewards + hp.GAMMA * next_q * (1.0 - dones)

    loss = loss_fn(q_val, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    args    = parse_args()
    hp      = HyperParams()
    run_tag = f"{args.exploration}_{args.run}"

    os.makedirs("training_logs", exist_ok=True)
    os.makedirs("policies",      exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Env: {args.env} | Exploration: {args.exploration} | Device: {device}")

    gym.register_envs(ale_py)
    env               = gym.make(args.env, render_mode="rgb_array")
    action_space_size = env.action_space.n

    q_net      = DQN(action_space_size).to(device)
    target_net = DQN(action_space_size).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=hp.LR)
    loss_fn   = nn.HuberLoss()

    strategy      = make_strategy(args.exploration, action_space_size)
    replay_buffer = ReplayBuffer(hp.BUFFER_SIZE, hp.SEED)
    frame_stack   = FrameStack(hp.FRAME_STACK)

    episode_rewards  = []
    episode_lengths  = []
    cumulative_steps = []
    best_avg_reward  = -float("inf")
    total_steps      = 0
    start            = time.time()

    for episode in range(1, hp.NUM_EPISODES + 1):
        obs, info = env.reset()
        frame_stack.reset()
        state = preprocess_frame(obs)
        for _ in range(hp.FRAME_STACK):
            frame_stack.append(state)

        ep_reward = 0.0
        ep_len    = 0
        done      = False
        life_lost = False
        curr_life = info.get("lives", 5)

        while ep_len <= hp.MAX_EPISODE_LENGTH:

            # Fire to launch ball at episode start or after losing a life
            if life_lost or ep_len == 0:
                obs, _, _, _, info = env.step(1)
                frame_stack.append(preprocess_frame(obs))
                life_lost = False

            # Capture state BEFORE the env step
            current_state = frame_stack.get()

            state_tensor = to_tensor(current_state, device)

            # Fill buffer with random actions, otherwise use strategy
            if len(replay_buffer) < hp.MIN_BUFFER_SIZE:
                action = env.action_space.sample()
                if total_steps % 10_000 == 0:
                    print(f"  Filling buffer {len(replay_buffer)}/{hp.MIN_BUFFER_SIZE}")
            else:
                action = strategy.act(q_net, state_tensor)

            next_obs, reward, done, truncated, info = env.step(action)
            total_steps += 1
            ep_len      += 1

            # DeepMind clips rewards to [-1, 1]
            reward = np.clip(reward, -1.0, 1.0)

            if info.get("lives", curr_life) < curr_life:
                life_lost = True
                curr_life = info["lives"]

            # Append next frame AFTER capturing current_state
            frame_stack.append(preprocess_frame(next_obs))

            replay_buffer.push((
                current_state,          # s  — pre-step frame stack
                action,
                reward,
                frame_stack.get(),      # s' — post-step frame stack
                done or life_lost,      # terminal signal
            ))

            ep_reward += reward

            if len(replay_buffer) >= hp.MIN_BUFFER_SIZE:
                if total_steps % hp.UPDATE_FREQ == 0:
                    gradient_update(q_net, target_net, replay_buffer,
                                    optimizer, loss_fn, hp, device)
                    strategy.update()

                if total_steps % hp.TARGET_UPDATE == 0:
                    target_net.load_state_dict(q_net.state_dict())
                    print(f"  [Step {total_steps:,}] Target network synced")

            if done or truncated:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)
        cumulative_steps.append(total_steps)

        if episode % 100 == 0:
            avg_r = np.mean(episode_rewards[-100:])
            avg_l = np.mean(episode_lengths[-100:])
            print(f"[Ep {episode:>5}/{hp.NUM_EPISODES}] "
                  f"steps={total_steps:>9,} | avg_rew={avg_r:>7.2f} | "
                  f"avg_len={avg_l:>6.0f} | {(time.time()-start)/60:.1f}min")

        if episode % 500 == 0:
            avg_100 = np.mean(episode_rewards[-100:])
            if avg_100 > best_avg_reward:
                best_avg_reward = avg_100
                torch.save(q_net.state_dict(), f"policies/{run_tag}_best.pth")
                print(f"  -> Best avg-100: {best_avg_reward:.2f} saved")

            with open(f"training_logs/{run_tag}.jsonl", "w") as f:
                f.write(json.dumps({
                    "rewards":          episode_rewards,
                    "episode_lengths":  episode_lengths,
                    "cumulative_steps": cumulative_steps,
                }) + "\n")

    torch.save(q_net.state_dict(), f"policies/{run_tag}_final.pth")
    with open(f"training_logs/{run_tag}.jsonl", "w") as f:
        f.write(json.dumps({
            "rewards":          episode_rewards,
            "episode_lengths":  episode_lengths,
            "cumulative_steps": cumulative_steps,
        }) + "\n")

    print(f"Done in {(time.time()-start)/60:.0f}min")
    env.close()


if __name__ == "__main__":
    main()