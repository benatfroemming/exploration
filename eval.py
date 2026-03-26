#!/usr/bin/env python
# eval.py
# Usage:
#   python eval.py --policy policies/epsilon_greedy_1_best.pth
#   python eval.py --policy policies/greedy_1_best.pth --env ALE/Pong-v5 --run 2

import argparse
import os

import ale_py
import gymnasium as gym
import torch

from dqn_base_model import DQN, FrameStack, HyperParams, preprocess_frame
from exploration import available_strategies


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy",      required=True)
    p.add_argument("--exploration", default="greedy", choices=available_strategies(),
                   help="Label for the output video folder.")
    p.add_argument("--env",  default="ALE/Breakout-v5")
    p.add_argument("--run",  type=int, default=1)
    return p.parse_args()


def to_tensor(stack, device):
    return stack.clone().detach().float().unsqueeze(0).div(255.0).to(device)


def main():
    args      = parse_args()
    hp        = HyperParams()
    video_dir = os.path.join("runs", f"{args.exploration}_{args.run}")
    os.makedirs(video_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Policy: {args.policy} | Env: {args.env} | Video: {video_dir}")

    gym.register_envs(ale_py)
    env = gym.wrappers.RecordVideo(
        gym.make(args.env, render_mode="rgb_array"),
        video_folder    = video_dir,
        episode_trigger = lambda _: True,
        name_prefix     = f"{args.exploration}_{args.run}",
    )

    q_net = DQN(env.action_space.n).to(device)
    q_net.load_state_dict(torch.load(args.policy, map_location=device))
    q_net.eval()

    frame_stack = FrameStack(hp.FRAME_STACK)
    obs, info   = env.reset()
    frame_stack.reset()
    state = preprocess_frame(obs)
    for _ in range(hp.FRAME_STACK):
        frame_stack.append(state)

    total_reward = 0.0
    done         = False
    step         = 0
    life_lost    = False
    curr_life    = info.get("lives", 5)

    while not done and step < 30_000:
        if life_lost or step == 0:
            obs, _, _, _, info = env.step(1)
            frame_stack.append(preprocess_frame(obs))
            life_lost = False

        with torch.no_grad():
            action = q_net(to_tensor(frame_stack.get(), device)).argmax().item()

        next_obs, reward, done, truncated, info = env.step(action)

        if info.get("lives", curr_life) < curr_life:
            life_lost = True
            curr_life = info["lives"]

        frame_stack.append(preprocess_frame(next_obs))
        total_reward += reward
        step         += 1

    print(f"Reward: {total_reward:.1f} | Steps: {step}")
    print(f"Video saved to: {video_dir}/")
    env.close()


if __name__ == "__main__":
    main()