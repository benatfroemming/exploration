# eval.py
import argparse
import os
import sys

import ale_py
import gymnasium as gym
import numpy as np
import torch

from core import DQN, FrameStack, SharedHyperParams, preprocess_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved DQN policy greedily on an Atari environment."
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to a saved policy .pth checkpoint file.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="ALE/Breakout-v5",
        help='Gymnasium ALE environment ID (default: "ALE/Breakout-v5").',
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes (default: 1).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation.",
    )
    return parser.parse_args()


def run_greedy_episode(
    env: gym.Env,
    policy_net: DQN,
    device: torch.device,
    frame_stack: FrameStack,
    max_episode_length: int = 20_000,
) -> float:
    """Run a single greedy episode (argmax Q, no exploration). Returns total reward."""
    obs, _ = env.reset()
    frame_stack.reset()

    frame = preprocess_frame(obs)
    for _ in range(frame_stack.k):
        frame_stack.append(frame)

    total_reward = 0.0
    policy_net.eval()

    for _ in range(max_episode_length):
        state = frame_stack.get_stack().unsqueeze(0).float().to(device) / 255.0

        with torch.no_grad():
            action = policy_net(state).argmax(dim=1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        frame_stack.append(preprocess_frame(obs))

        if terminated or truncated:
            break

    return total_reward


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.policy):
        raise FileNotFoundError(f"Checkpoint not found: {args.policy}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")
    print(f"Policy   : {args.policy}")
    print(f"Env      : {args.env}")
    print(f"Episodes : {args.episodes}\n")

    gym.register_envs(ale_py)
    env = gym.make(args.env, render_mode="human" if args.render else "rgb_array")
    action_dim = env.action_space.n

    checkpoint = torch.load(args.policy, map_location=device)
    state_dict = checkpoint.get("policy_net", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    policy_net = DQN(action_dim=action_dim).to(device)
    policy_net.load_state_dict(state_dict)

    shared_hp = SharedHyperParams()
    frame_stack = FrameStack(k=shared_hp.FRAME_STACK)

    rewards: list[float] = []
    for ep in range(1, args.episodes + 1):
        ep_reward = run_greedy_episode(
            env=env,
            policy_net=policy_net,
            device=device,
            frame_stack=frame_stack,
            max_episode_length=shared_hp.MAX_EPISODE_LENGTH,
        )
        rewards.append(ep_reward)
        print(f"Episode {ep:>4d} | Total reward: {ep_reward:.1f}")

    env.close()

    print(f"\n{'─' * 35}")
    print(f"Total reward : {sum(rewards):.1f}")
    if args.episodes > 1:
        print(f"Average      : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Min / Max    : {min(rewards):.1f} / {max(rewards):.1f}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()