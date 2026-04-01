# eval.py
import argparse
import importlib
import os
import sys

import ale_py
import gymnasium as gym
import torch
import imageio

from core import SharedHyperParams

STRATEGIES: dict[str, str] = {
    "epsilon_greedy": "exploration.epsilon_greedy.EpsilonGreedyAgent",
    "boltzmann": "exploration.boltzmann.BoltzmannAgent",
    "entropy_reg": "exploration.entropy_reg.EntropyRegAgent",
    "thompson": "exploration.thompson.ThompsonAgent",
    "rnd": "exploration.rnd.RNDAgent",
    "ucb": "exploration.ucb.UCBAgent",
    # add future strategies here
}

def _import_agent(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def _save_render(all_episodes_frames: list[list], env_id: str) -> None:
    base = env_id.replace("/", "_")
    for i, frames in enumerate(all_episodes_frames, start=1):
        fname = f"{base}_ep{i}.gif"
        imageio.mimsave(fname, frames, fps=30)
        print(f"Saved render → {fname}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved DQN policy greedily on an Atari environment."
    )
    parser.add_argument("--policy", type=str, required=True,
                        help="Path to a saved .pth checkpoint file.")
    parser.add_argument("--env", type=str, default="ALE/Breakout-v5",
                        help='ALE environment ID (default: "ALE/Breakout-v5").')
    parser.add_argument("--strategy", type=str, default="epsilon_greedy",
                        choices=list(STRATEGIES.keys()),
                        help="Strategy the policy was trained with (default: epsilon_greedy).")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of evaluation episodes (default: 1).")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.policy):
        raise FileNotFoundError(f"Checkpoint not found: {args.policy}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")
    print(f"Policy   : {args.policy}")
    print(f"Env      : {args.env}")
    print(f"Strategy : {args.strategy}")
    print(f"Episodes : {args.episodes}\n")

    gym.register_envs(ale_py)
    env = gym.make(args.env, render_mode="rgb_array")
    action_dim = env.action_space.n

    shared_hp = SharedHyperParams()
    AgentClass = _import_agent(STRATEGIES[args.strategy])
    module_path = STRATEGIES[args.strategy].rsplit(".", 1)[0]
    ExploreHP = _import_agent(f"{module_path}.HyperParams")

    agent = AgentClass(
        env_id=args.env,
        shared_hp=shared_hp,
        explore_hp=ExploreHP(),
        action_dim=action_dim,
        device=device,
        checkpoint=args.policy,  # loads weights in __init__
    )

    results = agent.evaluate(env=env, num_episodes=args.episodes, record=args.render)

    env.close()
    
    print(args.render)
    print(results['frames'])
    if args.render and "frames" in results:
        _save_render(results["frames"], args.env)
    
    print(f"Total reward : {results['total_reward']:.1f}")
    if args.episodes > 1:
        print(f"Average      : {results['mean']:.2f} ± {results['std']:.2f}")
        print(f"Min / Max    : {results['min']:.1f} / {results['max']:.1f}")

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()