# train.py
import argparse
import os
import sys
from datetime import datetime

import ale_py
import gymnasium as gym
import torch

import importlib
from core import SharedHyperParams

import random, numpy as np

# available exploration strategies
STRATEGIES: dict[str, str] = {
    "epsilon_greedy": "exploration.epsilon_greedy.EpsilonGreedyAgent",
    "boltzmann": "exploration.boltzmann.BoltzmannAgent",
    "entropy_reg": "exploration.entropy_reg.EntropyRegAgent",
    "thompson": "exploration.thompson.ThompsonAgent",
    "rnd": "exploration.rnd.RNDAgent",
    "ucb": "exploration.ucb.UCBAgent",
    # add future strategies here:
}

def _import_agent(dotted_path: str):
    """Dynamically import an agent class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def _make_run_dir(env_id: str, strategy: str, seed: int) -> str:
    env_slug = env_id.replace("/", "-").replace(" ", "_")
    run_dir = os.path.join("runs", env_slug, strategy)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on an Atari environment."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5_000,
        help="Number of training episodes (default: 5000).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="epsilon_greedy",
        choices=list(STRATEGIES.keys()),
        help="Exploration strategy (default: epsilon_greedy).",
    )
    parser.add_argument(
        "--env",
        type=str,
        # default="ALE/Breakout-v5",
        default="ALE/MontezumaRevenge-v5",
        help='Gymnasium environment ID (default: "ALE/Breakout-v5").',
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pth file to resume training from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main() -> None:
    args = parse_args()
    set_seed(args.seed) 
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"Env     : {args.env}")
    print(f"Strategy: {args.strategy}")
    print(f"Episodes: {args.episodes}")

    # environment
    gym.register_envs(ale_py)
    env = gym.make(args.env, render_mode="rgb_array")
    action_dim = env.action_space.n

    # shared hyperparameters
    sys.path.insert(0, os.path.dirname(__file__))
    shared_hp = SharedHyperParams()
    shared_hp.NUM_EPISODES = args.episodes
    shared_hp.SEED = args.seed

    # output paths
    stem = f"{args.strategy}_{args.seed}_{args.episodes}"
    run_dir = _make_run_dir(args.env, args.strategy, args.seed)
    log_path = os.path.join(run_dir, f"{stem}.jsonl")
    model_dir = run_dir
    print(f"Run dir : {run_dir}\n")
    
    AgentClass = _import_agent(STRATEGIES[args.strategy])
    module_path = STRATEGIES[args.strategy].rsplit(".", 1)[0]
    ExploreHP = _import_agent(f"{module_path}.HyperParams")
    agent = AgentClass(
        env_id=args.env,
        shared_hp=shared_hp,
        explore_hp=ExploreHP(),
        action_dim=action_dim,
        device=device,
        checkpoint=args.checkpoint,
    )

    # train
    agent.train(
        env=env,
        num_episodes=args.episodes,
        log_path=log_path,
        model_dir=model_dir,
    )

    env.close()


if __name__ == "__main__":
    main()
