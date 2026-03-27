# train.py
import argparse
import importlib
import os
import torch
import ale_py
import gymnasium as gym

from core import HyperParams, setup_logger

METHODS = {
    "greedy":         "methods.greedy",
    "epsilon_greedy": "methods.epsilon_greedy",
    "boltzmann":      "methods.boltzmann",
}

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",   required=True, choices=list(METHODS.keys()))
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--env",      type=str, default="ALE/Breakout-v5")

    args, _ = parser.parse_known_args(argv)

    method_module = importlib.import_module(METHODS[args.method])
    method_module.add_args(parser)
    args = parser.parse_args(argv)

    # derived defaults
    args.experiment = f"{args.method}_{args.episodes}ep"
    args.output_dir = os.path.join("runs", args.experiment)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args, method_module


def main(argv=None):
    args, method_module = parse_args(argv)

    logger = setup_logger(args.output_dir, f"{args.experiment}_training")
    logger.info(f"method={args.method}  episodes={args.episodes}  env={args.env}  device={args.device}")

    hyperparams = HyperParams()
    hyperparams.NUM_EPISODES = args.episodes

    gym.register_envs(ale_py)
    env = gym.make(args.env, render_mode="rgb_array")

    rewards, lengths, sample_rewards = method_module.train(
        args=args, hyperparams=hyperparams, env=env, logger=logger,
    )

    env.close()
    logger.info(f"done. avg100={sum(rewards[-100:]) / min(len(rewards), 100):.2f}")


if __name__ == "__main__":
    main()