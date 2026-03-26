"""
train.py — CLI entrypoint for DQN exploration method comparison.

Usage
-----
# Epsilon-greedy for 5000 episodes
python train.py --method epsilon_greedy --episodes 5000 --experiment eg_run1

# Pure greedy for 5000 episodes
python train.py --method greedy --episodes 5000 --experiment greedy_run1

# Resume from checkpoint
python train.py --method epsilon_greedy --episodes 5000 --checkpoint path/to/model.pth

# Override epsilon schedule
python train.py --method epsilon_greedy --episodes 5000 \\
    --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay-steps 500000

Run `python train.py --help` for all options.
"""

import argparse
import importlib
import os
import sys

import ale_py
import gymnasium as gym
import torch

from core import HyperParams, setup_logger


# ---------------------------------------------------------------------------
# Available methods — add new entries here when adding a new method module
# ---------------------------------------------------------------------------

METHODS = {
    "greedy":           "methods.greedy",
    "epsilon_greedy":   "methods.epsilon_greedy",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train DQN on ALE/Breakout with a chosen exploration method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--method", required=True, choices=list(METHODS.keys()),
        help="Exploration method to use."
    )

    # Training control
    parser.add_argument(
        "--episodes", type=int, default=5_000,
        help="Number of episodes to train for."
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment name used for checkpoint filenames and log names. "
             "Defaults to '<method>_<episodes>ep'."
    )

    # Paths
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for checkpoints, stats, and logs. "
             "Defaults to 'runs/<experiment>'."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a .pth checkpoint to resume from."
    )

    # Hardware
    parser.add_argument(
        "--device", type=str, default=None,
        help="torch device string, e.g. 'cuda', 'cuda:1', 'cpu'. "
             "Auto-detected if omitted."
    )

    # Environment
    parser.add_argument(
        "--env", type=str, default="ALE/Breakout-v5",
        help="Gymnasium environment ID."
    )

    # Parse known args first so method modules can add their own
    args, remaining = parser.parse_known_args(argv)

    # Resolve defaults that depend on other args
    if args.experiment is None:
        args.experiment = f"{args.method}_{args.episodes}ep"
    if args.output_dir is None:
        args.output_dir = os.path.join("runs", args.experiment)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Let the chosen method add and parse its own arguments
    method_module = importlib.import_module(METHODS[args.method])
    method_module.add_args(parser)
    args = parser.parse_args(argv)   # re-parse with method args registered

    # Re-apply defaults (parse_known_args doesn't set them a second time)
    if args.experiment is None:
        args.experiment = f"{args.method}_{args.episodes}ep"
    if args.output_dir is None:
        args.output_dir = os.path.join("runs", args.experiment)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args, method_module


def main(argv=None):
    args, method_module = parse_args(argv)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    logger = setup_logger(
        log_dir=args.output_dir,
        log_name=f"{args.experiment}_training",
    )
    logger.info("=" * 60)
    logger.info(f"Experiment  : {args.experiment}")
    logger.info(f"Method      : {args.method}")
    logger.info(f"Episodes    : {args.episodes}")
    logger.info(f"Device      : {args.device}")
    logger.info(f"Output dir  : {args.output_dir}")
    logger.info(f"Checkpoint  : {args.checkpoint or 'none'}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Hyperparameters  (patch NUM_EPISODES from CLI)
    # ------------------------------------------------------------------
    hyperparams = HyperParams()
    hyperparams.NUM_EPISODES = args.episodes

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    gym.register_envs(ale_py)
    env = gym.make(args.env, render_mode="rgb_array")
    logger.info(f"Environment : {args.env}  "
                f"(action_space={env.action_space.n})")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    rewards, lengths, sample_rewards = method_module.train(
        args=args,
        hyperparams=hyperparams,
        env=env,
        logger=logger,
    )

    env.close()

    logger.info(f"Done. Total episodes: {len(rewards)}")
    logger.info(f"Final avg100 reward : "
                f"{sum(rewards[-100:]) / min(len(rewards), 100):.2f}")


if __name__ == "__main__":
    main()