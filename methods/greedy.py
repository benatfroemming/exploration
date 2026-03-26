import os

import ale_py
import gymnasium as gym

from agent import Agent
from core import HyperParams, setup_logger


# ---------------------------------------------------------------------------
# Exploration function
# ---------------------------------------------------------------------------

def action_fn(agent, state_tensor):
    """Always return the greedy (argmax Q) action — no randomness."""
    return agent.greedy_action(state_tensor)


# ---------------------------------------------------------------------------
# Method-specific CLI arguments
# ---------------------------------------------------------------------------

def add_args(parser):
    """
    Greedy has no extra hyperparameters; this hook exists for API consistency
    so train.py can call method.add_args(parser) uniformly.
    """
    pass

# Train
def train(args, hyperparams: HyperParams, env, logger):
    """
    Set up and run a greedy-exploration DQN training session.

    Parameters
    ----------
    args        : argparse.Namespace — parsed CLI arguments
    hyperparams : HyperParams        — shared hyperparams (already patched
                                       with NUM_EPISODES from --episodes)
    env         : Gymnasium env
    logger      : logging.Logger

    Returns
    -------
    episode_rewards  : list[float]
    episode_lengths  : list[int]
    sample_rewards   : list[float]
    """
    logger.info("Exploration method: GREEDY (no randomness)")
    logger.info(f"Episodes: {hyperparams.NUM_EPISODES}")

    agent = Agent(
        experiment_name=args.experiment,
        hyperparams=hyperparams,
        action_space_size=env.action_space.n,
        device=args.device,
        chkpt=getattr(args, "checkpoint", None),
    )

    logger.info(f"Q-network architecture:\n{agent.q_network}")

    rewards, lengths, sample_rewards = agent.run(
        env=env,
        action_fn=action_fn,
        logger=logger,
        output_dir=args.output_dir,
        training=True,
    )

    return rewards, lengths, sample_rewards