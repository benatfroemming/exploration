import random

import numpy as np

from agent import Agent
from core import HyperParams, setup_logger


def _init_epsilon(agent):
    """Attach epsilon-greedy state to the agent the first time it's needed."""
    if not hasattr(agent, "_eg_epsilon"):
        hp = agent.hyperparams
        agent._eg_epsilon = hp.EPSILON_START
        agent._eg_epsilons = np.linspace(
            hp.EPSILON_START, hp.MIN_EPSILON, hp.EPSILON_DECAY_STEPS
        )

# Exploration function
def action_fn(agent, state_tensor):
    """
    Epsilon-greedy action selection.

    - With probability epsilon  → random action
    - With probability 1-epsilon → greedy (argmax Q)

    Epsilon is decayed linearly by the Agent.run() loop via the hook below;
    this function only reads the current value.
    """
    _init_epsilon(agent)
    if random.random() < agent._eg_epsilon:
        return random.randint(0, agent.action_space_size - 1)
    return agent.greedy_action(state_tensor)


def _make_action_fn_with_decay():
    """
    Returns an action_fn that also updates epsilon after each call so the
    decay happens in lock-step with env steps — identical to the notebook.
    """
    def _action_fn(agent, state_tensor):
        _init_epsilon(agent)

        # Decay epsilon (only after buffer is warm and only on trained steps)
        if agent.replay_buffer.size() >= agent.hyperparams.MIN_BUFFER_SIZE:
            idx = min(agent.total_env_steps, agent.hyperparams.EPSILON_DECAY_STEPS - 1)
            agent._eg_epsilon = agent._eg_epsilons[idx]
        else:
            agent._eg_epsilon = agent.hyperparams.EPSILON_START

        return action_fn(agent, state_tensor)

    return _action_fn

def add_args(parser):
    """Optional overrides for epsilon schedule via CLI."""
    parser.add_argument(
        "--epsilon-start", type=float, default=None,
        help="Override EPSILON_START (default: from HyperParams)"
    )
    parser.add_argument(
        "--epsilon-end", type=float, default=None,
        help="Override MIN_EPSILON (default: from HyperParams)"
    )
    parser.add_argument(
        "--epsilon-decay-steps", type=int, default=None,
        help="Override EPSILON_DECAY_STEPS (default: from HyperParams)"
    )

# Train
def train(args, hyperparams: HyperParams, env, logger):
    """
    Set up and run an epsilon-greedy DQN training session.

    Parameters
    ----------
    args        : argparse.Namespace
    hyperparams : HyperParams
    env         : Gymnasium env
    logger      : logging.Logger

    Returns
    -------
    episode_rewards  : list[float]
    episode_lengths  : list[int]
    sample_rewards   : list[float]
    """
    # Apply any CLI overrides to the epsilon schedule
    if getattr(args, "epsilon_start", None) is not None:
        hyperparams.EPSILON_START = args.epsilon_start
    if getattr(args, "epsilon_end", None) is not None:
        hyperparams.MIN_EPSILON = args.epsilon_end
    if getattr(args, "epsilon_decay_steps", None) is not None:
        hyperparams.EPSILON_DECAY_STEPS = args.epsilon_decay_steps

    logger.info("Exploration method: EPSILON-GREEDY")
    logger.info(
        f"Epsilon schedule: {hyperparams.EPSILON_START} → "
        f"{hyperparams.MIN_EPSILON} over {hyperparams.EPSILON_DECAY_STEPS} steps"
    )
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
        action_fn=_make_action_fn_with_decay(),
        logger=logger,
        output_dir=args.output_dir,
        training=True,
    )

    return rewards, lengths, sample_rewards