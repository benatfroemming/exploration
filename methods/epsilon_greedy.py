import random
import numpy as np
from agent import Agent
from core import HyperParams
from dataclasses import dataclass

@dataclass
class EpsilonParams:
    start: float = 1.0
    end:   float = 0.1
    steps: int   = 1_000_000

def add_args(parser):
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-end",   type=float, default=None)
    parser.add_argument("--epsilon-steps", type=int,   default=None)

def action_fn(agent, state_tensor):
    if random.random() < agent._epsilon:
        return random.randint(0, agent.action_space_size - 1)
    return agent.greedy_action(state_tensor)

def _make_action_fn(ep: EpsilonParams):
    schedule = np.linspace(ep.start, ep.end, ep.steps)

    def _fn(agent, state_tensor):
        if agent.replay_buffer.size() >= agent.hyperparams.MIN_BUFFER_SIZE:
            idx = min(agent.total_env_steps, ep.steps - 1)
            agent._epsilon = schedule[idx]
        else:
            agent._epsilon = ep.start
        return action_fn(agent, state_tensor)

    return _fn

def train(args, hyperparams, env, logger):
    ep = EpsilonParams(
        start=getattr(args, "epsilon_start", None) or EpsilonParams.start,
        end=  getattr(args, "epsilon_end",   None) or EpsilonParams.end,
        steps=getattr(args, "epsilon_steps", None) or EpsilonParams.steps,
    )
    logger.info(f"method: epsilon-greedy  {ep.start}→{ep.end} over {ep.steps} steps")

    agent = Agent(args.experiment, hyperparams, env.action_space.n, args.device)
    agent._epsilon = ep.start
    return agent.run(env=env, action_fn=_make_action_fn(ep), logger=logger, output_dir=args.output_dir)