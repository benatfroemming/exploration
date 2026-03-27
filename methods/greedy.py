from agent import Agent
from core import HyperParams

def add_args(parser):
    pass  # no extra args

def action_fn(agent, state_tensor):
    return agent.greedy_action(state_tensor)

def train(args, hyperparams, env, logger):
    logger.info("method: greedy")
    agent = Agent(args.experiment, hyperparams, env.action_space.n, args.device)
    return agent.run(env=env, action_fn=action_fn, logger=logger, output_dir=args.output_dir)