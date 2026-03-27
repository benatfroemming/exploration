import torch
import torch.nn.functional as F
from agent import Agent
from dataclasses import dataclass

@dataclass
class BoltzmannParams:
    temperature: float = 1.0
    min_temp:    float = 0.1
    decay:       float = 0.99995  # multiplicative per episode

def add_args(parser):
    parser.add_argument("--temperature",     type=float, default=None)
    parser.add_argument("--min-temperature", type=float, default=None)
    parser.add_argument("--temp-decay",      type=float, default=None)

def action_fn(agent, state_tensor):
    with torch.no_grad():
        q = agent.q_network(state_tensor).squeeze(0)
    probs = F.softmax(q / agent._temperature, dim=0).cpu().numpy()
    return int(agent._rng.choice(len(probs), p=probs))

def _make_action_fn(bp: BoltzmannParams):
    import numpy as np
    def _fn(agent, state_tensor):
        # decay temperature each env step
        agent._temperature = max(bp.min_temp, agent._temperature * bp.decay)
        return action_fn(agent, state_tensor)
    return _fn

def train(args, hyperparams, env, logger):
    import numpy as np
    bp = BoltzmannParams(
        temperature=getattr(args, "temperature",     None) or BoltzmannParams.temperature,
        min_temp=   getattr(args, "min_temperature", None) or BoltzmannParams.min_temp,
        decay=      getattr(args, "temp_decay",      None) or BoltzmannParams.decay,
    )
    logger.info(f"method: boltzmann  T={bp.temperature}→{bp.min_temp}  decay={bp.decay}")

    agent = Agent(args.experiment, hyperparams, env.action_space.n, args.device)
    agent._temperature = bp.temperature
    agent._rng = np.random.default_rng(hyperparams.SEED)
    return agent.run(env=env, action_fn=_make_action_fn(bp), logger=logger, output_dir=args.output_dir)