# DQN Exploration Method Comparison

Modular DQN training suite for comparing exploration strategies on ALE/Breakout.  
The DQN architecture, replay buffer, and training loop are **fixed** across all methods —
only the action-selection logic changes.

---

## Project layout

```
dqn/
├── core.py                  # DQN network, FrameStack, ReplayBuffer,
│                            # HyperParams, preprocess_frame, logging utils
├── agent.py                 # Agent class — fixed training loop,
│                            # receives action_fn at runtime
├── methods/
│   ├── greedy.py            # Pure greedy (argmax Q, no randomness)
│   └── epsilon_greedy.py    # Linear ε-decay (matches original notebook)
└── train.py                 # CLI entrypoint
```

---

## Quick start

```bash
# Epsilon-greedy — 5000 episodes (original notebook behaviour)
python train.py --method epsilon_greedy --episodes 5000

# Pure greedy baseline
python train.py --method greedy --episodes 5000

# Custom experiment name + output directory
python train.py --method epsilon_greedy --episodes 5000 \
    --experiment eg_seed42 --output-dir results/eg_seed42

# Resume from checkpoint
python train.py --method epsilon_greedy --episodes 5000 \
    --checkpoint runs/epsilon_greedy_5000ep/qnetwork_epsilon_greedy_5000ep_best.pth

# Override epsilon schedule
python train.py --method epsilon_greedy --episodes 5000 \
    --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay-steps 500000

# Force CPU / specific GPU
python train.py --method epsilon_greedy --episodes 5000 --device cpu
python train.py --method epsilon_greedy --episodes 5000 --device cuda:1

# Full help
python train.py --help
python train.py --method epsilon_greedy --help
```

---

## Outputs

All outputs land in `--output-dir` (default: `runs/<experiment>/`):

| File | Contents |
|---|---|
| `training_stats.jsonl` | JSON line written every 1000 episodes and at the end. Fields: `rewards`, `episode_length`, `sample_rewards` |
| `qnetwork_<experiment>_best.pth` | Checkpoint of the best avg-100 model |
| `<experiment>_training.log` | Full timestamped log |

### Loading stats for plotting

```python
import json, numpy as np, matplotlib.pyplot as plt

with open("runs/epsilon_greedy_5000ep/training_stats.jsonl") as f:
    data = json.loads(f.readlines()[-1])   # last (= final) snapshot

rewards        = data["rewards"]
episode_length = data["episode_length"]
sample_rewards = data["sample_rewards"]   # one per env step

# Avg-100 smoothing
avg100 = [np.mean(rewards[max(0,i-100):i]) for i in range(1, len(rewards)+1)]

plt.plot(avg100)
plt.xlabel("Episode"); plt.ylabel("Avg-100 reward")
plt.title("Epsilon-greedy — Breakout")
plt.savefig("avg100_rewards.png")
```

---

## Adding a new exploration method

1. Create `methods/your_method.py` with these three symbols:

```python
def action_fn(agent, state_tensor) -> int:
    ...

def add_args(parser):
    # add method-specific argparse arguments (or just pass)
    ...

def train(args, hyperparams, env, logger):
    # build Agent, call agent.run(..., action_fn=...), return rewards, lengths, sample_rewards
    ...
```

2. Register it in `train.py`:

```python
METHODS = {
    "greedy":         "methods.greedy",
    "epsilon_greedy": "methods.epsilon_greedy",
    "your_method":    "methods.your_method",   # ← add this
}
```

That's it — no other files need touching.

---

## Hyperparameter defaults

| Parameter | Value | Notes |
|---|---|---|
| `BUFFER_SIZE` | 500 000 | |
| `MIN_BUFFER_SIZE` | 50 000 | warm-up before training starts |
| `BATCH_SIZE` | 32 | |
| `EPSILON_START` | 1.0 | ε-greedy only |
| `MIN_EPSILON` | 0.1 | ε-greedy only |
| `EPSILON_DECAY_STEPS` | 1 000 000 | ε-greedy only |
| `TARGET_UPDATE` | 10 000 steps | hard copy |
| `LR` | 0.0001 | Adam |
| `GAMMA` | 0.99 | |
| `FRAME_STACK` | 4 | |
| `MAX_EPISODE_LENGTH` | 20 000 | |
| `UPDATE_FREQ` | 4 | gradient step every N env steps |
| `SEED` | 42 | replay buffer seed |