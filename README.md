# Exploration in RL

Modular Deep Q-Network implementation based on  
[Mnih et al. (2015) — *Human-level control through deep reinforcement learning*](https://www.nature.com/articles/nature14236).

---

## Repository structure

```
.
├── dqn_base_model.py   # DQN network, replay buffer, frame preprocessing, Agent class
├── exploration.py      # Pluggable exploration strategies (greedy, epsilon_greedy)
├── train.py            # Training script  — 5 000 episodes, configurable exploration
├── eval.py             # Evaluation script — single greedy episode + video recording
├── results.ipynb       # Jupyter notebook: plots & performance table
│
├── training_logs/      # Per-run JSONL logs  (auto-created)
│   └── <exploration>_<run>.jsonl
├── policies/           # Saved policy checkpoints  (auto-created)
│   ├── <exploration>_<run>_best.pth
│   └── <exploration>_<run>_final.pth
└── runs/               # Evaluation videos  (auto-created)
    └── <exploration>_<run>/
```

---

## Installation

```bash
pip install torch torchvision gymnasium[atari] ale-py
# Optional but recommended for GPU:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Training

```bash
# Default: greedy exploration on Breakout, run 1
python train.py

# Epsilon-greedy exploration
python train.py --exploration epsilon_greedy

# Different environment, run index 2
python train.py --exploration epsilon_greedy --env ALE/Pong-v5 --run 2

# Override the random seed
python train.py --exploration greedy --seed 0
```

**Arguments**

| Flag | Default | Description |
|------|---------|-------------|
| `--exploration` | `greedy` | Exploration strategy: `greedy` or `epsilon_greedy` |
| `--env` | `ALE/Breakout-v5` | Any ALE Gymnasium environment id |
| `--run` | `1` | Integer label appended to log / policy file names |
| `--seed` | `42` | Random seed (overrides `HyperParams.SEED`) |

Training always runs for **exactly 5 000 episodes**.  
Logs are flushed to `training_logs/<exploration>_<run>.jsonl` every 500 episodes.  
The best-performing policy (by 100-episode average) is saved to `policies/` automatically.

---

## Evaluation

Loads a saved policy, runs one greedy episode, and saves an MP4 to `runs/`.

```bash
# Evaluate a previously trained policy
python eval.py --policy policies/epsilon_greedy_1_best.pth

# With explicit labels (used for the output folder name only)
python eval.py --policy policies/epsilon_greedy_2_best.pth \
               --exploration epsilon_greedy --env ALE/Breakout-v5 --run 2
```

**Arguments**

| Flag | Default | Description |
|------|---------|-------------|
| `--policy` | *(required)* | Path to a `.pth` policy file |
| `--exploration` | `greedy` | Label for the output video folder |
| `--env` | `ALE/Breakout-v5` | Environment to evaluate on |
| `--run` | `1` | Run index for the output folder name |

The video is saved to `runs/<exploration>_<run>/`.

---

## Results notebook

Open `results.ipynb` with Jupyter (or VS Code) after training to generate:

- **Reward vs Episode** — raw + 100-episode rolling mean
- **Reward vs Environment Samples** — sample-efficiency curve
- **Episode Length vs Episode**
- **Epsilon Decay** schedule
- **Max vs Average reward** per 100-episode block
- **Performance summary table** (mean, max, std of reward, AUC, sample efficiency, learning-onset episode)

```bash
jupyter notebook results.ipynb
# or
jupyter lab results.ipynb
```

---

## Adding a new exploration strategy

1. Open `exploration.py`.
2. Create a class that inherits `ExplorationStrategy` and implements `name` and `select_action`.
3. Register it in `_REGISTRY`.

```python
class BoltzmannExploration(ExplorationStrategy):
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    @property
    def name(self):
        return "boltzmann"

    def select_action(self, q_network, state_tensor):
        with torch.no_grad():
            q = q_network(state_tensor).squeeze(0)
        probs = torch.softmax(q / self.temperature, dim=0).cpu().numpy()
        return int(np.random.choice(len(probs), p=probs))

_REGISTRY["boltzmann"] = BoltzmannExploration
```

Then run:
```bash
python train.py --exploration boltzmann
```

---

## Hyperparameters (fixed, see `dqn_base_model.py → HyperParams`)

| Parameter | Value |
|-----------|-------|
| Replay buffer | 500 000 transitions |
| Min buffer before training | 50 000 |
| Batch size | 32 |
| ε start / end / decay steps | 1.0 / 0.1 / 1 000 000 |
| Target network sync | every 10 000 steps |
| Learning rate | 1 × 10⁻⁴ (Adam) |
| Discount γ | 0.99 |
| Frame stack | 4 |
| Network update frequency | every 4 env steps |
| Max episode length | 20 000 steps |