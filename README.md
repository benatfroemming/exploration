# Exploration in Reinforcement Learning

This project is a modular PyTorch implementation of Deep Q-Network (DQN) for Atari environments via the [Arcade Learning Environment (ALE)](https://ale.farama.org/environments/), a testbed designed to make swapping exploration strategies easy. Based on DeepMind's original implementation [Mnih et al. (2015)](https://www.nature.com/articles/nature14236). The DQN parameters stay fixed to compare different exploration strategies.

<div align="center">
  <img src="./media/ALE_Breakout-v5_ep1.gif" alt="Alt Text">
</div>

## Setup

```bash
git clone https://github.com/benatfroemming/exploration.git
cd exploration
pip install -r requirements.txt
```

## Training

**Exploration strategies:**

- `epsilon_greedy` — Takes a random action with probability ε, decaying over time.
- `boltzmann` — Samples actions proportional to their Q-values via softmax with decaying temperature.
- `entropy_reg` — Adds a penalty to the training loss whenever the policy is too certain, rewarding uncertainty.
- `ucb` — Adds an uncertainty bonus to the Q-values that favors under-visited actions.
- `thompson` — Maintains an ensemble of Q-networks (bootstrapped heads) and samples one head per episode.
- `rnd` — Adds intrinsic curiosity to the rewards for novel states via predicting a fixed random network's output.

```bash
# Defaults: 5000 episodes, epsilon greedy, ALE/Breakout-v5, 42
python train.py

# Custom
python train.py --episodes 10000 --strategy boltzmann --env ALE/Pong-v5

# Resume from checkpoint
python train.py --checkpoint runs/.../dqn_ALE-Breakout-v5_epsilon_greedy_final.pth

# Set a seed
python train.py --seed 84
```

## Evaluation

Runs a saved policy greedily for one or more episodes and reports the total reward.

```bash
# Required: --policy, --strategy
# Defaults: 1 episode, ALE/Breakout-v5, not rendered

# Single episode
python eval.py --policy runs/.../model.pth --strategy epsilon_greedy

# Multiple episodes (also reports mean ± std and min/max)
python eval.py --policy runs/.../model.pth --strategy boltzmann --episodes 10

# Different environment
python eval.py --policy runs/.../model.pth --strategy rnd --env ALE/Pong-v5 --episodes 5

# Render flag saves one GIF per episode (e.g. ALE_Breakout-v5_ep1.gif) to current directory
python eval.py --policy runs/.../model.pth --strategy thompson --render --episodes 3
```

## Add a New Exploration Strategie

1) Create `exploration/your_strategy.py` following the `exploration/template.py`.
2) Register it in the `STRATEGIES` dict in `train.py` and `eval.py`.
3) Run with `--strategy your_strategy`.

```python
STRATEGIES: dict[str, str] = {
    "epsilon_greedy": "exploration.epsilon_greedy.EpsilonGreedyAgent",
    # add future strategies here:
}
```

## Fixed Hyperparameter

The components that don't change across exploration strategies, including the hyperparameters, policy network, and image processing, are in `core.py`.

| Parameter | Value | Description |
|----------|------|-------------|
| `BUFFER_SIZE` | 500,000 | Replay buffer capacity |
| `MIN_BUFFER_SIZE` | 50,000 | Warm-up before training starts |
| `BATCH_SIZE` | 32 | Training batch size |
| `TARGET_UPDATE` | 10,000 steps | Hard update frequency for target network |
| `LR` | 1e-4 | Learning rate (Adam optimizer) |
| `GAMMA` | 0.99 | Discount factor |
| `FRAME_STACK` | 4 | Number of stacked frames per state |
| `MAX_EPISODE_LENGTH` | 20,000 | Maximum steps per episode |
| `UPDATE_FREQ` | 4 | Perform a gradient step every N environment steps |
| `SEED` | 42 | Random seed for reproducibility |

## Experiments

Results are in `results.ipynb` and the experimental data can be found in this Google Drive [folder](https://drive.google.com/drive/folders/1qRDPAOwVGfdGju1yAcFSOH2zjnah2NZe?usp=sharing).
