# Exploration in Reinforcement Learning

A modular PyTorch implementation of DQN for Atari environments via the [Arcade Learning Environment (ALE)](https://ale.farama.org/environments/), designed to make swapping exploration strategies easy. Based on Deep Mind's original implementation [Mnih et al. (2015)](https://www.nature.com/articles/nature14236). The DQN parameters stay fixed to compare different exploration strategies.

## Setup

```bash
!git clone https://github.com/benatfroemming/exploration.git
%cd exploration
!pip install -r requirements.txt
```

## Training

**Exploration strategies:** `epsilon_greedy`, `boltzmann`, `rnd`, `ucb`, `thompson`

```bash
# Defaults: 5000 episodes, epsilon greedy, ALE/Breakout-v5
python train.py

# Custom
python train.py --episodes 10000 --strategy boltzmann --env ALE/Pong-v5

# Resume from checkpoint
python train.py --checkpoint runs/.../dqn_ALE-Breakout-v5_epsilon_greedy_final.pth
```

## Add a New Exploration Strategie

1) Create `exploration/your_strategy.py` following the `exploration/template.py`.
2) Register it in the STRATEGIES dict in `train.py`.
3) Run with `--strategy your_strategy`.

```python
# train.py
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

Results are in `results.ipynb` and the experimental data can be found in this [Google Drive folder](https://drive.google.com/drive/folders/1qRDPAOwVGfdGju1yAcFSOH2zjnah2NZe?usp=sharing).
