# Exploration in Reinforcement Learning

A modular PyTorch implementation of DQN for Atari, designed to make swapping
exploration strategies easy. Based on [Mnih et al. (2015)](https://www.nature.com/articles/nature14236). The DQN parameters stay fixed to compare different exploration strategies.

## Setup

```bash
!git clone https://github.com/benatfroemming/exploration.git
%cd exploration
!pip install -r requirements.txt
```

## Training

```bash
# Defaults: 5000 episodes, epsilon-greedy, ALE/Breakout-v5
python train.py

# Custom
python train.py --episodes 10000 --strategy boltzmann --env ALE/Breakout-v5

# Resume from checkpoint
python train.py --checkpoint runs/.../dqn_ALE-Breakout-v5_epsilon_greedy_final.pth
```

## Fixed Hyperparameter

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
