# core.py
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torchvision.transforms as T

# Hyperparameters
class SharedHyperParams:
    """
    Hyperparameters shared by every exploration strategy.
    Strategy-specific params (epsilon schedule, UCB c, etc.) belong
    in the corresponding exploration/*.py file.
    """

    def __init__(self):
        # Replay buffer
        self.BUFFER_SIZE = 500_000
        self.MIN_BUFFER_SIZE = 50_000
        self.BATCH_SIZE = 32

        # Target network
        self.TARGET_UPDATE = 10_000       # steps between target syncs

        # Optimisation
        self.LR = 1e-4
        self.GAMMA = 0.99
        self.UPDATE_FREQ = 4              # gradient update every N env steps

        # Frame stack & environment
        self.FRAME_STACK = 4
        self.STATE_DIM = (4, 84, 84)

        # Training budget
        self.NUM_EPISODES = 5_000
        self.MAX_TIMESTEPS = 8_000_000
        self.MAX_EPISODE_LENGTH = 20_000

        # Misc
        self.SEED = 42

# Preprocessing
def preprocess_frame(obs: np.ndarray) -> torch.Tensor:
    """
    Convert a raw RGB Atari frame (H x W x C, uint8) to a
    grayscale 84×84 tensor (1 x 84 x 84, uint8).

    Follows the preprocessing from Mnih et al. (2015):
      - Grayscale conversion
      - Crop to 160×160 (removes score bar)
      - Resize to 84×84
    """
    obs = torch.tensor(obs)
    obs = obs.permute(2, 0, 1).float()                       # HWC → CHW
    obs = T.functional.rgb_to_grayscale(obs)                 # [1, 210, 160]
    obs = obs[:, 34:34 + 160, :]                             # crop → [1, 160, 160]
    obs = T.functional.resize(
        obs, (84, 84), interpolation=T.InterpolationMode.NEAREST
    )
    return obs.to(torch.uint8)                               # [1, 84, 84]

# DQN network  (Mnih et al. 2015)
class DQN(nn.Module):
    """
    Three convolutional layers followed by two fully-connected layers.
    Input:  [B, 4, 84, 84]  (4 stacked grayscale frames, normalised to [0,1])
    Output: [B, action_dim] (Q-values for each action)
    """

    def __init__(self, action_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),   # → [B, 32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → [B, 64, 9, 9]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → [B, 64, 7, 7]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),                         # 3136 = 64*7*7
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
# Bootstrapped DQN network, used by thompson and ucb
class BootstrappedDQN(nn.Module):
    def __init__(self, action_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.action_dim = action_dim

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        self.heads = nn.ModuleList([
            nn.Linear(512, action_dim) for _ in range(num_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feature_extractor(x)
        q_list = [head(z) for head in self.heads]
        return torch.stack(q_list, dim=1)   # [B, K, A]

# Frame stack
class FrameStack:
    """Maintains a sliding window of the last k preprocessed frames."""

    def __init__(self, k: int = 4):
        self.k = k
        self.frames: deque = deque([], maxlen=k)

    def reset(self) -> None:
        self.frames.clear()

    def append(self, obs: torch.Tensor) -> None:
        self.frames.append(obs)

    def get_stack(self) -> torch.Tensor:
        assert len(self.frames) == self.k, (
            f"FrameStack has {len(self.frames)} frames, expected {self.k}"
        )
        return torch.cat(list(self.frames), dim=0)  # [k, 84, 84]


# Replay buffer
class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, capacity: int, seed: int | None = None):
        self.buffer: deque = deque([], maxlen=capacity)
        if seed is not None:
            random.seed(seed)

    def push(self, experience: tuple) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)