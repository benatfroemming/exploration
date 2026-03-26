# dqn_base_model.py
# Standard DQN components: hyperparameters, frame preprocessing,
# frame stacking, replay buffer, and the Q-network.

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T


class HyperParams:
    def __init__(self):
        self.BUFFER_SIZE        = 500_000
        self.MIN_BUFFER_SIZE    = 50_000
        self.BATCH_SIZE         = 32
        self.TARGET_UPDATE      = 10_000   # env steps between hard syncs
        self.LR                 = 1e-4
        self.GAMMA              = 0.99
        self.FRAME_STACK        = 4
        self.NUM_EPISODES       = 5_000
        self.MAX_EPISODE_LENGTH = 20_000
        self.UPDATE_FREQ        = 4        # gradient update every N env steps
        self.SEED               = 42


def preprocess_frame(obs: np.ndarray) -> torch.Tensor:
    """Raw RGB frame (H x W x 3, uint8) -> grayscale (1 x 84 x 84, uint8)."""
    t = torch.as_tensor(obs).permute(2, 0, 1).float()
    t = T.functional.rgb_to_grayscale(t)
    t = t[:, 34:34 + 160, :]
    t = T.functional.resize(t, (84, 84), interpolation=T.InterpolationMode.NEAREST)
    return t.to(torch.uint8)


class FrameStack:
    def __init__(self, k: int):
        self.k      = k
        self.frames = deque(maxlen=k)

    def reset(self):
        self.frames.clear()

    def append(self, frame: torch.Tensor):
        self.frames.append(frame)

    def get(self) -> torch.Tensor:
        assert len(self.frames) == self.k
        return torch.cat(list(self.frames), dim=0)  # [k, 84, 84]


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = None):
        self.buffer = deque(maxlen=capacity)
        if seed is not None:
            random.seed(seed)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    """
    Input : [B, 4, 84, 84] normalised to [0, 1]
    Output: [B, action_dim]
    """
    def __init__(self, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4,  32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)