# core.py
import os
import logging
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

# Preprocessing
def preprocess_frame(obs):
    """
    Convert a raw ALE RGB frame (210x160x3) to a grayscale 84x84 uint8 tensor.
    Follows the preprocessing from Mnih et al. (2015):
    - Convert to grayscale (luminance channel)
    - Crop top 34 rows (score area) → 160x160
    - Resize to 84x84
    Returns shape: [1, 84, 84] uint8 tensor
    """
    obs = torch.tensor(obs)
    obs = obs.permute(2, 0, 1).float()                        # HWC → CHW
    obs = T.functional.rgb_to_grayscale(obs)                  # [1, 210, 160]
    obs = obs[:, 34:34 + 160, :]                              # crop to 160x160
    obs = T.functional.resize(obs, (84, 84), interpolation=T.InterpolationMode.NEAREST)
    return obs.to(torch.uint8)                                 # [1, 84, 84]

# Network
class DQN(nn.Module):
    """
    Convolutional Q-network from Mnih et al. (2015).
    Input:  [B, 4, 84, 84]  (4 stacked grayscale frames, normalised to [0,1])
    Output: [B, action_dim] (Q-value per action)
    """
    def __init__(self, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),   # → [B, 32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → [B, 64, 9, 9]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → [B, 64, 7, 7]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_dim),
        )

    def forward(self, x):
        return self.net(x)

# Frame stack
class FrameStack:
    """Maintains a rolling window of the last k preprocessed frames."""

    def __init__(self, k):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self):
        self.frames.clear()

    def append(self, obs):
        self.frames.append(obs)

    def get_stack(self):
        assert len(self.frames) == self.k, (
            f"FrameStack not full: {len(self.frames)}/{self.k}"
        )
        return torch.cat(list(self.frames), dim=0)  # [k, 84, 84]

# Replay buffer
class ReplayBuffer:
    """Uniform-random experience replay buffer."""

    def __init__(self, capacity, seed=None):
        self.buffer = deque([], maxlen=capacity)
        if seed is not None:
            random.seed(seed)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# Hyperparameters
class HyperParams:
    def __init__(self):
        self.BUFFER_SIZE     = 500_000
        self.MIN_BUFFER_SIZE = 50_000
        self.BATCH_SIZE      = 32
        self.TARGET_UPDATE   = 10_000
        self.LR              = 0.0001
        self.GAMMA           = 0.99
        self.FRAME_STACK     = 4
        self.STATE_DIM       = (4, 84, 84)
        self.NUM_EPISODES    = 5_000
        self.MAX_EPISODE_LENGTH = 20_000
        self.UPDATE_FREQ     = 4
        self.SEED            = 42

# Logging
def setup_logger(log_dir, log_name="training"):
    """
    Create a logger that writes to both a file and stdout.
    Returns the configured logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{log_name}.log")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers on re-import
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(message)s")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger

# Checkpoint / stats helpers
def save_stats(stats: dict, path: str):
    """Append a JSON line with the current stats dict to *path*."""
    import json
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(stats) + "\n")


def load_stats(path: str):
    """Load the last JSON line written by save_stats."""
    import json
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    return json.loads(lines[-1])