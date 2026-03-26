# exploration.py

import random
from abc import ABC, abstractmethod

import numpy as np
import torch


class ExplorationStrategy(ABC):

    @abstractmethod
    def act(self, q_network: torch.nn.Module, state_tensor: torch.Tensor) -> int:
        """Select an action given the current Q-network and state."""
        ...

    @abstractmethod
    def update(self):
        """Called once per env step (only after buffer is full). Decay internal state."""
        ...


# ---------------------------------------------------------------------------
# Greedy
# ---------------------------------------------------------------------------

class Greedy(ExplorationStrategy):
    """Always selects argmax Q(s, a). No exploration — useful for evaluation only."""

    def act(self, q_network, state_tensor) -> int:
        with torch.no_grad():
            return q_network(state_tensor).argmax().item()

    def update(self):
        pass


# ---------------------------------------------------------------------------
# Epsilon-Greedy
# ---------------------------------------------------------------------------

class EpsilonGreedy(ExplorationStrategy):
    """
    Linearly decays epsilon from 1.0 -> 0.1 over 1M env steps.
    update() is called only after the replay buffer is full, so the
    decay schedule starts exactly when real training begins.
    """

    EPSILON_START = 1.0
    EPSILON_MIN   = 0.1
    DECAY_STEPS   = 1_000_000

    def __init__(self, action_space_size: int):
        self.action_space_size = action_space_size
        self._epsilon  = self.EPSILON_START
        self._schedule = np.linspace(self.EPSILON_START, self.EPSILON_MIN, self.DECAY_STEPS)
        self._step     = 0

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def act(self, q_network, state_tensor) -> int:
        if random.random() < self._epsilon:
            return random.randint(0, self.action_space_size - 1)
        with torch.no_grad():
            return q_network(state_tensor).argmax().item()

    def update(self):
        self._step   += 1
        self._epsilon = float(self._schedule[min(self._step, self.DECAY_STEPS - 1)])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY = {
    "greedy":         Greedy,
    "epsilon_greedy": EpsilonGreedy,
}


def make_strategy(name: str, action_space_size: int) -> ExplorationStrategy:
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(_REGISTRY.keys())}")
    cls = _REGISTRY[name]
    if cls is Greedy:
        return Greedy()
    return cls(action_space_size=action_space_size)


def available_strategies() -> list[str]:
    return list(_REGISTRY.keys())