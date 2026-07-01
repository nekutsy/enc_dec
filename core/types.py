"""Protocols defining subsystem boundaries.

These are structural interfaces (Protocol) — any object satisfying the
shape is accepted. No subclassing required. Enables loose coupling:
  - training.loop only cares that model has encode()/decode()/forward()
  - experiment.runner only cares that scheduler has step()/state_dict()

No PyTorch imports in signatures — just pure Protocols.
"""

from typing import Protocol, runtime_checkable
from typing import Any

import torch
from torch import Tensor


# ── Model ───────────────────────────────────────────────────

@runtime_checkable
class ModelLike(Protocol):
    """Any autoencoder-like model: encode, decode, forward, bottleneck."""

    def encode(self, x: Tensor) -> Tensor: ...
    def decode(self, z: Tensor) -> Tensor: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def parameters(self):
        """Iterator over trainable parameters."""
        ...
    def train(self, mode: bool = True): ...
    def eval(self): ...
    def to(self, device, *args, **kwargs): ...

    @property
    def bottleneck(self) -> int: ...


# ── Optimizer ────────────────────────────────────────────────

@runtime_checkable
class OptimizerLike(Protocol):
    """PyTorch optimizer interface."""

    param_groups: list[dict[str, Any]]

    def step(self, closure=None) -> Any: ...
    def zero_grad(self, set_to_none: bool = False) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict) -> None: ...


# ── Scheduler ────────────────────────────────────────────────

@runtime_checkable
class StepSchedulerLike(Protocol):
    """Per-batch scheduler (OneCycle, Cosine, GreedySimple, GreedyGrad)."""

    uses_loss: bool  # if True, step_batch passes loss.item()

    def step(self, metric: float | None = None) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict) -> None: ...


@runtime_checkable
class CheckpointSchedulerLike(Protocol):
    """Per-checkpoint scheduler (Plateau, GreedyLR)."""

    def step(self, val_loss: float) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state_dict: dict) -> None: ...
    def is_exploring(self) -> bool: ...
