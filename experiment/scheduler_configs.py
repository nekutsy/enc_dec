"""Per-scheduler configuration dataclasses.

Each scheduler gets its own dataclass — new schedulers add 1 class here,
no changes to TrainConfig, fingerprint, or CLI required (use --scheduler-params).

Old flat fields on TrainConfig are kept for backward compat.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


@dataclass
class GreedyConfig:
    """GreedyLR — zeroth-order adaptive LR scheduler with probing."""
    factor: float = 0.5
    beta: float = 0.9
    lock_steps: int = 3
    probe_patience: int = 4
    probe_factor: float = 0.5
    probe_spike_ratio: float = 2.5
    probe_lock_steps: int = 3
    cooldown_steps: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GreedySimpleConfig:
    """GreedySimpleLR — immediate raise / cautious decrease."""
    min_lr: float = 1e-6
    max_lr: float = 0.4
    increase_factor: float = 1.01
    decrease_factor: float = 0.75
    patience: int = 500
    warmup: int = 0
    warmup_start_factor: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GreedyGradConfig:
    """GreedyGradLR — gradient descent on LR."""
    window: int = 50
    alpha: float = 0.01
    momentum: float = 0.995
    explore: float = 0.01
    min_lr: float = 1e-7
    max_lr: float = 0.3
    warmup: int = 0
    plateau_patience: int = 500
    plateau_multiplier: float = 1.5
    plateau_cooldown: int = 500

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlateauConfig:
    """ReduceLROnPlateau config."""
    patience: int = 10
    factor: float = 0.7
    min_lr: float = 1e-6

    def to_dict(self) -> dict:
        return asdict(self)


# Registry: scheduler name → config builder
_SCHEDULER_CONFIGS = {
    'greedy': GreedyConfig,
    'greedy_simple': GreedySimpleConfig,
    'greedy_grad': GreedyGradConfig,
    'plateau': PlateauConfig,
}


def build_scheduler_config(scheduler_name: str, params: dict | None = None
                           ) -> dict | None:
    """Build a scheduler_config dict from name + optional overrides.

    Returns a dict ready for TrainConfig.scheduler_config.
    Returns None for schedulers without config (onecycle, cosine, none).

    Usage:
        config = build_scheduler_config('greedy', {'factor': 0.3})
        tc = TrainConfig(scheduler='greedy', scheduler_config=config)
    """
    cls = _SCHEDULER_CONFIGS.get(scheduler_name)
    if cls is None:
        return None
    instance = cls()
    if params:
        for k, v in params.items():
            if hasattr(instance, k):
                setattr(instance, k, type(getattr(instance, k))(v))
    return instance.to_dict()


def parse_scheduler_params(params_str: str | None) -> dict | None:
    """Parse JSON or key=val syntax for scheduler params.

    Usage:
        --scheduler-params '{"factor": 0.3}'
        --scheduler-params factor=0.3,beta=0.8
    """
    if not params_str:
        return None
    try:
        return json.loads(params_str)
    except json.JSONDecodeError:
        pass
    # key=val,key=val
    result = {}
    for item in params_str.split(','):
        item = item.strip()
        if not item:
            continue
        k, v = item.split('=', 1)
        try:
            result[k.strip()] = int(v.strip())
        except ValueError:
            try:
                result[k.strip()] = float(v.strip())
            except ValueError:
                result[k.strip()] = v.strip()
    return result
