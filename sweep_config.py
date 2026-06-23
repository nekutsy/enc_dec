"""Configuration dataclasses for enc_dec — JSON-serializable, zero runtime state.

Three clean layers:
  ModelConfig    — architecture, activations, norms
  TrainConfig    — optimizer, LR, scheduler, batch, early-stopping
  OutputConfig   — paths, device preference, sweep log

Runtime-only state lives in sweep_lib.RuntimeContext.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Any


# ── Model configuration ──────────────────────────────────────

@dataclass
class ModelConfig:
    """Autoencoder architecture definition.

    Serialisable. No runtime state.
    """
    seq_len: int = 32
    bottleneck: int | None = None      # None → seq_len
    activation: str = 'silu'           # silu | relu | gelu | leaky_relu
    normalization: str = 'batchnorm'   # batchnorm | layernorm | none
    shape: str = 'rectangular'         # rectangular | pyramid
    init: str = 'orthogonal'           # orthogonal | xavier | kaiming
    init_gain: float = 1.0
    dropout: float = 0.0
    norm_bottleneck: bool = False      # norm on bottleneck layer
    norm_last: bool = False            # norm on final decoder layer


# ── Training configuration ───────────────────────────────────

@dataclass
class TrainConfig:
    """Training recipe — optimizer, scheduler, budget, early-stopping.

    Serialisable. Independent of ModelConfig — mix and match freely.
    """
    target_samples: int = 5_000_000
    batch_size: int = 256
    lr: float = 0.001
    grad_clip: float = 1.0
    scheduler: str = 'onecycle'        # onecycle | plateau | cosine | greedy | none
    warmup_fraction: float = 0.02      # fraction of total steps for warmup
    pct_start: float = 0.3             # onecycle: fraction at which LR peaks
    plateau_patience: int = 10          # plateau: checkpoints without improvement
    greedy_factor: float = 0.5          # greedy: γ — adjustment aggressiveness
    greedy_beta: float = 0.9            # greedy: EMA smoothing coefficient
    greedy_lock_steps: int = 3           # greedy: lock checkpoints after LR decrease
    greedy_probe_patience: int = 3       # greedy: flat δ checkpoints before probing
    greedy_probe_factor: float = 0.5     # greedy: LR multiplier on probe
    greedy_probe_threshold: float = 0.02  # greedy: |δ| below this is \"flat\"
    greedy_probe_lock: int = 3           # greedy: probe observation window
    greedy_cooldown: int = 9             # greedy: cooldown after failed probe
    optimizer: str = 'adamw_fused'     # adamw_fused | adamw | sgd | nag | lion | sophia
    weight_decay: float = 0.01
    decay_linear_only: bool = True     # True → only Linear weights; False → all params
    early_stop_patience: int = 20
    train_ratio: float = 0.999
    checkpoint_interval: int = 100_000   # samples between validation/checkpoint passes
    num_workers: int = 2               # DataLoader workers (2 for GPU, 0 for CPU)


# ── Sweep specification ──────────────────────────────────────

@dataclass
class SweepSpec:
    """What to vary and how."""
    strategy: str = 'grid'             # grid | binary
    vary: str = 'n'                    # parameter name
    values: list = field(default_factory=list)
    solve: str | None = None           # b | n | None
    budget: int | None = None          # target parameter count
    fixed: dict = field(default_factory=dict)


# ── Output configuration ─────────────────────────────────────

@dataclass
class OutputConfig:
    """Paths and device preference. No runtime data."""
    workspace: str = 'sessions/sweep'
    sweep_log: str = 'sessions/sweep_summary.csv'
    device: str = 'auto'               # auto | cuda | cpu


# ── Top-level sweep config ───────────────────────────────────

@dataclass
class SweepConfig:
    """Complete sweep configuration — one file per experiment.

    All fields are serialisable. Runtime state (device, text, global_logger)
    lives in sweep_lib.RuntimeContext.
    """
    name: str = 'sweep'
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    sweep: SweepSpec = field(default_factory=SweepSpec)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ── Serialization ────────────────────────────────────────

    def to_dict(self) -> dict:
        def _convert(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _convert(v) for k, v in asdict(obj).items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj
        return _convert(self)

    def to_json(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> 'SweepConfig':
        model_data = data.get('model', {})
        training_data = data.get('training', {})
        sweep_data = data.get('sweep', {})
        output_data = data.get('output', {})

        # backward compat: old configs use target_symbols
        if 'target_symbols' in training_data and 'target_samples' not in training_data:
            training_data['target_samples'] = training_data.pop('target_symbols')

        return cls(
            name=data.get('name', 'sweep'),
            model=ModelConfig(**model_data),
            training=TrainConfig(**training_data),
            sweep=SweepSpec(**sweep_data),
            output=OutputConfig(**output_data),
        )

    @classmethod
    def from_json(cls, path: str) -> 'SweepConfig':
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    # ── Override helpers ─────────────────────────────────────

    def apply_override(self, path: str, value: Any) -> None:
        """Apply a dotted-path override, e.g. 'model.seq_len=64'."""
        parts = path.split('.')
        obj = self
        for part in parts[:-1]:
            obj = getattr(obj, part)
        field_name = parts[-1]
        if not hasattr(obj, field_name):
            raise KeyError(f"Unknown field: {path}")
        existing = getattr(obj, field_name)
        if existing is not None:
            try:
                value = type(existing)(value)
            except (ValueError, TypeError):
                pass
        setattr(obj, field_name, value)


# ── Preset generators ────────────────────────────────────────

def preset_ratio(budget_m: int) -> SweepConfig:
    """Ratio sweep at a given budget in millions."""
    return SweepConfig(
        name=f'ratio_{budget_m}m',
        sweep=SweepSpec(
            strategy='grid',
            vary='n',
            values=list(range(2, 7)) if budget_m == 20
                   else list(range(2, 11, 2)) if budget_m == 40
                   else list(range(2, 17)),
            solve='b',
            budget=budget_m * 1_000_000,
        ),
        output=OutputConfig(
            workspace=f'sessions/ratio{budget_m}',
            sweep_log=f'sessions/ratio{budget_m}_sweep_summary.csv',
        ),
    )


def preset_binary() -> SweepConfig:
    """Binary search for optimal width ratios across seq_lens."""
    return SweepConfig(
        name='binary_search',
        sweep=SweepSpec(
            strategy='grid',
            vary='seq_len',
            values=[4, 8, 16, 32, 64, 128],
            solve=None,
        ),
        output=OutputConfig(
            workspace='sessions/sweep',
            sweep_log='sessions/sweep_binary_summary.csv',
        ),
    )


def preset_width(seq_len: int = 32, n_hidden: int = 7) -> SweepConfig:
    """Width sweep at fixed n and seq_len."""
    return SweepConfig(
        name=f'width_s{seq_len}_n{n_hidden}',
        model=ModelConfig(seq_len=seq_len),
        sweep=SweepSpec(
            strategy='grid',
            vary='b',
            values=[1/7, 1/3, 1, 2, 4, 8],
            solve=None,
            fixed={'n': n_hidden},
        ),
        output=OutputConfig(
            workspace='sessions/width',
            sweep_log='sessions/width_sweep_summary.csv',
        ),
    )


def preset_batch(budget_m: int = 20, n_hidden: int = 3) -> SweepConfig:
    """Batch-size sweep at fixed architecture."""
    return SweepConfig(
        name=f'batch_{budget_m}m_n{n_hidden}',
        training=TrainConfig(
            batch_size=64,  # will be overridden by sweep values
        ),
        sweep=SweepSpec(
            strategy='grid',
            vary='batch_size',
            values=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
            solve='b',
            budget=budget_m * 1_000_000,
            fixed={'n': n_hidden},
        ),
        output=OutputConfig(
            workspace=f'sessions/ratio{budget_m}',
            sweep_log=f'sessions/batch_sweep_{budget_m}m.csv',
        ),
    )
