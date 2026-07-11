"""Configuration dataclasses for enc_dec — JSON-serializable, zero runtime state.

Three clean layers:
  ModelConfig    — architecture, activations, norms
  TrainConfig    — optimizer, LR, scheduler, batch, early-stopping
  OutputConfig   — paths, device preference, sweep log

Runtime-only state lives in experiment.context.RuntimeContext.
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
    normalization: str = 'layernorm'    # layernorm | batchnorm | none
    shape: str = 'rectangular'         # rectangular | pyramid | interleaved | trapezoid
    trapezoid_alpha: float = 0.1       # trapezoid: deviation from base hidden_dim
    init: str = 'orthogonal'           # orthogonal | xavier | kaiming
    init_gain: float = 1.0
    enc_n: int | None = None           # encoder layers (None → use n from sweep fixed)
    dec_n: int | None = None           # decoder layers (None → same as enc_n)
    dropout: float = 0.0
    norm_bottleneck: bool = False      # norm on bottleneck layer
    norm_last: bool = False            # norm on final decoder layer
    residual: bool = False             # classic residual: f(x) + x where dims match
    residual_norm: str = 'post'        # 'post' | 'pre' — norm placement in residual blocks
    vae: bool = False                  # VAE mode: μ/logvar split head + KL loss

    def to_dict(self) -> dict:
        return asdict(self)


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
    scheduler: str = 'onecycle'        # onecycle | plateau | cosine | greedy | greedy_simple | greedy_grad | none
    scheduler_config: dict | None = None  # per-scheduler params as dict (takes precedence over flat fields)
    warmup_fraction: float = 0.02      # fraction of total steps for warmup
    pct_start: float = 0.3             # onecycle: fraction at which LR peaks
    # Legacy flat fields — kept for backward compat. Prefer scheduler_config dict.
    plateau_patience: int = 10          # plateau: checkpoints without improvement
    greedy_factor: float = 0.5
    greedy_beta: float = 0.9
    greedy_lock_steps: int = 3
    greedy_probe_patience: int = 4
    greedy_probe_factor: float = 0.5
    greedy_probe_spike_ratio: float = 2.5
    greedy_probe_lock: int = 3
    greedy_cooldown: int = 3
    greedy_simple_min_lr: float = 1e-6
    greedy_simple_max_lr: float = 0.4
    greedy_simple_inc: float = 1.01
    greedy_simple_dec: float = 0.75
    greedy_simple_patience: int = 500
    greedy_simple_warmup: int = 0
    greedy_simple_warmup_start: float = 0.1
    greedy_grad_window: int = 50
    greedy_grad_alpha: float = 0.01
    greedy_grad_momentum: float = 0.995
    greedy_grad_explore: float = 0.01
    greedy_grad_min_lr: float = 1e-7
    greedy_grad_max_lr: float = 0.3
    greedy_grad_warmup: int = 0
    greedy_grad_plateau_patience: int = 500
    greedy_grad_plateau_multiplier: float = 1.5
    greedy_grad_plateau_cooldown: int = 500
    pretrain_run_id: str = ''           # run_id донора: загрузить веса вместо random init
    optimizer: str = 'adamw_fused'     # adamw_fused | adamw | sgd | nag | lion | sophia
    weight_decay: float = 0.01
    decay_linear_only: bool = True     # True → only Linear weights; False → all params
    use_tf32: bool = True               # enable tf32 matmul on Ampere+ (free ~2-8× speedup)
    noise_prob: float = 0.0              # 0→disabled; fraction of symbols perturbed
    noise_std: float = 3.0               # σ for Gaussian noise on uint21 values
    vae_beta: float = 1.0                # β-VAE: KL divergence weight (1.0 = standard VAE)
    early_stop_patience: int = 20
    train_ratio: float = 0.999
    val_interval: int = 100_000         # samples between validation passes
    checkpoint_interval: int = 1_000_000  # samples between model saves
    num_workers: int = 4               # DataLoader workers (4 for GPU, 0 for CPU)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Sweep specification ──────────────────────────────────────

@dataclass
class SweepSpec:
    """What to vary and how."""
    strategy: str = 'grid'             # grid | binary
    vary: str = 'n'                    # parameter name
    values: list = field(default_factory=list)
    solve: str | None = None           # b | n | None
    budget: int | None = None          # target parameter count
    total_n: int | None = None         # fixed enc_n+dec_n sum; complementary auto-set
    fixed: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Output configuration ─────────────────────────────────────

@dataclass
class OutputConfig:
    """Paths and device preference. No runtime data."""
    workspace: str = 'sessions/sweep'
    sweep_log: str = 'sessions/global.csv'
    device: str = 'auto'               # auto | cuda | cpu

    def to_dict(self) -> dict:
        return asdict(self)


# ── Top-level sweep config ───────────────────────────────────

@dataclass
class SweepConfig:
    """Complete sweep configuration — one file per experiment.

    All fields are serialisable. Runtime state (device, text, global_logger)
    lives in experiment.context.RuntimeContext.
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

            # backward compat: old configs had no residual field
        model_data.setdefault('residual', False)
        # backward compat: old configs had no residual_norm field
        model_data.setdefault('residual_norm', 'post')
        # backward compat: old configs had no vae field
        model_data.setdefault('vae', False)
        training_data.setdefault('vae_beta', 1.0)
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
