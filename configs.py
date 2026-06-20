"""Training configurations for enc_dec autoencoder pipeline."""

from dataclasses import dataclass

UNICODE_BITS = 21  # bits per character in unicode-21 encoding


@dataclass(slots=True)
class _BaseConfig:
    """Common training parameters shared across autoencoder configs."""
    learning_rate: float = 0.001
    batch_size: int = 1024
    device: str = "cuda"
    model_name: str = "model"
    grad_clip: float = 1.0
    num_workers: int = 0
    lr_scheduler: str = ""
    lr_warmup_epochs: float = 0.0
    cudnn_benchmark: bool = True
    init_gain: float = 0.5
    norm_bottleneck: bool = True
    norm_last: bool = True
    dropout: float = 0.0


@dataclass(slots=True)
class PrimaryConfig(_BaseConfig):
    """Configuration for the primary (text → latent) autoencoder."""

    seq_len: int = 128
    input_dim: int = 128 * UNICODE_BITS
    hidden_dim: int = 128 * UNICODE_BITS
    bottleneck: int = 128
    learning_rate: float = 0.00005
    train_ratio: float = 0.99
    batch_size: int = 1024
    model_name: str = "primary_base"
    early_stop_patience: int = 3
