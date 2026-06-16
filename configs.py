"""Training configurations for enc_dec autoencoder pipeline."""

from dataclasses import dataclass

UNICODE_BITS = 21  # bits per character in unicode-21 encoding


@dataclass(slots=True)
class PrimaryConfig:
    """Configuration for the primary (text → latent) autoencoder."""

    seq_len: int = 128
    input_dim: int = 128 * UNICODE_BITS
    hidden_dim: int = 128 * UNICODE_BITS
    bottleneck: int = 128
    learning_rate: float = 0.00005
    train_ratio: float = 0.99
    batch_size: int = 1024
    device: str = "cuda"
    model_name: str = "primary_base"
    grad_clip: float = 1.0
    num_workers: int = 0
    lr_scheduler: str = ""  # "cosine" | "plateau" | ""
    lr_warmup_epochs: float = 0.0  # warmup before max LR
    cudnn_benchmark: bool = True


@dataclass(slots=True)
class SecondaryConfig:
    """Configuration for the secondary (latent prediction) autoencoder."""

    n: int = 2
    bottleneck_primary: int = 8
    input_dim: int = n * bottleneck_primary
    hidden_dim: int = input_dim * 2
    bottleneck: int = input_dim // 4
    output_dim: int = input_dim + 1
    learning_rate: float = 0.001
    batch_size: int = 1024
    device: str = "cuda"
    model_name: str = "secondary"
    grad_clip: float = 1.0
    num_workers: int = 0
    lr_scheduler: str = ""
    lr_warmup_epochs: float = 0.0
    cudnn_benchmark: bool = True
