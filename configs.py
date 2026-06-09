from dataclasses import dataclass

@dataclass
class PrimaryConfig:
    seq_len: int = 64
    input_dim: int = 64 * 32
    hidden_dim: int = 64 * 32
    bottleneck: int = 64 * 2
    learning_rate: float = 0.00005
    train_ratio: float = 0.99
    batch_size: int = 1024
    device: str = "cuda"
    model_name: str = "primary_base"
    encoding: str = "utf8"

@dataclass
class SecondaryConfig:
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