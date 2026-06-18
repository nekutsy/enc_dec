"""Secondary latent-prediction autoencoder — learns to predict the next latent
vector from previous `n` latents produced by the primary model."""

import os
import torch
import torch.optim as optim
import torch.nn as nn
from configs import SecondaryConfig
from model import Autoencoder
from data import load_latent_vectors
from logger import CSVLogger, get_last_epoch
from trainers import run_training, build_scheduler, _save_checkpoint, _load_optimizer

torch.set_float32_matmul_precision('high')


def _compile_model(model, device):
    if device.type == "cuda":
        return torch.compile(model, mode="reduce-overhead")
    return model


def create_sequences(latents: torch.Tensor, n: int):
    """Sliding windows of `n` consecutive latent vectors.

    Returns (inputs, targets) where targets are inputs with an extra zero column
    appended — the autoencoder decodes this into the next-step prediction.
    """
    windows = latents.unfold(0, n, 1).transpose(1, 2).reshape(-1, n * latents.shape[1])
    targets = torch.cat([windows, torch.zeros(windows.size(0), 1)], dim=1)
    return windows, targets


def _secondary_layer_sizes(config: SecondaryConfig) -> list[int]:
    """9-layer autoencoder for secondary model."""
    h = config.hidden_dim
    return [
        config.input_dim,
        h,
        h // 2,
        h // 4,
        config.bottleneck,
        h // 4,
        h // 2,
        h,
        config.output_dim,
    ]


def main():
    config = SecondaryConfig()
    if config.device == "cuda" and not torch.cuda.is_available():
        config.device = "cpu"
    device = torch.device(config.device)
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = config.cudnn_benchmark

    latents = load_latent_vectors()
    print(f"Loaded latent vectors: {latents.shape}")

    bottleneck_dim = latents.shape[1]
    seq_len = bottleneck_dim // 2
    print(f"Inferred primary seq_len = {seq_len}")

    X, y = create_sequences(latents, config.n)
    print(f"Created sequences: X {X.shape}, y {y.shape}")

    indices = torch.randperm(len(X))
    split = int(0.99 * len(X))
    X_train, X_val = X[indices[:split]], X[indices[split:]]
    y_train, y_val = y[indices[:split]], y[indices[split:]]

    layer_sizes = _secondary_layer_sizes(config)

    model = Autoencoder(layer_sizes, name=config.model_name).to(device)
    model = _compile_model(model, device)

    key = "_".join(map(str, layer_sizes))
    os.makedirs("sessions/secondary", exist_ok=True)
    model_path = os.path.join("sessions/secondary", f"secondary_{key}_{config.model_name}.pth")
    csv_path = os.path.join("sessions/secondary", f"training_losses_secondary_{key}_{config.model_name}.csv")

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    logger = CSVLogger(csv_path)

    current_epoch = get_last_epoch(csv_path)
    if current_epoch > 0:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        _load_optimizer(optimizer, model_path, device)

    step_sch, ckpt_sch = build_scheduler(optimizer, config, total_steps=100)

    run_training(
        start_symbols=current_epoch,
        max_symbols=100,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_X=X_train,
        train_y=y_train,
        val_X=X_val,
        val_y=y_val,
        logger=logger,
        model_path=model_path,
        batch_size=config.batch_size,
        symbols_per_sample=seq_len,
        grad_clip=config.grad_clip,
        num_workers=config.num_workers,
        step_scheduler=step_sch,
        checkpoint_scheduler=ckpt_sch,
    )


if __name__ == "__main__":
    main()
