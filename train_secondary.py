import os
import torch
import torch.optim as optim
import torch.nn as nn
from configs import SecondaryConfig
from model import Autoencoder
from data import load_latent_vectors
from logger import CSVLogger, get_last_epoch
from trainers import run_training

def create_sequences(latents, n):
    num_vectors = latents.shape[0]
    bottleneck = latents.shape[1]
    sequences = []
    for i in range(num_vectors - n + 1):
        window = latents[i:i+n].reshape(-1)
        target = torch.cat([window, torch.tensor([0.0])])
        sequences.append((window, target))
    X = torch.stack([seq[0] for seq in sequences])
    y = torch.stack([seq[1] for seq in sequences])
    return X, y

def main():
    config = SecondaryConfig()
    if config.device == "cuda" and not torch.cuda.is_available():
        config.device = "cpu"
    device = torch.device(config.device)
    print(f"Using device: {device}")

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

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    input_dim = config.input_dim
    hidden = config.hidden_dim
    bottleneck = config.bottleneck
    output_dim = config.output_dim

    layer_sizes = [
        input_dim,
        hidden,
        hidden // 2,
        hidden // 4,
        bottleneck,
        hidden // 4,
        hidden // 2,
        hidden,
        output_dim
    ]

    model = Autoencoder(layer_sizes, name=config.model_name).to(device)
    if config.device == "cuda":
        model = torch.compile(model)

    layer_sizes_str = "_".join(map(str, layer_sizes))
    os.makedirs("sessions/secondary", exist_ok=True)
    model_path = os.path.join("sessions/secondary", f"secondary_{layer_sizes_str}_{config.model_name}.pth")
    csv_path = os.path.join("sessions/secondary", f"training_losses_secondary_{layer_sizes_str}_{config.model_name}.csv")

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    logger = CSVLogger(csv_path)

    current_epoch = get_last_epoch(csv_path) + 1 if os.path.isfile(csv_path) else 0
    if current_epoch > 0:
        model.load_state_dict(torch.load(model_path, map_location=device))

    run_training(
        start_epoch=current_epoch,
        max_epochs=100,
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
        symbols_per_sample=seq_len
    )

if __name__ == "__main__":
    main()