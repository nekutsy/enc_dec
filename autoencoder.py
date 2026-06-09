import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from configs import PrimaryConfig
from model import Autoencoder
from data import load_text, prepare_data, split_into_chunks, vec2seq, export_latent_vectors
from trainers import run_training
from logger import CSVLogger, get_last_epoch

def reconstruct_text(model, text: str, config: PrimaryConfig, device) -> str:
    model.eval()
    max_bits = config.seq_len * 32
    chunks = split_into_chunks(text, max_bits, encoding=config.encoding)
    reconstructed = []
    with torch.no_grad():
        for orig_chunk, bits in chunks:
            inp = torch.tensor([bits], dtype=torch.float32).to(device)
            out = model(inp).squeeze(0).cpu().tolist()
            rec_str = vec2seq(out, encoding=config.encoding)
            reconstructed.append(rec_str)
    return ''.join(reconstructed)

def run_experiments():
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_type}")
    text = load_text()
    os.makedirs("sessions", exist_ok=True)

    seq_lens = [2, 4, 8, 16, 32]
    epochs = 30
    encoding = "utf8"

    for seq_len in seq_lens:
        print(f"\n--- Experiment: seq_len={seq_len}, encoding={encoding} ---")
        config = PrimaryConfig(
            seq_len=seq_len,
            input_dim=seq_len * 32,
            hidden_dim=seq_len * 32 * 2,
            bottleneck=seq_len * 1,
            learning_rate=0.001,
            train_ratio=0.99,
            batch_size=256,
            device=device_type,
            model_name=f"exp_seq{seq_len}_{encoding}",
            encoding=encoding
        )

        x_train, x_val = prepare_data(text, config)
        if config.device == "cuda":
            x_train = x_train.to("cuda")
            x_val = x_val.to("cuda")

        layer_sizes = [
            config.input_dim,
            config.hidden_dim,
            config.hidden_dim // 2,
            config.hidden_dim // 4,
            config.hidden_dim // 8,
            config.hidden_dim // 16,
            config.bottleneck,
            config.hidden_dim // 16,
            config.hidden_dim // 8,
            config.hidden_dim // 4,
            config.hidden_dim // 2,
            config.hidden_dim,
            config.input_dim
        ]

        model = Autoencoder(layer_sizes, name=config.model_name).to(config.device)
        if config.device == "cuda":
            model = torch.compile(model)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()

        layer_sizes_str = "_".join(map(str, layer_sizes))
        base_filename = f"{layer_sizes_str}_{model.name}"
        model_path = os.path.join("sessions", f"{base_filename}.pth")
        csv_path = os.path.join("sessions", f"training_losses_{base_filename}.csv")

        logger = CSVLogger(csv_path)
        run_training(
            start_epoch=0,
            max_epochs=epochs,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_X=x_train,
            train_y=None,
            val_X=x_val,
            val_y=None,
            logger=logger,
            model_path=model_path,
            batch_size=config.batch_size,
            symbols_per_sample=config.seq_len
        )

def main():
    config = PrimaryConfig()
    if config.device == "cuda" and not torch.cuda.is_available():
        config.device = "cpu"
    device = torch.device(config.device)
    print(f"Using device: {device}")
    print(f"Encoding: {config.encoding}")

    if "--experiment" in sys.argv:
        run_experiments()
        return

    text = load_text()
    x_train, x_val = prepare_data(text, config)

    layer_sizes = [
        config.input_dim,
        config.hidden_dim * 4,
        config.hidden_dim,
        config.hidden_dim // 4,
        config.bottleneck,
        config.hidden_dim // 4,
        config.hidden_dim,
        config.hidden_dim * 4,
        config.input_dim
    ]

    model = Autoencoder(layer_sizes, name=config.model_name).to(device)
    if config.device == "cuda":
        model = torch.compile(model)

    layer_sizes_str = "_".join(map(str, layer_sizes))
    base_filename = f"{layer_sizes_str}_{model.name}"
    model_path = os.path.join("sessions", f"{base_filename}.pth")
    csv_path = os.path.join("sessions", f"training_losses_{base_filename}.csv")

    print(f"Model path: {model_path}")
    print(f"CSV path: {csv_path}")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss() # BCEWithLogitsLoss или MSELoss
    logger = CSVLogger(csv_path)

    current_epoch = get_last_epoch(csv_path) + 1 if os.path.isfile(csv_path) else 0
    if current_epoch > 0:
        print(f"Resuming from epoch {current_epoch}. Loading weights...")
        model.load_state_dict(torch.load(model_path, map_location=device))

    print("Commands: <text to reconstruct>, 'resume N', 'export', 'quit'")
    while True:
        user_input = input("> ")
        if user_input.lower() in ('quit', 'exit'):
            break
        if user_input.lower().startswith('resume'):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                extra = int(parts[1])
                new_max = current_epoch + extra
                print(f"Training for {extra} more epochs...")
                try:
                    run_training(current_epoch, new_max, model, optimizer, criterion,
                                 x_train, None, x_val, None, logger, model_path,
                                 config.batch_size, config.seq_len)
                    current_epoch = new_max
                except KeyboardInterrupt:
                    print("\nTraining interrupted. Model not saved.")
                print("Done.\n")
            else:
                print("Usage: resume <epochs>")
            continue
        if user_input.lower() == 'export':
            export_latent_vectors(model, text, config, device,
                                  output_path="data/latent/latent_vectors.pt")
            continue
        if not user_input:
            print("Empty input.")
            continue
        reconstructed = reconstruct_text(model, user_input, config, device)
        print("Reconstructed:", reconstructed, "\n")

if __name__ == "__main__":
    main()