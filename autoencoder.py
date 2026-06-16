"""Primary text autoencoder — interactive training and reconstruction."""

import sys
import torch
import torch.optim as optim
import torch.nn as nn
from configs import PrimaryConfig, UNICODE_BITS
from model import Autoencoder
from data import load_text, prepare_data, split_into_chunks, vec2seq, export_latent_vectors
from trainers import run_training, build_scheduler, _cuda_safe_cleanup
from logger import CSVLogger
from sweep_lib import save_paths, compile_model, train_setup

torch.set_float32_matmul_precision('high')


def _default_layer_sizes(config: PrimaryConfig) -> list[int]:
    """11-layer autoencoder: deep hourglass with wide middle."""
    h = config.hidden_dim
    return [
        config.input_dim,
        h * 2, h * 4, h * 2, h * 2,
        config.bottleneck,
        h * 2, h * 2, h * 4, h * 2,
        config.input_dim,
    ]


def reconstruct_text(model, text: str, config, device) -> str:
    model.eval()
    max_bits = config.seq_len * UNICODE_BITS
    chunks = split_into_chunks(text, max_bits)
    reconstructed = []
    with torch.inference_mode():
        for orig_chunk, bits in chunks:
            inp = torch.tensor([bits], dtype=torch.float32).to(device)
            out = model(inp).squeeze(0).cpu().tolist()
            rec_str = vec2seq(out)
            reconstructed.append(rec_str)
    return ''.join(reconstructed)


def main():
    config = PrimaryConfig()
    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
    device = torch.device(config.device)
    print(f'Using device: {device}')
    print(f'Encoding: unicode21')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = config.cudnn_benchmark

    text = load_text()
    train_ds, val_ds = prepare_data(text, config)
    layer_sizes = _default_layer_sizes(config)

    model = Autoencoder(layer_sizes, name=config.model_name).to(device)
    model = compile_model(model, device)

    model_path, csv_path = save_paths(layer_sizes, config.model_name)

    print(f'Model path: {model_path}')
    print(f'CSV path: {csv_path}')

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, fused=config.device == 'cuda')
    criterion = nn.MSELoss()
    logger = CSVLogger(csv_path)

    start_symbols = train_setup(config, model, optimizer, csv_path, model_path, device)

    total_epochs = 30
    total_symbols_per_epoch = len(train_ds) * config.seq_len
    target_symbols = total_epochs * total_symbols_per_epoch

    scheduler = build_scheduler(optimizer, config, total_epochs)

    print('Commands: <text to reconstruct>, "resume N", "export", "quit"')
    while True:
        user_input = input('> ')
        if user_input.lower() in ('quit', 'exit'):
            _cuda_safe_cleanup()
            break
        if user_input.lower().startswith('resume'):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                extra_epochs = int(parts[1])
                extra_symbols = extra_epochs * total_symbols_per_epoch
                new_max = start_symbols + extra_symbols
                print(f'Training for {extra_epochs} more epochs ({extra_symbols} symbols)...')
                try:
                    start_symbols = run_training(
                        start_symbols, new_max, model, optimizer, criterion,
                        train_ds, val_ds, logger, model_path,
                        config.batch_size, config.seq_len,
                        grad_clip=config.grad_clip,
                        num_workers=config.num_workers,
                        scheduler=scheduler,
                    )
                except KeyboardInterrupt:
                    print('\nTraining interrupted. Checkpoint saved.')
                print('Done.\n')
            else:
                print('Usage: resume <epochs>')
            continue
        if user_input.lower() == 'export':
            export_latent_vectors(model, text, config, device,
                                  output_path='data/latent/latent_vectors.pt')
            continue
        if not user_input:
            print('Empty input.')
            continue
        reconstructed = reconstruct_text(model, user_input, config, device)
        print('Reconstructed:', reconstructed, '\n')


if __name__ == '__main__':
    main()
