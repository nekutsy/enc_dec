"""Primary text autoencoder — model creation and reconstruction utilities."""

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
    """Non-interactive: train an autoencoder from command line."""
    config = PrimaryConfig()
    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
    device = torch.device(config.device)
    torch.set_float32_matmul_precision('high')
    print(f'Using device: {device}')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = config.cudnn_benchmark

    text = load_text()
    train_ds, val_ds = prepare_data(text, config)
    layer_sizes = _default_layer_sizes(config)

    model = Autoencoder(layer_sizes, name=config.model_name).to(device)
    model = compile_model(model, device)

    model_path, csv_path = save_paths(layer_sizes, config.model_name)
    print(f'Model: {model_path}')

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, fused=config.device == 'cuda')
    criterion = nn.MSELoss()
    logger = CSVLogger(csv_path)

    start_symbols = train_setup(config, model, optimizer, csv_path, model_path, device)

    total_epochs = 30
    total_symbols_per_epoch = len(train_ds) * config.seq_len
    target_symbols = total_epochs * total_symbols_per_epoch
    step_sch, ckpt_sch = build_scheduler(optimizer, config, total_epochs)

    print(f'Training {total_epochs} epochs ({target_symbols:,} symbols)...')
    try:
        run_training(
            start_symbols, target_symbols, model, optimizer, criterion,
            train_ds, val_ds, logger, model_path,
            config.batch_size, config.seq_len,
            grad_clip=config.grad_clip,
            num_workers=config.num_workers,
            step_scheduler=step_sch,
            checkpoint_scheduler=ckpt_sch,
        )
    except KeyboardInterrupt:
        print('\nTraining interrupted. Checkpoint saved.')
    finally:
        _cuda_safe_cleanup()
    print('Done.')


if __name__ == '__main__':
    main()
