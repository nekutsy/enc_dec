"""Primary text autoencoder — model creation and reconstruction utilities.

Standalone training entry point (legacy; prefer sweep.py for experiments).
"""

import torch
import torch.optim as optim
import torch.nn as nn

from configs import UNICODE_BITS
from model import Autoencoder
from data import load_text, prepare_data, split_into_chunks, vec2seq, export_latent_vectors
from training import run_training, build_scheduler
from utils import cuda_safe_cleanup
from sweep_lib import (
    save_paths, compile_model, train_setup, setup_runtime, RuntimeContext,
)
from sweep_config import TrainConfig, OutputConfig


def _default_layer_sizes(seq_len: int, hidden_dim: int, bottleneck: int) -> list[int]:
    """11-layer autoencoder: deep hourglass with wide middle."""
    input_dim = seq_len * UNICODE_BITS
    h = hidden_dim
    return [input_dim, h * 2, h * 4, h * 2, h * 2, bottleneck,
            h * 2, h * 2, h * 4, h * 2, input_dim]


def reconstruct_text(model, text: str, seq_len: int, device) -> str:
    model.eval()
    max_bits = seq_len * UNICODE_BITS
    chunks = split_into_chunks(text, max_bits)
    reconstructed = []
    with torch.inference_mode():
        for orig_chunk, bits in chunks:
            inp = torch.tensor([bits], dtype=torch.float32).to(device)
            out = model(inp).squeeze(0).cpu()
            rec_str = vec2seq(torch.sigmoid(out).tolist())
            reconstructed.append(rec_str)
    return ''.join(reconstructed)


def main():
    """Non-interactive: train an autoencoder from command line."""
    seq_len = 128
    hidden_dim = seq_len * UNICODE_BITS
    bottleneck = 128
    batch_size = 1024
    lr = 0.00005
    num_workers = 0
    model_name = "primary_base"

    output = OutputConfig(workspace='sessions')
    runtime = setup_runtime(output)

    device = runtime.device
    torch.set_float32_matmul_precision('high')
    print(f'Using device: {device}')

    train_ds, val_ds = prepare_data(runtime.text, seq_len, train_ratio=0.999)
    layer_sizes = _default_layer_sizes(seq_len, hidden_dim, bottleneck)

    model = Autoencoder(
        layer_sizes, name=model_name,
        init_gain=1.0,
        norm_bottleneck=False,
        norm_last=False,
        dropout=0.0,
    ).to(device)
    model = compile_model(model, device)

    model_path, csv_path = save_paths(layer_sizes, model_name, prefix=output.workspace)
    print(f'Model: {model_path}')

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, fused=(device.type == 'cuda'))
    criterion = nn.BCEWithLogitsLoss()

    from logger import TrainingLogger
    logger = TrainingLogger(csv_path)
    start_samples = train_setup(model, optimizer, csv_path, model_path, device)

    train_cfg = TrainConfig(
        target_samples=30 * len(train_ds),
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
    )
    total_batches = int(train_cfg.target_samples / batch_size) + 1
    step_sch, ckpt_sch = build_scheduler(
        optimizer, train_cfg, total_batches, start_samples=start_samples,
        greedy_diff_d=train_cfg.greedy_diff_d,
        greedy_diff_packet=train_cfg.greedy_diff_packet,
        greedy_diff_k=train_cfg.greedy_diff_k,
        greedy_diff_min_lr=train_cfg.greedy_diff_min_lr,
        greedy_diff_max_lr=train_cfg.greedy_diff_max_lr)

    print(f'Training 30 epochs ({train_cfg.target_samples:,} samples)...')
    try:
        run_training(
            start_samples, train_cfg.target_samples, model, optimizer, criterion,
            train_ds, val_ds, logger, model_path,
            batch_size, seq_len,
            grad_clip=train_cfg.grad_clip,
            num_workers=num_workers,
            step_scheduler=step_sch,
            checkpoint_scheduler=ckpt_sch,
        )
    except KeyboardInterrupt:
        print('\nTraining interrupted. Checkpoint saved.')
    finally:
        cuda_safe_cleanup()
    print('Done.')


if __name__ == '__main__':
    main()
