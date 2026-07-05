"""Overfit test: seq_len=bottleneck=96, same n=6, ~384M params, Lion lr=1e-4."""

import sys, os, time, signal
import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import UNICODE_BITS
from data import load_text, _build_full_bits, SlidingWindowDataset
from model import Autoencoder
from training.optimizers import build_optimizer
from training.step import step_batch
from experiment.config import ModelConfig, TrainConfig
from utils import cuda_safe_cleanup


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    seq_len = 96
    layer_sizes = [2016, 5987, 5987, 5987, 5987, 5987, 5987, 96,
                   5987, 5987, 5987, 5987, 5987, 5987, 2016]

    n_params = sum(
        layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1] + 2 * layer_sizes[i + 1]
        for i in range(len(layer_sizes) - 1)
    )
    img_s = seq_len * UNICODE_BITS
    print(f'Params: {n_params:,}  Input: {img_s}  Bottleneck: 96  n=6')
    print(f'Arch: {" → ".join(str(s) for s in layer_sizes)}')

    text = load_text(verbose=False)
    full_bits = _build_full_bits(text)
    dataset = SlidingWindowDataset(full_bits, seq_len)

    bs = 256
    indices = torch.randint(0, len(dataset), (bs,))
    x_batch = torch.stack([dataset[i][0] for i in range(bs)]).to(device)
    print(f'Batch: {x_batch.shape}')

    model = Autoencoder(
        layer_sizes, activation='silu', normalization='batchnorm',
        init_gain=1.0, dropout=0.0,
    ).to(device)

    mc = ModelConfig(seq_len=96, activation='silu', normalization='batchnorm')
    tc = TrainConfig(
        lr=0.0001, optimizer='lion', weight_decay=0.01,
        grad_clip=1.0, batch_size=bs, target_samples=1_000_000,
    )
    optimizer = build_optimizer(model, tc, device)
    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    with torch.inference_mode():
        out = model(x_batch)
        initial_loss = criterion(out, x_batch).item()
    print(f'\nInitial loss: {initial_loss:.6f}  (CE baseline: {torch.log(torch.tensor(2.0)):.6f})')

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    print(f'\nTraining on single batch... (target: loss < 0.001)')
    print(f'{"Step":>8s}  {"Loss":>10s}  {"LR":>10s}  {"Time":>8s}')

    interrupted = False

    def _on_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
        print('\n⚠ Interrupted')

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    t_start = time.time()
    best_loss = float('inf')

    try:
        for step in range(1, 500_001):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_val = step_batch(
                model, x_batch, x_batch, criterion, optimizer,
                scaler=scaler, grad_clip=1.0,
            )
            if loss_val < best_loss:
                best_loss = loss_val
            if step % 100 == 0 or step == 1:
                cur_lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - t_start
                print(f'{step:>8d}  {loss_val:>10.6f}  {cur_lr:>10.2e}  {elapsed:>7.1f}s')
                if loss_val < 0.001:
                    print(f'\n✅ Overfit at step {step}! Loss = {loss_val:.6f}')
                    break
            if interrupted:
                break
    finally:
        cuda_safe_cleanup()

    elapsed = time.time() - t_start
    print(f'\nResult: steps={step}  best_loss={best_loss:.6f}  time={elapsed:.0f}s')

    if best_loss < 0.001:
        print('✅ MODEL CAN OVERFIT')
    elif best_loss < 0.04:
        print('⚠ MODEL OVERFITS POORLY')
    else:
        print('❌ MODEL CANNOT OVERFIT')


if __name__ == '__main__':
    main()
