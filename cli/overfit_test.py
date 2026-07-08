#!/usr/bin/env python3
"""Overfit test — verify an architecture can memorise a single batch.

Wraps the low-level training loop with LoggerConfig + TrainingLogger for
structured output, and registers success/failure as a transient run in Registry.

Usage:
  enc-dec overfit                     # default: seq=96, n=6, 384M params
  enc-dec overfit --seq-len 128 --n 8 --b 2.2 --budget 384M
  enc-dec overfit --help
"""

import argparse
import os
import signal
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import UNICODE_BITS
from data import load_text, _build_full_bits, SlidingWindowDataset
from model import Autoencoder
from model.architecture import resolve_architecture, count_params
from training.optimizers import build_optimizer
from training.step import step_batch
from training.checkpoint import save_checkpoint
from experiment.config import SweepConfig, ModelConfig, TrainConfig, SweepSpec, OutputConfig
from experiment.context import setup_runtime
from logger import TrainingLogger, LoggerConfig
from registry.db import Registry, RunResult
from orchestration.workspace import Workspace
from utils import cuda_safe_cleanup


def build_parser():
    p = argparse.ArgumentParser(
        description='Overfit test — can the architecture memorise one batch?')
    p.add_argument('--seq-len', type=int, default=96, help='Sequence length (& bottleneck)')
    p.add_argument('--bottleneck', type=int, default=None,
                   help='Bottleneck dim (default: seq_len)')
    p.add_argument('--n', type=int, default=6, help='Hidden layers per side')
    p.add_argument('--b', type=float, default=None, help='Width ratio (hidden_dim/input_dim)')
    p.add_argument('--budget', type=str, default=None, help='Target param count, e.g. 384M')
    p.add_argument('--shape', default='rectangular',
                   choices=['rectangular', 'pyramid', 'interleaved', 'trapezoid'])
    p.add_argument('--activation', default='silu')
    p.add_argument('--normalization', default='batchnorm')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--optimizer', default='lion')
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--max-steps', type=int, default=500_000,
                   help='Max training steps before giving up')
    p.add_argument('--target-loss', type=float, default=0.001,
                   help='Loss threshold for "overfit" verdict')
    p.add_argument('--log-interval', type=int, default=100, help='Log every N steps')
    p.add_argument('--device', default='auto')
    return p


def main(argv: list[str] | None = None):
    p = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    args = p.parse_args(argv)

    # ── Resolve device ──
    device_str = args.device
    if device_str == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f'Device: {device}')

    # ── Resolve architecture ──
    bottleneck = args.bottleneck or args.seq_len
    input_dim = args.seq_len * UNICODE_BITS

    if args.b is not None and args.n is not None:
        hidden_dim = max(1, int(round(input_dim * args.b)))
    elif args.budget is not None:
        budget = _parse_size(args.budget)
        cfg = SweepConfig(
            model=ModelConfig(seq_len=args.seq_len, bottleneck=bottleneck,
                              shape=args.shape, activation=args.activation,
                              normalization=args.normalization),
            sweep=SweepSpec(
                strategy='grid', vary='n', values=[args.n],
                solve='b', budget=budget,
                fixed={'b': args.b} if args.b else {},
            ),
        )
        arch = resolve_architecture(args.n, 'n', cfg)
        hidden_dim = arch['hidden_dim']
    else:
        print("Specify --b or --budget to determine hidden_dim")
        sys.exit(1)

    sizes = _build_layer_sizes(args.n, input_dim, hidden_dim, bottleneck, args.shape)
    n_params = count_params(sizes)

    print(f'Arch: {" → ".join(str(s) for s in sizes)}')
    print(f'Params: {n_params:,}  Input: {input_dim}  Bottleneck: {bottleneck}  n={args.n}')

    # ── Data: single batch ──
    text = load_text(verbose=False)
    full_bits = _build_full_bits(text)
    dataset = SlidingWindowDataset(full_bits, args.seq_len)
    bs = args.batch_size
    indices = torch.randint(0, len(dataset), (bs,))
    x_batch = torch.stack([dataset[i][0] for i in range(bs)]).to(device)
    print(f'Batch: {x_batch.shape}')

    # ── Model ──
    model = Autoencoder(
        sizes, activation=args.activation, normalization=args.normalization,
        init_gain=1.0, dropout=0.0, residual=False,
    ).to(device)

    tc = TrainConfig(
        lr=args.lr, optimizer=args.optimizer, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, batch_size=bs,
    )
    optimizer = build_optimizer(model, tc, device)
    criterion = nn.BCEWithLogitsLoss()

    # Initial loss check
    model.eval()
    with torch.inference_mode():
        out = model(x_batch)
        initial_loss = criterion(out, x_batch).item()
    print(f'\nInitial loss: {initial_loss:.6f}  '
          f'(CE baseline: {torch.log(torch.tensor(2.0)):.6f})')

    # ── Logger ──
    mn = f'overfit_{args.shape[:4]}_s{args.seq_len}_n{args.n}_b{round(hidden_dim/input_dim, 4):.4g}'
    ws_dir = f'sessions/_overfit/{mn}'
    os.makedirs(ws_dir, exist_ok=True)
    csv_path = os.path.join(ws_dir, 'log.csv')
    log_path = os.path.join(ws_dir, 'train.log')
    model_path = os.path.join(ws_dir, 'model.pth')

    lc = LoggerConfig(epoch=False, total_samples=False, speed_sps=False,
                      train_loss=True, train_loss_ema=False, val_loss=False, lr=True)
    logger = TrainingLogger(csv_path, config=lc, model_name=mn, log_path=log_path)
    logger.log_header([
        f'overfit test: {mn}',
        f'arch: {" → ".join(str(s) for s in sizes)}',
        f'params: {n_params:,}  batch: {bs}  lr: {args.lr}  optimizer: {args.optimizer}',
    ])

    # ── Training loop ──
    use_amp = (device.type == 'cuda')
    print(f'\nTraining on single batch... (target: loss < {args.target_loss})')
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
    step = 0

    try:
        for step in range(1, args.max_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_val = step_batch(
                model, x_batch, x_batch, criterion, optimizer,
                use_amp=use_amp, grad_clip=args.grad_clip,
            )
            if loss_val < best_loss:
                best_loss = loss_val

            if step % args.log_interval == 0 or step == 1 or loss_val < args.target_loss:
                cur_lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - t_start
                logger.log_checkpoint(
                    step, loss_val, args.batch_size, lr=cur_lr)
                if loss_val < args.target_loss:
                    save_checkpoint(model, optimizer, model_path, step_scheduler=None,
                                    checkpoint_scheduler=None, total_samples=step)
                    print(f'\n✅ Overfit at step {step}! Loss = {loss_val:.6f}')
                    break

            if interrupted:
                break
    finally:
        cuda_safe_cleanup()

    elapsed = time.time() - t_start
    print(f'\nResult: steps={step}  best_loss={best_loss:.6f}  time={elapsed:.0f}s')

    if best_loss < args.target_loss:
        print('✅ MODEL CAN OVERFIT')
        return 0
    elif best_loss < args.target_loss * 40:
        print('⚠ MODEL OVERFITS POORLY')
        return 1
    else:
        print('❌ MODEL CANNOT OVERFIT')
        return 2


# ── Helpers ──────────────────────────────────────────────────

def _parse_size(s):
    if s.upper().endswith('M'):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)


def _build_layer_sizes(n, input_dim, hidden_dim, bottleneck, shape):
    """Build layer sizes for the specified shape."""
    from model.architecture import (
        make_rectangular, make_pyramid, make_interleaved, make_trapezoid,
    )
    if shape == 'pyramid':
        d = hidden_dim - bottleneck
        return make_pyramid(input_dim, bottleneck, n, d)
    elif shape == 'interleaved':
        return make_interleaved(input_dim, hidden_dim, bottleneck, n)
    elif shape == 'trapezoid':
        return make_trapezoid(input_dim, hidden_dim, bottleneck, n, alpha=0.1)
    else:
        return make_rectangular(input_dim, hidden_dim, bottleneck, n)


if __name__ == '__main__':
    sys.exit(main())
