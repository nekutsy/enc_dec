"""Train a single model — thin CLI over orchestration.Run.

Usage:
  python cli/train.py --n 3 --budget 160M --samples 120M
  python cli/train.py --config configs/n8_rect_bn160_50M.json
  python cli/train.py --n 3 --budget 160M --samples 120M --workspace sessions/my_exp
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiment.config import (
    SweepConfig, ModelConfig, TrainConfig, SweepSpec, OutputConfig,
)
from experiment.context import setup_runtime
from model.architecture import resolve_architecture
from orchestration import Run, Workspace
from registry import Registry
from utils import gpu_health_check


def _parse_size(s):
    if s.upper().endswith('M'):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)


def main():
    p = argparse.ArgumentParser(description='Train a single autoencoder')
    p.add_argument('--config', default=None, help='JSON config file')
    p.add_argument('--n', type=int, default=None)
    p.add_argument('--b', type=float, default=None)
    p.add_argument('--shape', default='rectangular',
                   choices=['rectangular', 'pyramid', 'interleaved', 'trapezoid'])
    p.add_argument('--budget', type=str, default=None)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--bottleneck', type=int, default=None)
    p.add_argument('--activation', default='silu')
    p.add_argument('--normalization', default='layernorm')
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--samples', type=str, default='50M')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--scheduler', default='onecycle')
    p.add_argument('--optimizer', default='adamw_fused')
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--noise-prob', type=float, default=0.0)
    p.add_argument('--noise-std', type=float, default=3.0)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='auto')
    p.add_argument('--no-val', action='store_true')

    args = p.parse_args()

    # ── Config ──
    if args.config:
        cfg = SweepConfig.from_json(args.config)
    else:
        if args.n is None:
            p.error('--n is required (or --config)')
        if args.budget is None:
            p.error('--budget is required (or --config)')

        budget = _parse_size(args.budget)
        target_samples = _parse_size(args.samples)

        solve = 'n' if args.b else 'b'
        fixed = {}
        if args.b is not None:
            fixed['b'] = args.b
        if args.n is not None:
            fixed['n'] = args.n

        cfg = SweepConfig(
            name=f'n{args.n}_s{args.seq_len}',
            model=ModelConfig(
                seq_len=args.seq_len, bottleneck=args.bottleneck,
                shape=args.shape, activation=args.activation,
                normalization=args.normalization, dropout=args.dropout,
            ),
            training=TrainConfig(
                target_samples=target_samples, batch_size=args.batch_size,
                lr=args.lr, grad_clip=args.grad_clip,
                scheduler=args.scheduler, optimizer=args.optimizer,
                weight_decay=args.weight_decay, num_workers=args.num_workers,
                noise_prob=args.noise_prob, noise_std=args.noise_std,
            ),
            sweep=SweepSpec(
                strategy='grid', vary='n', values=[args.n],
                solve=solve, budget=budget, fixed=fixed,
            ),
            output=OutputConfig(device=args.device),
        )

    # ── Runtime ──
    reg = Registry()
    ws = Workspace()

    if not gpu_health_check():
        print('GPU not available')
        sys.exit(1)

    runtime = setup_runtime(cfg.output)

    # ── Resolve architecture ──
    n_val = cfg.sweep.values[0]
    arch = resolve_architecture(n_val, cfg.sweep.vary, cfg)
    sizes = arch['sizes']
    print(f'Arch: {"→".join(str(s) for s in sizes)}  ({arch["n_params"]:,} params)')
    print(f'Hidden dim: {arch["hidden_dim"]}  b={arch["b"]:.4f}')
    print(f'Target: {cfg.training.target_samples // 1_000_000}M samples')
    print()

    # ── Run ──
    run, created = Run.find_or_create(arch, cfg.model, cfg.training, reg, ws)

    if not created:
        rd = reg.get_run(run.run_id)
        if rd and rd.get('status') == 'done':
            print(f'Already done: loss={rd.get("final_train_loss")} '
                  f'samples={rd.get("total_samples")}')
            return
        else:
            print(f'Run exists but status={rd.get("status")} — retrying')

    result = run.execute(runtime, no_val=args.no_val)

    if result.final_train_loss is not None:
        print(f'\n✅ Done: train_loss={result.final_train_loss:.6f} '
              f'at {result.total_samples:,} samples')
    else:
        print(f'\n❌ Failed: {result.status}')


if __name__ == '__main__':
    main()
