"""CLI command: enc-dec lr-find — Run LR range test for a model config.

Usage:
  enc-dec lr-find [opts]

Options:
  --seq-len N           Sequence length (default: 32)
  --shape SHAPE         Architecture shape (default: rectangular)
  --n N                 n_hidden layers
  --b B                 hidden_dim = input_dim * b
  --bottleneck B        Bottleneck size (default: same as seq_len)
  --budget M            Target parameter count (e.g. 40M)
  --activation ACT      Activation: silu|relu|gelu|leaky_relu
  --optimizer OPT       Optimizer (default: adamw_fused)
  --batch-size N        Batch size (default: 256)
  --steps N             Max steps for LR test (default: 200)
  --lr-start L          Starting LR (default: 1e-7)
  --lr-end L            Ending LR (default: 10.0)
  --output PATH         Save plot to file
  --device DEV          Device (default: auto)
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import UNICODE_BITS
from model import Autoencoder
from model.architecture import resolve_architecture
from data import prepare_data
from training.optimizers import build_optimizer
from training.lr_finder import find_lr, plot_lr_finder
from experiment.config import SweepConfig, ModelConfig, TrainConfig, OutputConfig, SweepSpec
from experiment.context import setup_runtime
def _arch_tag(arch: dict, args) -> str:
    """Build human-readable arch tag: rect_s128_en6_d3_b2.016."""
    enc_n = arch.get('enc_n', arch.get('n', '?'))
    dec_n = arch.get('dec_n', arch.get('n', '?'))
    if enc_n == dec_n:
        n_part = f'n{enc_n}'
    else:
        n_part = f'en{enc_n}_d{dec_n}'
    return f'{args.shape[:4]}_s{args.seq_len}_{n_part}_b{arch["b"]:.4g}'


def _arch_tag(arch: dict, args) -> str:
    """Build human-readable arch tag: rect_s128_en6_d3_b2.016."""
    enc_n = arch.get('enc_n', arch.get('n', '?'))
    dec_n = arch.get('dec_n', arch.get('n', '?'))
    if enc_n == dec_n:
        n_part = f'n{enc_n}'
    else:
        n_part = f'en{enc_n}_d{dec_n}'
    return f'{args.shape[:4]}_s{args.seq_len}_{n_part}_b{arch["b"]:.4g}'


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='LR Range Test for enc-dec models')
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--shape', default='rectangular',
                        choices=['rectangular', 'pyramid', 'interleaved', 'trapezoid'])
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--enc-n', type=int, default=None, help='Encoder layers')
    parser.add_argument('--dec-n', type=int, default=None, help='Decoder layers')
    parser.add_argument('--b', type=float, default=None)
    parser.add_argument('--bottleneck', type=int, default=None)
    parser.add_argument('--budget', type=str, default=None)
    parser.add_argument('--activation', default='silu')
    parser.add_argument('--residual', action='store_true', default=False)
    parser.add_argument('--optimizer', default='adamw_fused')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr-start', type=float, default=1e-7)
    parser.add_argument('--lr-end', type=float, default=10.0)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--device', default='auto')
    return parser.parse_args(argv)


def _parse_budget(s):
    if s is None:
        return None
    if s.upper().endswith('M'):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)


def main():
    args = parse_args(sys.argv[1:])

    if not gpu_health_check():
        print('⚠ GPU not available — check nvidia-smi')
        sys.exit(1)

    budget = _parse_budget(args.budget)
    bottleneck = args.bottleneck if args.bottleneck is not None else args.seq_len

    # Build config to resolve architecture
    vary_name = 'b' if args.b is not None else ('n' if args.n is not None else 'n')
    vary_value = args.b if args.b is not None else (args.n if args.n is not None else 2)
    fixed = {}
    if args.n is not None:
        fixed['n'] = args.n
    if args.enc_n is not None:
        fixed['enc_n'] = args.enc_n
    if args.dec_n is not None:
        fixed['dec_n'] = args.dec_n
    if args.b is not None:
        fixed['b'] = args.b

    cfg = SweepConfig(
        name='lr_find',
        model=ModelConfig(
            seq_len=args.seq_len,
            bottleneck=bottleneck,
            activation=args.activation,
            shape=args.shape,
            residual=args.residual,
            enc_n=args.enc_n, dec_n=args.dec_n,
        ),
        training=TrainConfig(
            batch_size=args.batch_size,
            optimizer=args.optimizer,
        ),
        sweep=SweepSpec(vary=vary_name, values=[vary_value], fixed=fixed,
                         budget=budget),
        output=OutputConfig(device=args.device),
    )

    arch = resolve_architecture(vary_value, vary_name, cfg)
    sizes = arch['sizes']
    n_params = arch['n_params']

    print(f'LR Range Test')
    print(f'  arch: {"→".join(str(s) for s in sizes)}')
    print(f'  params: {n_params:,}')
    print(f'  shape: {args.shape}')
    print(f'  seq_len={args.seq_len}  bottleneck={bottleneck}')
    print(f'  batch={args.batch_size}  steps={args.steps}')
    print(f'  lr range: {args.lr_start:.0e} → {args.lr_end:.1f}')
    print(f'  optimizer: {args.optimizer}')
    print()

    output = OutputConfig(device=args.device)
    runtime = setup_runtime(output)
    device = runtime.device
    texts = runtime.texts

    # Data
    train_ds, _ = prepare_data(texts, args.seq_len, train_ratio=0.999)

    # Model
    model = Autoencoder(
        sizes, activation=args.activation,
        enc_n=arch.get('enc_n'),
    ).to(device)

    # Optimizer (use TrainConfig for param grouping)
    tc = TrainConfig(optimizer=args.optimizer, batch_size=args.batch_size)
    optimizer = build_optimizer(model, tc, device)

    criterion = nn.BCEWithLogitsLoss()

    try:
        suggested_lr, history = find_lr(
            model=model,
            train_dataset=train_ds,
            device=device,
            batch_size=args.batch_size,
            num_workers=0,
            seq_len=args.seq_len,
            criterion=criterion,
            optimizer=optimizer,
            lr_start=args.lr_start,
            lr_end=args.lr_end,
            steps=args.steps,
        )
    except torch.cuda.OutOfMemoryError:
        print('⚠ OOM during LR test')
        del model
        cuda_safe_cleanup()
        sys.exit(1)
    finally:
        del model
        del optimizer
        cuda_safe_cleanup()

    print()
    print(f'{"=" * 50}')
    print(f'Suggested LR: {suggested_lr:.6f}  ({suggested_lr:.2e})')
    print(f'{"=" * 50}')

    if args.output:
        arch_tag = _arch_tag(arch, args)
        title = f'LR Range Test — {arch_tag}'
        plot_lr_finder(history, suggested_lr, title=title, save_path=args.output)
        print(f'Plot saved to {args.output}')
    else:
        import matplotlib
        matplotlib.use('Agg')
        arch_tag = _arch_tag(arch, args)
        title = f'LR Range Test — {arch_tag}'
        plot_lr_finder(history, suggested_lr, title=title, save_path='lr_find.png')
        print(f'Plot saved to lr_find.png')



def _format_arch_tag(seq_len: int, arch: dict) -> str:
    """Format arch tag: s128_en6_d3_b2.016."""
    enc_n = arch.get('enc_n', arch.get('n', '?'))
    dec_n = arch.get('dec_n', arch.get('n', '?'))
    if enc_n == dec_n:
        n_part = f'n{enc_n}'
    else:
        n_part = f'en{enc_n}_d{dec_n}'
    return f's{seq_len}_{n_part}_b{arch["b"]:.4g}'




def run_lr_find_for_sweep(
    arch: dict,
    mc: ModelConfig,
    runtime,
    batch_size: int = 256,
    lr_start: float = 1e-7,
    lr_end: float = 10.0,
    steps: int = 200,
    output_dir: str | None = None,
    model_name: str = '',
    no_plot: bool = False,
    lr_ranges: list[str] | None = None,
) -> dict:
    """Run LR finder from sweep context. Returns result dict.

    Returns:
        {'arch_tag': str, 'suggested_lr': float, 'lr_min': float, 'lr_max': float,
         'steps_used': int, 'history': list[dict], 'all_lr_ranges': list[dict]}
    """
    device = runtime.device
    texts = runtime.texts
    seq_len = mc.seq_len
    sizes = arch['sizes']

    train_ds, _ = prepare_data(texts, seq_len, train_ratio=0.999)

    results: list[dict] = []

    # Parse multiple lr_ranges if provided
    if lr_ranges is None:
        lr_ranges = [f'{lr_start}:{lr_end}']

    for range_idx, range_str in enumerate(lr_ranges):
        parts = range_str.split(':')
        start = float(parts[0])
        end = float(parts[1]) if len(parts) > 1 else lr_end

        model = Autoencoder(
            sizes, activation=mc.activation,
            normalization=mc.normalization,
            norm_bottleneck=mc.norm_bottleneck,
            norm_last=mc.norm_last,
            dropout=mc.dropout,
            residual=mc.residual, residual_norm=mc.residual_norm,
            enc_n=arch.get('enc_n'),
        ).to(device)

        tc = TrainConfig(optimizer='adamw_fused', batch_size=batch_size)
        optimizer = build_optimizer(model, tc, device)
        criterion = nn.BCEWithLogitsLoss()

        suggested_lr, history = find_lr(
            model, train_ds, device, batch_size, 0, seq_len,
            criterion, optimizer, lr_start=start, lr_end=end,
            steps=steps,
        )

        del model, optimizer
        cuda_safe_cleanup()

        results.append({
            'range': range_str,
            'suggested_lr': float(suggested_lr),
            'steps_used': len(history),
            'history': history,
        })

    # Use the first range's suggestion as primary
    primary = results[0]

    for i, res in enumerate(results):
        print(f'  suggested lr [{res["range"]}]: {res["suggested_lr"]:.6f} ({res["suggested_lr"]:.2e})')

    if not no_plot and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for i, res in enumerate(results):
            arch_tag = model_name or _format_arch_tag(seq_len, arch)
            suffix = f'_{res["range"].replace(":", "_")}' if len(results) > 1 else ''
            path = os.path.join(output_dir, f'lr_find_{arch_tag}{suffix}.png')
            title = f'LR Range Test — {arch_tag}'
            plot_lr_finder(res['history'], res['suggested_lr'], title=title, save_path=path)
            print(f'  lr plot: {path}')

    return {
        'arch_tag': model_name,
        'suggested_lr': primary['suggested_lr'],
        'steps_used': primary['steps_used'],
        'history': primary['history'],
        'all_lr_ranges': results,
    }


if __name__ == '__main__':
    main()
