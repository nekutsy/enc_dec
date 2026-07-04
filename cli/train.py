"""Train a single model — unified CLI with auto-resume.

Usage:
  # Basic
  python train.py --n 3 --budget 160M --samples 120M

  # Resume (auto)
  python train.py --n 3 --budget 160M --samples 120M
  # → picks up from last checkpoint automatically

  # Fresh start (ignore checkpoints)
  python train.py --n 3 --budget 160M --samples 120M --fresh

  # Reset LR (keep weights, fresh optimizer + scheduler)
  python train.py --n 4 --budget 160M --samples 100M --reset-lr --lr 0.001

  # Custom config
  python train.py --n 2 --budget 160M --samples 50M --lr 0.002 --scheduler plateau

  # Custom optimizer + scheduler
  python train.py --n 3 --budget 160M --samples 50M --optimizer lion --scheduler greedy

  # From JSON config
  python train.py --config train_best.json

  # Override workspace
  python train.py --n 3 --budget 160M --samples 120M --workspace sessions/my_exp
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sweep_lib import (
    resolve_architecture, train_one, setup_runtime, gpu_health_check,
)
from sweep_config import SweepConfig, ModelConfig, TrainConfig, SweepSpec, OutputConfig
from logger import GlobalLogger, LoggerConfig


def _parse_size(s):
    if s.upper().endswith('M'):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)


def main():
    p = argparse.ArgumentParser(
        description='Train a single autoencoder with auto-resume',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --n 3 --budget 160M --samples 50M
  python train.py --n 2 --budget 160M --samples 120M --lr 0.002 --scheduler plateau
  python train.py --n 4 --budget 160M --samples 100M --reset-lr
  python train.py --config train_best.json
  python train.py --n 3 --budget 160M --samples 50M --workspace sessions/my_exp
        """,
    )

    # ── Mode ──
    p.add_argument('--config', default=None, help='JSON config file')
    p.add_argument('--fresh', action='store_true',
                   help='Force fresh start (ignore existing checkpoints)')
    p.add_argument('--reset-lr', action='store_true',
                   help='Reset optimizer & scheduler state (keep model weights)')

    # ── Architecture ──
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--n', type=int, default=None, help='n_hidden layers')
    p.add_argument('--b', type=float, default=None, help='width ratio (hidden_dim / input_dim)')
    p.add_argument('--shape', default='rectangular', choices=['rectangular', 'pyramid', 'interleaved'],
                   help='Architecture shape')
    p.add_argument('--budget', type=str, default=None, help='Target params, e.g. 160M')
    p.add_argument('--bottleneck', type=int, default=None)
    p.add_argument('--activation', default='silu', choices=['silu', 'relu', 'gelu', 'leaky_relu'])
    p.add_argument('--normalization', default='batchnorm', choices=['batchnorm', 'layernorm', 'none'])
    p.add_argument('--init-gain', type=float, default=1.0)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--norm-bottleneck', action='store_true', default=False)
    p.add_argument('--norm-last', action='store_true', default=False)

    # ── Training ──
    p.add_argument('--samples', type=str, default='50M', help='Target samples, e.g. 120M')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--scheduler', default='onecycle',
                   choices=['onecycle', 'plateau', 'cosine', 'greedy', 'greedy_simple', 'greedy_grad', 'none'])
    p.add_argument('--optimizer', default='adamw_fused',
                   choices=['adamw_fused', 'adamw', 'sgd', 'nag', 'lion', 'sophia'])
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--early-stop', type=int, default=20, help='Early stop patience (checkpoints)')
    p.add_argument('--pct-start', type=float, default=0.3,
                   help='OneCycle peak position (default 0.3, lower = earlier peak)')
    p.add_argument('--plateau-patience', type=int, default=10,
                   help='Plateau patience (checkpoints)')
    # greedy_simple CLI args
    p.add_argument('--greedy-simple-warmup', type=int, default=0,
                   help='Warmup batches for greedy_simple (0 = none)')
    p.add_argument('--greedy-simple-min-lr', type=float, default=1e-6)
    p.add_argument('--greedy-simple-max-lr', type=float, default=0.4)
    p.add_argument('--greedy-simple-inc', type=float, default=1.01)
    p.add_argument('--greedy-simple-dec', type=float, default=0.75)
    p.add_argument('--greedy-simple-patience', type=int, default=500)
    # greedy_grad CLI args
    p.add_argument('--greedy-grad-window', type=int, default=50)
    p.add_argument('--greedy-grad-alpha', type=float, default=0.01)
    p.add_argument('--greedy-grad-momentum', type=float, default=0.995)
    p.add_argument('--greedy-grad-explore', type=float, default=0.01)
    p.add_argument('--greedy-grad-min-lr', type=float, default=1e-7)
    p.add_argument('--greedy-grad-max-lr', type=float, default=0.3)
    p.add_argument('--greedy-grad-warmup', type=int, default=0)
    p.add_argument('--greedy-grad-plateau-patience', type=int, default=500)
    p.add_argument('--greedy-grad-plateau-multiplier', type=float, default=1.5)
    p.add_argument('--greedy-grad-plateau-cooldown', type=int, default=500)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--noise-prob', type=float, default=0.0)
    p.add_argument('--noise-std', type=float, default=3.0)
    p.add_argument('--checkpoint-interval', type=int, default=100000,
                   help='Samples between validation/checkpoint passes')
    p.add_argument('--train-ratio', type=float, default=0.999)

    # ── Output ──
    p.add_argument('--workspace', default='sessions/train')
    p.add_argument('--name', default=None, help='Override model name prefix')
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    p.add_argument('--no-val', action='store_true',
                   help='Skip val logging in CSV (plateau still gets val loss)')

    args = p.parse_args()

    # ── Config from file ──
    if args.config:
        cfg = SweepConfig.from_json(args.config)
        # CLI overrides (everything that differs from defaults)
        overrides = {}
        for k, v in vars(args).items():
            if k in ('config', 'fresh', 'reset_lr', 'name', 'no_val'):
                continue
            # Only override if explicitly passed (vs parser defaults)
            if k in sys.argv:
                overrides[k] = v
        for path, value in _cli_to_overrides(overrides):
            cfg.apply_override(path, value)
    else:
        if args.n is None:
            p.error('--n is required (unless --config is given)')
        if args.budget is None:
            p.error('--budget is required (unless --config is given)')

        budget = _parse_size(args.budget)
        target_samples = _parse_size(args.samples)

        solve = 'n' if args.b else 'b'
        fixed = {}
        if args.b is not None:
            fixed['b'] = args.b
        if args.n is not None:
            fixed['n'] = args.n

        cfg = SweepConfig(
            name=args.name or f'n{args.n}_s{args.seq_len}',
            model=ModelConfig(
                seq_len=args.seq_len,
                bottleneck=args.bottleneck,
                shape=args.shape,
                activation=args.activation,
                normalization=args.normalization,
                init_gain=args.init_gain,
                dropout=args.dropout,
                norm_bottleneck=args.norm_bottleneck,
                norm_last=args.norm_last,
            ),
            training=TrainConfig(
                target_samples=target_samples,
                batch_size=args.batch_size,
                lr=args.lr,
                grad_clip=args.grad_clip,
                scheduler=args.scheduler,
                optimizer=args.optimizer,
                weight_decay=args.weight_decay,
                early_stop_patience=args.early_stop,
                pct_start=args.pct_start,
                plateau_patience=args.plateau_patience,
                warmup_fraction=0.02,
                greedy_simple_warmup=args.greedy_simple_warmup,
                greedy_simple_min_lr=args.greedy_simple_min_lr,
                greedy_simple_max_lr=args.greedy_simple_max_lr,
                greedy_simple_inc=args.greedy_simple_inc,
                greedy_simple_dec=args.greedy_simple_dec,
                greedy_simple_patience=args.greedy_simple_patience,
                greedy_grad_window=args.greedy_grad_window,
                greedy_grad_alpha=args.greedy_grad_alpha,
                greedy_grad_momentum=args.greedy_grad_momentum,
                greedy_grad_explore=args.greedy_grad_explore,
                greedy_grad_min_lr=args.greedy_grad_min_lr,
                greedy_grad_max_lr=args.greedy_grad_max_lr,
                greedy_grad_warmup=args.greedy_grad_warmup,
                greedy_grad_plateau_patience=args.greedy_grad_plateau_patience,
                greedy_grad_plateau_multiplier=args.greedy_grad_plateau_multiplier,
                greedy_grad_plateau_cooldown=args.greedy_grad_plateau_cooldown,
                num_workers=args.num_workers,
                noise_prob=args.noise_prob,
                noise_std=args.noise_std,
                checkpoint_interval=args.checkpoint_interval,
                train_ratio=args.train_ratio,
            ),
            sweep=SweepSpec(
                strategy='grid',
                vary='n',
                values=[args.n],
                solve=solve,
                budget=budget,
                fixed=fixed,
            ),
            output=OutputConfig(
                workspace=args.workspace,
                sweep_log='sessions/global.csv',
                device=args.device,
            ),
        )

    # ── Runtime ──
    global_logger = GlobalLogger(cfg.output.sweep_log)
    global_logger.init()
    runtime = setup_runtime(cfg.output, global_logger)

    if not gpu_health_check():
        print('GPU not available')
        sys.exit(1)

    # ── Resolve architecture ──
    n_val = cfg.sweep.values[0]
    arch = resolve_architecture(n_val, cfg.sweep.vary, cfg)
    sizes = arch['sizes']
    print(f'Arch: {"→".join(str(s) for s in sizes)}  ({arch["n_params"]:,} params)')
    print(f'Hidden dim: {arch["hidden_dim"]}  b={arch["b"]:.4f}')
    print(f'Target: {cfg.training.target_samples // 1_000_000}M samples')
    print(f'Scheduler: {cfg.training.scheduler}  LR: {cfg.training.lr}')
    print(f'Workspace: {cfg.output.workspace}')
    print()

    # ── Fresh / Resume ──
    resume_lr_reset = args.reset_lr
    if args.fresh:
        # Remove checkpoint files to force clean start
        from sweep_lib import save_paths
        model_name = args.name or f'n{arch["n"]}_s{cfg.model.seq_len}'
        model_path, csv_path, model_dir = save_paths(model_name,
                                                     prefix=cfg.output.workspace)
        for pattern in [model_path, model_path + '.opt', model_path + '.sch',
                        os.path.join(model_dir, 'best.pth'),
                        os.path.join(model_dir, 'best.pth') + '.sch',
                        csv_path]:
            if os.path.exists(pattern):
                os.remove(pattern)
                print(f'  Removed: {pattern}')
        print(f'  Starting fresh')

    # ── Train ──
    model_name = args.name or f'n{arch["n"]}_s{cfg.model.seq_len}'
    train, status, samples = train_one(
        arch, cfg, model_name, runtime,
        log_config=LoggerConfig.full(),
        resume_lr_reset=resume_lr_reset,
        no_val=args.no_val,
    )

    if train is not None:
        print(f'\n✅ Done: train_loss={train:.6f} at {samples:,} samples')
    else:
        print(f'\n❌ Failed: {status}')


def _cli_to_overrides(overrides):
    """Map CLI arg names to dotted config paths."""
    mapping = {
        'seq_len': 'model.seq_len',
        'bottleneck': 'model.bottleneck',
        'activation': 'model.activation',
        'normalization': 'model.normalization',
        'init_gain': 'model.init_gain',
        'dropout': 'model.dropout',
        'norm_bottleneck': 'model.norm_bottleneck',
        'norm_last': 'model.norm_last',
        'shape': 'model.shape',
        'batch_size': 'training.batch_size',
        'lr': 'training.lr',
        'scheduler': 'training.scheduler',
        'optimizer': 'training.optimizer',
        'weight_decay': 'training.weight_decay',
        'grad_clip': 'training.grad_clip',
        'early_stop': 'training.early_stop_patience',
        'pct_start': 'training.pct_start',
        'plateau_patience': 'training.plateau_patience',
        'num_workers': 'training.num_workers',
        'noise_prob': 'training.noise_prob',
        'noise_std': 'training.noise_std',
        'checkpoint_interval': 'training.checkpoint_interval',
        'train_ratio': 'training.train_ratio',
        'greedy_simple_warmup': 'training.greedy_simple_warmup',
        'greedy_simple_min_lr': 'training.greedy_simple_min_lr',
        'greedy_simple_max_lr': 'training.greedy_simple_max_lr',
        'greedy_simple_inc': 'training.greedy_simple_inc',
        'greedy_simple_dec': 'training.greedy_simple_dec',
        'greedy_simple_patience': 'training.greedy_simple_patience',
        'greedy_grad_window': 'training.greedy_grad_window',
        'greedy_grad_alpha': 'training.greedy_grad_alpha',
        'greedy_grad_momentum': 'training.greedy_grad_momentum',
        'greedy_grad_explore': 'training.greedy_grad_explore',
        'greedy_grad_min_lr': 'training.greedy_grad_min_lr',
        'greedy_grad_max_lr': 'training.greedy_grad_max_lr',
        'greedy_grad_warmup': 'training.greedy_grad_warmup',
        'greedy_grad_plateau_patience': 'training.greedy_grad_plateau_patience',
        'greedy_grad_plateau_multiplier': 'training.greedy_grad_plateau_multiplier',
        'greedy_grad_plateau_cooldown': 'training.greedy_grad_plateau_cooldown',
        'workspace': 'output.workspace',
        'device': 'output.device',
    }
    result = []
    for cli_name, value in overrides.items():
        if cli_name in mapping:
            result.append((mapping[cli_name], value))
        elif cli_name == 'samples':
            result.append(('training.target_samples', _parse_size(value)))
        elif cli_name == 'budget':
            result.append(('sweep.budget', _parse_size(value)))
    return result


if __name__ == '__main__':
    main()
