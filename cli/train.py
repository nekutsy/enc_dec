"""Train a single model — thin CLI over orchestration.Run.

Usage:
  python cli/train.py --n 3 --budget 160M --samples 120M
  python cli/train.py --config configs/n8_rect_bn160_50M.json
  python cli/train.py --n 3 --budget 160M --samples 120M --workspace sessions/my_exp
  python cli/train.py --pretrain-from abc123 --samples 50M --lr 0.0001

With --pretrain-from: architecture is taken from the donor run's meta.json.
ModelConfig CLI args (--n, --b, --seq-len, --shape, --activation, --normalization)
are ignored with a warning. TrainConfig args (--samples, --lr, --noise-prob, etc.)
still apply.
"""

import argparse
import json
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


_ARCH_CLI_FLAGS = {'n', 'b', 'seq_len', 'bottleneck', 'shape', 'activation', 'normalization', 'dropout'}


def _parse_size(s):
    if s.upper().endswith('M'):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)


def _load_donor_meta(pretrain_run_id: str, ws: Workspace) -> dict:
    """Load donor meta.json, return (model_config_dict, layer_sizes, run_id)."""
    donor_dir = ws._find_run_dir(pretrain_run_id)
    if donor_dir is None:
        print(f'Donor run {pretrain_run_id} not found in sessions/runs/')
        sys.exit(1)
    meta_path = donor_dir / 'meta.json'
    if not meta_path.is_file():
        print(f'Donor {pretrain_run_id}: no meta.json in {donor_dir}')
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)
    return meta


def main():
    p = argparse.ArgumentParser(description='Train a single autoencoder')
    p.add_argument('--config', default=None, help='JSON config file')
    p.add_argument('--pretrain-from', default=None, help='Donor run_id for weight init')
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

    # ── Runtime ──
    reg = Registry()
    ws = Workspace()

    if not gpu_health_check():
        print('GPU not available')
        sys.exit(1)

    # ── Pretrain: load donor meta, override model config ──
    donor_run_id = args.pretrain_from
    if donor_run_id:
        if args.config:
            print('⚠ --pretrain-from and --config are mutually exclusive')
            sys.exit(1)
        donor_meta = _load_donor_meta(donor_run_id, ws)
        donor_mc = donor_meta.get('model_config', {})
        donor_sizes = donor_meta.get('layer_sizes', [])
        if not donor_sizes:
            print(f'Donor {donor_run_id}: layer_sizes missing in meta.json')
            sys.exit(1)

        # Warn on CLI arch args that conflict with pretrain
        arch_defaults = {'n': None, 'b': None, 'seq_len': 128, 'bottleneck': None,
                         'shape': 'rectangular', 'activation': 'silu',
                         'normalization': 'layernorm', 'dropout': 0.0}
        for flag in _ARCH_CLI_FLAGS:
            cli_val = getattr(args, flag, None)
            if cli_val != arch_defaults.get(flag):
                print(f'  ⚠ --{flag}={cli_val} ignored (architecture from donor {donor_run_id})')

        # Build ModelConfig from donor meta
        mc = ModelConfig(**donor_mc)
        arch = {'sizes': donor_sizes, 'n_params': donor_meta.get('n_params', 0),
                'n': donor_meta.get('n', len([s for s in donor_sizes if s == donor_sizes[1]]) // 2),
                'b': round(donor_sizes[1] / donor_sizes[0], 6) if len(donor_sizes) > 1 else 0,
                'hidden_dim': donor_sizes[1] if len(donor_sizes) > 1 else 0}

        target_samples = _parse_size(args.samples)
        tc = TrainConfig(
            pretrain_run_id=donor_run_id,
            target_samples=target_samples, batch_size=args.batch_size,
            lr=args.lr, grad_clip=args.grad_clip,
            scheduler=args.scheduler, optimizer=args.optimizer,
            weight_decay=args.weight_decay, num_workers=args.num_workers,
            noise_prob=args.noise_prob, noise_std=args.noise_std,
        )

        cfg = SweepConfig(
            name=f'ft_{donor_run_id[:6]}',
            model=mc,
            training=tc,
            sweep=SweepSpec(strategy='grid', vary='n', values=[arch['n']]),
            output=OutputConfig(device=args.device),
        )

        print(f'Pretrain from: {donor_run_id}')
        print(f'Arch: {"→".join(str(s) for s in donor_sizes)}  ({arch["n_params"]:,} params)')
        print(f'Target: {target_samples // 1_000_000}M samples')
        print()

    # ── Normal config path (no pretrain) ──
    elif args.config:
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

        # ── Resolve architecture ──
        n_val = cfg.sweep.values[0]
        arch = resolve_architecture(n_val, cfg.sweep.vary, cfg)
        sizes = arch['sizes']
        print(f'Arch: {"→".join(str(s) for s in sizes)}  ({arch["n_params"]:,} params)')
        print(f'Hidden dim: {arch["hidden_dim"]}  b={arch["b"]:.4f}')
        print(f'Target: {cfg.training.target_samples // 1_000_000}M samples')
        print()

    # ── Runtime ──
    runtime = setup_runtime(cfg.output, use_tf32=cfg.training.use_tf32)

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
