"""Unified sweep runner — grid and binary search.

Uses registry + orchestration. Deduplicates runs across experiments.

Usage:
  python cli/sweep.py run --config configs/noise_sweep.json
  python cli/sweep.py grid --vary n=2,4,6,8,10 --solve b --budget 40M
  python cli/sweep.py binary --vary n --range 2 16 --solve b --budget 40M
  python cli/sweep.py lr-find [opts]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiment.config import (
    SweepConfig, ModelConfig, TrainConfig, SweepSpec, OutputConfig,
)
from orchestration import Sweep
from utils import cuda_safe_cleanup


# ── Helpers ──────────────────────────────────────────────────

def _parse_size(s):
    if s.upper().endswith('M'):
        return int(float(s[:-1]) * 1_000_000)
    return int(s)


def _parse_fixed(fixed_args):
    d = {}
    for item in fixed_args:
        k, v = item.split('=', 1)
        try:
            d[k] = int(v)
        except ValueError:
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return d


def _parse_overrides(override_args):
    result = []
    for item in override_args:
        path, value = item.rsplit('=', 1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        result.append((path, value))
    return result


def _cli_shorthand_to_config(args, vary_values) -> SweepConfig:
    budget = _parse_size(args.budget) if args.budget else None
    target_samples = _parse_size(args.target_samples)
    batch_size = getattr(args, 'batch_size', None) or 256

    return SweepConfig(
        name=f'{args.command}_{getattr(args, "vary", "sweep")}'.replace('=', '_'),
        model=ModelConfig(
            seq_len=getattr(args, 'seq_len', 32),
            bottleneck=getattr(args, 'bottleneck', None),
        ),
        training=TrainConfig(
            target_samples=target_samples,
            batch_size=batch_size,
            lr=getattr(args, 'lr', 0.001),
            scheduler=getattr(args, 'scheduler', 'onecycle'),
            optimizer=getattr(args, 'optimizer', 'adamw_fused'),
            early_stop_patience=getattr(args, 'early_stop', None) or 20,
        ),
        sweep=SweepSpec(
            strategy=args.command,
            vary=args.vary.split('=', 1)[0] if args.command == 'grid' else args.vary,
            values=vary_values,
            solve=args.solve if hasattr(args, 'solve') else None,
            budget=budget,
            fixed=_parse_fixed(getattr(args, 'fixed', [])),
        ),
        output=OutputConfig(
            workspace=getattr(args, 'workspace', 'sessions/sweep'),
            sweep_log=getattr(args, 'sweep_log', 'sessions/global.csv'),
            device=getattr(args, 'device', 'auto'),
        ),
    )


def build_parser():
    parser = argparse.ArgumentParser(description='Unified sweep runner')
    sub = parser.add_subparsers(dest='command', required=True)

    # run — from config file
    rp = sub.add_parser('run', help='Run sweep from config file')
    rp.add_argument('--config', required=True, help='Path to sweep config JSON')
    rp.add_argument('--override', nargs='+', default=[],
                    help='Override config fields: model.seq_len=64')
    rp.add_argument('--no-val', action='store_true', default=False,
                    help='Skip writing val loss to CSV')

    # grid — shorthand
    gp = sub.add_parser('grid', help='Grid search over parameter values')
    gp.add_argument('--vary', required=True, help='e.g. n=2,3,4,5')
    gp.add_argument('--solve', default=None, choices=['b', 'n'])
    gp.add_argument('--fixed', nargs='+', default=[])
    gp.add_argument('--budget', type=str, default=None)
    gp.add_argument('--seq-len', type=int, default=32)
    gp.add_argument('--bottleneck', type=int, default=None)
    gp.add_argument('--lr', type=float, default=0.001)
    gp.add_argument('--scheduler', default='onecycle')
    gp.add_argument('--optimizer', default='adamw_fused')
    gp.add_argument('--target-samples', type=str, default='5M')
    gp.add_argument('--workspace', default='sessions/sweep')
    gp.add_argument('--sweep-log', default='sessions/global.csv')
    gp.add_argument('--device', default='auto')
    gp.add_argument('--batch-size', type=int, default=None)
    gp.add_argument('--early-stop', type=int, default=None)
    gp.add_argument('--no-val', action='store_true')
    gp.add_argument('--shape', default='rectangular',
                    choices=['rectangular', 'pyramid', 'interleaved', 'trapezoid'])

    # binary — shorthand
    bp = sub.add_parser('binary', help='Binary search over a parameter')
    bp.add_argument('--vary', required=True)
    bp.add_argument('--range', type=float, nargs=2, required=True)
    bp.add_argument('--solve', default=None, choices=['b', 'n'])
    bp.add_argument('--fixed', nargs='+', default=[])
    bp.add_argument('--budget', type=str, default=None)
    bp.add_argument('--seq-len', type=int, default=32)
    bp.add_argument('--bottleneck', type=int, default=None)
    bp.add_argument('--lr', type=float, default=0.001)
    bp.add_argument('--scheduler', default='onecycle')
    bp.add_argument('--optimizer', default='adamw_fused')
    bp.add_argument('--target-samples', type=str, default='5M')
    bp.add_argument('--workspace', default='sessions/sweep')
    bp.add_argument('--device', default='auto')
    bp.add_argument('--batch-size', type=int, default=None)

    # lr-find — LR range test sweep
    lp = sub.add_parser('lr-find', help='Run LR range test across architecture variants')
    lp.add_argument('--vary', required=True, help='e.g. n=2,3,4,5 or b=0.5,1,2,4,8')
    lp.add_argument('--solve', default='b', choices=['b', 'n'],
                    help='Solve for the other dim (default: b when varying n)')
    lp.add_argument('--fixed', nargs='+', default=[],,
                    help='Fixed params: n=2 b=2.0')
    lp.add_argument('--seq-len', type=int, default=32)
    lp.add_argument('--bottleneck', type=int, default=None)
    lp.add_argument('--shape', default='rectangular',
                    choices=['rectangular', 'pyramid', 'interleaved', 'trapezoid'])
    lp.add_argument('--activation', default='silu')
    lp.add_argument('--budget', type=str, default=None)
    lp.add_argument('--batch-size', type=int, default=256)
    lp.add_argument('--steps', type=int, default=200)
    lp.add_argument('--lr-start', type=float, default=1e-7)
    lp.add_argument('--lr-end', type=float, default=10.0)
    lp.add_argument('--lr-ranges', nargs='+', default=None,
                    help='Multiple LR ranges: 1e-7:1.0 1e-5:10.0')
    lp.add_argument('--output', default='sessions/lr_find')
    lp.add_argument('--device', default='auto')
    lp.add_argument('--no-plot', action='store_true', default=False)
    lp.add_argument('--residual', action='store_true', default=False)
    lp.add_argument('--residual-norm', default='post', choices=['post', 'pre'])
    lp.add_argument('--normalization', default='layernorm',
                    choices=['layernorm', 'rmsnorm', 'batchnorm', 'none'])

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'run':
        cfg = SweepConfig.from_json(args.config)
        for path, value in _parse_overrides(args.override):
            cfg.apply_override(path, value)
            print(f'  override: {path} = {value}')
        sweep = Sweep(cfg)
        sweep.run(no_val=args.no_val)
        return

    if args.command == 'lr-find':
        _run_lr_find_sweep(args)
        return

    if args.command == 'grid':
        vary_str = args.vary.split('=', 1)[1]
        vary_values = []
        for v in vary_str.split(','):
            try:
                vary_values.append(int(v) if '.' not in v else float(v))
            except ValueError:
                vary_values.append(v)
    else:
        vary_values = list(args.range)

    cfg = _cli_shorthand_to_config(args, vary_values)

    # Apply shape if specified
    if hasattr(args, 'shape') and args.shape:
        cfg.model.shape = args.shape

    sweep = Sweep(cfg)
    sweep.run()


# ── LR Find Sweep ────────────────────────────────────────────

def _run_lr_find_sweep(args):
    """Run LR range test across architecture variants."""
    from experiment.context import setup_runtime
    from model.architecture import resolve_architecture
    from utils import gpu_health_check
    from cli.lr_finder import run_lr_find_for_sweep

    if not gpu_health_check():
        print('\u26a0 GPU not available — check nvidia-smi')
        sys.exit(1)

    vary_str = args.vary.split('=', 1)
    vary_name = vary_str[0]
    vary_values_str = vary_str[1] if len(vary_str) > 1 else ''

    vary_values = []
    for v in vary_values_str.split(','):
        try:
            vary_values.append(int(v) if '.' not in v else float(v))
        except ValueError:
            vary_values.append(v)

    budget = _parse_size(args.budget) if args.budget else None
    bottleneck = args.bottleneck if args.bottleneck is not None else args.seq_len

    print(f'LR Finder Sweep')
    print(f'  vary: {vary_name} = {",".join(str(v) for v in vary_values)}')
    print(f'  shape: {args.shape}  seq_len={args.seq_len}')
    print(f'  lr range: {args.lr_start:.0e} \u2192 {args.lr_end:.1f}  steps={args.steps}')
    print()

    runtime = setup_runtime(OutputConfig(device=args.device))

    for v in vary_values:
        fixed = dict(_parse_fixed(args.fixed))
        fixed[vary_name] = v
        if 'b' not in fixed:
            fixed['b'] = 2.0
        if 'n' not in fixed:
            fixed['n'] = 2

        cfg = SweepConfig(
            name='lr_find',
            model=ModelConfig(
                seq_len=args.seq_len,
                bottleneck=bottleneck,
                activation=args.activation,
                normalization=args.normalization,
                shape=args.shape,
                residual=args.residual,
                residual_norm=args.residual_norm,
            ),
            training=TrainConfig(batch_size=args.batch_size),
            sweep=SweepSpec(vary=vary_name, values=[v], fixed=fixed, budget=budget,
                             solve=args.solve),
        )

        arch = resolve_architecture(v, vary_name, cfg)
        n_params = arch['n_params']
        sizes = arch['sizes']

        arch_str = '\u2192'.join(str(s) for s in sizes)
        model_name = f'{args.shape[:4]}_s{args.seq_len}_n{arch["n"]}_b{arch["b"]:.4g}'
        if args.residual:
            model_name += '_res'

        print(f'{"─" * 55}')
        print(f'[{vary_name}={v}]  arch: {arch_str}  ({n_params:,} params)')

        result = run_lr_find_for_sweep(
            arch=arch,
            mc=cfg.model,
            runtime=runtime,
            batch_size=args.batch_size,
            lr_start=args.lr_start,
            lr_end=args.lr_end,
            steps=args.steps,
            output_dir=args.output,
            model_name=model_name,
            no_plot=args.no_plot,
            lr_ranges=args.lr_ranges,
        )

        cuda_safe_cleanup()
        print()

    print(f'{"=" * 55}')
    print(f'Plots saved to {args.output}/')


if __name__ == '__main__':
    main()
