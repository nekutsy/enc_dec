"""Unified sweep runner — grid and binary search.

Three interfaces:
  1. Config file:  python sweep.py run --config configs/ratio_40m.json
  2. Shorthand:    python sweep.py grid --vary n=2,4,6,8,10 --solve b --budget 40M
  3. Override:     python sweep.py run --config ... --override model.seq_len=64
"""

import argparse
import os
import sys
import time as time_mod

import torch

sys.path.insert(0, os.path.dirname(__file__))

from configs import UNICODE_BITS
from data import load_text
from trainers import _cuda_safe_cleanup
from sweep_lib import (
    resolve_architecture, train_one, init_log, gather_done, log_row,
    gpu_health_check, UNIFIED_COLUMNS,
    solve_b_for_n, count_params, make_rectangular,
)
from sweep_config import SweepConfig


# ── Helpers ──────────────────────────────────────────────────

def _parse_size(s):
    """Parse "40M" → 40_000_000, "40" → 40_000_000."""
    s = s.upper().rstrip('M')
    return int(float(s) * 1_000_000)


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
    """Parse --override model.seq_len=64 sweep.solve=n into [(path, value), ...]."""
    result = []
    for item in override_args:
        path, value = item.rsplit('=', 1)
        # coerce numeric values
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        result.append((path, value))
    return result


def _setup_device(sweep_config: SweepConfig):
    """Set device + load text; store on config object as transient state."""
    oc = sweep_config.output
    device_str = oc.device
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False

    sweep_config._device = device
    sweep_config._text = load_text()

    return device


def _log_arch_info(vary_value, vary_name, sweep_config, arch):
    """Format arch info string for display."""
    mc = sweep_config.model
    seq_len = mc.seq_len
    bottleneck = mc.bottleneck if mc.bottleneck is not None else seq_len
    return (f'[{vary_name}={vary_value}]  n={arch["n"]}  b={arch["b"]:.4g}  '
            f'in={seq_len * UNICODE_BITS}→{arch["hidden_dim"]}×{arch["n"]}→{bottleneck}  '
            f'params={arch["n_params"]:,}')


# ── Sweep runner ─────────────────────────────────────────────

class SweepRunner:
    """Generic sweep runner: grid or binary strategy, driven by SweepConfig."""

    def __init__(self, sweep_config: SweepConfig):
        self.cfg = sweep_config
        self.results = {}

    def run(self):
        cfg = self.cfg

        device = _setup_device(cfg)

        if not gpu_health_check():
            print('⚠ GPU not available — check nvidia-smi')
            return None

        oc = cfg.output
        init_log(oc.sweep_log)
        existing = gather_done(oc.sweep_log, cfg.training.target_samples, cfg.model.seq_len)

        print(f'Sweep: {cfg.name}  ({cfg.sweep.strategy} over {cfg.sweep.vary})')
        print(f'  seq_len={cfg.model.seq_len}  target={cfg.training.target_samples//1e6:.0f}M samples  '
              f'budget={cfg.sweep.budget//1e6 if cfg.sweep.budget else "auto"}M')
        print(f'  workspace: {oc.workspace}  |  log: {oc.sweep_log}')
        print()

        if cfg.sweep.strategy == 'grid':
            self._run_grid(existing)
        elif cfg.sweep.strategy == 'binary':
            self._run_binary(existing)

        self._print_summary()
        return self.results

    def _train_and_log(self, vary_value):
        """Resolve, train, log, store result. Returns (val, status)."""
        cfg = self.cfg
        vary_name = cfg.sweep.vary

        arch = resolve_architecture(vary_value, vary_name, cfg)
        n_params = arch['n_params']

        if n_params > 250_000_000:
            print(f'  ⚠ {n_params:,} > 250M — skipping')
            return None, 'skip'

        # Prefix must include model-level vary params to avoid path collision
        prefix = f'sweep_{vary_name}{vary_value}'
        if vary_name in ('normalization', 'activation', 'dropout', 'batch_size'):
            prefix = f'{vary_name}_{vary_value}_sweep'
        val, status, actual_samples = train_one(arch, cfg, prefix)
        _cuda_safe_cleanup()

        mc = cfg.model
        seq_len = mc.seq_len
        bottleneck = mc.bottleneck if mc.bottleneck is not None else seq_len

        log_row(cfg.output.sweep_log, {
            'sweep_type': cfg.sweep.strategy,
            'vary_param': vary_name,
            'vary_value': str(vary_value),
            'seq_len': seq_len,
            'n_hidden': arch['n'],
            'b': f'{arch["b"]:.6g}',
            'hidden_dim': arch['hidden_dim'],
            'bottleneck': bottleneck,
            'params': n_params,
            'batch_size': cfg.output.batch_size,
            'total_samples': actual_samples,
            'total_symbols': actual_samples * seq_len,
            'final_val_loss': val if val is not None else '',
            'status': status,
            'duration_seconds': '',
        })

        if val is not None:
            self.results[vary_value] = val
        return val, status

    def _run_grid(self, existing):
        cfg = self.cfg
        vary_values = cfg.sweep.values
        vary_name = cfg.sweep.vary

        # Plan
        print(f'{"vary":>8}  {"n":>3}  {"b":>7}  {"params":>10}')
        for v in vary_values:
            try:
                arch = resolve_architecture(v, vary_name, cfg)
                tag = ' ✓' if v in existing else ''
                print(f'{v:>8}  {arch["n"]:>3}  {arch["b"]:>7.4g}  {arch["n_params"]:>10,}{tag}')
            except Exception as e:
                print(f'{v:>8}  ERROR: {e}')
        print()

        for v in vary_values:
            if v in existing:
                print(f'[{vary_name}={v}] — already done ({existing[v]:.6f})')
                self.results[v] = existing[v]
                continue

            print(f'{"─"*50}')
            print(f'[{vary_name}={v}]')
            self._train_and_log(v)
            print()

    def _run_binary(self, existing):
        """Binary search over vary in [lo, hi].
        
        For float vary (like lr), step is auto-detected:
        if both values are floats, step = 10^k where k is min decimal places.
        """
        cfg = self.cfg
        lo, hi = cfg.sweep.values[0], cfg.sweep.values[1]
        vary_name = cfg.sweep.vary
        results = dict(existing)
        
        # Auto-detect step for convergence
        is_float = isinstance(lo, float) or isinstance(hi, float)
        if is_float:
            # Find granularity: e.g. 0.001 → step 0.001, 0.01 → 0.01
            def _decimals(v):
                s = f"{v:.10f}".rstrip('0')
                if '.' in s:
                    return len(s.split('.')[1])
                return 0
            step = 10 ** (-max(_decimals(lo), _decimals(hi)))
            def _to_step(v):
                return round(v / step) * step
        else:
            step = 1
            def _to_step(v):
                return int(v)

        # Probe boundaries
        for boundary in [lo, hi]:
            if boundary not in results:
                print(f'[{vary_name}={boundary}] probing boundary...')
                val, status = self._train_and_log(boundary)
                if val is not None:
                    results[boundary] = val
                else:
                    results[boundary] = 1e9
                print()

        for iteration in range(12):
            valid = {k: v for k, v in results.items() if v < 1e8}
            if len(valid) < 2:
                print('  → not enough valid results')
                break

            sorted_ns = sorted(valid, key=valid.get)
            best = sorted_ns[0]

            # Check neighbors of best — converge when both sides tested
            left_neighbor = _to_step(best - step)
            right_neighbor = _to_step(best + step)
            has_left = best <= lo or left_neighbor in results
            has_right = best >= hi or right_neighbor in results

            print(f'  best: {vary_name}={best} ({results[best]:.6f})')

            if has_left and has_right:
                print(f'  → converged (best surrounded by tested neighbors)\n')
                break

            # Determine what to test next
            # Priority: test missing neighbor of best
            if not has_left and left_neighbor not in results:
                mid = left_neighbor
            elif not has_right and right_neighbor not in results:
                mid = right_neighbor
            else:
                second = sorted_ns[1]
                print(f'  2nd: {vary_name}={second} ({results[second]:.6f})')
                mid = _to_step((best + second) / 2)
                if mid in results:
                    lo2, hi2 = min(best, second), max(best, second)
                    found = False
                    candidate = lo2 + step
                    while candidate < hi2:
                        c = _to_step(candidate)
                        if c not in results:
                            mid = c
                            found = True
                            break
                        candidate += step
                    if not found:
                        print(f'  → all values tested — converged\n')
                        break

            print(f'  → testing {vary_name}={mid}\n')
            val, status = self._train_and_log(mid)
            if val is not None:
                results[mid] = val
            else:
                results[mid] = 1e9
            print()

        self.results = {k: v for k, v in results.items() if v < 1e8}

    def _print_summary(self):
        if not self.results:
            return
        print(f'{"="*55}')
        print(f'RESULTS: {self.cfg.name}')
        print(f'{"vary":>8}  {"val_loss":>10}')
        print('-' * 25)
        for v in sorted(self.results):
            print(f'{v:>8}  {self.results[v]:>10.6f}')
        best = min(self.results, key=self.results.get)
        print(f'\n  ★ best: {self.cfg.sweep.vary}={best}  val={self.results[best]:.6f}')
        print(f'{"="*55}')


# ── Embedded binary inside grid ──────────────────────────────

def _run_grid_with_binary(outer_config: SweepConfig, binary_vary, binary_range):
    """Grid over sweep.vary values; for each, run binary search on binary_vary.

    Used by CLI shorthand: --binary-on n --range 1 16
    """
    cfg = outer_config
    device = _setup_device(cfg)

    if not gpu_health_check():
        print('⚠ GPU not available')
        return

    init_log(cfg.output.sweep_log)

    print(f'Grid over {cfg.sweep.vary} × binary on {binary_vary}')
    print(f'  {cfg.sweep.vary} values: {cfg.sweep.values}')
    print(f'  {binary_vary} range: {binary_range}')
    print()

    all_best = {}
    for outer_val in cfg.sweep.values:
        # Clone config with outer value fixed
        inner_fixed = dict(cfg.sweep.fixed)
        inner_fixed[cfg.sweep.vary] = outer_val

        from sweep_config import SweepSpec
        inner_sweep = SweepSpec(
            strategy='binary',
            vary=binary_vary,
            values=list(binary_range),
            solve=cfg.sweep.solve,
            budget=cfg.sweep.budget,
            fixed=inner_fixed,
        )

        binary_cfg = SweepConfig(
            name=f'{cfg.name}_{cfg.sweep.vary}{outer_val}',
            model=cfg.model,
            training=cfg.training,
            sweep=inner_sweep,
            output=cfg.output,
        )
        binary_cfg._device = cfg._device
        binary_cfg._text = cfg._text

        print(f'\n{"="*60}')
        print(f'[{cfg.sweep.vary}={outer_val}] binary search on {binary_vary}')
        print(f'{"="*60}')

        runner = SweepRunner(binary_cfg)
        results = runner.run()
        if results:
            best = min(results, key=results.get)
            all_best[outer_val] = best
            print(f'  ★ [{cfg.sweep.vary}={outer_val}] best {binary_vary}={best}  val={results[best]:.6f}')

    if all_best:
        print(f'\n{"="*60}')
        print('GRID × BINARY SUMMARY')
        for ov in sorted(all_best):
            print(f'  {cfg.sweep.vary}={ov:>4d}  →  {binary_vary}={all_best[ov]}')
        print(f'{"="*60}')


# ── CLI ──────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(description='Unified sweep runner')
    sub = parser.add_subparsers(dest='command', required=True)

    # run — from config file
    rp = sub.add_parser('run', help='Run sweep from config file')
    rp.add_argument('--config', required=True, help='Path to sweep config JSON')
    rp.add_argument('--override', nargs='+', default=[],
                    help='Override config fields: model.seq_len=64 sweep.solve=n')

    # grid — shorthand
    gp = sub.add_parser('grid', help='Grid search over parameter values')
    gp.add_argument('--vary', required=True, help='e.g. n=2,3,4,5 or batch_size=64,128,256')
    gp.add_argument('--solve', default=None, choices=['b', 'n'])
    gp.add_argument('--fixed', nargs='+', default=[])
    gp.add_argument('--budget', type=str, default=None, help='Target params, e.g. 40M')
    gp.add_argument('--seq-len', type=int, default=32)
    gp.add_argument('--bottleneck', type=int, default=None)
    gp.add_argument('--lr', type=float, default=0.001)
    gp.add_argument('--scheduler', default='cosine')
    gp.add_argument('--target-samples', type=str, default='5M')
    gp.add_argument('--workspace', default='sessions/sweep')
    gp.add_argument('--sweep-log', default='sessions/sweep_summary.csv')
    gp.add_argument('--device', default='auto')
    gp.add_argument('--batch-size', type=int, default=None)
    gp.add_argument('--binary-on', default=None,
                    help='Parameter to binary-search for each grid value')
    gp.add_argument('--range', type=int, nargs=2, default=None)

    # binary — shorthand
    bp = sub.add_parser('binary', help='Binary search over a parameter')
    bp.add_argument('--vary', required=True, help='Parameter to search, e.g. n')
    bp.add_argument('--range', type=int, nargs=2, required=True)
    bp.add_argument('--solve', default=None, choices=['b', 'n'])
    bp.add_argument('--fixed', nargs='+', default=[])
    bp.add_argument('--budget', type=str, default=None)
    bp.add_argument('--seq-len', type=int, default=32)
    bp.add_argument('--bottleneck', type=int, default=None)
    bp.add_argument('--lr', type=float, default=0.001)
    bp.add_argument('--scheduler', default='cosine')
    bp.add_argument('--target-samples', type=str, default='5M')
    bp.add_argument('--workspace', default='sessions/sweep')
    bp.add_argument('--sweep-log', default='sessions/sweep_summary.csv')
    bp.add_argument('--device', default='auto')
    bp.add_argument('--batch-size', type=int, default=None)

    return parser


def _cli_shorthand_to_config(args, vary_values) -> SweepConfig:
    """Convert CLI shorthand args to SweepConfig."""
    from sweep_config import ModelConfig, TrainingConfig, SweepSpec, OutputConfig

    budget = _parse_size(args.budget) if args.budget else None
    target_samples = _parse_size(args.target_samples)

    return SweepConfig(
        name=f'{args.command}_{getattr(args, "vary", "sweep")}'.replace('=', '_'),
        model=ModelConfig(
            seq_len=getattr(args, 'seq_len', 32),
            bottleneck=getattr(args, 'bottleneck', None),
        ),
        training=TrainingConfig(
            target_samples=target_samples,
            lr=getattr(args, 'lr', 0.001),
            scheduler=getattr(args, 'scheduler', 'cosine'),
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
            sweep_log=getattr(args, 'sweep_log', 'sessions/sweep_summary.csv'),
            device=getattr(args, 'device', 'auto'),
            batch_size=getattr(args, 'batch_size', None),
        ),
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── «run» — config file ──
    if args.command == 'run':
        cfg = SweepConfig.from_json(args.config)

        # Apply overrides
        for path, value in _parse_overrides(args.override):
            cfg.apply_override(path, value)
            print(f'  override: {path} = {value}')

        runner = SweepRunner(cfg)
        runner.run()
        return

    # ── «grid» / «binary» — shorthand ──
    if args.command == 'grid':
        vary_str = args.vary.split('=', 1)[1]
        vary_values = [int(v) if '.' not in v else float(v) for v in vary_str.split(',')]
    else:  # binary
        vary_values = list(args.range)

    cfg = _cli_shorthand_to_config(args, vary_values)

    if args.command == 'grid' and args.binary_on:
        _run_grid_with_binary(cfg, args.binary_on, args.range)
    else:
        runner = SweepRunner(cfg)
        runner.run()


if __name__ == '__main__':
    main()
