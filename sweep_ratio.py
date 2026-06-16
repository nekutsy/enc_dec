"""Universal ratio sweep — find optimal n_hidden at fixed param budget.

Usage:
  python sweep_ratio.py --budget 20 --n-values 2 3 4 5 6
  python sweep_ratio.py --budget 40 --n-values 2 4 6 8 10
  python sweep_ratio.py --budget 80                  # defaults: all n-values per budget

Each n → binary search for b (width) such that params ≈ budget.
seq_len=32, 120M symbols, cosine LR, AdamW.
"""

import sys
import argparse

import torch

from configs import UNICODE_BITS
from data import load_text
from trainers import _cuda_safe_cleanup
from sweep_lib import (solve_b_for_n, adaptive_batch_size, train_model,
                       init_log, gather_done, log_row)

SEQ_LEN = 32
INPUT_DIM = SEQ_LEN * UNICODE_BITS  # 672
BOTTLENECK = SEQ_LEN
TARGET_SYMBOLS = 120_000_000

# Defaults per budget
BUDGET_DEFAULTS = {
    20: {'session': 'sessions/ratio20', 'log': 'sessions/ratio20_sweep_summary.csv',
         'n_default': range(2, 7), 'prefix': 'ratio20'},
    40: {'session': 'sessions/ratio40', 'log': 'sessions/ratio40_sweep_summary.csv',
         'n_default': range(2, 11, 2), 'prefix': 'ratio40'},
    80: {'session': 'sessions/ratio',     'log': 'sessions/ratio_sweep_summary.csv',
         'n_default': range(2, 17), 'prefix': 'ratio'},
}

COLUMNS = ['seq_len', 'b', 'n_hidden', 'input_dim', 'hidden_dim',
           'bottleneck', 'params', 'batch_size', 'total_symbols',
           'final_val_loss', 'status', 'duration_seconds']


def main():
    parser = argparse.ArgumentParser(description="Ratio sweep")
    parser.add_argument('--budget', type=int, required=True, choices=[20, 40, 80],
                        help="Target parameter count in millions")
    parser.add_argument('--n-values', type=int, nargs='+', default=None,
                        help="n values to test (default: budget-specific)")
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    cfg = BUDGET_DEFAULTS[args.budget]
    target_params = args.budget * 1_000_000
    n_values = args.n_values if args.n_values else list(cfg['n_default'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == 'auto' else torch.device(args.device)
    torch.backends.cudnn.benchmark = False

    init_log(cfg['log'], COLUMNS)
    existing = gather_done(cfg['log'], TARGET_SYMBOLS)
    text = load_text()

    print(f"Ratio sweep @ {args.budget}M  |  seq_len={SEQ_LEN}  |  {TARGET_SYMBOLS//1e6:.0f}M sym")
    print(f"Session: {cfg['session']}")

    # Plan
    print(f"\n{'n':>3}  {'b':>7}  {'h_dim':>5}  {'params':>10}  {'off%':>6}")
    plan = {}
    for n in n_values:
        b, h, p = solve_b_for_n(n, target_params, INPUT_DIM, BOTTLENECK)
        off = (p - target_params) / target_params * 100
        tag = " ✓" if n in existing else ""
        print(f"{n:>3}  {b:>7.4g}  {h:>5}  {p:>10,}  {off:>+5.0f}%{tag}")
        plan[n] = (b, h, p)

    # Run
    results = dict(existing)
    for n in n_values:
        if n in existing:
            print(f"\n  [n={n}] — already done ({existing[n]:.6f})")
            continue

        print(f"\n{'─'*50}")
        print(f"[n={n}]")
        val, status = train_model(
            n, target_params=target_params, target_symbols=TARGET_SYMBOLS,
            seq_len=SEQ_LEN, input_dim=INPUT_DIM, bottleneck=BOTTLENECK,
            device=device, text=text, session_dir=cfg['session'],
            model_prefix=cfg['prefix'])
        _cuda_safe_cleanup()

        b, h, p = plan[n]
        log_row(cfg['log'], [SEQ_LEN, f"{b:.6g}", n, INPUT_DIM, h,
                BOTTLENECK, p, adaptive_batch_size(p), TARGET_SYMBOLS,
                val if val else '', status, ''])
        if val is not None:
            results[n] = val

    # Summary
    print(f"\n{'='*55}")
    print(f"RESULTS @ {args.budget}M  (↑ eff = better)")
    print(f"{'n':>3}  {'b':>7}  {'params':>8}  {'val':>10}  {'eff':>8}")
    print("-" * 45)
    valid = {n: v for n, v in results.items() if v < 1e8}
    for n in sorted(valid):
        v = valid[n]
        eff = 1.0 / (v * plan[n][2] / 1e6)
        print(f"{n:>3}  {plan[n][0]:>7.4g}  {plan[n][2]/1e6:>5.1f}M  {v:>10.6f}  {eff:>8.4f}")

    if valid:
        best_n = min(valid, key=valid.get)
        b_best = plan[best_n][0]
        print(f"\n  ★ optimal: n={best_n}  b={b_best:.4g}  val={valid[best_n]:.6f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
