"""Universal batch-size sweep — vary batch_size at fixed architecture.

Usage:
  python sweep_batch.py --budget 20 --n 3 --batch-values 256 512 1024 2048 4096

Trains the same model multiple times with different batch sizes.
seq_len=32, 120M symbols, cosine LR, AdamW.
"""

import sys
import argparse
import time as time_mod

import torch

from configs import UNICODE_BITS
from data import load_text
from trainers import _cuda_safe_cleanup
from sweep_lib import solve_b_for_n, train_model

SEQ_LEN = 32
INPUT_DIM = SEQ_LEN * UNICODE_BITS  # 672
BOTTLENECK = SEQ_LEN
TARGET_SYMBOLS = 120_000_000

BUDGET_SESSIONS = {
    20: 'sessions/ratio20',
    40: 'sessions/ratio40',
    80: 'sessions/ratio',
}


def main():
    parser = argparse.ArgumentParser(description="Batch-size sweep")
    parser.add_argument('--budget', type=int, required=True, choices=[20, 40, 80],
                        help="Target param count (M)")
    parser.add_argument('--n', type=int, required=True, help="n_hidden value")
    parser.add_argument('--batch-values', type=int, nargs='+', required=True,
                        help="Batch sizes to test")
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    target_params = args.budget * 1_000_000
    session_dir = BUDGET_SESSIONS[args.budget]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == 'auto' else torch.device(args.device)
    torch.backends.cudnn.benchmark = False

    text = load_text()

    b_val, hidden_dim, n_params = solve_b_for_n(
        args.n, target_params, INPUT_DIM, BOTTLENECK)

    print(f"Batch-size sweep: {args.budget}M model, n={args.n}")
    print(f"  b={b_val:.4g}  hidden_dim={hidden_dim}  params={n_params:,}")
    print(f"  batch sizes: {args.batch_values}")

    results = {}
    for bs in args.batch_values:
        print(f"\n{'─'*45}")
        print(f"[bs={bs}]  steps={int(TARGET_SYMBOLS / bs / SEQ_LEN) + 1}")

        model_prefix = f"n{args.n}_bs{bs}" if len(args.batch_values) > 1 else f"ratio_{args.budget}m"
        val, status = train_model(
            args.n, target_params=target_params, target_symbols=TARGET_SYMBOLS,
            seq_len=SEQ_LEN, input_dim=INPUT_DIM, bottleneck=BOTTLENECK,
            device=device, text=text, session_dir=session_dir,
            model_prefix=model_prefix, batch_size=bs)
        _cuda_safe_cleanup()

        if val is not None:
            results[bs] = val

    # Summary
    print(f"\n{'='*55}")
    print(f"RESULTS: batch-size sweep ({args.budget}M, n={args.n})")
    print(f"{'bs':>6}  {'val':>12}  {'steps':>8}  {'Δ% vs min':>10}")
    print("-" * 45)

    if results:
        sorted_bs = sorted(results.keys())
        best_val = results[sorted(sorted_bs, key=lambda b: results[b])[0]]
        for bs in sorted_bs:
            val = results[bs]
            steps = int(TARGET_SYMBOLS / bs / SEQ_LEN) + 1
            delta = ((val - best_val) / best_val * 100) if best_val else 0
            tag = "★" if val == best_val else ""
            print(f"{bs:>6}  {val:>12.6f}  {steps:>8}  {delta:>+9.1f}%  {tag}")

        best_bs = min(results, key=results.get)
        print(f"\n  ★ optimal: bs={best_bs}  val={results[best_bs]:.6f}")

    print(f"{'='*55}")


if __name__ == "__main__":
    main()
