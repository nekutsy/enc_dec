"""Shared utilities for enc_dec sweep scripts.

Provides: count_params, make_rectangular, solve_b_for_n, adaptive_batch_size,
train_model, init_log, gather_done.
"""

import os
import csv
import time as time_mod

import torch
import torch.optim as optim
import torch.nn as nn

from configs import PrimaryConfig, UNICODE_BITS
from model import Autoencoder
from data import load_text, prepare_data
from trainers import run_training, build_scheduler, _cuda_safe_cleanup
from logger import CSVLogger
from autoencoder import _save_paths, _compile_model, _train_setup


def count_params(layer_sizes):
    """Count Linear + BatchNorm1d parameters."""
    n = 0
    for i in range(len(layer_sizes) - 1):
        n += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
        n += 2 * layer_sizes[i + 1]
    return n


def make_rectangular(input_dim, hidden_dim, bottleneck, n_hidden):
    """[input] → [hidden]×n → [bottleneck] → [hidden]×n → [input]"""
    return [input_dim] + [hidden_dim] * n_hidden + [bottleneck] + [hidden_dim] * n_hidden + [input_dim]


def solve_b_for_n(n_hidden, target_params, input_dim, bottleneck):
    """Binary search b ∈ [0.1, 20] such that total params ≈ target_params.
    Returns (b_val, hidden_dim, actual_params)."""
    def _p(b_val):
        h = max(1, int(round(input_dim * b_val)))
        return count_params(make_rectangular(input_dim, h, bottleneck, n_hidden))

    lo, hi = 0.1, 20.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if _p(mid) < target_params:
            lo = mid
        else:
            hi = mid

    p_lo, p_hi = _p(lo), _p(hi)
    b_val = round(lo, 6) if abs(p_lo - target_params) <= abs(p_hi - target_params) else round(hi, 6)
    h = max(1, int(round(input_dim * b_val)))
    return b_val, h, _p(b_val)


def adaptive_batch_size(n_params):
    """Heuristic batch size based on estimated VRAM usage."""
    est_gb = n_params * 4 * 8 / 1e9
    if est_gb < 0.25:      return 32768
    elif est_gb < 1.0:      return 8192
    elif est_gb < 2.0:      return 4096
    elif est_gb < 3.5:      return 2048
    elif est_gb < 5.0:      return 1024
    elif est_gb < 6.5:      return 512
    else:                    return 256


def train_model(n_hidden, *, target_params, target_symbols, seq_len,
                input_dim, bottleneck, device, text, session_dir,
                model_prefix, batch_size=None, lr=0.001,
                max_params=250_000_000):
    """Train one rectangular model at target param budget.
    Returns (val_loss, status)."""

    b_val, hidden_dim, n_params = solve_b_for_n(
        n_hidden, target_params, input_dim, bottleneck)

    if n_params > max_params:
        print(f"  ⚠ {n_params:,} > {max_params // 1e6:.0f}M — skipping")
        return None, "skip"

    bs = batch_size if batch_size is not None else adaptive_batch_size(n_params)
    sizes = make_rectangular(input_dim, hidden_dim, bottleneck, n_hidden)
    model_name = f"{model_prefix}_s{seq_len}_n{n_hidden}"
    model_path, csv_path = _save_paths(sizes, model_name, prefix=session_dir)

    off = (n_params - target_params) / target_params * 100
    arch_str = f"{sizes[0]}→[{hidden_dim}×{n_hidden}]→{bottleneck}"
    print(f"\n  n={n_hidden}  b={b_val:.4g}  hidden_dim={hidden_dim}  "
          f"params={n_params:,} ({off:+.0f}%)  batch={bs}")
    print(f"  {arch_str}")

    if os.path.isfile(csv_path):
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        if rows:
            last_sym = int(float(rows[-1][0]))
            if last_sym >= target_symbols:
                val = float(rows[-1][2])
                print(f"  already done ({last_sym:,} sym, val={val:.6f})")
                return val, "done"

    config = PrimaryConfig(
        seq_len=seq_len, input_dim=input_dim, hidden_dim=hidden_dim,
        bottleneck=bottleneck, learning_rate=lr, train_ratio=0.99,
        batch_size=bs, device=device.type, model_name=model_name,
        grad_clip=1.0, num_workers=2 if device.type == "cuda" else 0,
        lr_scheduler="cosine", lr_warmup_epochs=0.05, cudnn_benchmark=False,
    )

    train_ds, val_ds = prepare_data(text, config)

    try:
        model = Autoencoder(sizes, name=config.model_name).to(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
            raise
        print(f"  ⚠ OOM")
        _cuda_safe_cleanup()
        return None, "oom"

    try:
        model = _compile_model(model, device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
            raise
        print(f"  ⚠ OOM during compile")
        _cuda_safe_cleanup()
        del model
        return None, "oom"

    optimizer = optim.AdamW(model.parameters(), lr=lr, fused=device.type == 'cuda')
    total_batches = int(target_symbols / bs / seq_len) + 1
    scheduler = build_scheduler(optimizer, config, total_batches)
    criterion = nn.MSELoss()
    logger = CSVLogger(csv_path)

    start_sym = _train_setup(config, model, optimizer, csv_path, model_path, device)
    rem = max(0, target_symbols - start_sym)
    if rem <= 0:
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        val = float(rows[-1][2]) if rows and len(rows[-1]) > 2 else 0
        return val, "done"

    print(f"  training {rem:,} symbols...")
    t_start = time_mod.time()

    try:
        _ = run_training(
            start_sym, target_symbols, model, optimizer, criterion,
            train_ds, val_ds, logger, model_path, bs,
            seq_len, 1.0, 2, scheduler)

        with open(csv_path) as f:
            rows = list(csv.reader(f))
        val = float(rows[-1][2]) if rows and len(rows[-1]) > 2 else 0
        dur = time_mod.time() - t_start
        print(f"  done: {target_symbols:,} sym in {dur:.0f}s  val={val:.6f}")
        return val, "done"
    except torch.cuda.OutOfMemoryError:
        print(f"  ⚠ OOM")
        _cuda_safe_cleanup()
        del model, optimizer
        return None, "oom"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  ⚠ OOM")
            _cuda_safe_cleanup()
            del model, optimizer
            return None, "oom"
        raise
    except KeyboardInterrupt:
        _cuda_safe_cleanup()
        raise


def init_log(sweep_log, columns):
    """Create CSV log file with header if missing."""
    os.makedirs(os.path.dirname(sweep_log), exist_ok=True)
    if not os.path.isfile(sweep_log):
        with open(sweep_log, 'w', newline='') as f:
            csv.writer(f).writerow(columns)


def gather_done(sweep_log, target_symbols, key_col=2, val_col=9, sym_col=8, status_col=10):
    """Read completed models from CSV log. Returns {key: val_loss}."""
    done = {}
    if not os.path.isfile(sweep_log):
        return done
    with open(sweep_log) as f:
        for row in csv.reader(f):
            try:
                key = int(row[key_col])
                val = float(row[val_col]) if row[val_col] else None
                status = row[status_col] if len(row) > status_col else ''
                sym = int(float(row[sym_col])) if len(row) > sym_col and row[sym_col] else 0
                if status == 'done' and sym >= target_symbols * 0.85 and val is not None:
                    done[key] = val
            except (ValueError, IndexError):
                continue
    return done


def log_row(sweep_log, row):
    """Append a row to the sweep CSV."""
    with open(sweep_log, 'a', newline='') as f:
        csv.writer(f).writerow(row)
