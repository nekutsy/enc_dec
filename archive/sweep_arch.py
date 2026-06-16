"""Architecture sweep — compare shape families at fixed param budget.

Families: wide-shallow, rectangle, narrow-deep, pyramid.
All models: same seq_len, same target symbols, ~same total param count.
"""

import sys
import os
import argparse
import time as time_mod

import torch
import torch.optim as optim
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from configs import PrimaryConfig, UNICODE_BITS
from model import Autoencoder
from data import load_text, prepare_data
from trainers import run_training, build_scheduler, _cuda_safe_cleanup
from sweep_lib import count_params, make_rectangular, adaptive_batch_size
from autoencoder import _save_paths, _compile_model, _train_setup
from logger import CSVLogger

SEQ_LEN = 32
INPUT_DIM = SEQ_LEN * UNICODE_BITS  # 672
BOTTLENECK = SEQ_LEN
TARGET_SYMBOLS = 120_000_000
TARGET_PARAMS = 80_000_000
MAX_PARAMS = 250_000_000


def _make_pyramid(d, bottleneck, b_start, n_enc, taper):
    """Tapered encoder: d → b_start*d → ... → bottleneck. Mirror decoder."""
    h_enc = []
    cur = round(d * b_start)
    for _ in range(n_enc):
        h_enc.append(max(cur, bottleneck * 4))
        cur = round(cur * taper)
    h_dec = list(reversed(h_enc))
    return [d] + h_enc + [bottleneck] + h_dec + [d]


def _build_architectures():
    d, b = INPUT_DIM, BOTTLENECK
    archs = {}

    # Wide-shallow: b=8, just 2 layers
    h_wide = 8 * d
    archs["wide-shallow"] = {
        "desc": "Few wide layers (b=8, n=2)",
        "sizes": make_rectangular(d, h_wide, b, 2),
    }

    # Rectangle baseline: b=4, n=6
    h_rect = 4 * d
    archs["rectangle"] = {
        "desc": "Rectangle baseline (b=4, n=6)",
        "sizes": make_rectangular(d, h_rect, b, 6),
    }

    # Narrow-deep: b=2, find n to hit target params
    h_narrow = 2 * d
    for n in range(10, 40):
        test = make_rectangular(d, h_narrow, b, n)
        if count_params(test) >= TARGET_PARAMS * 0.9:
            break
    archs["narrow-deep"] = {
        "desc": f"Many narrow layers (b=2, n={n})",
        "sizes": make_rectangular(d, h_narrow, b, n),
    }

    # Pyramid: start wide (b=8), taper down
    archs["pyramid"] = {
        "desc": "Pyramid (b_start=8, n=5, taper=0.7)",
        "sizes": _make_pyramid(d, b, 8, 5, 0.7),
    }

    return archs


def _train_arch(name, info, text, device):
    sizes = info["sizes"]
    n_params = count_params(sizes)
    bs = adaptive_batch_size(n_params)

    print(f"\n  {name}: {info['desc']}")
    print(f"  arch: {'→'.join(str(s) for s in sizes)}")
    print(f"  params: {n_params:,}  |  batch: {bs}")

    if n_params > MAX_PARAMS:
        print(f"  ⚠ {n_params:,} > {MAX_PARAMS//1e6:.0f}M — skipping")
        return None, "skip"

    model_name = f"arch_{name}_s{SEQ_LEN}"
    model_path, csv_path = _save_paths(sizes, model_name, prefix="sessions/arch")

    if os.path.isfile(csv_path):
        with open(csv_path) as f:
            import csv
            rows = list(csv.reader(f))
        if rows:
            last_sym = int(float(rows[-1][0]))
            if last_sym >= TARGET_SYMBOLS:
                val = float(rows[-1][2])
                print(f"  already done ({last_sym:,} sym, val={val:.6f})")
                return val, "done"

    config = PrimaryConfig(
        seq_len=SEQ_LEN, input_dim=INPUT_DIM, hidden_dim=sizes[1],
        bottleneck=BOTTLENECK, learning_rate=0.001, train_ratio=0.99,
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

    optimizer = optim.AdamW(model.parameters(), lr=0.001, fused=device.type == 'cuda')
    total_batches = int(TARGET_SYMBOLS / bs / SEQ_LEN) + 1
    scheduler = build_scheduler(optimizer, config, total_batches)
    criterion = nn.MSELoss()
    logger = CSVLogger(csv_path)

    start_sym = _train_setup(config, model, optimizer, csv_path, model_path, device)
    rem = max(0, TARGET_SYMBOLS - start_sym)
    if rem <= 0:
        return None, "skipped"

    print(f"  training {rem:,} symbols...")
    t_start = time_mod.time()

    try:
        _ = run_training(
            start_sym, TARGET_SYMBOLS, model, optimizer, criterion,
            train_ds, val_ds, logger, model_path, bs,
            SEQ_LEN, 1.0, 2, scheduler)

        import csv
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        val = float(rows[-1][2]) if rows and len(rows[-1]) > 2 else float('inf')
        dur = time_mod.time() - t_start
        print(f"  done: val={val:.6f} ({dur:.0f}s)")
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    text = load_text()

    archs = _build_architectures()

    print(f"Architecture sweep: {len(archs)} families")
    for name, info in archs.items():
        sizes = info["sizes"]
        n = count_params(sizes)
        tag = "✓" if abs(n - TARGET_PARAMS) / TARGET_PARAMS < 0.2 else "⚠"
        print(f"  {name:<16}  params={n:>10,}  {tag}  {info['desc']}")

    results = {}
    for name, info in archs.items():
        print(f"\n{'─'*40}\n[{name}]")
        val, status = _train_arch(name, info, text, device)
        _cuda_safe_cleanup()
        if val is not None:
            results[name] = val

    print(f"\n{'='*60}\nRESULTS")
    for name in ["wide-shallow", "rectangle", "narrow-deep", "pyramid"]:
        if name in results:
            r = results[name]
            info = archs[name]
            p = count_params(info["sizes"])
            print(f"  {name:<16}  val={r:.6f}  p={p:,}  {info['desc']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
