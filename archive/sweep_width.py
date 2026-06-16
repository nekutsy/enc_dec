"""Width sweep — grid search over hidden_dim multiplier (b = hidden_dim / input_dim).

For each seq_len, tests all b ∈ {1/7, 1/3, 1, 2, 4, 8} at optimal n_hidden.
Fixed budget: 120M symbols per model.
"""

import sys, os, csv, time as time_mod
import torch
import torch.optim as optim
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from configs import PrimaryConfig, UNICODE_BITS
from model import Autoencoder
from data import load_text, prepare_data
from trainers import run_training, build_scheduler, _cuda_safe_cleanup
from logger import CSVLogger
from autoencoder import _save_paths, _compile_model, _train_setup
from sweep_lib import count_params, make_rectangular, adaptive_batch_size, init_log

TARGET_SYMBOLS = 120_000_000
MAX_PARAMS = 250_000_000
B_VALUES = [1/7, 1/3, 1, 2, 4, 8]
SWEEP_LOG = "sessions/width_sweep_summary.csv"

# Best n_hidden from binary search (per seq_len)
BEST_H = {4: 14, 8: 9, 16: 8, 32: 6, 64: 3, 128: 1}


def _gather_done(seq_len):
    done = {}
    if not os.path.isfile(SWEEP_LOG):
        return done
    with open(SWEEP_LOG) as f:
        rows = list(csv.reader(f))
    for row in rows[1:]:
        try:
            sl = int(row[0])
            b_val = row[1]
            nh = int(row[2])
            status = row[12] if len(row) > 12 else ''
            val = float(row[11]) if row[11] else None
            sym = int(float(row[7])) if row[7] else 0
            if sl == seq_len and status == 'done' and sym >= TARGET_SYMBOLS * 0.85:
                done[(b_val, nh)] = val
        except (ValueError, IndexError):
            continue
    return done


def _train_model(seq_len, b_mult, n_hidden, text, device):
    input_dim = seq_len * UNICODE_BITS
    hidden_dim = max(1, int(round(input_dim * b_mult)))
    bottleneck = seq_len

    sizes = make_rectangular(input_dim, hidden_dim, bottleneck, n_hidden)
    n_params = count_params(sizes)

    if n_params > MAX_PARAMS:
        print(f"  ⚠ {n_params:,} > {MAX_PARAMS//1e6:.0f}M — skipping")
        return None, "skip"

    batch_size = adaptive_batch_size(n_params)
    model_name = f"width_s{seq_len}_b{b_mult:.4g}_h{n_hidden}"
    model_path, csv_path = _save_paths(sizes, model_name, prefix="sessions/width")

    arch_str = "→".join(str(s) for s in sizes)
    print(f"\n  s{seq_len} b={b_mult:.4g} h={n_hidden}  arch: ({arch_str})  "
          f"params: {n_params:,}  batch: {batch_size}")

    if os.path.isfile(csv_path):
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        if rows:
            last_sym = int(float(rows[-1][0]))
            if last_sym >= TARGET_SYMBOLS:
                val = float(rows[-1][2])
                print(f"  already done ({last_sym:,} sym, val={val:.6f})")
                return val, "done"

    config = PrimaryConfig(
        seq_len=seq_len, input_dim=input_dim, hidden_dim=hidden_dim,
        bottleneck=bottleneck, learning_rate=0.001, train_ratio=0.99,
        batch_size=batch_size, device=device.type, model_name=model_name,
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
    total_batches = int(TARGET_SYMBOLS / batch_size / seq_len) + 1
    scheduler = build_scheduler(optimizer, config, total_batches)
    criterion = nn.MSELoss()
    logger = CSVLogger(csv_path)

    start_sym = _train_setup(config, model, optimizer, csv_path, model_path, device)
    rem = max(0, TARGET_SYMBOLS - start_sym)
    if rem <= 0:
        return 0.0, "skipped"

    print(f"  training {rem:,} symbols...")
    t_start = time_mod.time()

    try:
        final_symbols = run_training(
            start_sym, TARGET_SYMBOLS, model, optimizer, criterion,
            train_ds, val_ds, logger, model_path, batch_size,
            seq_len, 1.0, 2, scheduler)
        status = "done"
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

    dur = time_mod.time() - t_start
    chars = final_symbols // seq_len if seq_len else 0
    final_val = 0.0
    if os.path.isfile(csv_path):
        with open(csv_path) as f:
            rows = list(csv.reader(f))
            if len(rows) >= 2:
                try:
                    final_val = float(rows[-1][2]) if len(rows[-1]) >= 3 else 0.0
                except ValueError:
                    pass

    print(f"  done: {final_symbols:,} sym in {dur:.0f}s  val={final_val:.6f}")

    with open(SWEEP_LOG, 'a', newline='') as f:
        csv.writer(f).writerow([
            seq_len, f"{b_mult:.4g}", n_hidden, input_dim, hidden_dim, bottleneck,
            n_params, batch_size, final_symbols, chars,
            '', final_val, status, int(dur)
        ])

    return final_val, status


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False

    init_log(SWEEP_LOG, [
        'seq_len', 'b', 'n_hidden', 'input_dim', 'hidden_dim',
        'bottleneck', 'params', 'batch_size', 'total_symbols',
        'chars_processed', 'final_train_loss', 'final_val_loss',
        'status', 'duration_seconds'
    ])

    text = load_text()
    seq_lens = [16, 32, 64, 128]

    print(f"Width sweep: {len(seq_lens)} seq_lens × {len(B_VALUES)} b-values")
    print(f"b-values: {[f'{v:.4g}' for v in B_VALUES]}")

    for seq_len in seq_lens:
        h_center = BEST_H[seq_len]
        print(f"\n{'─'*50}\nseq_len={seq_len}  h={h_center}\n{'─'*50}")

        existing = _gather_done(seq_len)

        for b_val in B_VALUES:
            key = (f"{b_val:.4g}", h_center)
            if key in existing:
                print(f"  b={b_val:.4g} h={h_center} — done (val={existing[key]:.6f})")
                continue
            print(f"  b={b_val:.4g} → hidden_dim={max(1, int(round(seq_len * UNICODE_BITS * b_val)))}")
            val, status = _train_model(seq_len, b_val, h_center, text, device)
            _cuda_safe_cleanup()

    print(f"\nWidth sweep complete: {SWEEP_LOG}")


if __name__ == "__main__":
    main()
