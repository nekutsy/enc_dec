"""Binary search sweep — finds optimal n_hidden per seq_len.

Strategy: probe min/max boundaries → binary search between best and second.
Stops when best and second are adjacent.
All hidden_dims = 4 × input_dim (no special cases).
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
MAX_HIDDEN = 16
MAX_PARAMS = 250_000_000
SWEEP_LOG = "sessions/sweep_binary_summary.csv"


def _read_val_at(csv_path, target_sym):
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return None
    data = [(int(float(r[0])), float(r[2])) for r in rows[1:]]
    if not data:
        return None
    best = min(data, key=lambda r: abs(r[0] - target_sym))
    return best[1], best[0]


def _gather_existing(seq_len):
    results = {}
    if not os.path.isfile(SWEEP_LOG):
        return results
    with open(SWEEP_LOG) as f:
        rows = list(csv.reader(f))
    for row in rows[1:]:
        try:
            sl, nh = int(row[0]), int(row[1])
            val = float(row[11]) if row[11] else None
            sym = int(float(row[7])) if row[7] else 0
            status = row[12]
            if sl == seq_len and status == 'done' and sym >= TARGET_SYMBOLS * 0.85:
                results[nh] = val
        except (ValueError, IndexError):
            continue
    return results


def _train_model(seq_len, n_hidden, text, device):
    input_dim = seq_len * UNICODE_BITS
    hidden_dim = 4 * input_dim
    bottleneck = seq_len

    sizes = make_rectangular(input_dim, hidden_dim, bottleneck, n_hidden)
    n_params = count_params(sizes)
    batch_size = adaptive_batch_size(n_params)

    model_name = f"sweep_s{seq_len}_h{n_hidden}"
    model_path, csv_path = _save_paths(sizes, model_name, prefix="sessions/sweep")

    train_n = int((len(text) - seq_len + 1) * 0.99)
    symbols_per_pass = train_n * seq_len
    eq_epochs = TARGET_SYMBOLS / symbols_per_pass if symbols_per_pass > 0 else 0

    arch_str = "→".join(str(s) for s in sizes)
    print(f"\n  s{seq_len}_h{n_hidden}  arch: ({arch_str})  params: {n_params:,}  batch: {batch_size}")
    print(f"  target: {TARGET_SYMBOLS:,} sym = ~{TARGET_SYMBOLS / len(text):.1f} exposures/char ({eq_epochs:.2f} eps)")

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
    except torch.cuda.OutOfMemoryError:
        print(f"  ⚠ OOM")
        _cuda_safe_cleanup()
        return None, "oom"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  ⚠ OOM")
            _cuda_safe_cleanup()
            return None, "oom"
        raise

    try:
        model = _compile_model(model, device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
            raise
        print(f"  ⚠ OOM during compile")
        _cuda_safe_cleanup()
        del model
        return None, "oom"

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, fused=device.type == 'cuda')
    total_batches = int(TARGET_SYMBOLS / config.batch_size / config.seq_len) + 1
    scheduler = build_scheduler(optimizer, config, total_batches)
    criterion = nn.MSELoss()
    logger = CSVLogger(csv_path)

    start_symbols = _train_setup(config, model, optimizer, csv_path, model_path, device)
    remaining_sym = max(0, TARGET_SYMBOLS - start_symbols)

    if remaining_sym == 0:
        entry = _read_val_at(csv_path, TARGET_SYMBOLS)
        val = entry[0] if entry else None
        print(f"  already complete ({start_symbols:,} sym), val≈{val:.6f}")
        return val, "skipped"

    print(f"  resuming from {start_symbols:,} sym ({remaining_sym:,} remaining)")
    t_start = time_mod.time()

    try:
        final_symbols = run_training(
            start_symbols=start_symbols, max_symbols=TARGET_SYMBOLS,
            model=model, optimizer=optimizer, criterion=criterion,
            train_dataset=train_ds, val_dataset=val_ds,
            logger=logger, model_path=model_path, batch_size=config.batch_size,
            symbols_per_sample=config.seq_len, grad_clip=config.grad_clip,
            num_workers=config.num_workers, scheduler=scheduler,
        )
        final_entry = _read_val_at(csv_path, TARGET_SYMBOLS)
        final_val = final_entry[0] if final_entry else 0.0
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

    duration = time_mod.time() - t_start
    chars = final_symbols // seq_len if seq_len else 0
    print(f"  done: {final_symbols:,} sym in {duration:.0f}s  val={final_val:.6f}")

    with open(SWEEP_LOG, 'a', newline='') as f:
        csv.writer(f).writerow([
            seq_len, n_hidden, input_dim, hidden_dim, bottleneck,
            sum(p.numel() for p in model.parameters()), batch_size,
            final_symbols, chars, '', '', final_val, status, int(duration)
        ])

    return final_val, status


def _binary_search(seq_len, text, device, existing):
    results = dict(existing)

    # Probe h=1
    if 1 not in results:
        print(f"  probing h=1 (min boundary)...")
        val, status = _train_model(seq_len, 1, text, device)
        if status != "oom":
            results[1] = val
        print()

    # Probe right boundary
    for candidate_n in [MAX_HIDDEN, 8, 4, 2, 1]:
        if count_params(make_rectangular(
                seq_len * UNICODE_BITS, 4 * seq_len * UNICODE_BITS, seq_len, candidate_n)) > MAX_PARAMS:
            continue
        if candidate_n in results:
            print(f"  right boundary: h={candidate_n} (done, val={results[candidate_n]:.6f})\n")
            break
        print(f"  probing h={candidate_n} (right boundary)...")
        val, status = _train_model(seq_len, candidate_n, text, device)
        if status == "oom":
            _cuda_safe_cleanup()
            continue
        results[candidate_n] = val
        print()
        break

    # Binary search
    for _ in range(12):
        if len(results) < 2:
            break
        sorted_ns = sorted(results.keys(), key=lambda n: results[n])
        best, second = sorted_ns[0], sorted_ns[1]
        print(f"  best: h={best} ({results[best]:.6f})  2nd: h={second} ({results[second]:.6f})")

        if abs(best - second) <= 1:
            print(f"  → converged (adjacent)\n")
            break

        mid = (best + second) // 2
        if mid in results:
            lo, hi = min(best, second), max(best, second)
            found = False
            for candidate in range(lo + 1, hi):
                if candidate not in results:
                    mid = candidate
                    found = True
                    break
            if not found:
                print(f"  → all values between {lo} and {hi} tested — converged\n")
                break

        print(f"  → testing h={mid} between h={best} and h={second}\n")
        sizes = make_rectangular(seq_len * UNICODE_BITS, 4 * seq_len * UNICODE_BITS, seq_len, mid)
        if count_params(sizes) > MAX_PARAMS:
            print(f"  h={mid} exceeds MAX_PARAMS — marking as boundary\n")
            results[mid] = 1e9
            continue
        val, status = _train_model(seq_len, mid, text, device)
        if status == "oom":
            results[mid] = 1e9
            continue
        results[mid] = val
        print()

    return results


def _is_converged(seq_len, existing):
    if len(existing) < 2:
        return False
    sorted_ns = sorted(existing.keys(), key=lambda n: existing[n])
    return abs(sorted_ns[0] - sorted_ns[1]) <= 1


def main():
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_type}")
    device = torch.device(device_type)

    if device_type == "cuda":
        torch.backends.cudnn.benchmark = False

    init_log(SWEEP_LOG, [
        'seq_len', 'n_hidden', 'input_dim', 'hidden_dim', 'bottleneck',
        'params', 'batch_size', 'total_symbols', 'chars_processed', 'epochs',
        'final_train_loss', 'final_val_loss', 'status', 'duration_seconds'
    ])

    text = load_text()
    seq_lens = [4, 8, 16, 32, 64, 128]
    all_best = {}

    print(f"\n{'='*60}")
    print(f"Binary sweep: {len(seq_lens)} seq_lens  |  hidden = 4× input_dim")
    print(f"Target: {TARGET_SYMBOLS:,} sym/model")
    print(f"{'='*60}")

    for seq_len in seq_lens:
        input_dim = seq_len * UNICODE_BITS
        existing = _gather_existing(seq_len)

        if _is_converged(seq_len, existing):
            best_n = min(existing, key=existing.get)
            all_best[seq_len] = best_n
            print(f"\n  ★ seq_len={seq_len}: converged → n_hidden={best_n}  val={existing[best_n]:.6f}")
            continue

        print(f"\n{'─'*40}")
        print(f"seq_len={seq_len}  input_dim={input_dim}")
        results = _binary_search(seq_len, text, device, existing)

        valid = {n: v for n, v in results.items() if v < 1e8}
        if valid:
            best_n = min(valid, key=valid.get)
            all_best[seq_len] = best_n
            print(f"  ★ BEST: n_hidden={best_n}  val={valid[best_n]:.6f}")
        else:
            print(f"  ★ No valid results")

    print(f"\n{'='*60}\nSUMMARY")
    for sl, nh in sorted(all_best.items()):
        print(f"  seq_len={sl:3d}  →  n_hidden={nh}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
