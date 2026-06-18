"""Training loop — shared by primary and secondary models."""

import torch
import os
import sys
import time
import csv
import signal
import atexit
from torch.utils.data import DataLoader

def _cuda_safe_cleanup():
    """Sync CUDA to avoid GPU ERR — call from MAIN THREAD only.

    Never call from signal handlers or subprocess forks.
    Robust against already-broken CUDA contexts.
    """
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass  # context may already be broken
        torch.cuda.empty_cache()

# Safety net: if the process exits without explicit cleanup,
# atexit runs in the main thread after all non-daemon threads join.
atexit.register(_cuda_safe_cleanup)


def _validate(model, val_loader, criterion, device):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            out = model(x_batch)
            loss = criterion(out, y_batch)
            n = x_batch.size(0)
            total_loss += loss.item() * n
            total_samples += n
    return total_loss / total_samples if total_samples > 0 else 0.0


def _log_checkpoint(csv_path, total_samples, total_symbols, train_loss, val_loss):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([total_samples, total_symbols, train_loss, val_loss])


def _progress_line(total_processed, max_total, loss, speed, eta):
    progress = (total_processed / max_total) * 100
    return f"\r\033[KProgress: {progress:.1f}% | Loss: {loss:.6f} | Speed: {speed:.0f} samples/s | ETA: {eta:.0f}s"


def _save_checkpoint(model, optimizer, model_path):
    _cuda_safe_cleanup()
    # Always save unwrapped model state — compiled models add _orig_mod. prefix
    unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save(unwrapped.state_dict(), model_path)
    opt_path = model_path + ".opt"
    torch.save(optimizer.state_dict(), opt_path)


def _load_optimizer(optimizer, model_path, device):
    opt_path = model_path + ".opt"
    if os.path.isfile(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device, weights_only=True))


def build_scheduler(optimizer, config, total_steps: int):
    """Return (per_step_scheduler, per_checkpoint_scheduler).
    
    per_step_scheduler: called every batch step (e.g. warmup, cosine).
    per_checkpoint_scheduler: called at validation checkpoints with val_loss (plateau).
    Returns (None, None) if no scheduler configured.
    """
    if not config.lr_scheduler:
        return None, None

    total_steps = max(total_steps, 1)
    warmup_steps = int(config.lr_warmup_epochs * total_steps)

    if config.lr_scheduler == "cosine":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        ) if warmup_steps > 0 else None
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps
        )
        if warmup:
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine],
                milestones=[warmup_steps]
            ), None
        return cosine, None

    if config.lr_scheduler == "plateau":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        ) if warmup_steps > 0 else None
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return warmup, plateau

    return None, None


def run_training(start_samples, max_samples, model, optimizer, criterion,
                 train_dataset, val_dataset, logger, model_path, batch_size,
                 seq_len, grad_clip=1.0, num_workers=0,
                 step_scheduler=None, checkpoint_scheduler=None,
                 early_stop_patience=3):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    device = next(model.parameters()).device

    # Register signal handlers — ONLY set a flag, never call CUDA from handler.
    # CUDA API calls from signal context can corrupt the driver → ERR! state.
    _interrupted = False

    def _graceful_exit(signum, frame):
        nonlocal _interrupted
        _interrupted = True

    prev_sigint = signal.signal(signal.SIGINT, _graceful_exit)
    prev_sigterm = signal.signal(signal.SIGTERM, _graceful_exit)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers,
                              pin_memory=(device.type == 'cuda'),
                              prefetch_factor=4 if num_workers > 0 else None,
                              persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers,
                            pin_memory=(device.type == 'cuda'),
                            prefetch_factor=4 if num_workers > 0 else None)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = scaler is not None

    total_samples_processed = start_samples
    LOG_INTERVAL = 500_000   # samples between validation checkpoints
    UPDATE_INTERVAL = 25_000  # samples between progress updates

    best_val_loss = float('inf')
    stale_checkpoints = 0
    best_model_path = model_path.replace('.pth', '_best.pth')
    _early_stopped = False

    interval_train_loss_sum = 0.0
    interval_train_count = 0

    next_update = total_samples_processed + UPDATE_INTERVAL
    next_log = total_samples_processed + LOG_INTERVAL
    last_update_time = time.time()
    last_update_samples = total_samples_processed

    if not os.path.isfile(logger.csv_path):
        with open(logger.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['total_samples', 'total_symbols', 'train_loss', 'val_loss'])

    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

    try:
        while total_samples_processed < max_samples and not _early_stopped:
            if _interrupted:
                raise KeyboardInterrupt
            model.train()
            for x_batch, y_batch in train_loader:
                if _interrupted or total_samples_processed >= max_samples:
                    break

                # ── safe interruption point (no CUDA ops in flight between batches) ──
                if _interrupted:
                    break
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        out = model(x_batch)
                        loss = criterion(out, y_batch)
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(x_batch)
                    loss = criterion(out, y_batch)
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                if step_scheduler is not None:
                    step_scheduler.step()

                batch_size_actual = x_batch.size(0)

                interval_train_loss_sum += loss.item() * batch_size_actual
                interval_train_count += batch_size_actual
                total_samples_processed += batch_size_actual

                if total_samples_processed >= next_update:
                    current_time = time.time()
                    time_delta = current_time - last_update_time
                    samples_delta = total_samples_processed - last_update_samples
                    speed = samples_delta / time_delta if time_delta > 0 else 0
                    remaining = max_samples - total_samples_processed
                    eta = remaining / speed if speed > 0 else 0
                    avg_loss = interval_train_loss_sum / interval_train_count if interval_train_count > 0 else 0

                    sys.stdout.write(_progress_line(total_samples_processed, max_samples, avg_loss, speed, eta))
                    sys.stdout.flush()

                    next_update += UPDATE_INTERVAL
                    last_update_time = current_time
                    last_update_samples = total_samples_processed

                if total_samples_processed >= next_log:
                    avg_train_loss = interval_train_loss_sum / interval_train_count if interval_train_count > 0 else 0
                    avg_val_loss = _validate(model, val_loader, criterion, device)
                    model.train()

                    if checkpoint_scheduler is not None:
                        checkpoint_scheduler.step(avg_val_loss)

                    # Early stopping
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        stale_checkpoints = 0
                        _save_checkpoint(model, optimizer, best_model_path)
                    else:
                        stale_checkpoints += 1

                    total_symbols_val = total_samples_processed * seq_len
                    _log_checkpoint(logger.csv_path, total_samples_processed, total_symbols_val, avg_train_loss, avg_val_loss)

                    # Write to global training log
                    from pathlib import Path
                    global_log = os.path.join(os.path.dirname(logger.csv_path), '..', 'training.log')
                    global_log = os.path.abspath(global_log)
                    os.makedirs(os.path.dirname(global_log), exist_ok=True)
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    model_tag = os.path.basename(model_path).replace('.pth', '').replace('_best', '')
                    with open(global_log, 'a') as gl:
                        gl.write(f'{timestamp} | {model_tag} | samples={total_samples_processed:,} | train={avg_train_loss:.6f} | val={avg_val_loss:.6f}\n')

                    sys.stdout.write(f"\r\033[K[{total_samples_processed} samples / {total_symbols_val} sym] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}\n")
                    sys.stdout.flush()

                    if stale_checkpoints >= early_stop_patience:
                        print(f"  Early stop: val loss not improved for {early_stop_patience} checkpoints (best={best_val_loss:.6f})")
                        _early_stopped = True
                        break

                    interval_train_loss_sum = 0.0
                    interval_train_count = 0
                    next_log += LOG_INTERVAL

                    last_update_time = time.time()
                    last_update_samples = total_samples_processed
                    next_update = total_samples_processed + UPDATE_INTERVAL

            if total_samples_processed >= max_samples:
                break

            if _interrupted:
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\nTraining interrupted. Cleaning up GPU...", file=sys.stderr, flush=True)
        _cuda_safe_cleanup()
        print("Saving checkpoint...", file=sys.stderr, flush=True)
        _save_checkpoint(model, optimizer, model_path)
        raise
    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)

    # Normal exit — sync before final save (at safe point)
    _cuda_safe_cleanup()

    if interval_train_count > 0:
        avg_train_loss = interval_train_loss_sum / interval_train_count
        avg_val_loss = _validate(model, val_loader, criterion, device)
        total_symbols_val = total_samples_processed * seq_len
        _log_checkpoint(logger.csv_path, total_samples_processed, total_symbols_val, avg_train_loss, avg_val_loss)
        sys.stdout.write(f"\r\033[K[{total_samples_processed} samples / {total_symbols_val} sym] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}\n")
        sys.stdout.flush()

    _save_checkpoint(model, optimizer, model_path)
    print(f"Training finished. Model saved to {model_path}")
    return total_samples_processed
