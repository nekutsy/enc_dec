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


atexit.register(_cuda_safe_cleanup)


# ── Validation ──────────────────────────────────────────────

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


# ── Checkpoint save/load ────────────────────────────────────

def _save_checkpoint(model, optimizer, model_path):
    _cuda_safe_cleanup()
    unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save(unwrapped.state_dict(), model_path)
    opt_path = model_path + ".opt"
    torch.save(optimizer.state_dict(), opt_path)


def _load_optimizer(optimizer, model_path, device):
    opt_path = model_path + ".opt"
    if os.path.isfile(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device, weights_only=True))


# ── Scheduler builder ───────────────────────────────────────

def build_scheduler(optimizer, config, total_steps: int, start_samples: int = 0):
    """Return (per_step_scheduler, per_checkpoint_scheduler).
    
    per_step_scheduler: called every batch step (e.g. warmup, cosine).
    per_checkpoint_scheduler: called at validation checkpoints with val_loss (plateau).
    Returns (None, None) if no scheduler configured.
    
    If start_samples > 0 (resume): warmup is skipped.
    """
    if not config.lr_scheduler:
        return None, None

    total_steps = max(total_steps, 1)
    warmup_steps = int(config.lr_warmup_epochs * total_steps)

    # Skip warmup on resume — LR is already at operating value
    if start_samples > 0:
        warmup_steps = 0

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

    if config.lr_scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.learning_rate,
            total_steps=total_steps, pct_start=0.3,
            anneal_strategy='cos', div_factor=25.0,
            final_div_factor=10000.0,
        )
        return scheduler, None

    return None, None


# ── Training loop ────────────────────────────────────────────

def run_training(start_samples: int, max_samples: int, model, optimizer, criterion,
                 train_dataset, val_dataset, train_logger, model_path, batch_size,
                 seq_len, grad_clip=1.0, num_workers=0,
                 step_scheduler=None, checkpoint_scheduler=None,
                 early_stop_patience=3, no_val=False):
    """Main training loop.

    Args:
        train_logger: TrainingLogger instance (replaces old CSVLogger).
    """
    from logger import TrainingLogger

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    device = next(model.parameters()).device

    # ── Signal handlers — only set a flag, never call CUDA ──
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

    if no_val:
        LOG_INTERVAL = 250_000  # log train_loss twice as often (fast, no val)
        val_loader = None
    else:
        LOG_INTERVAL = 500_000
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers,
                                pin_memory=(device.type == 'cuda'),
                                prefetch_factor=4 if num_workers > 0 else None)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = scaler is not None

    total_samples_processed = start_samples
    UPDATE_INTERVAL = 25_000  # samples between progress updates
    epoch_size = len(train_dataset)  # samples per epoch

    best_val_loss = float('inf')
    stale_checkpoints = 0
    best_model_path = model_path.replace('.pth', '_best.pth')
    _early_stopped = False

    # ── Resume early-stopping state from per-model CSV ──
    csv_path = train_logger.csv_path
    if os.path.isfile(csv_path):
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and header[0] == 'total_samples':
                    # new format: find val_loss column
                    try:
                        val_col = header.index('val_loss')
                    except ValueError:
                        val_col = None
                    train_col = header.index('train_loss') if 'train_loss' in header else 2
                    vals = []
                    for row in reader:
                        if len(row) > max(val_col or 3, train_col):
                            sam = int(float(row[0]))
                            v = float(row[val_col]) if val_col is not None and row[val_col] else float(row[train_col])
                            vals.append((sam, v))
                    if vals:
                        best_idx = min(range(len(vals)), key=lambda i: vals[i][1])
                        best_val_loss = vals[best_idx][1]
                        stale_checkpoints = len(vals) - 1 - best_idx
                        print(f'  Resumed early-stopping: best_val={best_val_loss:.6f} stale={stale_checkpoints}', flush=True)
                elif len(header) >= 3:
                    # old 3/4-col format
                    vals = []
                    for row in reader:
                        if len(row) >= 3:
                            sam = int(float(row[0])) if len(row) >= 4 else 0
                            v = float(row[3]) if len(row) >= 4 else float(row[2])
                            vals.append((sam, v))
                    if vals:
                        best_idx = min(range(len(vals)), key=lambda i: vals[i][1])
                        best_val_loss = vals[best_idx][1]
                        stale_checkpoints = len(vals) - 1 - best_idx
                        print(f'  Resumed early-stopping: best_val={best_val_loss:.6f} stale={stale_checkpoints}', flush=True)
        except Exception:
            pass

    sum_train_loss = 0.0
    sum_train_count = 0

    next_update = total_samples_processed + UPDATE_INTERVAL
    next_log = total_samples_processed + LOG_INTERVAL

    sys.stderr.write("\r\033[K")
    sys.stderr.flush()

    try:
        while total_samples_processed < max_samples and not _early_stopped:
            if _interrupted:
                raise KeyboardInterrupt
            model.train()
            for x_batch, y_batch in train_loader:
                if _interrupted or total_samples_processed >= max_samples:
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

                # ── Update EMA in logger ──
                train_logger.on_batch_end(total_samples_processed + batch_size_actual,
                                          loss.item())

                sum_train_loss += loss.item() * batch_size_actual
                sum_train_count += batch_size_actual
                total_samples_processed += batch_size_actual

                # ── Progress update (stderr, in-place) ──
                if total_samples_processed >= next_update:
                    avg_loss = sum_train_loss / sum_train_count if sum_train_count > 0 else 0
                    line = train_logger.format_progress(
                        total_samples_processed, max_samples, avg_loss, epoch_size)
                    sys.stderr.write(line)
                    sys.stderr.flush()
                    next_update = total_samples_processed + UPDATE_INTERVAL

                # ── Checkpoint (CSV + stdout) ──
                if total_samples_processed >= next_log:
                    avg_train_loss = sum_train_loss / sum_train_count if sum_train_count > 0 else 0

                    if no_val:
                        avg_val_loss = avg_train_loss
                    else:
                        avg_val_loss = _validate(model, val_loader, criterion, device)
                        model.train()

                    if checkpoint_scheduler is not None:
                        checkpoint_scheduler.step(avg_train_loss if no_val else avg_val_loss)

                    if not no_val:
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            stale_checkpoints = 0
                            _save_checkpoint(model, optimizer, best_model_path)
                        else:
                            stale_checkpoints += 1

                    # Get current LR
                    cur_lr = optimizer.param_groups[0]['lr']

                    train_logger.log_checkpoint(
                        total_samples_processed, avg_train_loss, epoch_size,
                        val_loss=avg_val_loss if not no_val else None,
                        lr=cur_lr)

                    if not no_val and stale_checkpoints >= early_stop_patience:
                        print(f"  Early stop: val loss not improved for "
                              f"{early_stop_patience} checkpoints "
                              f"(best={best_val_loss:.6f})")
                        _early_stopped = True
                        break

                    sum_train_loss = 0.0
                    sum_train_count = 0
                    next_log = total_samples_processed + LOG_INTERVAL

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

    # ── Normal exit ──
    _cuda_safe_cleanup()

    if sum_train_count > 0:
        avg_train_loss = sum_train_loss / sum_train_count
        if no_val:
            avg_val_loss = avg_train_loss
        else:
            avg_val_loss = _validate(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]['lr']
        train_logger.log_checkpoint(
            total_samples_processed, avg_train_loss, epoch_size,
            val_loss=avg_val_loss if not no_val else None,
            lr=cur_lr)

    _save_checkpoint(model, optimizer, model_path)
    print(f"Training finished. Model saved to {model_path}")
    return total_samples_processed
