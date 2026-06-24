"""Main training loop — validation, checkpointing, progress."""

import os
import sys
import signal

import torch
from torch.utils.data import DataLoader

from training.step import step_batch
from training.checkpoint import save_checkpoint, resume_early_stopping_state
from utils import cuda_safe_cleanup as _cuda_safe_cleanup


# ── Validation ──────────────────────────────────────────────

def _validate(model, val_loader, criterion, device):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            out = model(x_batch)
            loss = criterion(out, y_batch)
            n = x_batch.size(0)
            total_loss += loss.item() * n
            total_samples += n
    return total_loss / total_samples if total_samples > 0 else 0.0


# ── Signal handler (context manager) ────────────────────────

class Interruptible:
    """Context manager that sets _interrupted on SIGINT/SIGTERM.

    No CUDA calls in the handler — safe for GPU training.
    Usage:
        with Interruptible() as flag:
            if flag.interrupted: ...
    """

    def __init__(self):
        self.interrupted = False

    def _on_signal(self, signum, frame):
        self.interrupted = True

    def __enter__(self):
        self._prev_sigint = signal.signal(signal.SIGINT, self._on_signal)
        self._prev_sigterm = signal.signal(signal.SIGTERM, self._on_signal)
        return self

    def __exit__(self, *args):
        signal.signal(signal.SIGINT, self._prev_sigint)
        signal.signal(signal.SIGTERM, self._prev_sigterm)


# ── Training loop ────────────────────────────────────────────

def run_training(start_samples: int, max_samples: int, model, optimizer, criterion,
                 train_dataset, val_dataset, train_logger, model_path, batch_size,
                 seq_len, grad_clip=1.0, num_workers=0,
                 step_scheduler=None, checkpoint_scheduler=None,
                 early_stop_patience=3, no_val=False,
                 val_interval: int | None = None):
    """Main training loop.

    Args:
        train_logger: TrainingLogger instance.
        no_val: if True, skip validation for logging (faster CSV).
            Plateau scheduler and early-stopping still receive val loss
            (computed at the same checkpoints, just not persisted to CSV).
        val_interval: samples between validation passes. Defaults to
            250k when no_val, 500k otherwise.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    device = next(model.parameters()).device

    # ── DataLoaders ──
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    LOG_INTERVAL = val_interval or (100_000 if no_val else 500_000)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    total_samples = start_samples
    UPDATE_INTERVAL = 25_000
    epoch_size = len(train_dataset)

    best_val_loss = float('inf')
    stale_checkpoints = 0
    best_model_path = model_path.replace('.pth', '_best.pth')
    _early_stopped = False

    # ── Resume early-stopping counters ──
    best_val_loss, stale_checkpoints = resume_early_stopping_state(train_logger.csv_path)
    # Cap stale on resume — model behaviour shifts after restart
    if stale_checkpoints > 3:
        print(f'  Capping stale from {stale_checkpoints} to 3 (post-resume reset)', flush=True)
        stale_checkpoints = 3

    sum_train_loss = 0.0
    sum_train_count = 0

    next_update = total_samples + UPDATE_INTERVAL
    next_log = total_samples + LOG_INTERVAL

    sys.stderr.write("\r\033[K")
    sys.stderr.flush()

    try:
        with Interruptible() as interrupt:
            while total_samples < max_samples and not _early_stopped:
                model.train()
                for x_batch, y_batch in train_loader:
                    if interrupt.interrupted or total_samples >= max_samples:
                        break

                    x_batch = x_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    bs = x_batch.size(0)

                    optimizer.zero_grad(set_to_none=True)
                    loss_val = step_batch(
                        model, x_batch, y_batch, criterion, optimizer,
                        scaler=scaler, grad_clip=grad_clip,
                        step_scheduler=step_scheduler,
                    )

                    train_logger.on_batch_end(total_samples + bs, loss_val)
                    sum_train_loss += loss_val * bs
                    sum_train_count += bs
                    total_samples += bs

                    # ── Progress (stderr, in-place) ──
                    if total_samples >= next_update:
                        avg_loss = sum_train_loss / sum_train_count if sum_train_count > 0 else 0
                        cur_lr = optimizer.param_groups[0]['lr']
                        debug = None
                        if step_scheduler is not None and hasattr(step_scheduler, 'get_debug_info'):
                            debug = step_scheduler.get_debug_info()
                        line = train_logger.format_progress(
                            total_samples, max_samples, avg_loss, epoch_size,
                            lr=cur_lr, debug=debug)
                        sys.stderr.write(line)
                        sys.stderr.flush()
                        next_update = total_samples + UPDATE_INTERVAL

                    # ── Checkpoint ──
                    if total_samples >= next_log:
                        avg_train_loss = sum_train_loss / sum_train_count if sum_train_count > 0 else 0
                        avg_val_loss = _validate(model, val_loader, criterion, device)
                        model.train()

                        if checkpoint_scheduler is not None:
                            checkpoint_scheduler.step(avg_val_loss)

                        # Early-stopping + best model (always on val loss)
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            stale_checkpoints = 0
                            save_checkpoint(model, optimizer, best_model_path,
                                            checkpoint_scheduler=checkpoint_scheduler,
                                            step_scheduler=step_scheduler)
                        elif checkpoint_scheduler is not None and hasattr(checkpoint_scheduler, 'is_exploring') and checkpoint_scheduler.is_exploring():
                            # Don't count probe/cooldown checkpoints as stale
                            pass
                        else:
                            stale_checkpoints += 1

                        cur_lr = optimizer.param_groups[0]['lr']
                        debug = None
                        if step_scheduler is not None and hasattr(step_scheduler, 'get_debug_info'):
                            debug = step_scheduler.get_debug_info()
                        train_logger.log_checkpoint(
                            total_samples, avg_train_loss, epoch_size,
                            val_loss=None if no_val else avg_val_loss,
                            lr=cur_lr, debug=debug)

                        # early-stop disabled
                        # if stale_checkpoints >= early_stop_patience:
                        #     print(f"  Early stop: val loss not improved for "
                        #           f"{early_stop_patience} checkpoints "
                        #           f"(best={best_val_loss:.6f})")
                        #     _early_stopped = True
                        #     break

                        sum_train_loss = 0.0
                        sum_train_count = 0
                        next_log = total_samples + LOG_INTERVAL

                if total_samples >= max_samples:
                    break
    except KeyboardInterrupt:
        print("\nTraining interrupted. Cleaning up GPU...", file=sys.stderr, flush=True)
        _cuda_safe_cleanup()
        print("Saving checkpoint...", file=sys.stderr, flush=True)
        save_checkpoint(model, optimizer, model_path,
                        checkpoint_scheduler=checkpoint_scheduler,
                        step_scheduler=step_scheduler)
        raise

    # ── Normal exit ──
    _cuda_safe_cleanup()

    if sum_train_count > 0:
        avg_train_loss = sum_train_loss / sum_train_count
        avg_val_loss = _validate(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]['lr']
        debug = None
        if step_scheduler is not None and hasattr(step_scheduler, 'get_debug_info'):
            debug = step_scheduler.get_debug_info()
        train_logger.log_checkpoint(
            total_samples, avg_train_loss, epoch_size,
            val_loss=None if no_val else avg_val_loss,
            lr=cur_lr, debug=debug)

    save_checkpoint(model, optimizer, model_path,
                    checkpoint_scheduler=checkpoint_scheduler,
                    step_scheduler=step_scheduler)
    print(f"Training finished. Model saved to {model_path}")
    return total_samples
