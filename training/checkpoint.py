"""Checkpoint save/load and resume utilities."""

import csv
import os

import torch

from utils import cuda_safe_cleanup as _cuda_safe_cleanup


def save_checkpoint(model, optimizer, model_path: str,
                    checkpoint_scheduler=None):
    """Save model weights + optimizer state atomically.

    Handles compiled models (_orig_mod unwrapping).
    Side effect: syncs CUDA before save.
    Optionally saves ReduceLROnPlateau scheduler state.
    """
    _cuda_safe_cleanup()
    unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save(unwrapped.state_dict(), model_path)
    opt_path = model_path + ".opt"
    torch.save(optimizer.state_dict(), opt_path)
    if checkpoint_scheduler is not None:
        sch_path = model_path + ".sch"
        torch.save(checkpoint_scheduler.state_dict(), sch_path)


def load_optimizer(optimizer, model_path: str, device):
    """Load optimizer state from model_path.opt. Skips if missing."""
    opt_path = model_path + ".opt"
    if os.path.isfile(opt_path):
        optimizer.load_state_dict(
            torch.load(opt_path, map_location=device, weights_only=True))


def load_plat_scheduler(checkpoint_scheduler, model_path: str):
    """Load ReduceLROnPlateau scheduler state from model_path.sch.

    Skips silently if file missing or scheduler is None.
    """
    if checkpoint_scheduler is None:
        return
    sch_path = model_path + ".sch"
    if os.path.isfile(sch_path):
        checkpoint_scheduler.load_state_dict(
            torch.load(sch_path, map_location='cpu', weights_only=True))


def resume_early_stopping_state(csv_path: str):
    """Parse per-model CSV to restore early-stopping counters on restart.

    Returns (best_val_loss, stale_checkpoints).
    Handles both new format (header: total_samples,…) and legacy 3/4-col format.
    """
    best_val_loss = float('inf')
    stale_checkpoints = 0

    if not os.path.isfile(csv_path):
        return best_val_loss, stale_checkpoints

    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return best_val_loss, stale_checkpoints

            if header[0] == 'total_samples':
                # New format — find val_loss or train_loss column
                try:
                    val_col = header.index('val_loss')
                except ValueError:
                    val_col = None
                try:
                    train_col = header.index('train_loss')
                except ValueError:
                    train_col = 2

                vals = []
                for row in reader:
                    if len(row) <= max(val_col or 3, train_col):
                        continue
                    v = (float(row[val_col]) if val_col is not None and row[val_col]
                         else float(row[train_col]))
                    vals.append(v)

                if vals:
                    best_idx = min(range(len(vals)), key=lambda i: vals[i])
                    best_val_loss = vals[best_idx]
                    stale_checkpoints = len(vals) - 1 - best_idx
            elif len(header) >= 3:
                # Legacy 3/4-col format
                vals = []
                val_col = 3 if len(header) >= 4 else 2
                for row in reader:
                    if len(row) > val_col:
                        vals.append(float(row[val_col]))
                if vals:
                    best_idx = min(range(len(vals)), key=lambda i: vals[i])
                    best_val_loss = vals[best_idx]
                    stale_checkpoints = len(vals) - 1 - best_idx
    except Exception:
        pass

    if best_val_loss < float('inf'):
        print(f'  Resumed early-stopping: best_val={best_val_loss:.6f} '
              f'stale={stale_checkpoints}', flush=True)
    return best_val_loss, stale_checkpoints
