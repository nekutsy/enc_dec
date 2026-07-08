"""LR Range Test — find optimal learning rate via exponential sweep.

Based on Leslie Smith's LR Range Test (arXiv:1506.01186):
  1. Start with tiny LR (e.g. 1e-7)
  2. For each minibatch: forward, backward, step, record (lr, loss)
  3. Multiply LR by constant factor each step
  4. Stop when loss explodes or step limit reached
  5. Suggested LR = point of steepest descent on smoothed curve

Caller is responsible for: model creation, optimizer creation, GPU cleanup.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def find_lr(
    model: torch.nn.Module,
    train_dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seq_len: int,
    criterion,
    optimizer: torch.optim.Optimizer,
    lr_start: float = 1e-7,
    lr_end: float = 10.0,
    steps: int = 200,
    stop_factor: float = 4.0,
    smooth_window: int = 5,
) -> tuple[float, list[dict]]:
    """Run LR range test.

    Args:
        model: nn.Module (already on device, initialised weights).
        train_dataset: Dataset for training samples.
        device: torch.device.
        batch_size: minibatch size.
        num_workers: DataLoader workers.
        seq_len: sequence length (unused, kept for API consistency).
        criterion: loss function.
        optimizer: torch.optim.Optimizer with starting LR already set.
        lr_start: initial learning rate.
        lr_end: final learning rate.
        steps: max number of minibatches to test.
        stop_factor: stop if loss > stop_factor * min_loss_seen.
        smooth_window: moving-average window for suggested LR calculation.

    Returns:
        (suggested_lr: float, history: list[dict])
        Each history entry: {'lr': float, 'loss': float, 'loss_smooth': float}
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    # Set starting LR on all param groups
    for pg in optimizer.param_groups:
        pg['lr'] = lr_start

    # Exponential factor: lr_end = lr_start * factor^(steps-1)
    if steps > 1:
        factor = (lr_end / lr_start) ** (1.0 / (steps - 1))
    else:
        factor = 1.0

    history: list[dict] = []
    min_loss = float('inf')
    use_amp = (device.type == 'cuda')

    iterator = iter(train_loader)
    model.train()

    for step_i in range(steps):
        # Fetch next batch; reset iterator on epoch end
        try:
            x_batch, y_batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x_batch, y_batch = next(iterator)

        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(x_batch)
                loss = criterion(out, y_batch)
        else:
            out = model(x_batch)
            loss = criterion(out, y_batch)

        loss.backward()
        optimizer.step()

        cur_lr = optimizer.param_groups[0]['lr']
        loss_val = loss.item()

        history.append({'lr': cur_lr, 'loss': loss_val})

        if loss_val < min_loss:
            min_loss = loss_val

        # Early stop if loss explodes
        if loss_val > stop_factor * min_loss and len(history) > 10:
            break

        # Increase LR exponentially
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr * factor

    # Compute smoothed losses
    raw_losses = [h['loss'] for h in history]
    smoothed = _smooth(raw_losses, smooth_window)
    for i, h in enumerate(history):
        h['loss_smooth'] = smoothed[i]

    suggested_lr = _suggest_lr(history)

    return suggested_lr, history


def _smooth(values: list[float], window: int) -> list[float]:
    """Simple moving average."""
    if len(values) < window or window <= 1:
        return list(values)
    result = []
    half = window // 2
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        result.append(sum(values[start:end]) / (end - start))
    return result


def _suggest_lr(history: list[dict]) -> float:
    """Find LR at point of steepest loss decrease (most negative gradient).

    Uses smoothed loss curve. Returns LR from the history entry just after
    the steepest gradient segment.
    """
    if len(history) < 3:
        return history[-1]['lr'] if history else 0.001

    smoothed = [h['loss_smooth'] for h in history]
    lrs = [h['lr'] for h in history]

    # Gradient of loss w.r.t. log10(LR) — positive when loss decreases
    gradients = []
    for i in range(1, len(smoothed)):
        dl = smoothed[i - 1] - smoothed[i]  # loss decrease
        dlog = np.log10(lrs[i]) - np.log10(lrs[i - 1])
        grad = dl / dlog if dlog > 0 else 0.0
        gradients.append(grad)

    if not gradients:
        return lrs[-1]

    best_idx = int(np.argmax(gradients))
    # Gradient is between points, pick the later one (where the better LR is)
    actual_idx = min(best_idx + 1, len(lrs) - 1)
    return lrs[actual_idx]


def plot_lr_finder(
    history: list[dict],
    suggested_lr: float,
    title: str = 'LR Range Test',
    save_path: str | None = None,
) -> None:
    """Plot loss vs learning rate (log scale) with suggested LR marker.

    Args:
        history: list of {'lr', 'loss', 'loss_smooth'} dicts.
        suggested_lr: the suggested learning rate to mark.
        title: plot title.
        save_path: if set, save figure to this path and close.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lrs = [h['lr'] for h in history]
    losses = [h['loss'] for h in history]
    smoothed = [h['loss_smooth'] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lrs, losses, alpha=0.3, color='blue', linewidth=0.8, label='loss')
    ax.plot(lrs, smoothed, color='blue', linewidth=1.5, label='smoothed loss')
    ax.axvline(suggested_lr, color='red', linestyle='--', linewidth=1.5,
               label=f'suggested lr = {suggested_lr:.2e}')
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
