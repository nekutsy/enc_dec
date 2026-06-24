"""Single training step: forward + backward + AMP + grad clip + optimizer step.

Caller is responsible for: .to(device), zero_grad(), and interruption checks.
"""

import torch


def step_batch(model, x_batch, y_batch, criterion, optimizer,
               scaler=None, grad_clip: float = 1.0,
               step_scheduler=None) -> float:
    """Execute one training step. Returns loss value (float).

    Args:
        model, x_batch, y_batch: model and inputs (already on device).
        criterion: loss function.
        optimizer: PyTorch optimizer.
        scaler: torch.amp.GradScaler or None (CPU training).
        grad_clip: max grad norm; <= 0 to disable.
        step_scheduler: optional per-step LR scheduler.

    Returns loss.item() — the loss scalar.
    """
    if scaler is not None:
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
        if getattr(step_scheduler, 'uses_loss', False):
            step_scheduler.step(loss.item())
        else:
            step_scheduler.step()

    return loss.item()
