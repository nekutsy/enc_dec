"""Single training step: forward + backward + AMP + grad clip + optimizer step.

Caller is responsible for: .to(device), zero_grad(), and interruption checks.
"""

from __future__ import annotations

import torch
from torch import Tensor

from core.types import ModelLike, OptimizerLike, StepSchedulerLike


def step_batch(model: ModelLike, x_batch: Tensor, y_batch: Tensor,
               criterion, optimizer: OptimizerLike,
               use_amp: bool = False, grad_clip: float = 1.0,
               step_scheduler: StepSchedulerLike | None = None,
               vae_beta: float = 1.0) -> float:
    """Execute one training step. Returns scalar loss value.

    Args:
        model, x_batch, y_batch: model and inputs (already on device).
        criterion: loss function (BCEWithLogitsLoss).
        optimizer: PyTorch optimizer.
        use_amp: enable bfloat16 autocast (no scaler — bfloat16 doesn't need it).
        grad_clip: max grad norm; <= 0 to disable.
        step_scheduler: optional per-step LR scheduler.
        vae_beta: β weight for KL term (only used when model.vae=True).

    Returns loss.item() — the total loss scalar (recon + β·KL).
    """
    vae_mode = getattr(model, 'vae', False)

    if use_amp:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            fwd = model(x_batch)
            if vae_mode:
                out, mu, logvar = fwd
                recon_loss = criterion(out, y_batch)
                kl = model.kl_loss(mu, logvar)
                loss = recon_loss + vae_beta * kl
            else:
                loss = criterion(fwd, y_batch)
    else:
        fwd = model(x_batch)
        if vae_mode:
            out, mu, logvar = fwd
            recon_loss = criterion(out, y_batch)
            kl = model.kl_loss(mu, logvar)
            loss = recon_loss + vae_beta * kl
        else:
            loss = criterion(fwd, y_batch)

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
