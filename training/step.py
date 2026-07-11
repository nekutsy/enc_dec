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
               vae_beta: float = 1.0) -> tuple[float, float | None, float | None]:
    """Execute one training step. Returns (total_loss, recon_loss, kl_loss).

    For non-VAE models, recon_loss and kl_loss are None.
    """
    vae_mode = getattr(model, 'vae', False)

    if use_amp:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            fwd = model(x_batch)
            if vae_mode:
                out, mu, logvar = fwd
                recon_val = criterion(out, y_batch)
                kl_val = model.kl_loss(mu, logvar)
                loss = recon_val + vae_beta * kl_val
            else:
                loss = criterion(fwd, y_batch)
                recon_val = None
                kl_val = None
    else:
        fwd = model(x_batch)
        if vae_mode:
            out, mu, logvar = fwd
            recon_val = criterion(out, y_batch)
            kl_val = model.kl_loss(mu, logvar)
            loss = recon_val + vae_beta * kl_val
        else:
            loss = criterion(fwd, y_batch)
            recon_val = None
            kl_val = None

    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    if step_scheduler is not None:
        if getattr(step_scheduler, 'uses_loss', False):
            step_scheduler.step(loss.item())
        else:
            step_scheduler.step()

    return loss.item(), \
           recon_val.item() if recon_val is not None else None, \
           kl_val.item() if kl_val is not None else None
