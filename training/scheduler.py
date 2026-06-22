"""LR scheduler builder — one function per scheduler type."""

import torch


def build_scheduler(optimizer, train_config, total_steps: int, start_samples: int = 0):
    """Build (per_step_scheduler, per_checkpoint_scheduler) from TrainConfig.

    per_step_scheduler: called every batch (warmup, cosine, onecycle).
    per_checkpoint_scheduler: called at validation checkpoints (plateau).
    Returns (None, None) when scheduler == 'none'.

    If start_samples > 0 (resume): warmup phase is skipped.
    """
    scheduler = train_config.scheduler
    if scheduler == 'none':
        return None, None

    total_steps = max(total_steps, 1)
    warmup_steps = int(train_config.warmup_fraction * total_steps)

    if start_samples > 0:
        warmup_steps = 0  # skip warmup on resume

    if scheduler == "cosine":
        return _build_cosine(optimizer, warmup_steps, total_steps)

    if scheduler == "plateau":
        return _build_plateau(optimizer, warmup_steps)

    if scheduler == "onecycle":
        return _build_onecycle(optimizer, train_config.lr, total_steps), None

    return None, None


def _build_cosine(optimizer, warmup_steps, total_steps):
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


def _build_plateau(optimizer, warmup_steps):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_steps
    ) if warmup_steps > 0 else None
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5,
        min_lr=1e-6,
    )
    return warmup, plateau


def _build_onecycle(optimizer, max_lr, total_steps):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr,
        total_steps=total_steps, pct_start=0.3,
        anneal_strategy='cos', div_factor=25.0,
        final_div_factor=10000.0,
    )
