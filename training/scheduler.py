"""LR scheduler builder — one function per scheduler type."""

import torch


# ── GreedyLR ──────────────────────────────────────────────

class GreedyLR:
    """Per-checkpoint LR scheduler: multiply LR by `factor` when val loss
    does not improve for `patience` consecutive checkpoints.

    More aggressive than ReduceLROnPlateau — no cooldown, no threshold.
    Compatible with checkpoint save/load via state_dict().
    """

    def __init__(self, optimizer, factor=0.5, patience=5, min_lr=1e-7):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.num_bad = 0
        self._last_lr = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.num_bad = 0
            self._last_lr = [pg['lr'] for pg in self.optimizer.param_groups]
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = max(pg['lr'] * self.factor, self.min_lr)
                self.num_bad = 0

    def state_dict(self):
        return {'best_loss': self.best_loss, 'num_bad': self.num_bad,
                '_last_lr': self._last_lr}

    def load_state_dict(self, sd):
        self.best_loss = sd['best_loss']
        self.num_bad = sd['num_bad']
        self._last_lr = sd['_last_lr']


# ── Builder ───────────────────────────────────────────────

def build_scheduler(optimizer, train_config, total_steps: int, start_samples: int = 0,
                     pct_start: float = 0.3, plateau_patience: int = 10,
                     greedy_factor: float = 0.5, greedy_patience: int = 5):
    """Build (per_step_scheduler, per_checkpoint_scheduler) from TrainConfig.

    per_step_scheduler: called every batch (warmup, cosine, onecycle).
    per_checkpoint_scheduler: called at validation checkpoints (plateau, greedy).
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
        return _build_plateau(optimizer, warmup_steps, plateau_patience)

    if scheduler == "onecycle":
        return _build_onecycle(optimizer, train_config.lr, total_steps, pct_start), None

    if scheduler == "greedy":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        ) if warmup_steps > 0 else None
        greedy = GreedyLR(optimizer, factor=greedy_factor,
                          patience=greedy_patience)
        return warmup, greedy

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


def _build_plateau(optimizer, warmup_steps, patience=10):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_steps
    ) if warmup_steps > 0 else None
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=patience,
        min_lr=1e-6,
    )
    return warmup, plateau


def _build_onecycle(optimizer, max_lr, total_steps, pct_start=0.3):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr,
        total_steps=total_steps, pct_start=pct_start,
        anneal_strategy='cos', div_factor=25.0,
        final_div_factor=10000.0,
    )
