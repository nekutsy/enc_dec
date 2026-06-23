"""LR scheduler builder — one function per scheduler type."""

import torch


# ── GreedyLR ──────────────────────────────────────────────

class GreedyLR:
    """Zeroth-order GreedyLR — bidirectional adaptive LR scheduler.

    Based on: "Zeroth Order GreedyLR" (Amazon Science).

    Algorithm:
      1. Track EMA of val_loss:  smoothed = β·smoothed + (1-β)·current
      2. Compute relative change:  δ = (smoothed_now - smoothed_prev) / smoothed_prev
      3. Adjust LR:  LR = LR · (1 - γ·δ)
         - Loss decreased (δ < 0) → LR increases (greedy: push when winning)
         - Loss increased (δ > 0) → LR decreases (retreat when losing)
      4. Clamp multiplier to [1-γ, 1+γ] — protects against outlier changes
      5. Lock for `lock_steps` checkpoints after a decrease to let LR stabilise
      6. Clamp LR to [min_lr, max_lr]

    Key property: bidirectional — raises LR on improvement, lowers on
    degradation. Unlike plateau-based schedulers that only cut LR.

    Compatible with checkpoint save/load via state_dict().
    """

    def __init__(self, optimizer, factor=0.5, beta=0.9, lock_steps=3,
                 min_lr=1e-7, max_lr=0.1):
        self.optimizer = optimizer
        self.factor = factor        # γ: controls adjustment aggressiveness
        self.beta = beta            # EMA smoothing coefficient
        self.lock_steps = lock_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self._ema = None            # exponential moving average of loss
        self._prev_ema = None       # previous checkpoint's EMA
        self._locked = 0            # remaining lock checkpoints

    def step(self, val_loss):
        # Update EMA
        if self._ema is None:
            self._ema = val_loss
        else:
            self._ema = self.beta * self._ema + (1 - self.beta) * val_loss

        # Lock phase: track EMA but don't change LR
        if self._locked > 0:
            self._locked -= 1
            if self._locked == 0:
                self._prev_ema = self._ema  # reset baseline after lock
            return

        # Not locked — compute δ and adjust
        if self._prev_ema is None:
            self._prev_ema = self._ema
            return

        if self._prev_ema == 0:
            self._prev_ema = self._ema
            return

        delta = (self._ema - self._prev_ema) / self._prev_ema
        multiplier = 1.0 - self.factor * delta

        # Clamp multiplier to prevent wild swings
        multiplier = max(1.0 - self.factor, min(1.0 + self.factor, multiplier))

        if multiplier < 1.0:
            # LR decreased — enter lock
            self._locked = self.lock_steps

        self._prev_ema = self._ema

        for pg in self.optimizer.param_groups:
            pg['lr'] = max(self.min_lr, min(self.max_lr, pg['lr'] * multiplier))

    def state_dict(self):
        return {
            '_ema': self._ema,
            '_prev_ema': self._prev_ema,
            '_locked': self._locked,
        }

    def load_state_dict(self, sd):
        self._ema = sd['_ema']
        self._prev_ema = sd['_prev_ema']
        self._locked = sd['_locked']


# ── Builder ───────────────────────────────────────────────

def build_scheduler(optimizer, train_config, total_steps: int, start_samples: int = 0,
                     pct_start: float = 0.3, plateau_patience: int = 10,
                     greedy_factor: float = 0.5, greedy_beta: float = 0.9,
                     lock_steps: int = 3):
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
                          beta=greedy_beta, lock_steps=lock_steps)
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
