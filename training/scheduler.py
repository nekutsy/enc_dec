"""LR scheduler builder — one function per scheduler type."""

import torch


# ── GreedyLR ──────────────────────────────────────────────

class GreedyLR:
    """Greedy per-checkpoint LR scheduler.

    Algorithm (fundamentally different from plateau):
      1. At each checkpoint, evaluate val_loss against best_loss.
      2. If improved → keep LR, update best, reset bad counter.
      3. If NOT improved → immediately halve LR (greedy — act on first
         sign of stalling, not after N confirmations).
      4. Enter "lock" phase: don't reduce again for `lock_steps`
         checkpoints, giving the new LR time to show results.
      5. After lock expires:
         - If loss improved over the pre-reduction best → LR was a good
           greedy move, continue normally.
         - If loss did NOT improve → reduction didn't help, halve again
           and enter a new lock.

    Key difference from ReduceLROnPlateau:
      - patience=0 by design — acts on the first bad checkpoint
      - lock_steps prevents panic-chaining reductions
      - the question is "did the reduction help?" not "are we stuck?"

    Compatible with checkpoint save/load via state_dict().
    """

    def __init__(self, optimizer, factor=0.5, lock_steps=3, min_lr=1e-7):
        self.optimizer = optimizer
        self.factor = factor
        self.lock_steps = lock_steps
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.bad_counter = 0
        self._locked = 0           # remaining lock checkpoints (>0 → locked)
        self._lock_best = float('inf')  # best_loss at moment of reduction

    def step(self, val_loss):
        if self._locked > 0:
            # Lock phase: collect evidence, don't reduce yet.
            self._locked -= 1
            if val_loss < self.best_loss:
                self.best_loss = val_loss
            if self._locked == 0:
                # Lock expired — check if the reduction paid off.
                if self.best_loss >= self._lock_best:
                    # Did not beat pre-reduction best → reduce again.
                    self._reduce_lr()
                # else: reduction helped, stay at current LR, exit lock silently.
            return

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.bad_counter = 0
        else:
            self.bad_counter += 1
            # Act immediately on first bad checkpoint.
            self._reduce_lr()

    def _reduce_lr(self):
        self._lock_best = self.best_loss
        self._locked = self.lock_steps
        self.bad_counter = 0
        for pg in self.optimizer.param_groups:
            pg['lr'] = max(pg['lr'] * self.factor, self.min_lr)

    def state_dict(self):
        return {
            'best_loss': self.best_loss,
            'bad_counter': self.bad_counter,
            '_locked': self._locked,
            '_lock_best': self._lock_best,
        }

    def load_state_dict(self, sd):
        self.best_loss = sd['best_loss']
        self.bad_counter = sd['bad_counter']
        self._locked = sd['_locked']
        self._lock_best = sd['_lock_best']


# ── Builder ───────────────────────────────────────────────

def build_scheduler(optimizer, train_config, total_steps: int, start_samples: int = 0,
                     pct_start: float = 0.3, plateau_patience: int = 10,
                     greedy_factor: float = 0.5, lock_steps: int = 3):
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
                          lock_steps=lock_steps)
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
