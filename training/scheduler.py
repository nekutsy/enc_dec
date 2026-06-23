"""LR scheduler builder — one function per scheduler type."""

import torch


# ── GreedyLR ──────────────────────────────────────────────

class GreedyLR:
    """Zeroth-order GreedyLR — bidirectional adaptive LR scheduler with probing.

    Based on: "Zeroth Order GreedyLR" (Amazon Science).

    Algorithm:
      1. Track EMA of val_loss:  smoothed = β·smoothed + (1-β)·current
      2. Compute relative change:  δ = (smoothed_now - smoothed_prev) / smoothed_prev
      3. Adjust LR:  LR = LR · (1 - γ·δ)
         - Loss decreased (δ < 0) → LR increases (push when winning)
         - Loss increased (δ > 0) → LR decreases (retreat when losing)
      4. Clamp multiplier to [1-γ, 1+γ] — protects against outlier changes
      5. Lock for `lock_steps` checkpoints after a decrease to let LR stabilise
      6. Probe mode: when δ ≈ 0 for `probe_patience` checkpoints, temporarily
         cut LR by `probe_factor` to test if a lower LR helps. After probe
         lock expires:
         - Improved over pre-probe best → keep lower LR
         - Did not improve → restore old LR + enter cooldown
      7. Clamp LR to [min_lr, max_lr]

    Key property: bidirectional with exploration — probes plateaus instead
    of freezing.

    Compatible with checkpoint save/load via state_dict().
    """

    def __init__(self, optimizer, factor=0.5, beta=0.9, lock_steps=3,
                 probe_patience=3, probe_factor=0.5, probe_lock_steps=3,
                 probe_threshold=0.02,
                 cooldown_steps=3,
                 min_lr=1e-7, max_lr=0.1):
        self.optimizer = optimizer
        self.factor = factor            # γ: controls adjustment aggressiveness
        self.beta = beta                # EMA smoothing coefficient
        self.lock_steps = lock_steps    # lock after LR decrease
        self.probe_patience = probe_patience    # flat δ count before probing
        self.probe_factor = probe_factor        # LR multiply on probe
        self.probe_lock_steps = probe_lock_steps  # probe observation window
        self.probe_threshold = probe_threshold    # |δ| threshold for "flat"
        self.cooldown_steps = cooldown_steps      # cooldown after failed probe
        self.min_lr = min_lr
        self.max_lr = max_lr
        self._ema = None                # exponential moving average of loss
        self._prev_ema = None           # previous checkpoint's EMA
        self._locked = 0                # remaining lock checkpoints
        self._flat_count = 0            # consecutive checkpoints with tiny δ
        self._phase = 'normal'          # normal | probing | cooldown
        self._phase_steps = 0           # remaining steps in current phase
        self._probe_backup = None       # (lr, ema, prev_ema, best_probe_loss)

    def step(self, val_loss):
        # Update EMA
        if self._ema is None:
            self._ema = val_loss
            self._prev_ema = self._ema
            return
        prev_ema = self._ema
        self._ema = self.beta * prev_ema + (1 - self.beta) * val_loss

        # Lock check (from normal-mode LR decrease)
        if self._locked > 0:
            self._locked -= 1
            self._prev_ema = self._ema
            return

        # ── Phase machine ──
        if self._phase == 'cooldown':
            self._phase_steps -= 1
            self._prev_ema = self._ema
            if self._phase_steps <= 0:
                self._phase = 'normal'
                self._flat_count = 0
            return

        if self._phase == 'probing':
            self._phase_steps -= 1
            old_lr, old_ema, old_prev_ema, best_probe = self._probe_backup
            if val_loss < best_probe:
                self._probe_backup = (old_lr, old_ema, old_prev_ema, val_loss)
            if self._phase_steps <= 0:
                self._evaluate_probe()
            return

        # ── Normal mode: compute δ ──
        if self._prev_ema is None or self._prev_ema == 0:
            self._prev_ema = self._ema
            return

        delta = (self._ema - self._prev_ema) / abs(self._prev_ema)

        # Detect plateau: |δ| < probe_threshold
        if abs(delta) < self.probe_threshold:
            self._flat_count += 1
            if self._flat_count >= self.probe_patience:
                self._start_probe()
                return
        else:
            self._flat_count = 0

        # Apply δ-based adjustment
        multiplier = 1.0 - self.factor * delta
        multiplier = max(1.0 - self.factor, min(1.0 + self.factor, multiplier))

        if multiplier < 1.0:
            self._locked = self.lock_steps

        self._prev_ema = self._ema

        for pg in self.optimizer.param_groups:
            pg['lr'] = max(self.min_lr, min(self.max_lr, pg['lr'] * multiplier))

    def _start_probe(self):
        lr = self.optimizer.param_groups[0]['lr']
        self._probe_backup = (lr, self._ema, self._prev_ema, float('inf'))
        self._phase = 'probing'
        self._phase_steps = self.probe_lock_steps
        self._flat_count = 0
        probe_lr = max(self.min_lr, lr * self.probe_factor)
        for pg in self.optimizer.param_groups:
            pg['lr'] = probe_lr

    def is_exploring(self):
        """True during probe or cooldown — early-stop should not count these."""
        return self._phase != 'normal'

    def _evaluate_probe(self):
        old_lr, old_ema, old_prev_ema, best_probe = self._probe_backup
        self._probe_backup = None
        # Compare best probe val against pre-probe EMA — robust to noise
        if best_probe < float('inf') and best_probe < old_ema:
            # Probe helped — keep lower LR, reset from improved state
            self._ema = old_ema  # keep continuity with pre-probe EMA
            self._prev_ema = self._ema
            self._phase = 'normal'
            self._flat_count = 0
        else:
            # Probe didn't help — restore LR, enter cooldown
            for pg in self.optimizer.param_groups:
                pg['lr'] = old_lr
            self._ema = old_ema
            self._prev_ema = old_prev_ema
            self._phase = 'cooldown'
            self._phase_steps = min(self.cooldown_steps, 3)

    def state_dict(self):
        return {
            '_ema': self._ema,
            '_prev_ema': self._prev_ema,
            '_locked': self._locked,
            '_flat_count': self._flat_count,
            '_phase': self._phase,
            '_phase_steps': self._phase_steps,
            '_probe_backup': self._probe_backup,
        }

    def load_state_dict(self, sd):
        self._ema = sd['_ema']
        self._prev_ema = sd['_prev_ema']
        self._locked = sd['_locked']
        self._flat_count = sd.get('_flat_count', 0)
        self._phase = sd.get('_phase', 'normal')
        self._phase_steps = sd.get('_phase_steps', 0)
        self._probe_backup = sd.get('_probe_backup', None)


# ── Builder ───────────────────────────────────────────────

def build_scheduler(optimizer, train_config, total_steps: int, start_samples: int = 0,
                     pct_start: float = 0.3, plateau_patience: int = 10,
                     greedy_factor: float = 0.5, greedy_beta: float = 0.9,
                     lock_steps: int = 3,
                     probe_patience: int = 3, probe_factor: float = 0.5,
                     probe_threshold: float = 0.02,
                     probe_lock_steps: int = 3, cooldown_steps: int = 3):
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
                          beta=greedy_beta, lock_steps=lock_steps,
                          probe_patience=probe_patience, probe_factor=probe_factor,
                          probe_threshold=probe_threshold,
                          probe_lock_steps=probe_lock_steps, cooldown_steps=cooldown_steps)
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
