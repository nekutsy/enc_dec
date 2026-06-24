"""LR scheduler builder — one function per scheduler type."""

from __future__ import annotations

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
      6. Probe mode: plateau detection via running-minimum staleness
         (not δ-EMA — immune to single-batch val spikes). When best val
         hasn't improved for `probe_patience` checkpoints, temporarily
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
                 probe_patience=4, probe_factor=0.5, probe_lock_steps=3,
                 probe_spike_ratio=2.5,
                 cooldown_steps=3,
                 min_lr=1e-7, max_lr=0.1):
        self.optimizer = optimizer
        self.factor = factor            # γ: controls adjustment aggressiveness
        self.beta = beta                # EMA smoothing coefficient
        self.lock_steps = lock_steps    # lock after LR decrease
        self.probe_patience = probe_patience    # stale chkpts before probing
        self.probe_factor = probe_factor        # LR multiply on probe
        self.probe_lock_steps = probe_lock_steps  # probe observation window
        self.probe_spike_ratio = probe_spike_ratio  # skip probe if val spike > this × best
        self.cooldown_steps = cooldown_steps      # cooldown after failed probe
        self.min_lr = min_lr
        self.max_lr = max_lr
        self._ema = None                # exponential moving average of loss
        self._prev_ema = None           # previous checkpoint's EMA
        self._locked = 0                # remaining lock checkpoints
        self._running_best = float('inf')  # best val in current plateau window
        self._stale_count = 0           # chkpts since running_best was set
        self._phase = 'normal'          # normal | probing | cooldown
        self._phase_steps = 0           # remaining steps in current phase
        self._probe_backup = None       # dict: {lr, ema, prev_ema, running_best, best_probe}

    def step(self, val_loss):
        # Update EMA
        if self._ema is None:
            self._ema = val_loss
            self._prev_ema = self._ema
            self._running_best = val_loss
            self._stale_count = 0
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
                self._running_best = val_loss
                self._stale_count = 0
            return

        if self._phase == 'probing':
            self._phase_steps -= 1
            bp = self._probe_backup
            if val_loss < bp['best_probe']:
                bp['best_probe'] = val_loss
            if self._phase_steps <= 0:
                self._evaluate_probe()
            return

        # ── Normal mode ──
        if self._prev_ema is None or self._prev_ema == 0:
            self._prev_ema = self._ema
            return

        # Update running best
        if val_loss < self._running_best:
            self._running_best = val_loss
            self._stale_count = 0
        else:
            self._stale_count += 1

        # Detect plateau: running best hasn't improved for probe_patience chkpts
        if self._stale_count >= self.probe_patience:
            # Suppress probe if last chkpt is a spike (single-batch noise)
            if val_loss > self._running_best * self.probe_spike_ratio:
                # Skip probe — this is a spike, not a plateau
                self._stale_count -= 1  # don't lose the count
                self._prev_ema = self._ema
                return
            self._start_probe()
            return

        # Apply δ-based adjustment (clipped to damp single-batch spikes)
        delta = (self._ema - self._prev_ema) / abs(self._prev_ema)
        delta = max(-0.1, min(0.1, delta))  # single spikes can't slam LR

        multiplier = 1.0 - self.factor * delta
        multiplier = max(1.0 - self.factor, min(1.0 + self.factor, multiplier))

        if multiplier < 1.0:
            self._locked = self.lock_steps

        self._prev_ema = self._ema

        for pg in self.optimizer.param_groups:
            pg['lr'] = max(self.min_lr, min(self.max_lr, pg['lr'] * multiplier))

    def _start_probe(self):
        lr = self.optimizer.param_groups[0]['lr']
        self._probe_backup = {
            'lr': lr,
            'ema': self._ema,
            'prev_ema': self._prev_ema,
            'running_best': self._running_best,
            'best_probe': float('inf'),
        }
        self._phase = 'probing'
        self._phase_steps = self.probe_lock_steps
        self._running_best = float('inf')
        self._stale_count = 0
        probe_lr = max(self.min_lr, lr * self.probe_factor)
        for pg in self.optimizer.param_groups:
            pg['lr'] = probe_lr

    def is_exploring(self):
        """True during probe or cooldown — early-stop should not count these."""
        return self._phase != 'normal'

    def _evaluate_probe(self):
        bp = self._probe_backup
        old_lr = bp['lr']
        old_ema = bp['ema']
        old_prev_ema = bp['prev_ema']
        old_best = bp['running_best']
        best_probe = bp['best_probe']
        self._probe_backup = None
        if best_probe < float('inf') and best_probe < old_best:
            # Probe helped — keep lower LR, reset from improved state
            self._ema = old_ema
            self._prev_ema = self._ema
            self._phase = 'normal'
            self._running_best = best_probe
            self._stale_count = 0
        else:
            # Probe didn't help — restore LR, enter cooldown
            for pg in self.optimizer.param_groups:
                pg['lr'] = old_lr
            self._ema = old_ema
            self._prev_ema = old_prev_ema
            self._phase = 'cooldown'
            self._phase_steps = min(self.cooldown_steps, 3)
            self._running_best = old_best
            self._stale_count = self.probe_patience

    def state_dict(self):
        return {
            '_ema': self._ema,
            '_prev_ema': self._prev_ema,
            '_locked': self._locked,
            '_running_best': self._running_best,
            '_stale_count': self._stale_count,
            '_phase': self._phase,
            '_phase_steps': self._phase_steps,
            '_probe_backup': self._probe_backup,
        }

    def load_state_dict(self, sd):
        self._ema = sd['_ema']
        self._prev_ema = sd['_prev_ema']
        self._locked = sd['_locked']
        self._running_best = sd.get('_running_best', float('inf'))
        self._stale_count = sd.get('_stale_count', 0)
        self._phase = sd.get('_phase', 'normal')
        self._phase_steps = sd.get('_phase_steps', 0)
        bp = sd.get('_probe_backup', None)
        if bp is not None and isinstance(bp, (list, tuple)):
            # Backward compat: old tuple format → dict
            self._probe_backup = {
                'lr': bp[0], 'ema': bp[1], 'prev_ema': bp[2],
                'running_best': bp[3], 'best_probe': bp[4],
            }
        else:
            self._probe_backup = bp


# ── GreedyDiffLR (second-order) ──────────────────────────

class GreedyDiffLR:
    """Second-order GreedyLR — maximises convergence rate via D'(loss).

    Instead of driving delta-loss to zero (original GreedyLR), this tracks
    the second derivative to find and ride extrema of relative change.

    Algorithm:
      1. Group batches into packets of size `packet_size`, average loss.
      2. D(p) = (L_p - L_{p-d}) / L_p          — relative change (d in packets)
      3. D'(p) = (D_p - D_{p-d}) / d            — derivative (convergence accel)
      4. l'(p) = (l_p - l_{p-d}) / d            — LR velocity
      5. l(p) = l(p-d) + k * (l'(p-d) + K * l(p-d)) * D'(p-d)
         K < 0, |K| ≪ 1 — prevents l' from decaying to zero
      6. Clamp to [min_lr, max_lr]

    Zeroes of D' are extrema of D (max/min convergence rate). The update
    pushes LR toward regions of maximum |D|.

    Requires warmup (≥ 3*d packets) before first LR adjustment.
    Operates on train loss — not suitable when overfitting is a risk.

    Compatible with checkpoint save/load via state_dict().
    """

    uses_loss = True  # signals step_batch to feed loss value

    def __init__(self, optimizer, d=1, packet_size=100, k=1.0,
                 K=-0.01,
                 min_lr=1e-7, max_lr=0.1,
                 warmup_steps=0, warmup_start_factor=0.1):
        self.optimizer = optimizer
        self.d = max(1, d)                           # measurement spacing (packets)
        self.packet_size = max(1, packet_size)        # batches per packet
        self.k = k                                     # damping coefficient
        self.K = K                                     # anti-stagnation (negative, small)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_factor = warmup_start_factor
        self._base_lr = optimizer.param_groups[0]['lr']

        self._losses = []            # avg loss per packet, indexed by packet idx
        self._D_values = {}          # packet_idx -> D value
        self._Dprime_values = {}     # packet_idx -> D' value
        self._lr_history = []        # lr at each packet boundary [lr_0, lr_1, ...]
        self._loss_sum = 0.0
        self._loss_count = 0
        self._p = 0                  # current packet index
        self._step = 0               # batch count (for warmup tracking)

        if warmup_steps > 0:
            for pg in self.optimizer.param_groups:
                pg['lr'] = self._base_lr * warmup_start_factor

    def step(self, loss):
        """Feed a single batch loss; adjusts LR at packet boundaries."""
        self._step += 1
        self._loss_sum += loss
        self._loss_count += 1
        if self._loss_count < self.packet_size:
            return

        # ── Packet complete ──
        avg = self._loss_sum / self._loss_count
        self._losses.append(avg)
        p = self._p
        self._p += 1
        self._loss_sum = 0.0
        self._loss_count = 0

        d = self.d

        # Record current LR at this packet boundary
        self._lr_history.append(self.optimizer.param_groups[0]['lr'])

        # ── D(p) = (L_p - L_{p-d}) / L_p ──
        if p >= d:
            L_p = self._losses[p]
            L_pd = self._losses[p - d]
            denom = abs(L_p) + 1e-12
            self._D_values[p] = (L_p - L_pd) / denom

        # ── D'(p-d) = (D_{p-d} - D_{p-2d}) / d ──
        if p >= 2 * d:
            D_pd = self._D_values.get(p - d)
            D_p2d = self._D_values.get(p - 2 * d)
            if D_pd is not None and D_p2d is not None:
                self._Dprime_values[p - d] = (D_pd - D_p2d) / d

        # ── Warmup: linear ramp ──
        if self._step < self.warmup_steps:
            progress = self._step / max(1, self.warmup_steps)
            lr = self._base_lr * (self.warmup_start_factor
                                  + (1.0 - self.warmup_start_factor) * progress)
            self._lr_history[-1] = lr  # correct recorded LR
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            return

        # ── Apply: l(p) = l(p-d) + k * (l'(p-d) + K * l(p-d)) * D'(p-d) ──
        if p >= 3 * d:
            Dprime = self._Dprime_values.get(p - d)
            if Dprime is not None:
                lr_pd = self._lr_history[p - d]
                lr_p2d = self._lr_history[p - 2 * d]
                lr_velocity = (lr_pd - lr_p2d) / d
                effective = lr_velocity + self.K * lr_pd
                delta = self.k * effective * Dprime
                new_lr = lr_pd + delta
                new_lr = max(self.min_lr, min(self.max_lr, new_lr))
                self._last_delta = delta
                for pg in self.optimizer.param_groups:
                    pg['lr'] = new_lr

    # ── Debug info ────────────────────────────────────────

    def get_debug_info(self) -> dict | None:
        """Return latest D, D', lr_delta for progress logging."""
        p = self._p
        d = self.d
        D_val = self._D_values.get(p - 1) if p > 0 else None
        Dprime_val = None
        if p > 2 * d:
            Dprime_val = self._Dprime_values.get(p - 1 - d)
        if D_val is None and not hasattr(self, '_last_delta'):
            return None
        return {
            'D': D_val,
            'Dprime': Dprime_val,
            'lr_delta': getattr(self, '_last_delta', 0.0),
            'lr': self.optimizer.param_groups[0]['lr'],
        }

    # ── Checkpointing ─────────────────────────────────────

    def state_dict(self):
        return {
            '_losses': self._losses,
            '_D_values': self._D_values,
            '_Dprime_values': self._Dprime_values,
            '_lr_history': self._lr_history,
            '_loss_sum': self._loss_sum,
            '_loss_count': self._loss_count,
            '_p': self._p,
            '_step': self._step,
            '_base_lr': self._base_lr,
            '_last_delta': getattr(self, '_last_delta', 0.0),
        }

    def load_state_dict(self, sd):
        self._losses = sd['_losses']
        self._D_values = sd['_D_values']
        self._Dprime_values = sd['_Dprime_values']
        self._lr_history = sd.get('_lr_history', [])
        self._loss_sum = sd.get('_loss_sum', 0.0)
        self._loss_count = sd.get('_loss_count', 0)
        self._p = sd.get('_p', len(self._losses))
        self._step = sd.get('_step', self._p * self.packet_size)
        self._base_lr = sd.get('_base_lr', self._base_lr)


# ── Builder ───────────────────────────────────────────────

def build_scheduler(optimizer, train_config, total_steps: int,
                     start_samples: int = 0):
    """Build (per_step_scheduler, per_checkpoint_scheduler) from TrainConfig.

    per_step_scheduler: called every batch (warmup, cosine, onecycle).
    per_checkpoint_scheduler: called at validation checkpoints (plateau, greedy).
    Returns (None, None) when scheduler == 'none'.

    If start_samples > 0 (resume): warmup phase is skipped.
    """
    tc = train_config
    scheduler = tc.scheduler
    if scheduler == 'none':
        return None, None

    total_steps = max(total_steps, 1)
    warmup_steps = int(tc.warmup_fraction * total_steps)
    if start_samples > 0:
        warmup_steps = 0  # skip warmup on resume

    if scheduler == "cosine":
        return _build_cosine(optimizer, warmup_steps, total_steps)

    if scheduler == "plateau":
        return _build_plateau(optimizer, warmup_steps, tc.plateau_patience)

    if scheduler == "onecycle":
        return _build_onecycle(optimizer, tc.lr, total_steps, tc.pct_start), None

    if scheduler == "greedy":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps
        ) if warmup_steps > 0 else None
        greedy = GreedyLR(
            optimizer, factor=tc.greedy_factor, beta=tc.greedy_beta,
            lock_steps=tc.greedy_lock_steps,
            probe_patience=tc.greedy_probe_patience,
            probe_factor=tc.greedy_probe_factor,
            probe_spike_ratio=tc.greedy_probe_spike_ratio,
            probe_lock_steps=tc.greedy_probe_lock,
            cooldown_steps=tc.greedy_cooldown,
        )
        return warmup, greedy

    if scheduler == "greedy_diff":
        wsteps = tc.greedy_diff_warmup
        if wsteps == 0:
            wsteps = warmup_steps  # fallback: use warmup_fraction
        if start_samples > 0:
            wsteps = 0  # skip warmup on resume
        gd = GreedyDiffLR(
            optimizer, d=tc.greedy_diff_d, packet_size=tc.greedy_diff_packet,
            k=tc.greedy_diff_k, K=tc.greedy_diff_K,
            min_lr=tc.greedy_diff_min_lr, max_lr=tc.greedy_diff_max_lr,
            warmup_steps=wsteps,
        )
        return gd, None

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
