"""Shared utilities for enc_dec sweep scripts.

Architecture helpers: count_params, make_rectangular, solve_b_for_n, solve_n_for_b.
Training: train_one, compile_model, train_setup, save_paths.
Runtime: RuntimeContext, setup_runtime.
Optimizer: build_optimizer — NAG, AdamW, Lion, Sophia.
Logging: re-exports from logger.
"""

import math
import gc
import os
import time as time_mod

import torch
import torch.optim as optim
import torch.nn as nn

from configs import UNICODE_BITS
from model import Autoencoder
from data import load_text, prepare_data
from training import (
    run_training, build_scheduler,
    save_checkpoint, load_optimizer, load_plat_scheduler, load_step_scheduler,
)
from utils import cuda_safe_cleanup, gpu_health_check
from logger import (
    TrainingLogger, GlobalLogger, LoggerConfig, get_last_samples,
    init_log, gather_done, log_row, UNIFIED_COLUMNS,  # backward-compat
)
from sweep_config import SweepConfig, ModelConfig, TrainConfig, OutputConfig


# ══════════════════════════════════════════════════════════════
# Custom Optimizers
# ══════════════════════════════════════════════════════════════

class Lion(torch.optim.Optimizer):
    """Lion (EvoLved Sign Momentum) — PyTorch port from Google.

    Implements the Lion optimizer with sign-based updates.
    Reference: https://arxiv.org/abs/2302.06675
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99),
                 weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                # weight decay (decoupled, not L2-regularisation)
                if wd > 0:
                    p.mul_(1 - lr * wd)

                # Lion update: c = beta1*m + (1-beta1)*g, then m = beta2*m + (1-beta2)*g
                update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)
                p.add_(update.sign(), alpha=-lr)

                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


class Sophia(torch.optim.Optimizer):
    """Sophia — second-order clipped optimizer.

    Uses exponential moving average of Hessian diagonal estimate (hutchinson)
    for gradient clipping before Adam-style update.
    Reference: https://arxiv.org/abs/2305.14342
    """

    def __init__(self, params, lr=1e-3, betas=(0.965, 0.99),
                 weight_decay=0.01, rho=0.01):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, rho=rho)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']
            rho = group['rho']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.zeros_like(p)

                state['step'] += 1
                exp_avg = state['exp_avg']
                hessian = state['hessian']

                # Move averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                hessian.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias1 = 1 - beta1 ** state['step']
                bias2 = 1 - beta2 ** state['step']

                # Reuse grad as scratch (zero_grad clears it each step):
                #   grad = sqrt(hessian / bias2).clamp(min=eps)
                #   grad = (exp_avg / bias1) / grad
                #   grad = clip(grad, -rho, rho)
                grad.copy_(hessian).div_(bias2).sqrt_().clamp_(min=1e-15)
                grad.mul_(bias1)
                torch.div(exp_avg, grad, out=grad)
                grad.clamp_(-rho, rho)

                # Decoupled weight decay
                if wd > 0:
                    p.mul_(1 - lr * wd)

                p.add_(grad, alpha=-lr)
        return loss


def build_optimizer(model, train_config: TrainConfig, device):
    """Build optimizer from TrainConfig.

    Handles param groups (decay_linear_only) and returns the optimizer
    with appropriate constructor.
    """
    tc = train_config

    if tc.decay_linear_only:
        decay_params = []
        no_decay_params = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)
        optim_groups = [
            {'params': decay_params, 'weight_decay': tc.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
    else:
        optim_groups = model.parameters()

    opt_name = tc.optimizer
    lr = tc.lr
    wd = tc.weight_decay

    if opt_name == 'nag':
        return optim.SGD(optim_groups, lr=lr, weight_decay=wd,
                         momentum=0.9, nesterov=True)

    if opt_name == 'sgd':
        return optim.SGD(optim_groups, lr=lr, weight_decay=wd, momentum=0.9)

    if opt_name == 'adamw_fused' and device.type == 'cuda':
        return optim.AdamW(optim_groups, lr=lr, weight_decay=wd, fused=True)

    if opt_name in ('adamw', 'adamw_fused'):
        return optim.AdamW(optim_groups, lr=lr, weight_decay=wd)

    if opt_name == 'lion':
        return Lion(optim_groups, lr=lr, weight_decay=wd,
                    betas=(0.9, 0.99))

    if opt_name == 'sophia':
        return Sophia(optim_groups, lr=lr, weight_decay=wd,
                      betas=(0.965, 0.99), rho=0.01)

    # fallback
    return optim.AdamW(optim_groups, lr=lr, weight_decay=wd)


# ══════════════════════════════════════════════════════════════
# RuntimeContext
# ══════════════════════════════════════════════════════════════

class RuntimeContext:
    """Transient runtime state — device, text corpus, global logger.

    Not serialisable. Not part of SweepConfig.
    """
    def __init__(self, device: torch.device, text: str,
                 global_logger: GlobalLogger | None = None):
        self.device = device
        self.text = text
        self.global_logger = global_logger


def setup_runtime(output: OutputConfig,
                  global_logger: GlobalLogger | None = None,
                  text: str | None = None) -> RuntimeContext:
    """Resolve device + load text → RuntimeContext.

    Centralised replacement for the 8-line pattern duplicated across 5 scripts.
    """
    if output.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(output.device)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False

    if text is None:
        text = load_text()

    return RuntimeContext(device, text, global_logger)


# ══════════════════════════════════════════════════════════════
# Architecture helpers
# ══════════════════════════════════════════════════════════════

def count_params(layer_sizes):
    """Count Linear + BatchNorm1d parameters."""
    n = 0
    for i in range(len(layer_sizes) - 1):
        n += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
        n += 2 * layer_sizes[i + 1]
    return n


def make_rectangular(input_dim, hidden_dim, bottleneck, n_hidden):
    """[input] → [hidden]×n → [bottleneck] → [hidden]×n → [input]"""
    return (
        [input_dim]
        + [hidden_dim] * n_hidden
        + [bottleneck]
        + [hidden_dim] * n_hidden
        + [input_dim]
    )


def solve_d_for_n(n, target_params, D, B, max_d=None):
    """Binary search d ∈ [0.1, max_d] for pyramid."""
    if max_d is None:
        max_d = D * 10
    def _p(d_val):
        return count_params(make_pyramid(D, B, n, d_val))
    lo, hi = 0.1, max_d
    for _ in range(50):
        mid = (lo + hi) / 2
        if _p(mid) < target_params:
            lo = mid
        else:
            hi = mid
    p_lo, p_hi = _p(lo), _p(hi)
    d = lo if abs(p_lo - target_params) <= abs(p_hi - target_params) else hi
    d = round(d, 6)
    h_start = B + d
    return d, h_start, _p(d)


def make_pyramid(input_dim, bottleneck, n_hidden, d):
    """Build pyramid sizes: D→h1→h2→…→hn→B→hn→…→h2→h1→D"""
    B, n = bottleneck, n_hidden
    enc = [input_dim]
    for i in range(1, n + 1):
        enc.append(int(B + d * (n - i + 1) / n))
    enc.append(B)
    dec = []
    for i in range(n - 1, -1, -1):
        dec.append(int(B + d * (i + 1) / n))
    dec.append(input_dim)
    return enc + dec


def solve_b_for_n(n_hidden, target_params, input_dim, bottleneck):
    """Binary search b ∈ [0.1, 20] for rectangular."""
    def _p(b_val):
        h = max(1, int(round(input_dim * b_val)))
        return count_params(make_rectangular(input_dim, h, bottleneck, n_hidden))
    lo, hi = 0.1, 20.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if _p(mid) < target_params:
            lo = mid
        else:
            hi = mid
    p_lo, p_hi = _p(lo), _p(hi)
    b_val = round(lo, 6) if abs(p_lo - target_params) <= abs(p_hi - target_params) else round(hi, 6)
    h = max(1, int(round(input_dim * b_val)))
    return b_val, h, _p(b_val)


def solve_n_for_b(b_val, target_params, input_dim, bottleneck, max_n=20):
    """Binary search n ∈ [1, max_n] for rectangular."""
    def _p(n):
        h = max(1, int(round(input_dim * b_val)))
        return count_params(make_rectangular(input_dim, h, bottleneck, int(n)))
    lo, hi = 1, max_n
    for _ in range(30):
        mid = int((lo + hi) // 2)
        if _p(mid) < target_params:
            lo = mid + 1
        else:
            hi = mid
    p_lo, p_hi = _p(lo), _p(hi)
    n = lo if abs(p_lo - target_params) <= abs(p_hi - target_params) else hi
    n = max(1, min(n, max_n))
    h = max(1, int(round(input_dim * b_val)))
    return n, h, _p(n)


# ══════════════════════════════════════════════════════════════
# Architecture resolution
# ══════════════════════════════════════════════════════════════

MODEL_LEVEL_VARY = {'normalization', 'activation', 'dropout',
                    'norm_bottleneck', 'norm_last'}
TRAIN_LEVEL_VARY = {'lr', 'scheduler', 'grad_clip', 'optimizer', 'weight_decay',
                    'batch_size', 'num_workers', 'greedy_diff_packet',
                    'greedy_diff_k'}


def resolve_architecture(vary_value, vary_name, sweep_config: SweepConfig) -> dict:
    """Resolve full architecture given a candidate vary_value and SweepConfig."""
    mc = sweep_config.model
    sc = sweep_config.sweep

    if vary_name in MODEL_LEVEL_VARY:
        setattr(mc, vary_name, vary_value)
    elif vary_name in TRAIN_LEVEL_VARY:
        setattr(sweep_config.training, vary_name, vary_value)

    if vary_name in MODEL_LEVEL_VARY | TRAIN_LEVEL_VARY:
        fixed = dict(sc.fixed)
        if 'n' in fixed:
            vary_name, vary_value = 'n', fixed['n']
        elif 'b' in fixed:
            vary_name, vary_value = 'b', fixed['b']
        else:
            raise ValueError(f'vary={vary_name} needs either n= or b= in fixed')

    seq_len = mc.seq_len
    input_dim = seq_len * UNICODE_BITS
    bottleneck = mc.bottleneck if mc.bottleneck is not None else seq_len
    shape = getattr(mc, 'shape', 'rectangular')

    fixed = dict(sc.fixed)
    fixed[vary_name] = vary_value
    n = fixed.get('n', None)
    b_val = fixed.get('b', None)
    budget = sc.budget

    if shape == 'pyramid':
        if sc.solve == 'b':
            assert n is not None, "need fixed n when solve=b"
            d, h_start, n_params = solve_d_for_n(n, budget, input_dim, bottleneck)
            sizes = make_pyramid(input_dim, bottleneck, n, d)
            return {'sizes': sizes, 'b': round(h_start / input_dim, 6),
                    'n': n, 'hidden_dim': h_start, 'n_params': n_params}
        elif sc.solve == 'n':
            raise NotImplementedError("solve=n not supported for pyramid shape")
        elif n is not None and b_val is not None:
            h_start = int(input_dim * b_val)
            d = h_start - bottleneck
            sizes = make_pyramid(input_dim, bottleneck, n, d)
            return {'sizes': sizes, 'b': b_val, 'n': n,
                    'hidden_dim': h_start, 'n_params': count_params(sizes)}
        else:
            raise ValueError("Pyramid shape needs solve=b with n, or n+b fixed")

    # rectangular
    if sc.solve == 'b':
        assert n is not None, "need fixed n when solve=b"
        b_val, hidden_dim, n_params = solve_b_for_n(n, budget, input_dim, bottleneck)
    elif sc.solve == 'n':
        assert b_val is not None, "need fixed b when solve=n"
        n, hidden_dim, n_params = solve_n_for_b(b_val, budget, input_dim, bottleneck)
    elif n is not None and b_val is not None:
        hidden_dim = max(1, int(round(input_dim * b_val)))
        n_params = count_params(make_rectangular(input_dim, hidden_dim, bottleneck, n))
    elif n is not None:
        raise ValueError(f"n={n} given but no solve/b — cannot determine hidden_dim")
    elif b_val is not None:
        hidden_dim = max(1, int(round(input_dim * b_val)))
        if budget is not None:
            n, _, _ = solve_n_for_b(b_val, budget, input_dim, bottleneck)
        else:
            n = 1
        n_params = count_params(make_rectangular(input_dim, hidden_dim, bottleneck, n))
    else:
        raise ValueError("Cannot determine architecture: need n+b, or solve+fixed")

    sizes = make_rectangular(input_dim, hidden_dim, bottleneck, n)
    n_params = count_params(sizes)
    return {'sizes': sizes, 'b': round(hidden_dim / input_dim, 6) if b_val is None else b_val,
            'n': n, 'hidden_dim': hidden_dim, 'n_params': n_params}


# ══════════════════════════════════════════════════════════════
# File paths
# ══════════════════════════════════════════════════════════════

def save_paths(layer_sizes, model_name, prefix='sessions'):
    """Return (model_path, csv_path) for a given layer configuration."""
    mid = len(layer_sizes) // 2
    encoder_half = layer_sizes[:mid + 1]
    key = '_'.join(map(str, encoder_half))
    if len(key) > 200:
        import hashlib
        key = hashlib.md5(key.encode()).hexdigest()[:16]
    base = f'{key}_{model_name}'
    os.makedirs(prefix, exist_ok=True)
    return os.path.join(prefix, f'{base}.pth'), os.path.join(prefix, f'training_losses_{base}.csv')


# ══════════════════════════════════════════════════════════════
# Model compilation
# ══════════════════════════════════════════════════════════════

def compile_model(model, device):
    """Compile model for GPU — skip for tiny models.

    Uses mode="default" (no CUDA graphs) to avoid ERR on RTX 3070.
    Skipped when BatchNorm is present — known fake-tensor device propagation
    bug on CUDA 13.x drivers.
    """
    if device.type != 'cuda':
        return model
    n_params = sum(p.numel() for p in model.parameters())
    if n_params <= 50_000:
        return model
    has_bn = any(isinstance(m, torch.nn.BatchNorm1d) for m in model.modules())
    if has_bn:
        return model
    try:
        return torch.compile(model, mode='default')
    except Exception as e:
        print(f'  ⚠ torch.compile failed ({e}) — running uncompiled')
        return model


# ══════════════════════════════════════════════════════════════
# Checkpoint resume
# ══════════════════════════════════════════════════════════════

def train_setup(model, optimizer, csv_path, model_path, device):
    """Load checkpoint if available, return start_samples."""
    start_samples = get_last_samples(csv_path)
    if start_samples > 0:
        if not os.path.isfile(model_path):
            best_path = model_path.replace('.pth', '_best.pth')
            if os.path.isfile(best_path):
                model_path = best_path
                print(f'  Using _best checkpoint (main .pth missing)')
        if not os.path.isfile(model_path):
            print(f'  No checkpoint found — starting from scratch')
            return 0
        print(f'  Resuming from {start_samples:,} samples. Loading checkpoint...')
        state = torch.load(model_path, map_location=device, weights_only=True)
        has_prefix = any(k.startswith('_orig_mod.') for k in state.keys())
        unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
        if has_prefix:
            state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
        unwrapped.load_state_dict(state)
        load_optimizer(optimizer, model_path, device)
    return start_samples


# ══════════════════════════════════════════════════════════════
# Core training
# ══════════════════════════════════════════════════════════════

def train_one(arch: dict, sweep_config: SweepConfig, model_prefix: str,
              runtime: RuntimeContext,
              log_config: LoggerConfig | None = None,
              resume_lr_reset: bool = False,
              no_val: bool = False):
    """Train a single model.

    Args:
        arch: resolved dict from resolve_architecture()
        sweep_config: serialisable sweep config (Model + Train + Output)
        model_prefix: short name for file naming
        runtime: RuntimeContext with device, text, global_logger
        log_config: optional LoggerConfig for per-model TrainingLogger
        resume_lr_reset: if True, start fresh optimizer (ignore saved state);
            used for mid-epoch resume with new LR schedule.

    Returns (final_loss, status, total_samples).
    """
    mc = sweep_config.model
    tc = sweep_config.training
    oc = sweep_config.output

    sizes = arch['sizes']
    n_params = arch['n_params']
    hidden_dim = arch['hidden_dim']
    seq_len = mc.seq_len
    input_dim = seq_len * UNICODE_BITS
    bottleneck = mc.bottleneck if mc.bottleneck is not None else seq_len

    device = runtime.device
    text = runtime.text
    global_logger = runtime.global_logger

    ws = oc.workspace
    target_samples = tc.target_samples
    bs = tc.batch_size

    model_path, csv_path = save_paths(sizes, model_prefix, prefix=ws)

    arch_str = '→'.join(str(s) for s in sizes)
    print(f'  arch: {arch_str}')
    print(f'  params: {n_params:,}  batch: {bs}')

    # Already done?
    if os.path.isfile(csv_path):
        last_samples = get_last_samples(csv_path)
        if last_samples >= target_samples:
            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader)
                last_row = None
                for row in reader:
                    last_row = row
                if last_row:
                    try:
                        col = header.index('train_loss')
                        val = float(last_row[col])
                    except (ValueError, IndexError):
                        val = float(last_row[2]) if len(last_row) > 2 else 0.0
                else:
                    val = 0.0
            print(f'  already done ({last_samples:,} samples, train={val:.6f})')
            return val, 'done', last_samples

    # ── Data ──
    train_ds, val_ds = prepare_data(text, seq_len, tc.train_ratio)

    # ── Model ──
    try:
        model = Autoencoder(
            sizes,
            activation=mc.activation,
            normalization=mc.normalization,
            init_gain=mc.init_gain,
            norm_bottleneck=mc.norm_bottleneck,
            norm_last=mc.norm_last,
            dropout=mc.dropout,
        ).to(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
            raise
        print(f'  ⚠ OOM during creation')
        cuda_safe_cleanup()
        return None, 'oom', 0

    try:
        model = compile_model(model, device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
            raise
        print(f'  ⚠ OOM during compile')
        cuda_safe_cleanup()
        del model
        return None, 'oom', 0

    # ── Optimizer ──
    optimizer = build_optimizer(model, tc, device)

    total_batches = int(target_samples / bs) + 1
    start_samples = train_setup(model, optimizer, csv_path, model_path, device)
    if resume_lr_reset:
        print('  Starting fresh optimizer (LR reset)')
        start_samples = get_last_samples(csv_path)  # load weights, skip opt state

    step_scheduler, checkpoint_scheduler = build_scheduler(
        optimizer, tc, total_batches, start_samples=start_samples,
        pct_start=tc.pct_start,
        plateau_patience=tc.plateau_patience,
        greedy_factor=tc.greedy_factor,
        greedy_beta=tc.greedy_beta,
        lock_steps=tc.greedy_lock_steps,
        probe_patience=tc.greedy_probe_patience,
        probe_factor=tc.greedy_probe_factor,
        probe_spike_ratio=tc.greedy_probe_spike_ratio,
        probe_lock_steps=tc.greedy_probe_lock,
        cooldown_steps=tc.greedy_cooldown,
        greedy_diff_d=tc.greedy_diff_d,
        greedy_diff_packet=tc.greedy_diff_packet,
        greedy_diff_k=tc.greedy_diff_k,
        greedy_diff_min_lr=tc.greedy_diff_min_lr,
        greedy_diff_max_lr=tc.greedy_diff_max_lr,
        greedy_diff_warmup=tc.greedy_diff_warmup)
    if start_samples > 0 and checkpoint_scheduler is not None:
        load_plat_scheduler(checkpoint_scheduler, model_path)
    if start_samples > 0 and step_scheduler is not None:
        load_step_scheduler(step_scheduler, model_path)
    criterion = nn.BCEWithLogitsLoss()

    # ── TrainingLogger ──
    lc = log_config or LoggerConfig()
    train_logger = TrainingLogger(csv_path, config=lc, model_name=model_prefix)

    rem = max(0, target_samples - start_samples)
    if rem <= 0:
        return 0.0, 'done', start_samples

    print(f'  training {rem:,} samples...')
    t_start = time_mod.time()

    train_done = None  # result tuple or None

    try:
        final_samples = run_training(
            start_samples, target_samples, model, optimizer, criterion,
            train_ds, val_ds, train_logger, model_path, bs,
            seq_len, tc.grad_clip, tc.num_workers,
            step_scheduler=step_scheduler,
            checkpoint_scheduler=checkpoint_scheduler,
            early_stop_patience=tc.early_stop_patience,
            no_val=no_val,
            val_interval=tc.checkpoint_interval)

        dur = time_mod.time() - t_start

        # Read final train_loss from CSV
        final_train_loss = train_logger.ema_loss or 0.0
        try:
            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader)
                last_row = None
                for row in reader:
                    last_row = row
                if last_row:
                    try:
                        col = header.index('train_loss')
                        final_train_loss = float(last_row[col])
                    except (ValueError, IndexError):
                        final_train_loss = float(last_row[2]) if len(last_row) > 2 else final_train_loss
        except Exception:
            pass

        speed_avg = final_samples / dur if dur > 0 else 0.0
        print(f'  done: {final_samples:,} samples in {dur:.0f}s '
              f'({speed_avg:,.0f} sps)  train={final_train_loss:.6f}')

        # Global summary
        if global_logger is not None:
            epoch = final_samples / len(train_ds) if len(train_ds) > 0 else 0.0
            global_logger.log_result({
                'sweep_type': sweep_config.sweep.strategy,
                'vary_param': sweep_config.sweep.vary,
                'vary_value': str(sweep_config.sweep.values[0])
                    if len(sweep_config.sweep.values) == 1 else '',
                'seq_len': seq_len,
                'n_hidden': arch['n'],
                'b': f'{arch["b"]:.6g}',
                'hidden_dim': arch['hidden_dim'],
                'bottleneck': bottleneck,
                'params': n_params,
                'batch_size': bs,
                'total_samples': final_samples,
                'total_symbols': final_samples * seq_len,
                'final_train_loss': final_train_loss,
                'final_val_loss': '',
                'final_epoch': round(epoch, 4),
                'train_loss_ema': train_logger.ema_loss or '',
                'speed_avg_sps': round(speed_avg, 1),
                'status': 'done',
                'duration_seconds': round(dur, 1),
            })

        train_done = (final_train_loss, 'done', final_samples)

    except torch.cuda.OutOfMemoryError:
        print(f'  ⚠ OOM')
        train_done = (None, 'oom', 0)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f'  ⚠ OOM')
            train_done = (None, 'oom', 0)
        else:
            raise

    finally:
        # Always release GPU memory before returning
        del model
        del optimizer
        gc.collect()
        cuda_safe_cleanup()

    if train_done is None:
        raise RuntimeError('unreachable: train_done was not set')
    return train_done
