"""Shared utilities for enc_dec sweep scripts.

Architecture helpers: count_params, make_rectangular, solve_b_for_n, solve_n_for_b.
Training: train_one, compile_model, train_setup, save_paths.
Logging: GlobalLogger, TrainingLogger.
"""

import os
import time as time_mod

import torch
import torch.optim as optim
import torch.nn as nn

from configs import PrimaryConfig, UNICODE_BITS
from model import Autoencoder
from data import load_text, prepare_data
from trainers import run_training, build_scheduler, _cuda_safe_cleanup, _load_optimizer
from logger import (
    TrainingLogger, GlobalLogger, LoggerConfig, get_last_samples,
    init_log, gather_done, log_row, UNIFIED_COLUMNS,  # backward-compat
)
from sweep_config import SweepConfig


# ── Architecture helpers ──────────────────────────────────────

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
    """Binary search d ∈ [0.1, max_d] such that count_params(make_pyramid(D,B,n,d)) ≈ target_params."""
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
    """Binary search b ∈ [0.1, 20] such that total params ≈ target_params."""
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
    """Binary search n ∈ [1, max_n] such that total params ≈ target_params."""
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


MODEL_LEVEL_VARY = {'normalization', 'activation', 'dropout', 'norm_bottleneck', 'norm_last'}
TRAINING_LEVEL_VARY = {'lr', 'scheduler', 'grad_clip', 'optimizer', 'weight_decay'}
OUTPUT_LEVEL_VARY = {'batch_size'}


def resolve_architecture(vary_value, vary_name, sweep_config: SweepConfig):
    """Resolve full architecture given a candidate vary_value and SweepConfig."""
    mc = sweep_config.model
    sc = sweep_config.sweep

    if vary_name in MODEL_LEVEL_VARY:
        setattr(mc, vary_name, vary_value)
    elif vary_name in TRAINING_LEVEL_VARY:
        setattr(sweep_config.training, vary_name, vary_value)
    elif vary_name in OUTPUT_LEVEL_VARY:
        sweep_config.output.batch_size = vary_value

    if vary_name in MODEL_LEVEL_VARY | TRAINING_LEVEL_VARY | OUTPUT_LEVEL_VARY:
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
            return {
                'sizes': sizes,
                'b': round(h_start / input_dim, 6),
                'n': n,
                'hidden_dim': h_start,
                'n_params': n_params,
            }
        elif sc.solve == 'n':
            raise NotImplementedError("solve=n not supported for pyramid shape")
        elif n is not None and b_val is not None:
            h_start = int(input_dim * b_val)
            d = h_start - bottleneck
            sizes = make_pyramid(input_dim, bottleneck, n, d)
            n_params = count_params(sizes)
            return {
                'sizes': sizes,
                'b': b_val,
                'n': n,
                'hidden_dim': h_start,
                'n_params': n_params,
            }
        else:
            raise ValueError("Pyramid shape needs solve=b with n, or n+b fixed")

    # --- rectangular ---
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

    return {
        'sizes': sizes,
        'b': round(hidden_dim / input_dim, 6) if b_val is None else b_val,
        'n': n,
        'hidden_dim': hidden_dim,
        'n_params': n_params,
    }


# ── File paths ────────────────────────────────────────────────

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


# ── Model compilation ────────────────────────────────────────

def compile_model(model, device):
    """Compile model for GPU — skip for tiny models where overhead dominates."""
    if device.type == 'cuda':
        n_params = sum(p.numel() for p in model.parameters())
        if n_params > 50_000:
            return torch.compile(model, mode='reduce-overhead')
    return model


# ── Checkpoint resume ────────────────────────────────────────

def train_setup(config, model, optimizer, csv_path, model_path, device):
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
        _load_optimizer(optimizer, model_path, device)
    return start_samples


# ── Core training ────────────────────────────────────────────

def train_one(arch, sweep_config: SweepConfig, model_prefix,
              global_logger: GlobalLogger | None = None,
              log_config: LoggerConfig | None = None):
    """Train a single model given resolved architecture dict + SweepConfig.

    Args:
        arch: resolved architecture dict from resolve_architecture()
        sweep_config: full SweepConfig
        model_prefix: short name for the model (e.g. 'sweep_n4')
        global_logger: optional GlobalLogger for writing final summary
        log_config: optional LoggerConfig for per-model TrainingLogger

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
    device = sweep_config._device
    text = sweep_config._text
    workspace = oc.workspace
    target_samples = tc.target_samples
    lr = tc.lr
    batch_size = oc.batch_size

    bs = batch_size if batch_size is not None else 256
    model_path, csv_path = save_paths(sizes, model_prefix, prefix=workspace)

    arch_str = '→'.join(str(s) for s in sizes)
    print(f'  arch: {arch_str}')
    print(f'  params: {n_params:,}  batch: {bs}')

    # ── Already done? ──
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
                    # Try to read train_loss — prefers 'train_loss' column
                    try:
                        col = header.index('train_loss')
                        val = float(last_row[col])
                    except (ValueError, IndexError):
                        val = float(last_row[2]) if len(last_row) > 2 else 0.0
                else:
                    val = 0.0
            print(f'  already done ({last_samples:,} samples, train={val:.6f})')
            return val, 'done', last_samples

    config = PrimaryConfig(
        seq_len=seq_len, input_dim=input_dim, hidden_dim=hidden_dim,
        bottleneck=bottleneck, learning_rate=lr, train_ratio=tc.train_ratio,
        batch_size=bs, device=device.type, model_name=model_prefix,
        grad_clip=tc.grad_clip, num_workers=2 if device.type == 'cuda' else 0,
        lr_scheduler=tc.scheduler if tc.scheduler != 'none' else '',
        lr_warmup_epochs=tc.warmup_fraction,
        early_stop_patience=tc.early_stop_patience,
        cudnn_benchmark=False,
    )

    train_ds, val_ds = prepare_data(text, config)

    # ── Build model ──
    try:
        model = Autoencoder(
            sizes, name=config.model_name,
            activation=mc.activation, normalization=mc.normalization,
            init_gain=mc.init_gain,
            norm_bottleneck=mc.norm_bottleneck,
            norm_last=mc.norm_last,
            dropout=mc.dropout,
        ).to(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
            raise
        print(f'  ⚠ OOM during creation')
        _cuda_safe_cleanup()
        return None, 'oom', 0

    try:
        model = compile_model(model, device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
            raise
        print(f'  ⚠ OOM during compile')
        _cuda_safe_cleanup()
        del model
        return None, 'oom', 0

    # ── Optimizer ──
    if tc.decay_linear_only:
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)
        optim_groups = [
            {'params': decay_params, 'weight_decay': tc.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
    else:
        optim_groups = model.parameters()

    if tc.optimizer == 'adamw_fused' and device.type == 'cuda':
        optimizer = optim.AdamW(optim_groups, lr=lr,
                                weight_decay=tc.weight_decay, fused=True)
    elif tc.optimizer in ('adamw', 'adamw_fused'):
        optimizer = optim.AdamW(optim_groups, lr=lr,
                                weight_decay=tc.weight_decay)
    elif tc.optimizer == 'sgd':
        optimizer = optim.SGD(optim_groups, lr=lr,
                              weight_decay=tc.weight_decay, momentum=0.9)
    else:
        optimizer = optim.AdamW(optim_groups, lr=lr,
                                weight_decay=tc.weight_decay)

    total_batches = int(target_samples / bs) + 1
    start_samples = train_setup(config, model, optimizer, csv_path, model_path, device)
    step_scheduler, checkpoint_scheduler = build_scheduler(
        optimizer, config, total_batches, start_samples=start_samples)
    criterion = nn.BCEWithLogitsLoss()

    # ── TrainingLogger — per-model CSV + stdout ──
    lc = log_config or LoggerConfig()
    train_logger = TrainingLogger(
        csv_path, config=lc, model_name=model_prefix)

    rem = max(0, target_samples - start_samples)
    if rem <= 0:
        return 0.0, 'done', start_samples

    print(f'  training {rem:,} samples...')
    t_start = time_mod.time()

    try:
        final_samples = run_training(
            start_samples, target_samples, model, optimizer, criterion,
            train_ds, val_ds, train_logger, model_path, bs,
            seq_len, tc.grad_clip, 2,
            step_scheduler=step_scheduler,
            checkpoint_scheduler=checkpoint_scheduler,
            early_stop_patience=config.early_stop_patience,
            no_val=True)

        dur = time_mod.time() - t_start

        # Read final result from CSV
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

        # Speed
        speed_avg = final_samples / dur if dur > 0 else 0.0

        print(f'  done: {final_samples:,} samples in {dur:.0f}s '
              f'({speed_avg:,.0f} sps)  train={final_train_loss:.6f}')

        # ── Write to global summary ──
        if global_logger is not None:
            epoch = final_samples / len(train_ds) if len(train_ds) > 0 else 0.0
            global_logger.log_result({
                'sweep_type': sweep_config.sweep.strategy,
                'vary_param': sweep_config.sweep.vary,
                'vary_value': str(sweep_config.sweep.values[0]) if len(sweep_config.sweep.values) == 1 else '',
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

        return final_train_loss, 'done', final_samples

    except torch.cuda.OutOfMemoryError:
        print(f'  ⚠ OOM')
        _cuda_safe_cleanup()
        del model, optimizer
        return None, 'oom', 0
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f'  ⚠ OOM')
            _cuda_safe_cleanup()
            del model, optimizer
            return None, 'oom', 0
        raise
    except KeyboardInterrupt:
        _cuda_safe_cleanup()
        raise


# ── GPU health check ─────────────────────────────────────────

def gpu_health_check():
    """Check GPU is usable before starting a sweep. Returns True if OK."""
    if not torch.cuda.is_available():
        return True
    try:
        torch.cuda.get_device_properties(0)
        t = torch.zeros(1, device='cuda')
        del t
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False
