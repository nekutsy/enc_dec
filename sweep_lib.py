"""Shared utilities for enc_dec sweep scripts.

Architecture helpers: count_params, make_rectangular, solve_b_for_n, solve_n_for_b.
Training: train_one, compile_model, train_setup, save_paths.
Logging: init_log, gather_done, log_row.
"""

import os
import csv
import time as time_mod

import torch
import torch.optim as optim
import torch.nn as nn

from configs import PrimaryConfig, UNICODE_BITS
from model import Autoencoder
from data import load_text, prepare_data
from trainers import run_training, build_scheduler, _cuda_safe_cleanup
from logger import CSVLogger, get_last_symbols
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


def solve_b_for_n(n_hidden, target_params, input_dim, bottleneck):
    """Binary search b ∈ [0.1, 20] such that total params ≈ target_params.
    Returns (b_val, hidden_dim, actual_params)."""

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
    """Binary search n ∈ [1, max_n] such that total params ≈ target_params.
    Returns (n, hidden_dim, actual_params)."""
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



MODEL_LEVEL_VARY = {'normalization', 'activation', 'dropout'}
OUTPUT_LEVEL_VARY = {'batch_size'}


def resolve_architecture(vary_value, vary_name, sweep_config: SweepConfig):
    """Resolve full architecture given a candidate vary_value and SweepConfig.

    Model-level params (normalization, activation) → set on cfg.model,
    output-level params (batch_size) → set on cfg.output.batch_size,
    then architecture is resolved from sweep.fixed.

    Returns dict: {sizes, b, n, hidden_dim, n_params}
    """
    mc = sweep_config.model
    sc = sweep_config.sweep

    # Params that don't affect architecture shape
    if vary_name in MODEL_LEVEL_VARY:
        setattr(mc, vary_name, vary_value)
    elif vary_name in OUTPUT_LEVEL_VARY:
        sweep_config.output.batch_size = vary_value

    if vary_name in MODEL_LEVEL_VARY | OUTPUT_LEVEL_VARY:
        # Resolve architecture from fixed params (n or b)
        fixed = dict(sc.fixed)
        if 'n' in fixed:
            vary_name, vary_value = 'n', fixed['n']
        elif 'b' in fixed:
            vary_name, vary_value = 'b', fixed['b']
        else:
            raise ValueError(
                f'vary={vary_name} needs either n= or b= in fixed')

    seq_len = mc.seq_len
    input_dim = seq_len * UNICODE_BITS
    bottleneck = mc.bottleneck if mc.bottleneck is not None else seq_len

    # Build fixed dict — sweep fixed params + model-derived constants
    fixed = dict(sc.fixed)
    fixed[vary_name] = vary_value

    n = fixed.get('n', None)
    b_val = fixed.get('b', None)
    budget = sc.budget

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
    """Load checkpoint if available, return start_symbols."""
    start_symbols = get_last_symbols(csv_path)
    if start_symbols > 0:
        print(f'  Resuming from {start_symbols} symbols. Loading checkpoint...')
        state = torch.load(model_path, map_location=device, weights_only=True)
        has_prefix = any(k.startswith('_orig_mod.') for k in state.keys())
        unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
        if has_prefix:
            state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
        unwrapped.load_state_dict(state)
        _load_optimizer(optimizer, model_path, device)
    return start_symbols


def _load_optimizer(optimizer, model_path, device):
    opt_path = model_path + '.opt'
    if os.path.isfile(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device, weights_only=True))


# ── CSV logging helpers ──────────────────────────────────────

UNIFIED_COLUMNS = [
    'sweep_type', 'vary_param', 'vary_value',
    'seq_len', 'n_hidden', 'b', 'hidden_dim', 'bottleneck',
    'params', 'batch_size', 'total_symbols',
    'final_train_loss', 'final_val_loss', 'status', 'duration_seconds',
]


def init_log(sweep_log, columns=None):
    """Create CSV log file with header if missing."""
    cols = columns or UNIFIED_COLUMNS
    os.makedirs(os.path.dirname(sweep_log) or '.', exist_ok=True)
    if not os.path.isfile(sweep_log):
        with open(sweep_log, 'w', newline='') as f:
            csv.writer(f).writerow(cols)


def gather_done(sweep_log, target_symbols):
    """Read completed models from unified CSV. Returns {vary_value: val_loss}."""
    done = {}
    if not os.path.isfile(sweep_log):
        return done
    with open(sweep_log) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                status = row.get('status', '')
                sym = int(float(row.get('total_symbols', 0)))
                val_str = row.get('final_val_loss', '')
                vary_str = row.get('vary_value', '')
                if status == 'done' and sym >= target_symbols * 0.85 and val_str and vary_str:
                    try:
                        vary_val = float(vary_str) if '.' in vary_str or vary_str.lstrip('-').isdigit() else vary_str
                        if isinstance(vary_val, str):
                            vary_val = vary_str
                        else:
                            vary_val = int(vary_val) if vary_val == int(vary_val) else vary_val
                    except ValueError:
                        vary_val = vary_str
                    done[vary_val] = float(val_str)
            except (ValueError, TypeError):
                continue
    return done


def log_row(sweep_log, row_dict):
    """Append a dict row to the sweep CSV."""
    with open(sweep_log, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_COLUMNS)
        writer.writerow(row_dict)


# ── Core training ────────────────────────────────────────────

def train_one(arch, sweep_config: SweepConfig, model_prefix):
    """Train a single model given resolved architecture dict + SweepConfig.

    Returns (val_loss, status).
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
    target_symbols = tc.target_symbols
    lr = tc.lr
    batch_size = oc.batch_size

    bs = batch_size if batch_size is not None else 256
    model_path, csv_path = save_paths(sizes, model_prefix, prefix=workspace)

    arch_str = '→'.join(str(s) for s in sizes)
    print(f'  arch: {arch_str}')
    print(f'  params: {n_params:,}  batch: {bs}')

    # Resume check
    if os.path.isfile(csv_path):
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        if rows:
            last_sym = int(float(rows[-1][0]))
            if last_sym >= target_symbols:
                val = float(rows[-1][2])
                print(f'  already done ({last_sym:,} sym, val={val:.6f})')
                return val, 'done'

    config = PrimaryConfig(
        seq_len=seq_len, input_dim=input_dim, hidden_dim=hidden_dim,
        bottleneck=bottleneck, learning_rate=lr, train_ratio=tc.train_ratio,
        batch_size=bs, device=device.type, model_name=model_prefix,
        grad_clip=tc.grad_clip, num_workers=2 if device.type == 'cuda' else 0,
        lr_scheduler=tc.scheduler if tc.scheduler != 'none' else '',
        lr_warmup_epochs=tc.warmup_fraction, cudnn_benchmark=False,
    )

    train_ds, val_ds = prepare_data(text, config)

    # Build model with configurable activation/norm
    try:
        model = Autoencoder(
            sizes, name=config.model_name,
            activation=mc.activation, normalization=mc.normalization,
        ).to(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
            raise
        print(f'  ⚠ OOM during creation')
        _cuda_safe_cleanup()
        return None, 'oom'

    try:
        model = compile_model(model, device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
            raise
        print(f'  ⚠ OOM during compile')
        _cuda_safe_cleanup()
        del model
        return None, 'oom'

    # Optimizer selection
    if tc.optimizer == 'adamw_fused' and device.type == 'cuda':
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=tc.weight_decay, fused=True)
    elif tc.optimizer in ('adamw', 'adamw_fused'):
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=tc.weight_decay)
    elif tc.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              weight_decay=tc.weight_decay, momentum=0.9)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=tc.weight_decay)

    total_batches = int(target_symbols / bs / seq_len) + 1
    scheduler = build_scheduler(optimizer, config, total_batches)
    criterion = nn.MSELoss()
    logger = CSVLogger(csv_path)

    start_sym = train_setup(config, model, optimizer, csv_path, model_path, device)
    rem = max(0, target_symbols - start_sym)
    if rem <= 0:
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        val = float(rows[-1][2]) if rows and len(rows[-1]) > 2 else 0
        return val, 'done'

    print(f'  training {rem:,} symbols...')
    t_start = time_mod.time()

    try:
        final_symbols = run_training(
            start_sym, target_symbols, model, optimizer, criterion,
            train_ds, val_ds, logger, model_path, bs,
            seq_len, tc.grad_clip, 2, scheduler)

        with open(csv_path) as f:
            rows = list(csv.reader(f))
        val = float(rows[-1][2]) if rows and len(rows[-1]) > 2 else 0
        dur = time_mod.time() - t_start
        print(f'  done: {final_symbols:,} sym in {dur:.0f}s  val={val:.6f}')
        return val, 'done'
    except torch.cuda.OutOfMemoryError:
        print(f'  ⚠ OOM')
        _cuda_safe_cleanup()
        del model, optimizer
        return None, 'oom'
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f'  ⚠ OOM')
            _cuda_safe_cleanup()
            del model, optimizer
            return None, 'oom'
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
        result = torch.cuda.get_device_properties(0)
        t = torch.zeros(1, device='cuda')
        del t
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False
