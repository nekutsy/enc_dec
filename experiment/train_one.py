"""train_one — single model training orchestrator.

Orchestrates: data → model → compile → optimizer → checkpoint resume →
scheduler → training loop → global log. Delegates to domain subsystems.
"""

import csv
import gc
import os
import time as time_mod

import torch
import torch.nn as nn

from configs import UNICODE_BITS
from model import Autoencoder
from data import prepare_data
from training import (
    run_training, build_scheduler,
    save_checkpoint, load_optimizer, load_plat_scheduler, load_step_scheduler,
)
from training.optimizers import build_optimizer
from utils import cuda_safe_cleanup, gpu_health_check
from logger import (
    TrainingLogger, GlobalLogger, LoggerConfig, get_last_samples,
)
from sweep_config import SweepConfig
from experiment.context import RuntimeContext


# ── Helpers ────────────────────────────────────────────────

def save_paths(model_name: str, prefix: str = 'sessions') -> tuple[str, str, str]:
    """Return (model_path, csv_path, model_dir) — dir-per-model.

    Creates: {prefix}/{model_name}/model.pth, log.csv
    """
    model_dir = os.path.join(prefix, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return (os.path.join(model_dir, 'model.pth'),
            os.path.join(model_dir, 'log.csv'),
            model_dir)


def compile_model(model, device):
    """Compile model for GPU — skip when unsafe.

    Skip conditions: CPU, tiny model (≤50k params), BatchNorm (CUDA 13.x bug).
    Uses mode="default" (no CUDA graphs) to avoid ERR on RTX 3070.
    """
    if device.type != 'cuda':
        return model
    n_params = sum(p.numel() for p in model.parameters())
    if n_params <= 50_000:
        return model
    has_bn = any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    if has_bn:
        return model
    try:
        return torch.compile(model, mode='default')
    except Exception as e:
        print(f'  ⚠ torch.compile failed ({e}) — running uncompiled')
        return model


def _train_setup(model, optimizer, csv_path, model_path, model_dir, device):
    """Load checkpoint if available → (start_samples, effective_path)."""
    start_samples = get_last_samples(csv_path)
    effective_path = model_path
    if start_samples > 0:
        if not os.path.isfile(effective_path):
            best_path = os.path.join(model_dir, 'best.pth')
            if os.path.isfile(best_path):
                effective_path = best_path
                print('  Using best checkpoint (model.pth missing)')
        if not os.path.isfile(effective_path):
            print('  No checkpoint found — starting from scratch')
            return 0, model_path
        print(f'  Resuming from {start_samples:,} samples. '
              f'Loading checkpoint...')
        state = torch.load(effective_path, map_location=device,
                           weights_only=True)
        has_prefix = any(k.startswith('_orig_mod.') for k in state.keys())
        unwrapped = (
            model._orig_mod if hasattr(model, '_orig_mod') else model)
        if has_prefix:
            state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
        unwrapped.load_state_dict(state)
        load_optimizer(optimizer, effective_path, device)
    return start_samples, effective_path


def _read_final_loss(csv_path, train_logger) -> float:
    """Read final train_loss from the last row of CSV."""
    loss = train_logger.ema_loss or 0.0
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
                    loss = float(last_row[col])
                except (ValueError, IndexError):
                    loss = (float(last_row[2])
                            if len(last_row) > 2 else loss)
    except Exception:
        pass
    return loss


# ── Main orchestrator ──────────────────────────────────────

def train_one(arch: dict, sweep_config: SweepConfig, model_prefix: str,
              runtime: RuntimeContext,
              log_config: LoggerConfig | None = None,
              resume_lr_reset: bool = False,
              no_val: bool = False):
    """Train a single model.

    Returns (final_loss, status, total_samples).
    """
    mc = sweep_config.model
    tc = sweep_config.training
    oc = sweep_config.output

    sizes = arch['sizes']
    n_params = arch['n_params']
    seq_len = mc.seq_len
    bottleneck = mc.bottleneck if mc.bottleneck is not None else seq_len
    device = runtime.device
    text = runtime.text
    global_logger = runtime.global_logger

    ws = oc.workspace
    target_samples = tc.target_samples
    bs = tc.batch_size

    model_path, csv_path, model_dir = save_paths(model_prefix, prefix=ws)

    # Write meta.json
    import json
    meta = {
        'layer_sizes': sizes,
        'n_params': n_params,
        'seq_len': seq_len,
        'bottleneck': bottleneck,
        'n_hidden': arch['n'],
        'experiment': os.path.basename(ws),
        'model_name': model_prefix,
        'activation': mc.activation,
        'normalization': mc.normalization,
        'init_gain': mc.init_gain,
        'dropout': mc.dropout,
        'norm_bottleneck': mc.norm_bottleneck,
        'norm_last': mc.norm_last,
    }
    from datetime import datetime, timezone
    meta['created'] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(model_dir, 'model.meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    arch_str = '→'.join(str(s) for s in sizes)
    print(f'  arch: {arch_str}')
    print(f'  params: {n_params:,}  batch: {bs}')

    # Already done?
    if os.path.isfile(csv_path):
        last_samples = get_last_samples(csv_path)
        if last_samples >= target_samples:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                last_row = None
                for row in reader:
                    last_row = row
            val = float(last_row.get('train_loss', 0) or 0) if last_row else 0.0
            print(f'  already done ({last_samples:,} samples, '
                  f'train={val:.6f})')
            return val, 'done', last_samples

    # ── Data ──
    train_ds, val_ds = prepare_data(text, seq_len, tc.train_ratio)

    # ── Model ──
    try:
        model = Autoencoder(
            sizes, activation=mc.activation,
            normalization=mc.normalization,
            init_gain=mc.init_gain,
            norm_bottleneck=mc.norm_bottleneck,
            norm_last=mc.norm_last,
            dropout=mc.dropout,
        ).to(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
            raise
        print('  ⚠ OOM during creation')
        cuda_safe_cleanup()
        return None, 'oom', 0

    try:
        model = compile_model(model, device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
            raise
        print('  ⚠ OOM during compile')
        cuda_safe_cleanup()
        del model
        return None, 'oom', 0

    # ── Optimizer ──
    optimizer = build_optimizer(model, tc, device)

    total_batches = int(target_samples / bs) + 1
    start_samples, ckpt_path = _train_setup(
        model, optimizer, csv_path, model_path, model_dir, device)
    if resume_lr_reset:
        print('  Starting fresh optimizer (LR reset)')
        start_samples = get_last_samples(csv_path)

    step_scheduler, checkpoint_scheduler = build_scheduler(
        optimizer, tc, total_batches, start_samples=start_samples)
    if start_samples > 0 and checkpoint_scheduler is not None:
        load_plat_scheduler(checkpoint_scheduler, ckpt_path)
    if start_samples > 0 and step_scheduler is not None:
        load_step_scheduler(step_scheduler, ckpt_path)
    criterion = nn.BCEWithLogitsLoss()

    # ── Logger ──
    lc = log_config or LoggerConfig()
    train_logger = TrainingLogger(
        csv_path, config=lc, model_name=model_prefix)

    rem = max(0, target_samples - start_samples)
    if rem <= 0:
        return 0.0, 'done', start_samples

    print(f'  training {rem:,} samples...')
    t_start = time_mod.time()
    train_done = None

    try:
        final_samples = run_training(
            start_samples, target_samples, model, optimizer, criterion,
            train_ds, val_ds, train_logger, model_path, bs,
            seq_len, tc.grad_clip, tc.num_workers,
            step_scheduler=step_scheduler,
            checkpoint_scheduler=checkpoint_scheduler,
            early_stop_patience=tc.early_stop_patience,
            no_val=no_val,
            val_interval=tc.checkpoint_interval,
        )

        dur = time_mod.time() - t_start
        final_train_loss = _read_final_loss(csv_path, train_logger)
        speed_avg = final_samples / dur if dur > 0 else 0.0
        print(f'  done: {final_samples:,} samples in {dur:.0f}s '
              f'({speed_avg:,.0f} sps)  train={final_train_loss:.6f}')

        # Global summary (sessions/global.csv)
        if global_logger is not None:
            epoch = (final_samples / len(train_ds)
                     if len(train_ds) > 0 else 0.0)
            result = {
                'experiment': os.path.basename(ws),
                'model_name': model_prefix,
                'sweep_type': sweep_config.sweep.strategy,
                'vary_param': sweep_config.sweep.vary,
                'vary_value': (
                    str(sweep_config.sweep.values[0])
                    if len(sweep_config.sweep.values) == 1 else ''),
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
            }
            global_logger.log_result(result)
            # Also write sweep-local summary
            local_logger = GlobalLogger(os.path.join(ws, 'summary.csv'))
            local_logger.init()
            local_logger.log_result(result)

        train_done = (final_train_loss, 'done', final_samples)

    except torch.cuda.OutOfMemoryError:
        print('  ⚠ OOM')
        train_done = (None, 'oom', 0)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print('  ⚠ OOM')
            train_done = (None, 'oom', 0)
        else:
            raise
    finally:
        del model
        del optimizer
        gc.collect()
        cuda_safe_cleanup()

    if train_done is None:
        raise RuntimeError('unreachable: train_done was not set')
    return train_done
