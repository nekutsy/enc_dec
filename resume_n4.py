#!/usr/bin/env python3
"""Resume training for n=4 from checkpoint to TARGET_SAMPLES, resetting LR schedule."""

import os
import sys
import signal
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from configs import PrimaryConfig, UNICODE_BITS
from model import Autoencoder
from data import load_text, prepare_data
from trainers import _cuda_safe_cleanup
from logger import TrainingLogger, LoggerConfig

# ── Config ───────────────────────────────────────────────────
SEQ_LEN = 128
HIDDEN_DIM = 4713
BOTTLENECK = 128
N_HIDDEN = 4
LR = 0.001
TARGET_SAMPLES = 100_000_000
BATCH_SIZE = 256
GRAD_CLIP = 1.0
DEVICE = 'cuda'
WORKSPACE = 'sessions/n_binary_160m'

sizes = [2688, 4713, 4713, 4713, 4713, 128, 4713, 4713, 4713, 4713, 2688]
model_prefix = '2688_4713_4713_4713_4713_128_sweep_n4'

model_path = os.path.join(WORKSPACE, f'{model_prefix}.pth')
csv_path = os.path.join(WORKSPACE, f'training_losses_{model_prefix}.csv')

# Clean up from any previous runs
try:
    import gc; gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
except Exception:
    pass

device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')

# ── Load text & data ─────────────────────────────────────────
text = load_text()
config = PrimaryConfig(
    seq_len=SEQ_LEN, input_dim=sizes[0], hidden_dim=HIDDEN_DIM,
    bottleneck=BOTTLENECK, learning_rate=LR, train_ratio=0.99,
    batch_size=BATCH_SIZE, device=device.type, model_name=model_prefix,
    grad_clip=GRAD_CLIP, num_workers=2,
    lr_scheduler='', early_stop_patience=3, cudnn_benchmark=False,
)
train_ds, val_ds = prepare_data(text, config)
epoch_size = len(train_ds)

# ── Model ────────────────────────────────────────────────────
model = Autoencoder(sizes, init_gain=1.0).to(device)

# ── Resume checkpoint ────────────────────────────────────────
print(f'Loading checkpoint: {model_path}')
if not os.path.isfile(model_path):
    alt = model_path.replace('.pth', '_best.pth')
    if os.path.isfile(alt):
        model_path = alt
        print(f'  Using _best checkpoint')

state = torch.load(model_path, map_location=device, weights_only=True)
has_prefix = any(k.startswith('_orig_mod.') for k in state.keys())
if has_prefix:
    state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
model.load_state_dict(state)
print('  Model loaded')

# ── Optimizer (fresh — LR reset for new epoch) ───────────────
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
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0},
]
optimizer = optim.AdamW(optim_groups, lr=LR, fused=True)
print('  Starting fresh optimizer (LR reset)')

# ── Compile after optimizer creation ─────────────────────────
model = torch.compile(model, mode="default")
unwrapped = model._orig_mod

# ── Read CSV for start position ──────────────────────────────
import csv
start_samples = 0
if os.path.isfile(csv_path):
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        val_col = header.index('train_loss') if header and 'train_loss' in header else 2
        for row in reader:
            pass
        if row:
            start_samples = int(float(row[0]))
print(f'  Resume from {start_samples:,} samples')

remaining = TARGET_SAMPLES - start_samples
if remaining <= 0:
    print('Already done!')
    sys.exit(0)

total_steps = int(remaining / BATCH_SIZE) + 1
print(f'  Remaining: {remaining:,} samples, ~{total_steps} steps')

# ── Fresh LR schedule ───────────────────────────────────────
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, total_steps=total_steps,
    pct_start=0.3, anneal_strategy='cos',
    div_factor=25.0, final_div_factor=10000.0,
)
print(f'  LR schedule: OneCycleLR (max_lr={LR}, pct_start=0.3)')

# ── Training logger ──────────────────────────────────────────
logger = TrainingLogger(csv_path, config=LoggerConfig.full(), model_name=model_prefix)

# ── Training loop ────────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()

_interrupted = False
def _on_signal(signum, frame):
    global _interrupted
    _interrupted = True
signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)

loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=2, pin_memory=True, drop_last=False)

model.train()
total = start_samples
CKPT_EVERY = 500_000  # samples between checkpoints
next_ckpt = total + CKPT_EVERY
interval_loss_sum = 0.0
interval_loss_count = 0
t_start = time.time()

print(f'\nTraining {remaining:,} samples → {TARGET_SAMPLES:,} total')
for batch_idx, batch in enumerate(loader):
    if _interrupted:
        print('\n  Interrupted — saving checkpoint...')
        torch.save(unwrapped.state_dict(), model_path)
        torch.save(optimizer.state_dict(), opt_path := model_path + '.opt')
        _cuda_safe_cleanup()
        sys.exit(0)

    if isinstance(batch, (list, tuple)):
        inp = batch[0].to(device, non_blocking=True)
    else:
        inp = batch.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    out = model(inp)
    loss = criterion(out, inp)
    loss.backward()
    if GRAD_CLIP:
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    scheduler.step()

    bs = inp.size(0)
    total += bs
    interval_loss_sum += loss.item() * bs
    interval_loss_count += bs
    logger.on_batch_end(total, loss.item())

    # ── Checkpoint log ──
    if total >= next_ckpt:
        avg_loss = interval_loss_sum / interval_loss_count if interval_loss_count > 0 else 0
        cur_lr = scheduler.get_last_lr()[0]
        logger.log_checkpoint(total, avg_loss, epoch_size, lr=cur_lr)

        torch.save(unwrapped.state_dict(), model_path)
        torch.save(optimizer.state_dict(), model_path + '.opt')

        interval_loss_sum = 0.0
        interval_loss_count = 0
        next_ckpt += CKPT_EVERY

    if total >= TARGET_SAMPLES:
        break

# ── Final ───────────────────────────────────────────────────
avf_loss = interval_loss_sum / interval_loss_count if interval_loss_count > 0 else loss.item()
logger.log_final(total, avf_loss, epoch_size,
                 duration_seconds=time.time() - t_start)
torch.save(unwrapped.state_dict(), model_path)
torch.save(optimizer.state_dict(), model_path + '.opt')
_cuda_safe_cleanup()
print(f'\nDone! Final loss: {avf_loss:.6f} at {total:,} samples')
