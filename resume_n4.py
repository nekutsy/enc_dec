#!/usr/bin/env python3
"""Resume training for n=4 from 25M to 50M samples, resetting LR schedule."""
import os, sys, signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from configs import PrimaryConfig, UNICODE_BITS
from model import Autoencoder
from data import load_text, prepare_data
from trainers import _cuda_safe_cleanup
from logger import CSVLogger

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

# Match the exact architecture from the sweep
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

# ── Optimizer (before compile — parameter groups must match original sweep) ─
# decay_linear_only=True: decay on params with dim >= 2 (Linear weights), skip biases/norm
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
opt_path = model_path + '.opt'
if os.path.isfile(opt_path):
    # Start fresh optimizer — LR is reset anyway, momentum state stale after pause
    print('  Starting fresh optimizer (LR reset for new epoch)')
else:
    print('  ⚠ No optimizer checkpoint — starting fresh optimizer')

# ── Compile after optimizer creation ─────────────────────────
# mode="default" — no CUDA graphs to avoid ERR risk
model = torch.compile(model, mode="default")
unwrapped = model._orig_mod

# ── Read CSV for start position ──────────────────────────────
import csv
start_samples = 0
if os.path.isfile(csv_path):
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            pass  # iterate to last
        start_samples = int(row[0])
print(f'  Resume from {start_samples:,} samples')

remaining = TARGET_SAMPLES - start_samples
if remaining <= 0:
    print('Already done!')
    sys.exit(0)

total_steps = int(remaining / BATCH_SIZE) + 1
print(f'  Remaining: {remaining:,} samples, ~{total_steps} steps')

# ── Fresh LR schedule for remaining steps ───────────────────
# OneCycleLR: fast warmup to max_lr, then cosine down to near-zero
# pct_start=0.3 → 30% warmup, 70% cosine decay
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, total_steps=total_steps,
    pct_start=0.3, anneal_strategy='cos',
    div_factor=25.0, final_div_factor=10000.0,
)
print(f'  LR schedule: OneCycleLR (max_lr={LR}, pct_start=0.3, div=25, final_div=10000)')

# ── Training loop ────────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()
logger = CSVLogger(csv_path)

# Signal handler — just set flag, no CUDA calls
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
ckpt_interval = 512  # save every 512 batches (~131k samples)

print(f'\nTraining {remaining:,} samples → {TARGET_SAMPLES:,} total')
for batch_idx, batch in enumerate(loader):
    if _interrupted:
        print('\n  Interrupted — saving checkpoint...')
        torch.save(unwrapped.state_dict(), model_path)
        torch.save(optimizer.state_dict(), opt_path)
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

    total += BATCH_SIZE
    if batch_idx % 250 == 0:
        print(f'  {total:>10,} / {TARGET_SAMPLES:,}  '
              f'loss={loss.item():.6f}  lr={scheduler.get_last_lr()[0]:.2e}')

    if batch_idx > 0 and batch_idx % ckpt_interval == 0:
        logger.log_row({
            'total_samples': total, 'total_symbols': total * SEQ_LEN,
            'train_loss': loss.item(), 'val_loss': loss.item(),
        })
        torch.save(unwrapped.state_dict(), model_path)
        torch.save(optimizer.state_dict(), opt_path)

    if total >= TARGET_SAMPLES:
        break

# ── Final save ───────────────────────────────────────────────
torch.save(unwrapped.state_dict(), model_path)
torch.save(optimizer.state_dict(), opt_path)
logger.log_row({
    'total_samples': total, 'total_symbols': total * SEQ_LEN,
    'train_loss': loss.item(), 'val_loss': loss.item(),
})
_cuda_safe_cleanup()
print(f'\nDone! Final loss: {loss.item():.6f} at {total:,} samples')
