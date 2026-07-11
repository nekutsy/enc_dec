#!/usr/bin/env python3
"""Fine-tune 41eab71ae154: noise=0, greedy_simple inc=1.02 dec=0.75 patience=1000 warmup=2000."""
import os, sys, gc, json, time, signal, csv, io
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch, torch.nn as nn
from torch.utils.data import DataLoader

from configs import UNICODE_BITS
from model import Autoencoder
from data import prepare_data, load_text
from training import step_batch, save_checkpoint, build_scheduler
from training.optimizers import build_optimizer
from experiment.config import TrainConfig
from utils import cuda_safe_cleanup

DONOR = '41eab71ae154'
TARGET = 40_000_000
BS = 16
LR = 0.0001
GCLIP = 1.0
OPTIM = 'sgd'
VAL_INTERVAL = 100_000
CKPT_INTERVAL = 1_000_000
NUM_WORKERS = 0
NO_VAL = True
SCHED = 'greedy_simple'
GS_INC = 1.02
GS_DEC = 0.75
GS_PAT = 1000
GS_WARMUP = 2000
GS_MIN_LR = 1e-6
GS_MAX_LR = 0.4

device = torch.device('cuda')
torch.backends.cuda.matmul.allow_tf32 = True

# ── Data ──
texts = load_text()
train_ds, val_ds = prepare_data(texts, 128, 0.999)
print(f'Train: {len(train_ds):,} windows  |  Val: {len(val_ds):,}')
print(f'VRAM after data: {torch.cuda.memory_allocated()/1024**3:.2f} GB')

# ── Model ──
meta = json.load(open(f'sessions/runs/{DONOR}-rect_s128_en3_d5_b3.042_res_pre/meta.json'))
sizes = meta['layer_sizes']
model = Autoencoder(
    sizes, activation='silu', normalization='rmsnorm', residual=True,
    residual_norm='pre', enc_n=meta.get('enc_n', 3),
).to(device)
n = sum(p.numel() for p in model.parameters())
print(f'Model: {n:,} params  batch={BS}  opt={OPTIM}')
print(f'VRAM after model: {torch.cuda.memory_allocated()/1024**3:.2f} GB')

# ── Weights ──
state = torch.load(f'sessions/runs/{DONOR}-rect_s128_en3_d5_b3.042_res_pre/model.pth',
                   map_location='cpu', weights_only=True)
model.load_state_dict(state)
del state
print(f'VRAM after weights: {torch.cuda.memory_allocated()/1024**3:.2f} GB')

# ── Optimizer & Scheduler ──
tc = TrainConfig(
    target_samples=TARGET, batch_size=BS, lr=LR, grad_clip=GCLIP,
    scheduler=SCHED, optimizer=OPTIM, noise_prob=0.0, weight_decay=0.01,
    greedy_simple_inc=GS_INC, greedy_simple_dec=GS_DEC,
    greedy_simple_patience=GS_PAT, greedy_simple_warmup=GS_WARMUP,
    greedy_simple_min_lr=GS_MIN_LR, greedy_simple_max_lr=GS_MAX_LR,
    decay_linear_only=True,
)
opt = build_optimizer(model, tc, device)
print(f'VRAM after optimizer: {torch.cuda.memory_allocated()/1024**3:.2f} GB')

# DataLoader
loader = DataLoader(train_ds, batch_size=BS, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=True)
print(f'VRAM after loader: {torch.cuda.memory_allocated()/1024**3:.2f} GB')

run_dir = f'sessions/runs/ft_n0_{int(time.time())}'
os.makedirs(run_dir, exist_ok=True)
model_path = f'{run_dir}/model.pth'
best_path = f'{run_dir}/best.pth'
csv_path = f'{run_dir}/log.csv'

# Init CSV
csv_f = open(csv_path, 'w')
csv_w = csv.writer(csv_f)
csv_w.writerow(['step', 'total_samples', 'train_loss', 'val_loss', 'lr'])
csv_f.flush()

total_batches = int(TARGET / BS) + 1
step_sch, ckpt_sch = build_scheduler(opt, tc, total_batches, start_samples=0)

interrupted = False
def on_signal(sig, frame):
    global interrupted; interrupted = True
signal.signal(signal.SIGINT, on_signal)
signal.signal(signal.SIGTERM, on_signal)

total = 0
step_count = 0
next_ckpt = CKPT_INTERVAL
t0 = time.time()
best_loss = float('inf')
loss_ema = None
loss_ema_beta = 0.95

criterion = nn.BCEWithLogitsLoss()
use_amp = (device.type == 'cuda')

print(f'training {TARGET:,} samples...')
sys.stdout.flush()

while total < TARGET and not interrupted:
    model.train()
    for x, y in loader:
        if interrupted or total >= TARGET:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss_val = step_batch(model, x, y, criterion, opt,
                              use_amp=use_amp, grad_clip=GCLIP,
                              step_scheduler=step_sch)
        bs = x.size(0)
        total += bs
        step_count += 1

        # EMA loss
        loss_ema = (loss_ema_beta * loss_ema + (1 - loss_ema_beta) * loss_val) if loss_ema is not None else loss_val

        if step_count % 100 == 0:
            elapsed = time.time() - t0
            sps = total / elapsed if elapsed > 0 else 0
            sys.stdout.write(f'\r  {total/1e6:7.2f}M  lr={opt.param_groups[0]["lr"]:.2e}  loss={loss_ema:.6f}  {sps:,.0f} sps')
            sys.stdout.flush()

        if total >= next_ckpt:
            csv_f.flush()
            # Save
            torch.save(model.state_dict(), model_path)
            csv_line = [step_count, total, loss_ema, '', opt.param_groups[0]['lr']]
            csv_w.writerow(csv_line)
            csv_f.flush()
            # Track best
            if loss_ema and loss_ema < best_loss:
                best_loss = loss_ema
                torch.save(model.state_dict(), best_path)
            next_ckpt += CKPT_INTERVAL
            sys.stdout.write(f'\n  [ckpt @ {total/1e6:.1f}M] loss_ema={loss_ema:.6f}\n')
            sys.stdout.flush()

        if total >= TARGET or interrupted:
            break

# ── Final ──
torch.save(model.state_dict(), model_path)
csv_f.close()
dur = time.time() - t0
del model, opt, loader
gc.collect()
cuda_safe_cleanup()
print(f'\ndone: {total:,} samples in {dur:.0f}s ({total/dur:,.0f} sps)  loss_ema={loss_ema:.6f if loss_ema else 0}')
