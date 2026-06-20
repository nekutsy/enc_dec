"""Norm ablation sweep — 4 combos of norm_bottleneck × norm_last.

Fixed: 160M budget, n=3, seq_len=128, batch_size=256, lr=0.002, plateau, 5M samples.
"""

import sys
import time as time_mod
import torch
import torch.optim as optim
import torch.nn as nn

from configs import PrimaryConfig, UNICODE_BITS
from model import Autoencoder
from data import load_text, prepare_data
from sweep_lib import (
    resolve_architecture, train_one,
    init_log, gather_done, log_row, UNIFIED_COLUMNS,
    gpu_health_check,
)
from sweep_config import SweepConfig, ModelConfig, TrainingConfig, SweepSpec, OutputConfig


def main():
    # ── Sweep config ──
    cfg = SweepConfig(
        name='norm_ablation',
        model=ModelConfig(
            seq_len=128,
            activation='silu',
            normalization='batchnorm',
            shape='rectangular',
            dropout=0.0,
        ),
        training=TrainingConfig(
            target_samples=5_000_000,
            lr=0.002,
            grad_clip=1.0,
            scheduler='plateau',
            warmup_fraction=0.05,
            optimizer='adamw_fused',
            weight_decay=0.01,
            early_stop_patience=3,
            train_ratio=0.99,
        ),
        sweep=SweepSpec(
            strategy='grid',
            vary='n',
            values=[3],
            solve='b',
            budget=160_000_000,
        ),
        output=OutputConfig(
            workspace='sessions/norm_ablation',
            sweep_log='sessions/norm_ablation_summary.csv',
            device='auto',
            batch_size=256,
        ),
    )

    # ── Device setup ──
    device_str = cfg.output.device
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False

    cfg._device = device
    cfg._text = load_text()

    if not gpu_health_check():
        print('GPU not available')
        return

    init_log(cfg.output.sweep_log)

    # ── Architecture (same for all 4 combos) ──
    arch = resolve_architecture(3, 'n', cfg)
    sizes = arch['sizes']
    print(f'Arch: {"→".join(str(s) for s in sizes)}  ({arch["n_params"]:,} params)')
    print()

    # ── 4 combos ──
    combos = [
        (True,  True,  'nbT_nlT'),
        (True,  False, 'nbT_nlF'),
        (False, True,  'nbF_nlT'),
        (False, False, 'nbF_nlF'),
    ]

    results = {}
    for nb, nl, label in combos:
        cfg.model.norm_bottleneck = nb
        cfg.model.norm_last = nl

        print(f'{"="*50}')
        print(f'  norm_bottleneck={nb}  norm_last={nl}')
        print(f'{"="*50}')

        train, status, samples = train_one(arch, cfg, label)
        mc = cfg.model
        seq_len = mc.seq_len
        bottleneck = mc.bottleneck if mc.bottleneck is not None else seq_len

        log_row(cfg.output.sweep_log, {
            'sweep_type': 'grid',
            'vary_param': 'norm',
            'vary_value': label,
            'seq_len': seq_len,
            'n_hidden': arch['n'],
            'b': f'{arch["b"]:.6g}',
            'hidden_dim': arch['hidden_dim'],
            'bottleneck': bottleneck,
            'params': arch['n_params'],
            'batch_size': cfg.output.batch_size,
            'total_samples': samples,
            'total_symbols': samples * seq_len,
            'final_train_loss': train if train is not None else '',
            'final_val_loss': '',
            'status': status,
            'duration_seconds': '',
        })

        if train is not None:
            results[label] = train
        print()

    # ── Summary ──
    print(f'{"="*50}')
    print('RESULTS')
    for nb, nl, label in combos:
        r = results.get(label, '—')
        print(f'  nb={nb} nl={nl} : {r}')
    best = min(results, key=results.get, default=None)
    if best:
        print(f'\n  ★ best: {best}  train={results[best]:.6f}')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
