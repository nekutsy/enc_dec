"""Norm ablation — model with zero normalization (normalization='none').

Same as sweep_norm.py but with normalization='none' everywhere.
Fixed: 160M budget, n=3, seq_len=128, batch_size=256, lr=0.002, plateau, 5M samples.
"""

import sys
import torch

from configs import UNICODE_BITS
from data import load_text
from sweep_lib import (
    resolve_architecture, train_one,
    init_log, gather_done, log_row, UNIFIED_COLUMNS,
    gpu_health_check,
)
from sweep_config import SweepConfig, ModelConfig, TrainingConfig, SweepSpec, OutputConfig


def main():
    cfg = SweepConfig(
        name='norm_none_test',
        model=ModelConfig(
            seq_len=128,
            activation='silu',
            normalization='none',   # ← zero normalization
            shape='rectangular',
            dropout=0.0,
            norm_bottleneck=False,
            norm_last=False,
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
            workspace='sessions/norm_none',
            sweep_log='sessions/norm_none_summary.csv',
            device='auto',
            batch_size=256,
        ),
    )

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

    arch = resolve_architecture(3, 'n', cfg)
    sizes = arch['sizes']
    print(f'Arch: {"→".join(str(s) for s in sizes)}  ({arch["n_params"]:,} params)')
    print(f'Normalization: none  (no BatchNorm anywhere)')
    print()

    train, status, samples = train_one(arch, cfg, 'norm_none')
    mc = cfg.model
    seq_len = mc.seq_len
    bottleneck = mc.bottleneck if mc.bottleneck is not None else seq_len

    log_row(cfg.output.sweep_log, {
        'sweep_type': 'single',
        'vary_param': 'normalization',
        'vary_value': 'none',
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

    print(f'\nResult: train_loss={train}')


if __name__ == '__main__':
    main()
