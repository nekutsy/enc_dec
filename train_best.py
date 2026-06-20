"""Long training run — 160M model, n=2, best config (nbF_nlF, gain=1.0).

Usage: python3 train_best.py
Sessions: sessions/best_160m_n2/
"""

import torch

from data import load_text
from sweep_lib import (
    resolve_architecture, train_one,
    init_log, log_row, UNIFIED_COLUMNS,
    gpu_health_check,
)
from sweep_config import SweepConfig, ModelConfig, TrainingConfig, SweepSpec, OutputConfig


def main():
    cfg = SweepConfig(
        name='best_160m_n2',
        model=ModelConfig(
            seq_len=128,
            activation='silu',
            normalization='batchnorm',
            shape='rectangular',
            dropout=0.0,
            init_gain=1.0,
            norm_bottleneck=False,
            norm_last=False,
        ),
        training=TrainingConfig(
            target_samples=120_000_000,
            lr=0.002,
            grad_clip=1.0,
            scheduler='plateau',
            warmup_fraction=0.05,
            optimizer='adamw_fused',
            weight_decay=0.01,
            early_stop_patience=10,
            train_ratio=0.99,
        ),
        sweep=SweepSpec(
            strategy='grid',
            vary='n',
            values=[2],
            solve='b',
            budget=160_000_000,
        ),
        output=OutputConfig(
            workspace='sessions/best_160m_n2',
            sweep_log='sessions/best_160m_n2_summary.csv',
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

    arch = resolve_architecture(2, 'n', cfg)
    sizes = arch['sizes']
    print(f'Arch: {"->".join(str(s) for s in sizes)}  ({arch["n_params"]:,} params)')
    print(f'Hidden dim: {arch["hidden_dim"]}  b={arch["b"]:.4f}')
    print(f'Target: {cfg.training.target_samples // 1_000_000}M samples')
    print(f'Estimated time: ~5-7h at 4000-6000 samples/s')
    print()

    train, status, samples = train_one(arch, cfg, 'best_160m_n2')

    mc = cfg.model
    seq_len = mc.seq_len
    bottleneck = mc.bottleneck if mc.bottleneck is not None else seq_len

    log_row(cfg.output.sweep_log, {
        'sweep_type': 'train',
        'vary_param': 'n',
        'vary_value': '2',
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
        print(f'\nDone: train_loss={train:.6f}')
    else:
        print(f'\nFailed: {status}')


if __name__ == '__main__':
    main()
