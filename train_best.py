"""Long training run — 160M model, n=2, best config (nbF_nlF, gain=1.0).

Usage: python3 train_best.py
Sessions: sessions/best_160m_n2/
"""

import torch

from data import load_text
from sweep_lib import (
    resolve_architecture, train_one,
    gpu_health_check,
)
from sweep_config import SweepConfig, ModelConfig, TrainingConfig, SweepSpec, OutputConfig
from logger import GlobalLogger


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

    global_logger = GlobalLogger(cfg.output.sweep_log)
    global_logger.init()

    arch = resolve_architecture(2, 'n', cfg)
    sizes = arch['sizes']
    print(f'Arch: {"->".join(str(s) for s in sizes)}  ({arch["n_params"]:,} params)')
    print(f'Hidden dim: {arch["hidden_dim"]}  b={arch["b"]:.4f}')
    print(f'Target: {cfg.training.target_samples // 1_000_000}M samples')
    print(f'Estimated time: ~5-7h at 4000-6000 samples/s')
    print()

    train, status, samples = train_one(arch, cfg, 'best_160m_n2',
                                       global_logger=global_logger)

    if train is not None:
        print(f'\nDone: train_loss={train:.6f}')
    else:
        print(f'\nFailed: {status}')


if __name__ == '__main__':
    main()
