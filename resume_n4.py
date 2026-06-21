#!/usr/bin/env python3
"""Resume n=4 training with fresh LR schedule — thin wrapper over train_one.

Usage: python3 resume_n4.py

Old implementation had a duplicative manual training loop.
Now just calls train_one(resume_lr_reset=True).
"""

import torch

from sweep_lib import (
    resolve_architecture, train_one, setup_runtime,
)
from sweep_config import SweepConfig, ModelConfig, TrainConfig, SweepSpec, OutputConfig
from logger import GlobalLogger, LoggerConfig
from utils import gpu_health_check


def main():
    cfg = SweepConfig(
        name='n4_resume',
        model=ModelConfig(seq_len=128),
        training=TrainConfig(
            target_samples=100_000_000,
            lr=0.001,
            scheduler='onecycle',
            num_workers=2,
        ),
        sweep=SweepSpec(
            strategy='grid', vary='n', values=[4],
            solve='b', budget=160_000_000,
        ),
        output=OutputConfig(
            workspace='sessions/n_binary_160m',
            sweep_log='sessions/n_binary_160m_summary.csv',
        ),
    )

    global_logger = GlobalLogger(cfg.output.sweep_log)
    global_logger.init()
    runtime = setup_runtime(cfg.output, global_logger)

    if not gpu_health_check():
        print('GPU not available')
        return

    arch = resolve_architecture(4, 'n', cfg)
    sizes = arch['sizes']
    print(f'Arch: {"→".join(str(s) for s in sizes)}  ({arch["n_params"]:,} params)')
    print(f'Target: {cfg.training.target_samples // 1_000_000}M samples')
    print()

    train, status, samples = train_one(
        arch, cfg, 'sweep_n4', runtime,
        log_config=LoggerConfig.full(),
        resume_lr_reset=True,
    )

    if train is not None:
        print(f'\nDone: train_loss={train:.6f} at {samples:,} samples')
    else:
        print(f'\nFailed: {status}')


if __name__ == '__main__':
    main()
