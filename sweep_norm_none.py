"""Norm ablation — model with zero normalization (normalization='none').

Same as sweep_norm.py but with normalization='none' everywhere.
Fixed: 160M budget, n=3, seq_len=128, batch_size=256, lr=0.002, plateau, 5M samples.
"""

from sweep_lib import (
    resolve_architecture, train_one, gpu_health_check, setup_runtime,
)
from sweep_config import SweepConfig, ModelConfig, TrainConfig, SweepSpec, OutputConfig
from logger import GlobalLogger


def main():
    cfg = SweepConfig(
        name='norm_none_test',
        model=ModelConfig(seq_len=128, normalization='none'),
        training=TrainConfig(
            target_samples=5_000_000, lr=0.002, scheduler='plateau',
        ),
        sweep=SweepSpec(
            strategy='grid', vary='n', values=[3],
            solve='b', budget=160_000_000,
        ),
        output=OutputConfig(
            workspace='sessions/norm_none',
            sweep_log='sessions/norm_none_summary.csv',
        ),
    )

    global_logger = GlobalLogger(cfg.output.sweep_log)
    global_logger.init()
    runtime = setup_runtime(cfg.output, global_logger)

    if not gpu_health_check():
        print('GPU not available')
        return

    arch = resolve_architecture(3, 'n', cfg)
    sizes = arch['sizes']
    print(f'Arch: {"→".join(str(s) for s in sizes)}  ({arch["n_params"]:,} params)')
    print(f'Normalization: none  (no BatchNorm anywhere)\n')

    train, status, samples = train_one(arch, cfg, 'norm_none', runtime)
    print(f'\nResult: train_loss={train}')


if __name__ == '__main__':
    main()
