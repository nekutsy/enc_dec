"""Long training run — 160M model, n=2, best config (nbF_nlF, gain=1.0).

Usage: python3 train_best.py
Sessions: sessions/best_160m_n2/
"""

from sweep_lib import (
    resolve_architecture, train_one, gpu_health_check, setup_runtime,
)
from sweep_config import SweepConfig, ModelConfig, TrainConfig, SweepSpec, OutputConfig
from logger import GlobalLogger


def main():
    cfg = SweepConfig(
        name='best_160m_n2',
        model=ModelConfig(seq_len=128),
        training=TrainConfig(
            target_samples=120_000_000,
            lr=0.002,
            scheduler='plateau',
            early_stop_patience=10,
        ),
        sweep=SweepSpec(
            strategy='grid', vary='n', values=[2],
            solve='b', budget=160_000_000,
        ),
        output=OutputConfig(
            workspace='sessions/best_160m_n2',
            sweep_log='sessions/best_160m_n2_summary.csv',
        ),
    )

    global_logger = GlobalLogger(cfg.output.sweep_log)
    global_logger.init()
    runtime = setup_runtime(cfg.output, global_logger)

    if not gpu_health_check():
        print('GPU not available')
        return

    arch = resolve_architecture(2, 'n', cfg)
    sizes = arch['sizes']
    print(f'Arch: {"->".join(str(s) for s in sizes)}  ({arch["n_params"]:,} params)')
    print(f'Hidden dim: {arch["hidden_dim"]}  b={arch["b"]:.4f}')
    print(f'Target: {cfg.training.target_samples // 1_000_000}M samples')
    print()

    train, status, samples = train_one(arch, cfg, 'best_160m_n2', runtime)

    if train is not None:
        print(f'\nDone: train_loss={train:.6f}')
    else:
        print(f'\nFailed: {status}')


if __name__ == '__main__':
    main()
