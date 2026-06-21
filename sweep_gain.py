"""Init gain sweep — 4 values over best norm config (nbF_nlF).

Fixed: 160M budget, n=3, seq_len=128, batch_size=256, lr=0.002, plateau, 4M samples.
"""

import torch

from sweep_lib import (
    resolve_architecture, train_one,
    init_log, log_row, UNIFIED_COLUMNS,
    gpu_health_check, setup_runtime,
)
from sweep_config import SweepConfig, ModelConfig, TrainConfig, SweepSpec, OutputConfig
from logger import GlobalLogger


def main():
    gains = [0.1, 0.25, 0.5, 1.0]
    results = {}
    base_cfg = SweepConfig(
        name='init_gain_sweep',
        model=ModelConfig(seq_len=128),
        training=TrainConfig(
            target_samples=4_000_000, lr=0.002, scheduler='plateau',
        ),
        sweep=SweepSpec(
            strategy='grid', vary='n', values=[3],
            solve='b', budget=160_000_000,
        ),
        output=OutputConfig(
            workspace='sessions/init_gain_sweep',
            sweep_log='sessions/init_gain_sweep_summary.csv',
        ),
    )

    global_logger = GlobalLogger(base_cfg.output.sweep_log)
    global_logger.init()
    runtime = setup_runtime(base_cfg.output, global_logger)

    if not gpu_health_check():
        print('GPU not available')
        return

    arch = resolve_architecture(3, 'n', base_cfg)
    sizes = arch['sizes']
    print(f'Arch: {"→".join(str(s) for s in sizes)}  ({arch["n_params"]:,} params)\n')

    for gain in gains:
        base_cfg.model.init_gain = gain
        print(f'{"="*50}')
        print(f'  init_gain={gain}')
        print(f'{"="*50}')

        label = f'gain{str(gain).replace(".", "_")}'
        train, status, samples = train_one(arch, base_cfg, label, runtime)

        if train is not None:
            results[gain] = train
        print()

    print(f'{"="*50}')
    print('RESULTS')
    for gain in gains:
        r = results.get(gain, '—')
        print(f'  gain={gain:.2f} : {r}')
    best = min(results, key=results.get, default=None)
    if best:
        print(f'\n  ★ best: gain={best}  train={results[best]:.6f}')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
