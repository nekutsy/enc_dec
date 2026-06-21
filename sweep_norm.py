"""Norm ablation sweep — 4 combos of norm_bottleneck × norm_last.

Fixed: 160M budget, n=3, seq_len=128, batch_size=256, lr=0.002, plateau, 5M samples.
"""

import torch

from sweep_lib import (
    resolve_architecture, train_one, gpu_health_check, setup_runtime,
)
from sweep_config import SweepConfig, ModelConfig, TrainConfig, SweepSpec, OutputConfig
from logger import GlobalLogger


def main():
    cfg = SweepConfig(
        name='norm_ablation',
        model=ModelConfig(seq_len=128),
        training=TrainConfig(
            target_samples=5_000_000, lr=0.002, scheduler='plateau',
        ),
        sweep=SweepSpec(
            strategy='grid', vary='n', values=[3],
            solve='b', budget=160_000_000,
        ),
        output=OutputConfig(
            workspace='sessions/norm_ablation',
            sweep_log='sessions/norm_ablation_summary.csv',
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
    print(f'Arch: {"→".join(str(s) for s in sizes)}  ({arch["n_params"]:,} params)\n')

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

        train, status, samples = train_one(arch, cfg, label, runtime)

        if train is not None:
            results[label] = train
        print()

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
