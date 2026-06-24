"""Preset sweep config generators — ready-to-use experiments."""

from sweep_config import SweepConfig, ModelConfig, TrainConfig, SweepSpec, OutputConfig


def preset_ratio(budget_m: int) -> SweepConfig:
    """Ratio sweep at a given budget in millions."""
    return SweepConfig(
        name=f'ratio_{budget_m}m',
        sweep=SweepSpec(
            strategy='grid',
            vary='n',
            values=(list(range(2, 7)) if budget_m == 20
                    else list(range(2, 11, 2)) if budget_m == 40
                    else list(range(2, 17))),
            solve='b',
            budget=budget_m * 1_000_000,
        ),
        output=OutputConfig(
            workspace=f'sessions/ratio{budget_m}',
            sweep_log=f'sessions/ratio{budget_m}_sweep_summary.csv',
        ),
    )


def preset_binary() -> SweepConfig:
    """Binary search for optimal width ratios across seq_lens."""
    return SweepConfig(
        name='binary_search',
        sweep=SweepSpec(
            strategy='grid',
            vary='seq_len',
            values=[4, 8, 16, 32, 64, 128],
            solve=None,
        ),
        output=OutputConfig(
            workspace='sessions/sweep',
            sweep_log='sessions/sweep_binary_summary.csv',
        ),
    )


def preset_width(seq_len: int = 32, n_hidden: int = 7) -> SweepConfig:
    """Width sweep at fixed n and seq_len."""
    return SweepConfig(
        name=f'width_s{seq_len}_n{n_hidden}',
        model=ModelConfig(seq_len=seq_len),
        sweep=SweepSpec(
            strategy='grid',
            vary='b',
            values=[1 / 7, 1 / 3, 1, 2, 4, 8],
            solve=None,
            fixed={'n': n_hidden},
        ),
        output=OutputConfig(
            workspace='sessions/width',
            sweep_log='sessions/width_sweep_summary.csv',
        ),
    )


def preset_batch(budget_m: int = 20, n_hidden: int = 3) -> SweepConfig:
    """Batch-size sweep at fixed architecture."""
    return SweepConfig(
        name=f'batch_{budget_m}m_n{n_hidden}',
        training=TrainConfig(batch_size=64),
        sweep=SweepSpec(
            strategy='grid',
            vary='batch_size',
            values=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
            solve='b',
            budget=budget_m * 1_000_000,
            fixed={'n': n_hidden},
        ),
        output=OutputConfig(
            workspace=f'sessions/ratio{budget_m}',
            sweep_log=f'sessions/batch_sweep_{budget_m}m.csv',
        ),
    )
