"""DataPipeline — текст → подготовленные датасеты.

Single entry point: DataPipeline.prepare(config) → (train_ds, val_ds).

Handles: загрузку текста, кэширование битов, разбиение train/val, шум.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import data as _data_mod
from experiment.config import TrainConfig


@dataclass
class DataConfig:
    """What DataPipeline needs to know — extracted from ModelConfig + TrainConfig."""
    seq_len: int
    train_ratio: float
    noise_prob_min: float
    noise_prob_max: float
    noise_std_min: float
    noise_std_max: float


class DataPipeline:
    """Produces (train_dataset, val_dataset) from raw text files.

    Usage:
        pipeline = DataPipeline(texts=rt.texts)
        train_ds, val_ds = pipeline.prepare(data_config)
    """

    def __init__(self, texts: list[str] | None = None,
                 data_dir: str = "data/dataset"):
        self._texts = texts
        self._data_dir = data_dir

    @property
    def texts(self) -> list[str]:
        if self._texts is None:
            self._texts = _data_mod.load_text(self._data_dir, verbose=False)
        return self._texts

    def prepare(self, config: DataConfig) -> tuple[Dataset, Dataset]:
        """Build train/val datasets. Returns (train_ds, val_ds)."""
        train_ds, val_ds = _data_mod.prepare_data(
            self.texts, config.seq_len, config.train_ratio)

        if config.noise_prob_max > 0.0:
            train_ds = _data_mod.NoisyDataset(
                train_ds,
                noise_prob_min=config.noise_prob_min,
                noise_prob_max=config.noise_prob_max,
                noise_std_min=config.noise_std_min,
                noise_std_max=config.noise_std_max,
            )
        return train_ds, val_ds

    @classmethod
    def from_train_config(cls, tc: TrainConfig, seq_len: int,
                          texts: list[str] | None = None) -> DataConfig:
        return DataConfig(
            seq_len=seq_len,
            train_ratio=tc.train_ratio,
            noise_prob_min=tc.noise_prob_min,
            noise_prob_max=tc.noise_prob_max,
            noise_std_min=tc.noise_std_min,
            noise_std_max=tc.noise_std_max,
        )
